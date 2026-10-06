
# VGGT-SLAM Metric Scaling: Updated High-Level Design

## Purpose

This document records the current high-level design for adding metric
scale to VGGT-SLAM in the Go2 mapping pipeline.

The central objective has not changed: **VGGT-SLAM remains the visual
mapping system, while synchronized Go2 odometry supplies metric
information needed to make the resulting geometry useful for robot
planning.**

However, the implementation strategy has evolved from the earlier idea
of estimating one scale for an already globally aligned arbitrary-scale
map. The current pipeline instead **metricizes each ordinary raw VGGT
submap before it enters VGGT-SLAM's normal visual inter-submap
alignment**.

This document describes that current architecture.

------------------------------------------------------------------------

## 1. Overall Goal

VGGT-SLAM remains responsible for:

-   visual reconstruction;
-   keyframe/submap creation;
-   visual alignment between overlapping submaps;
-   residual inter-submap scale estimation;
-   pose-graph optimization;
-   loop closure;
-   production of the globally consistent map.

The Go2 odometry is **not** intended to replace VGGT-SLAM's visual
trajectory or optimization.

Its main role is narrower:

> **Use synchronized local Go2 camera motion as a metric ruler for each
> raw VGGT submap.**

This lets ordinary submaps enter the visual SLAM system with coordinates
that are already approximately expressed in metres, while leaving
VGGT-SLAM free to determine their spatial relationships from visual
evidence.

The planner-facing result should ultimately support physical queries
such as:

``` python
points = map.get_points_within_box(
    center=estimated_robot_position,
    size_m=(1.0, 1.0, 1.0),
)
```

without the planner having to understand arbitrary monocular VGGT units.

------------------------------------------------------------------------

## 2. Why the Earlier "One Global Scale After SLAM" Model Is No Longer the Implementation

An earlier design treated the final VGGT-SLAM map as globally consistent
but arbitrary-scale and proposed estimating a single quantity

$$
s_{\mathrm{to\_metric}}
$$

afterward.

That remains a useful conceptual description of the *physical ambiguity
of a perfectly scale-consistent monocular reconstruction*, but it is no
longer how the current code resolves metric scale.

In practice, raw VGGT submaps are reconstructed independently and can
have substantially different arbitrary local scales. The current
implementation therefore resolves the metric ambiguity **locally, before
submap insertion**:

``` text
raw VGGT submap
        |
        v
Go2-derived local metric scale
        |
        v
metricized VGGT submap
        |
        v
VGGT-SLAM visual inter-submap alignment
        |
        v
pose-graph optimization / loop closure
        |
        v
globally consistent approximately metric map
```

The current design therefore does **not** robustly aggregate per-submap
scale observations into one global post-hoc scale and then multiply the
final map by that value.

Instead, each accepted ordinary submap receives its own
raw-VGGT-to-metric preprocessing scale.

------------------------------------------------------------------------

## 3. Inputs Available for Each Keyframe

Each ordinary VGGT keyframe has two relevant camera poses.

### 3.1 Raw VGGT camera pose

VGGT estimates the camera trajectory within the raw submap in arbitrary
monocular units.

For scale estimation, the relevant quantity is the raw VGGT camera
center

$$
\mathbf{p}_j^V.
$$

### 3.2 Synchronized metric Go2 camera pose

The Go2 camera stream associates each image with synchronized
`/utlidar/robot_odom` body odometry.

Because VGGT estimates the **camera** trajectory rather than the
`base_link` trajectory, the body pose is converted to the physical front
optical-camera pose before metric scale estimation.

The nominal front-camera origin in the Go2 body frame is

$$
{}^B\mathbf{t}_C =
\begin{bmatrix}
0.32715\\
-0.00003\\
0.04297
\end{bmatrix}
\mathrm{\ m}.
$$

The optical-axis convention is:

``` text
optical x/right = -body y
optical y/down  = -body z
optical z/fwd   =  body x
```

For body pose $(R_B^O,\mathbf{p}_B^O)$, the metric camera origin is

$$
\mathbf{p}_C^O =
\mathbf{p}_B^O + R_B^O\,{}^B\mathbf{t}_C.
$$

This camera-origin correction matters particularly during rotation
because the front camera is offset from the body origin.

The `.g2rec` protocol retains the synchronized body odometry; the
camera-pose conversion is performed inside VGGT-SLAM ingestion.

------------------------------------------------------------------------

## 4. Go2 Translation Calibration

Tape-measured experiments showed that `/utlidar/robot_odom`
underestimates translational distance.

The current implementation therefore applies a configurable calibration
factor

$$
c_{\mathrm{odom}},
$$

currently defaulting to

$$
c_{\mathrm{odom}} = 1.20.
$$

It is exposed as:

``` bash
--go2_odom_translation_scale 1.20
```

For one submap, the local calibrated metric trajectory used for fitting
is

$$
\tilde{\mathbf{p}}_j^O =
c_{\mathrm{odom}}
\left(
\mathbf{p}_j^O - \mathbf{p}_0^O
\right).
$$

Subtracting the first camera position removes the arbitrary global
odometry origin. Only relative camera translation within the submap is
needed to determine metric scale.

The calibration factor is still provisional and should be regarded as
part of the metric-accuracy error budget.

------------------------------------------------------------------------

## 5. Per-Submap Metric Scale Estimation

For each ordinary raw VGGT submap, the implementation fits a
proper-rotation similarity transform between:

-   raw VGGT camera centers; and
-   calibrated synchronized Go2 optical-camera positions.

Conceptually,

$$
\tilde{\mathbf{p}}_j^O
\approx
s_i R_i \mathbf{p}_j^V + \mathbf{t}_i.
$$

The fit uses a multi-frame Umeyama/Kabsch-style similarity alignment.

The resulting scalar

$$
\boxed{s_i}
$$

has units

$$
\mathrm{metres/raw\ VGGT\ unit}.
$$

This is the quantity injected into the SLAM pipeline.

The fitted $R_i$ and $\mathbf{t}_i$ are retained only as fit
diagnostics. They are **not** used to place the submap in the Go2
odometry frame.

This distinction is fundamental:

> **Go2 supplies local scale; VGGT-SLAM supplies visual spatial
> alignment.**

------------------------------------------------------------------------

## 6. Applying Metric Scale Before Map Insertion

If the scale estimate is accepted, it is applied to the raw submap
**before the submap is inserted into the map and before `add_edge()`
performs visual inter-submap alignment**.

### Point geometry

Every raw VGGT point is multiplied by $s_i$:

$$
\mathbf{x}^{V,\mathrm{metric}} = s_i \mathbf{x}^V.
$$

### Camera-pose translations

The translation component of every raw VGGT camera pose is multiplied by
the same factor:

$$
\mathbf{t}_j^{V,\mathrm{metric}} = s_i \mathbf{t}_j^V.
$$

Camera rotations remain unchanged.

Conceptually, the implementation performs:

``` python
world_points *= s_i
world_to_cam[:, :3, 3] *= s_i
```

while leaving rotations, intrinsics, confidence values, colors, and
image data untouched.

Scaling both geometry and camera translations is essential to preserve
their internal consistency.

------------------------------------------------------------------------

## 7. What Go2 Does *Not* Inject Into Visual SLAM

The metricization stage deliberately does not use the fitted similarity
rotation or translation to orient or position a raw submap.

In particular, the preprocessing stage does **not**:

-   rotate the submap according to Go2;
-   translate the submap into the Go2 `odom` frame;
-   replace VGGT-SLAM poses with odometry poses;
-   add Go2 pose constraints to the visual pose graph.

Therefore:

$$
\boxed{\text{Go2 contributes local metric scale}}
$$

while

$$
\boxed{\text{VGGT-SLAM remains responsible for visual map alignment}}
$$

This prevents raw odometric position/orientation drift from directly
determining the internal visual map geometry.

------------------------------------------------------------------------

## 8. VGGT-SLAM Still Estimates Inter-Submap Scale

VGGT-SLAM's existing visual alignment remains active.

When a new metricized submap overlaps the previous submap, `add_edge()`
still estimates the normal visual scale factor from overlapping
geometry.

The interpretation of this scale factor has changed.

### Without Go2 metricization

The visual scale factor may need to compensate for large arbitrary
monocular scale differences between independently reconstructed raw
submaps.

### With Go2 metricization

Both neighboring submaps should already be approximately metric, so the
visual scale factor becomes a **residual correction**.

Ideally,

$$
s_{\mathrm{visual,residual}} \approx 1.
$$

This is useful rather than redundant: the visual overlap acts as an
independent consistency check and lets VGGT-SLAM reconcile remaining
disagreement between locally metricized submaps.

The final optimized map is therefore not simply a collection of
independently scaled Go2 submaps. It is still the result of VGGT-SLAM's
visual alignment and graph optimization.

------------------------------------------------------------------------

## 9. Weak-Motion and Degenerate Submaps

A metric scale cannot be estimated reliably when the local camera
trajectory provides insufficient translational information.

The current implementation requires:

-   at least 3 matched camera positions;
-   calibrated Go2 path length of at least 0.20 m by default;
-   non-degenerate raw VGGT camera-position variance;
-   a finite, positive proper-rotation similarity fit.

If these checks fail, the metricization estimate is rejected.

The preprocessing scale then remains

$$
s_i = 1,
$$

so the submap enters the map at its raw VGGT scale.

VGGT-SLAM's ordinary visual inter-submap scale estimation can
subsequently reconcile that submap with already metricized neighboring
geometry.

This fallback is important: weak local Go2 scale observability does not
prevent the submap from participating in the visual SLAM map.

------------------------------------------------------------------------

## 10. Per-Submap Diagnostics

The current pipeline retains raw VGGT camera centers and records
metricization diagnostics for each ordinary submap.

With:

``` bash
--metric_submap_diagnostics_path <file.csv>
```

the diagnostics include:

``` text
submap_id
num_frames
status
odom_translation_scale
metric_path_m
metric_displacement_m
raw_path_units
raw_displacement_units
raw_to_metric_scale_m_per_unit
fit_rmse_m
fit_max_error_m
applied_scale_m_per_unit
incoming_visual_scale
```

The key quantities are:

### `raw_to_metric_scale_m_per_unit`

The fitted local scale $s_i$, in metres per raw VGGT unit.

### `fit_rmse_m` / `fit_max_error_m`

How well one similarity transform explains the raw VGGT trajectory
versus the calibrated local Go2 camera trajectory.

### `incoming_visual_scale`

The residual scale subsequently estimated by VGGT-SLAM from visual
overlap.

For well-observed, successfully metricized neighboring submaps, this
value should generally remain reasonably close to 1.

These diagnostics are the right place to evaluate whether the local
metricization and the visual reconstruction agree.

------------------------------------------------------------------------

## 11. Meaning of the Internal Optimized Map

With `--metricize_submaps_from_go2` enabled, accepted ordinary submaps
enter VGGT-SLAM with their coordinate scale approximately expressed in
metres.

After visual inter-submap alignment and graph optimization, the
resulting global map therefore has an **intended metric coordinate
unit**:

$$
\boxed{1\ \text{map coordinate unit} \approx 1\ \text{metre}}
$$

This should be described as **metricized** or **approximately metric**,
not assumed to be metrically exact.

Potential errors include:

-   uncertainty in `--go2_odom_translation_scale`;
-   local Go2 odometry error;
-   temporal synchronization error;
-   camera-extrinsic error;
-   VGGT reconstruction error;
-   per-submap scale-fit error;
-   residual visual scale corrections;
-   graph optimization effects;
-   weak-motion submaps.

The distinction remains:

> **Metric units do not imply perfect metric accuracy.**

------------------------------------------------------------------------

## 12. Metric Scale and Coordinate-Frame Alignment Are Separate

The current code now explicitly separates two questions:

1.  **What physical scale does the VGGT geometry have?**
2.  **In which coordinate frame should the planner consume that
    geometry?**

The per-submap metricization stage answers the first question.

It does **not** place the map in Go2 `odom`.

After graph optimization, the map can separately be rigidly aligned to
Go2 `odom`.

This rigid frame transform contains only rotation and translation:

$$
\mathbf{p}^{O} =
R_{OV}\mathbf{p}^{V_m} + \mathbf{t}_{OV},
$$

where $\mathbf{p}^{V_m}$ is already metricized VGGT geometry.

No additional scale is fitted at this stage.

------------------------------------------------------------------------

## 13. Odom-Frame Alignment

For the odom-aligned export/planning map, the current implementation
constructs a fixed transform

``` text
T_odom_vggt
```

from the **first common synchronized optical-camera pose** between:

-   the metric Go2 camera trajectory; and
-   the optimized metricized VGGT camera trajectory.

Conceptually,

$$
T_{\mathrm{odom}\leftarrow\mathrm{VGGT}} =
T_{\mathrm{odom}\leftarrow C_0}
\left(
T_{\mathrm{VGGT}\leftarrow C_0}
\right)^{-1}.
$$

This transform is a rigid SE(3) transform.

It establishes:

-   a common origin;
-   a common orientation;
-   the Go2 `odom` planner-facing frame.

It does **not** change metric scale.

The transform is deliberately anchored once rather than repeatedly
refitted to the full Go2 trajectory. Thus later Go2 drift is not used to
warp or continually realign the visual map.

Full-trajectory Go2-versus-VGGT residuals can still be computed as
diagnostics.

------------------------------------------------------------------------

## 14. Planner-Facing Map

The current architecture supports an explicit planner-facing map in Go2
`odom`.

With:

``` bash
--planning_map_output_path <map.ply>
```

and Go2 metricization enabled, the system maintains a complete
graph-optimized map snapshot after ordinary-submap updates.

The sequence is:

``` text
raw camera frames + synchronized Go2 odometry
                    |
                    v
             VGGT raw submap
                    |
                    v
       local Go2 scale estimation
                    |
                    v
          metricized raw submap
                    |
                    v
       VGGT-SLAM visual alignment
                    |
                    v
         graph-optimized metric map
                    |
                    v
 fixed first-frame rigid T_odom_vggt
                    |
                    v
       planner-facing map in odom
```

This gives the planner geometry whose:

-   coordinate units are intended to be metres; and
-   coordinate frame is Go2 `odom`.

That is a stronger and more concrete planner-facing contract than the
earlier design, where the final frame convention was left open.

------------------------------------------------------------------------

## 15. Current Separation of Responsibilities

The architecture can be summarized as follows.

### Go2 synchronized odometry

Provides:

-   metric local camera translation;
-   the provisional translation calibration source;
-   the optical-camera pose used for local scale estimation;
-   the reference pose used to rigidly anchor the final map to `odom`.

It does **not** provide the ongoing global map trajectory or visual
graph constraints.

### VGGT

Provides:

-   raw monocular submap reconstruction;
-   raw camera geometry within each submap.

### Metricization layer

Provides:

-   one accepted/rejected raw-to-metric scale estimate per ordinary
    submap;
-   scaling of raw point geometry and pose translations;
-   diagnostics of local scale observability and fit quality.

### VGGT-SLAM

Provides:

-   visual inter-submap alignment;
-   residual scale correction;
-   pose-graph optimization;
-   loop closure;
-   globally consistent optimized geometry.

### Odom-frame export/planning-map layer

Provides:

-   a fixed rigid `T_odom_vggt`;
-   conversion of the already metric optimized map into Go2 `odom`;
-   planner-facing map snapshots.

------------------------------------------------------------------------

## 16. Guiding Principles

Future work should follow these principles:

1.  **Do not replace VGGT-SLAM with Go2 odometry.**\
    VGGT-SLAM remains responsible for visual spatial consistency,
    inter-submap alignment, graph optimization, and loop closure.

2.  **Use Go2 as a local metric ruler.**\
    Synchronized Go2 camera translation resolves the raw monocular scale
    of each sufficiently observable ordinary submap.

3.  **Compare camera motion with camera motion.**\
    Convert synchronized body odometry to the physical front
    optical-camera pose using the known Go2 camera extrinsic.

4.  **Metricize before visual inter-submap alignment.**\
    Accepted raw VGGT points and pose translations are scaled before map
    insertion, so VGGT-SLAM sees approximately metric submaps.

5.  **Keep Go2 rotation/translation out of the submap metricization
    transform.**\
    The fitted Sim(3) rotation and translation are diagnostics, not
    visual-map placement constraints.

6.  **Retain VGGT-SLAM's visual scale estimation.**\
    After metricization it becomes a residual visual correction and an
    important consistency signal.

7.  **Allow weak-motion submaps to fall back to visual alignment.**\
    Do not force unreliable metric scale estimates.

8.  **Keep metric scale separate from coordinate-frame alignment.**\
    Metricization establishes metres; the later rigid `T_odom_vggt`
    establishes the planner-facing `odom` frame.

9.  **Do not continually fit the visual map to drifting odometry.**\
    The current odom-frame alignment is anchored from the first common
    optical-camera pose and remains rigid.

10. **Treat the map as approximately metric until accuracy is
    quantified.**\
    The intended unit is metres, but physical accuracy must be measured
    empirically.

11. **The planner-facing contract is the ultimate criterion.**\
    The planner should receive geometry in physical units and a
    well-defined robot-relative/world frame without needing to know
    VGGT's monocular-scale internals.

------------------------------------------------------------------------

## 17. Current Representative Configuration

A representative metricized run uses:

``` bash
python main_realtime.py \
    --camera go2 \
    --go2_host 127.0.0.1 \
    --go2_port 5432 \
    --go2_exit_on_disconnect \
    --metricize_submaps_from_go2 \
    --go2_odom_translation_scale 1.20 \
    --metric_submap_diagnostics_path new_office_trajectory_5_metric_submaps.csv \
    --metric_trajectory_path new_office_trajectory_5_metricized_metric.txt \
    --vggt_trajectory_path new_office_trajectory_5_metricized_vggt.txt \
    --vggt_submap_trajectory_path new_office_trajectory_5_metricized_vggt_submaps.txt \
    --vggt_scale_diagnostics_path new_office_trajectory_5_metricized_scale_diagnostics.txt \
    --map_output_path new_office_trajectory_5_metricized.ply
```

For a planner-facing map in Go2 `odom`, the current code additionally
supports:

``` bash
--planning_map_output_path <planning_map.ply>
```

and a final odom-aligned export via:

``` bash
--odom_aligned_map_output_path <odom_aligned_map.ply>
```

The metricization feature remains opt-in through:

``` bash
--metricize_submaps_from_go2
```

Without it, VGGT-SLAM retains its original arbitrary-scale behavior.

------------------------------------------------------------------------

## 18. Scope Check for Future Changes

When considering a proposed change, ask:

> Does this improve the reliability, accuracy, or planner usability of
> the pipeline\
> **raw VGGT submap → local metricization → visual SLAM optimization →
> metric map → rigid odom-frame alignment**?

Changes are likely in scope if they improve:

-   Go2 translation calibration;
-   camera-pose synchronization/extrinsics;
-   local scale observability;
-   scale-estimation robustness;
-   residual-scale diagnostics;
-   metric-accuracy validation;
-   stable planner-facing map publication.

Changes are probably outside the current objective if they
unnecessarily:

-   replace VGGT-SLAM's visual trajectory with odometry;
-   inject raw Go2 pose drift into the graph;
-   redesign unrelated VGGT-SLAM internals;
-   continually deform the visual map to match Go2 odometry.

The current central architecture is:

$$
\boxed{
\text{raw arbitrary-scale VGGT submaps}
\rightarrow
\text{Go2 local metric scale}
\rightarrow
\text{VGGT-SLAM visual optimization}
\rightarrow
\text{approximately metric global map}
\rightarrow
\text{rigid alignment to Go2 odom}
\rightarrow
\text{planner}
}
$$

