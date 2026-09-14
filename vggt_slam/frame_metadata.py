"""Immutable source metadata associated with selected SLAM keyframes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricCameraPose:
    """Authoritative upstream metric transform ``T_odom_camera``."""

    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class MetricTrajectorySample:
    """One canonical Go2 metric camera sample in the ``odom`` frame.

    ``frame_id`` remains available only for diagnostics and attribution; the
    exact source-camera ``timestamp_ns`` is the trajectory identity.
    """

    timestamp_ns: int
    frame_id: int | float
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class VGGTTrajectorySample:
    """One canonical optimized VGGT camera sample.

    Positions live in VGGT's optimized reconstruction frame, not in metric
    ``odom``.  The quaternion is the decomposed VGGT camera-to-world
    orientation and is retained for diagnostics only.
    """

    timestamp_ns: int
    frame_id: int | float
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    submap_id: int
    frame_index: int


@dataclass(frozen=True)
class SubmapTrajectoryDiagnostic:
    """Diagnostic comparison of one local VGGT window against Go2 odometry."""

    submap_id: int
    num_frames: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    incoming_stitch_scale: float | None
    local_vggt_endpoint_displacement: float
    metric_endpoint_displacement_m: float
    endpoint_implied_scale_m_per_vggt: float | None
    local_umeyama_scale_m_per_vggt: float | None
    local_umeyama_rmse_m: float | None
    local_umeyama_median_error_m: float | None
    local_umeyama_max_error_m: float | None
    global_vggt_endpoint_displacement: float
    global_implied_scale_m_per_vggt: float | None


@dataclass(frozen=True)
class KeyframeRecord:
    """Identity, storage location, and optional upstream metadata of a keyframe.

    ``frame_id`` is the existing/local VGGT realtime identity. ``timestamp_ns``
    is the exact external source-camera identity and is intentionally separate.
    """

    image_path: str
    frame_id: int | float
    timestamp_ns: int | None = None
    metric_pose: MetricCameraPose | None = None
    sequence_id: int | None = None
