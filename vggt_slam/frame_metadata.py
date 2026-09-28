"""Immutable source metadata associated with selected SLAM keyframes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricCameraPose:
    """Authoritative derived metric transform ``T_odom_camera``.

    For Go2 data this is derived from synchronized ``T_odom_body`` packet
    odometry plus the fixed nominal front-camera extrinsic at ingestion.
    """

    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class ImuSample:
    """One raw IMU measurement as transported by the Go2 bridge."""

    timestamp_ns: int
    orientation_xyzw: tuple[float, float, float, float]
    angular_velocity_xyz: tuple[float, float, float]
    linear_acceleration_xyz: tuple[float, float, float]


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
class VGGTSubmapTrajectorySample:
    """One optimized VGGT camera-pose occurrence in an ordinary submap.

    This intentionally preserves duplicate source timestamps for overlap
    keyframes: each ordinary submap contributes its complete local window.
    """

    timestamp_ns: int
    frame_id: int | float
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    submap_id: int
    frame_index: int


@dataclass(frozen=True)
class VGGTScaleDiagnosticSample:
    """One ordinary-submap occurrence for raw/final/metric scale analysis.

    ``raw_vggt_position_xyz`` is the local VGGT camera center before graph
    alignment; ``optimized_vggt_position_xyz`` is the final graph-optimized
    camera center; and ``metric_position_xyz`` is the synchronized Go2
    optical-camera position in metric odom coordinates.  The optional
    ``incoming_scale_factor`` is the existing current-raw to previous-raw
    point-cloud alignment factor, not a metric scale.
    """

    timestamp_ns: int
    submap_id: int
    frame_index: int
    frame_id: int | float
    raw_vggt_position_xyz: tuple[float, float, float]
    optimized_vggt_position_xyz: tuple[float, float, float]
    metric_position_xyz: tuple[float, float, float]
    incoming_scale_factor: float | None


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
    # This is the source-camera interval, not necessarily the interval since
    # the prior selected VGGT keyframe; rejected frames are not aggregated.
    imu_samples: tuple[ImuSample, ...] = ()
