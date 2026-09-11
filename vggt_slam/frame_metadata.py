"""Immutable source metadata associated with selected SLAM keyframes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricCameraPose:
    """Authoritative upstream metric transform ``T_odom_camera``."""

    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


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
