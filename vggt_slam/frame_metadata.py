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
class KeyframeImageQuality:
    """One image-sharpness measurement for a keyframe appearance in a submap."""

    submap_id: int
    frame_index: int
    frame_id: int | float
    timestamp_ns: int
    image_path: str
    width_px: int
    height_px: int
    laplacian_variance: float


@dataclass(frozen=True)
class SubmapImageQualityDiagnostic:
    """Continuous sharpness statistics paired with existing local pose errors."""

    submap_id: int
    num_frames: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    mean_laplacian_variance: float
    median_laplacian_variance: float
    min_laplacian_variance: float
    max_laplacian_variance: float
    std_laplacian_variance: float
    local_umeyama_rmse_m: float | None
    local_umeyama_median_error_m: float | None
    local_umeyama_max_error_m: float | None
    min_sharpness_frame_index: int
    min_sharpness_timestamp_ns: int
    num_below_blur_threshold: int | None = None
    fraction_below_blur_threshold: float | None = None


@dataclass(frozen=True)
class SubmapMetricAlignment:
    """Independent local-VGGT-to-Go2-odom similarity fit for one submap."""

    submap_id: int
    num_frames: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    scale_m_per_vggt: float
    rotation_matrix: tuple[tuple[float, float, float], ...]
    translation_m: tuple[float, float, float]
    rmse_m: float
    median_error_m: float
    max_error_m: float


@dataclass(frozen=True)
class CameraAxisAlignment:
    """Fixed ``C_vggtcam_from_go2cam`` rotation, from Go2-camera to VGGT-camera coordinates."""
    rotation_matrix: tuple[tuple[float, float, float], ...]
    quaternion_xyzw: tuple[float, float, float, float]
    num_samples: int
    candidate_median_deviation_deg: float
    candidate_mean_deviation_deg: float
    candidate_max_deviation_deg: float
    candidate_rmse_deviation_deg: float


@dataclass(frozen=True)
class KeyframeOrientationDiagnostic:
    submap_id: int
    frame_index: int
    frame_id: int | float
    timestamp_ns: int
    raw_orientation_error_deg: float
    global_axis_corrected_error_deg: float
    loso_axis_corrected_error_deg: float | None
    go2_step_rotation_deg: float | None
    vggt_step_rotation_deg: float | None
    step_rotation_magnitude_error_deg: float | None


@dataclass(frozen=True)
class SubmapOrientationDiagnostic:
    submap_id: int
    num_frames: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    position_rmse_m: float
    raw_orientation_rmse_deg: float
    raw_orientation_median_deg: float
    raw_orientation_max_deg: float
    global_axis_orientation_rmse_deg: float
    global_axis_orientation_median_deg: float
    global_axis_orientation_max_deg: float
    loso_orientation_rmse_deg: float | None
    loso_orientation_median_deg: float | None
    loso_orientation_max_deg: float | None
    relative_step_rotation_mae_deg: float | None
    relative_step_rotation_rmse_deg: float | None
    relative_step_rotation_max_error_deg: float | None


@dataclass(frozen=True)
class SubmapMotionStep:
    """One authoritative Go2 camera-motion transition in a local submap."""

    submap_id: int
    from_frame_index: int
    to_frame_index: int
    from_timestamp_ns: int
    to_timestamp_ns: int
    dt_s: float
    translation_m: float
    rotation_deg: float
    cumulative_translation_m: float
    cumulative_rotation_deg: float
    aligned_vggt_residual_m: float | None


@dataclass(frozen=True)
class SubmapMotionDiagnostic:
    """Motion geometry and local aligned-VGGT error for one ordinary submap."""

    submap_id: int
    num_frames: int
    num_steps: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    duration_s: float
    metric_path_length_m: float
    metric_endpoint_displacement_m: float
    path_efficiency: float | None
    mean_step_translation_m: float
    median_step_translation_m: float
    max_step_translation_m: float
    total_rotation_deg: float
    mean_step_rotation_deg: float
    median_step_rotation_deg: float
    max_step_rotation_deg: float
    translation_per_rotation_m_per_rad: float | None
    local_umeyama_scale_m_per_vggt: float
    local_umeyama_rmse_m: float
    local_umeyama_median_error_m: float
    local_umeyama_max_error_m: float
    aligned_vggt_xy_rmse_m: float
    aligned_vggt_xy_max_error_m: float
    window_policy: str | None = None
    window_trigger_reason: str | None = None
    window_keyframes: int | None = None
    window_trigger_translation_m: float | None = None
    window_trigger_rotation_deg: float | None = None


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
