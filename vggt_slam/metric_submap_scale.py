"""Go2-derived local metric scale estimation for ordinary VGGT submaps.

Only the scalar returned here is used by SLAM.  The fitted rotation and
translation are deliberately retained solely as fit diagnostics.
"""

from dataclasses import dataclass

import numpy as np


DEFAULT_MIN_METRIC_PATH_M = 0.20


@dataclass(frozen=True)
class SimilarityTransformEstimate:
    """A proper, positive-scale transform satisfying target ~= s R source + t."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float
    max_error: float
    errors: np.ndarray


def fit_similarity_transform(source_points, target_points) -> SimilarityTransformEstimate:
    """Fit a full Sim(3) with Umeyama/Kabsch SVD.

    This intentionally has no Go2 calibration or acceptance policy so it can
    also be used by observational map branches.
    """
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape:
        raise ValueError("source and target points must be matching N x 3 arrays")
    if len(source) < 3:
        raise ValueError("similarity fitting requires at least 3 points")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and target points must be finite")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    source_variance = float(np.mean(np.sum(source_centered ** 2, axis=1)))
    if source_variance <= 1e-12:
        raise ValueError("source point variance is near zero")
    try:
        u, singular_values, vt = np.linalg.svd(target_centered.T @ source_centered / len(source))
    except np.linalg.LinAlgError as exc:
        raise ValueError("similarity SVD failed") from exc
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 0 or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("similarity fit did not yield a positive proper-rotation scale")
    translation = target_mean - scale * rotation @ source_mean
    errors = np.linalg.norm((scale * (rotation @ source.T)).T + translation - target, axis=1)
    return SimilarityTransformEstimate(
        scale, rotation, translation, float(np.sqrt(np.mean(errors ** 2))),
        float(errors.max()), errors,
    )


@dataclass(frozen=True)
class MetricSubmapScaleEstimate:
    scale_m_per_raw_unit: float | None
    rotation: np.ndarray | None
    translation: np.ndarray | None
    rmse_m: float | None
    max_error_m: float | None
    num_frames: int
    metric_path_length_m: float
    metric_displacement_m: float
    raw_path_length: float
    raw_displacement: float
    status: str
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "applied"


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _rejected(raw, metric, status, reason):
    return MetricSubmapScaleEstimate(
        None, None, None, None, None, len(raw), _path_length(metric),
        float(np.linalg.norm(metric[-1] - metric[0])) if len(metric) else 0.0,
        _path_length(raw), float(np.linalg.norm(raw[-1] - raw[0])) if len(raw) else 0.0,
        status, reason,
    )


def estimate_metric_submap_scale(
    raw_vggt_centers,
    metric_camera_positions,
    odom_translation_scale,
    min_metric_path_m: float = DEFAULT_MIN_METRIC_PATH_M,
) -> MetricSubmapScaleEstimate:
    """Fit calibrated Go2 positions ~= scale * R * raw VGGT positions + t.

    Expected weak-motion cases return a rejected result; malformed input raises
    ``ValueError`` so callers cannot accidentally pair mismatched keyframes.
    """
    raw = np.asarray(raw_vggt_centers, dtype=float)
    odom = np.asarray(metric_camera_positions, dtype=float)
    if raw.ndim != 2 or raw.shape[1:] != (3,) or odom.shape != raw.shape:
        raise ValueError("raw VGGT centers and metric camera positions must be matching N x 3 arrays")
    if not np.isfinite(raw).all() or not np.isfinite(odom).all():
        raise ValueError("raw VGGT centers and metric camera positions must be finite")
    if not np.isfinite(odom_translation_scale) or odom_translation_scale <= 0:
        raise ValueError("odom translation scale must be finite and positive")
    if not np.isfinite(min_metric_path_m) or min_metric_path_m < 0:
        raise ValueError("minimum metric path must be finite and non-negative")

    metric = float(odom_translation_scale) * (odom - odom[0]) if len(odom) else odom.copy()
    if len(raw) < 3:
        return _rejected(raw, metric, "rejected_insufficient_frames", "requires at least 3 matched positions")
    metric_path = _path_length(metric)
    if metric_path < min_metric_path_m:
        return _rejected(raw, metric, "rejected_low_motion", f"metric path {metric_path:.6g} m is below {min_metric_path_m:.6g} m")

    try:
        fitted = fit_similarity_transform(raw, metric)
    except ValueError as exc:
        if "variance" in str(exc):
            status = "rejected_degenerate_raw"
        elif "SVD" in str(exc):
            status = "rejected_svd_failure"
        else:
            status = "rejected_invalid_fit"
        return _rejected(raw, metric, status, str(exc))
    return MetricSubmapScaleEstimate(
        fitted.scale, fitted.rotation, fitted.translation, fitted.rmse, fitted.max_error,
        len(raw), metric_path, float(np.linalg.norm(metric[-1] - metric[0])),
        _path_length(raw), float(np.linalg.norm(raw[-1] - raw[0])), "applied",
    )


def apply_metric_submap_scale(world_points, world_to_cam, scale):
    """Return scaled geometry; rotations/intrinsics/confidences are untouched."""
    scale = float(scale)
    poses = np.asarray(world_to_cam)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("metric submap scale must be finite and positive")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("world_to_cam must have shape N x 4 x 4")
    scaled_points = np.asarray(world_points).copy() * scale
    scaled_poses = poses.copy()
    scaled_poses[:, :3, 3] *= scale
    return scaled_points, scaled_poses
