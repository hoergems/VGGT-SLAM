"""Rigid first-camera alignment between the optimized VGGT and Go2 odom frames."""

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
from scipy.spatial.transform import Rotation

from vggt_slam.ply_map import PLYCameraPose, PLYObjectOBB


POSITION_TOLERANCE_M = 1e-6
ORIENTATION_TOLERANCE_DEG = 1e-4


@dataclass(frozen=True)
class FirstFrameOdomAlignment:
    """The fixed ``T_odom_vggt`` derived from one synchronized camera pose."""

    transform_odom_vggt: np.ndarray
    metric_sample: object
    vggt_sample: object


def _validate_rotation(rotation_matrix, name="rotation"):
    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    if rotation_matrix.shape != (3, 3) or not np.isfinite(rotation_matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=1e-7):
        raise ValueError(f"{name} is not orthonormal")
    if not np.isclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-7):
        raise ValueError(f"{name} is not a proper rotation")
    return rotation_matrix


def validate_se3(transform, name="transform"):
    """Validate and return a finite camera-to-world rigid transform."""
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{name} must have homogeneous final row [0, 0, 0, 1]")
    _validate_rotation(transform[:3, :3], f"{name} rotation")
    return transform


def pose_matrix_from_position_quaternion(position_xyz, quaternion_xyzw):
    """Build camera-to-world ``T_world_camera`` from an xyzw quaternion."""
    position = np.asarray(position_xyz, dtype=float)
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position_xyz must be three finite values")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) == 0:
        raise ValueError("quaternion_xyzw must be four finite non-zero values")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = position
    return validate_se3(transform, "pose")


def invert_se3(transform):
    transform = validate_se3(transform)
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def transform_points_se3(points, transform):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    transform = validate_se3(transform)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def transform_camera_pose_se3(pose, transform):
    """Transform a PLY camera pose while retaining its source frame id."""
    transform = validate_se3(transform)
    pose_matrix = pose_matrix_from_position_quaternion(pose.position_xyz, pose.quaternion_xyzw)
    transformed = transform @ pose_matrix
    validate_se3(transformed, "transformed camera pose")
    return PLYCameraPose(
        frame_id=pose.frame_id,
        position_xyz=tuple(float(value) for value in transformed[:3, 3]),
        quaternion_xyzw=tuple(float(value) for value in Rotation.from_matrix(transformed[:3, :3]).as_quat()),
    )


def transform_object_obb_se3(obb, transform):
    """Rigidly transform OBB pose, preserving its dimensions and identity."""
    transform = validate_se3(transform)
    center = np.asarray(obb.center_xyz, dtype=float)
    rotation = _validate_rotation(np.asarray(obb.rotation_matrix, dtype=float).reshape(3, 3), "object OBB rotation")
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("object OBB center must be three finite values")
    transformed_rotation = transform[:3, :3] @ rotation
    return PLYObjectOBB(
        object_id=obb.object_id,
        center_xyz=tuple(float(value) for value in transform[:3, :3] @ center + transform[:3, 3]),
        extent_xyz=obb.extent_xyz,
        rotation_matrix=tuple(float(value) for value in transformed_rotation.ravel(order="C")),
    )


def _samples_by_timestamp(samples, name):
    if not samples:
        raise ValueError(f"cannot align odom map: {name} trajectory is empty")
    result = {}
    for sample in samples:
        if type(sample.timestamp_ns) is not int:
            raise TypeError(f"{name} trajectory timestamp_ns must be an exact Python int")
        if sample.timestamp_ns in result:
            raise ValueError(f"{name} trajectory has duplicate timestamp {sample.timestamp_ns}")
        result[sample.timestamp_ns] = sample
    return result


def build_first_frame_odom_alignment(metric_samples, vggt_samples):
    """Build the fixed first-common-camera ``T_odom_vggt`` alignment."""
    metric_by_timestamp = _samples_by_timestamp(metric_samples, "metric")
    vggt_by_timestamp = _samples_by_timestamp(vggt_samples, "VGGT")
    common_timestamps = sorted(metric_by_timestamp.keys() & vggt_by_timestamp.keys())
    if not common_timestamps:
        raise ValueError("cannot align odom map: metric and VGGT trajectories have no common timestamp")
    timestamp_ns = common_timestamps[0]
    metric_sample = metric_by_timestamp[timestamp_ns]
    vggt_sample = vggt_by_timestamp[timestamp_ns]
    metric_pose = pose_matrix_from_position_quaternion(metric_sample.position_xyz, metric_sample.quaternion_xyzw)
    vggt_pose = pose_matrix_from_position_quaternion(vggt_sample.position_xyz, vggt_sample.quaternion_xyzw)
    transform = validate_se3(metric_pose @ invert_se3(vggt_pose), "T_odom_vggt")
    alignment = FirstFrameOdomAlignment(transform, metric_sample, vggt_sample)
    position_error, orientation_error = pose_residual(transform @ vggt_pose, metric_pose)
    if position_error > POSITION_TOLERANCE_M or orientation_error > ORIENTATION_TOLERANCE_DEG:
        raise ValueError(
            "first-frame odom alignment residual exceeds tolerance: "
            f"position={position_error:.9g}m orientation={orientation_error:.9g}deg"
        )
    return alignment


def rotation_angle_deg(rotation_matrix):
    rotation_matrix = _validate_rotation(rotation_matrix)
    cosine = np.clip((np.trace(rotation_matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pose_residual(predicted_pose, target_pose):
    predicted_pose = validate_se3(predicted_pose, "predicted pose")
    target_pose = validate_se3(target_pose, "target pose")
    position_error = float(np.linalg.norm(predicted_pose[:3, 3] - target_pose[:3, 3]))
    orientation_error = rotation_angle_deg(target_pose[:3, :3].T @ predicted_pose[:3, :3])
    return position_error, orientation_error


def trajectory_alignment_diagnostics(metric_samples, vggt_samples, transform_odom_vggt):
    """Compare all common poses; this deliberately never refits the transform."""
    metric_by_timestamp = _samples_by_timestamp(metric_samples, "metric")
    vggt_by_timestamp = _samples_by_timestamp(vggt_samples, "VGGT")
    transform_odom_vggt = validate_se3(transform_odom_vggt, "T_odom_vggt")
    rows = []
    for timestamp_ns in sorted(metric_by_timestamp.keys() & vggt_by_timestamp.keys()):
        metric = metric_by_timestamp[timestamp_ns]
        vggt = vggt_by_timestamp[timestamp_ns]
        predicted = transform_odom_vggt @ pose_matrix_from_position_quaternion(vggt.position_xyz, vggt.quaternion_xyzw)
        target = pose_matrix_from_position_quaternion(metric.position_xyz, metric.quaternion_xyzw)
        position_error, orientation_error = pose_residual(predicted, target)
        rows.append({
            "timestamp_ns": timestamp_ns,
            "frame_id": getattr(metric, "frame_id", vggt.frame_id),
            "vggt_x": vggt.position_xyz[0], "vggt_y": vggt.position_xyz[1], "vggt_z": vggt.position_xyz[2],
            "pred_odom_x": predicted[0, 3], "pred_odom_y": predicted[1, 3], "pred_odom_z": predicted[2, 3],
            "go2_odom_x": metric.position_xyz[0], "go2_odom_y": metric.position_xyz[1], "go2_odom_z": metric.position_xyz[2],
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error,
        })
    return rows


def summarize_trajectory_alignment(rows):
    if not rows:
        raise ValueError("cannot summarize odom alignment: no common trajectory samples")
    return {
        "count": len(rows),
        "position_error_m": _summary([row["position_error_m"] for row in rows]),
        "orientation_error_deg": _summary([row["orientation_error_deg"] for row in rows]),
    }


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {key: float(value) for key, value in {
        "mean": np.mean(values), "median": np.median(values),
        "p95": np.percentile(values, 95), "max": np.max(values),
    }.items()}


def alignment_csv_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_alignment.csv")


def write_trajectory_alignment_csv(output_path, rows):
    output_path = alignment_csv_path(output_path)
    fieldnames = [
        "timestamp_ns", "frame_id", "vggt_x", "vggt_y", "vggt_z",
        "pred_odom_x", "pred_odom_y", "pred_odom_z", "go2_odom_x", "go2_odom_y", "go2_odom_z",
        "position_error_m", "orientation_error_deg",
    ]
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def verify_rigid_point_transform(points_before, points_after):
    """Cheap deterministic production invariant for a rigid point transform."""
    points_before = np.asarray(points_before, dtype=float)
    points_after = np.asarray(points_after, dtype=float)
    if points_before.shape != points_after.shape or points_before.ndim != 2 or points_before.shape[1] != 3:
        raise ValueError("rigid-transform points must be matching Nx3 arrays")
    if not np.isfinite(points_after).all():
        raise ValueError("transformed point geometry contains non-finite values")
    if len(points_before) < 2:
        return
    indices = np.unique(np.linspace(0, len(points_before) - 1, min(8, len(points_before)), dtype=int))
    for index, first in enumerate(indices):
        for second in indices[index + 1:]:
            before = np.linalg.norm(points_before[first] - points_before[second])
            after = np.linalg.norm(points_after[first] - points_after[second])
            if not np.isclose(before, after, rtol=1e-9, atol=1e-9):
                raise ValueError("rigid-transform distance invariant failed")
