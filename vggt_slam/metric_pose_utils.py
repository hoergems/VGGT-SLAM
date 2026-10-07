"""Go2 body-pose to front optical-camera-pose conversion utilities.

The Go2 protocol-v3 TCP stream (``go2_camera_bridge``) carries the Jetson
bridge's interpolated ``/utlidar/robot_odom`` pose, i.e. ``T_odom_body``.
This module derives the physical front optical-camera pose using the nominal
fixed Go2 body-to-camera extrinsic. Packet data remains the authoritative body
odometry; the derived ``MetricCameraPose`` is ``T_odom_camera``.
"""

import numpy as np

from vggt_slam.frame_metadata import MetricCameraPose


# Maps body-frame vectors to optical-frame coordinates:
# optical x/right = -body y, optical y/down = -body z, optical z/forward = body x.
R_BODY_TO_OPTICAL = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])

# Nominal front-camera origin expressed in the Go2 body frame, in metres.
GO2_FRONT_CAMERA_TRANSLATION_BODY_M = np.array(
    [0.32715, -0.00003, 0.04297], dtype=float
)


def _rotation_from_xyzw(quaternion_xyzw):
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("metric quaternion must be finite xyzw")
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
        raise ValueError("metric quaternion must have non-zero norm")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quaternion_xyzw_from_rotation(rotation):
    """Return a normalized xyzw quaternion for a proper rotation matrix."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")

    # This branch-based form is stable near 180-degree rotations and avoids a
    # runtime dependency solely for matrix-to-quaternion conversion.
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        quaternion = np.array([
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
        ])
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        quaternion = np.array([
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
        ])
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        quaternion = np.array([
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    return quaternion / np.linalg.norm(quaternion)


def go2_body_pose_to_metric_camera_pose(position_xyz, quaternion_xyzw):
    """Convert synchronized Go2 ``T_odom_body`` into ``T_odom_camera``.

    The returned pose represents the physical front optical camera. Its origin
    includes the nominal body-frame mounting offset and its orientation uses
    the established body-to-optical axis convention.
    """
    position = np.asarray(position_xyz, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("metric position must be a finite xyz vector")

    rotation_odom_body = _rotation_from_xyzw(quaternion_xyzw)
    rotation_odom_camera = rotation_odom_body @ R_BODY_TO_OPTICAL.T
    position_odom_camera = (
        position + rotation_odom_body @ GO2_FRONT_CAMERA_TRANSLATION_BODY_M
    )
    quaternion_odom_camera = _quaternion_xyzw_from_rotation(rotation_odom_camera)
    return MetricCameraPose(
        position_xyz=tuple(float(value) for value in position_odom_camera),
        quaternion_xyzw=tuple(float(value) for value in quaternion_odom_camera),
    )


def metric_pose_to_optical_camera_to_odom(metric_pose):
    """Return ``R_odom_camera`` represented by a camera-oriented metric pose."""
    return _rotation_from_xyzw(metric_pose.quaternion_xyzw)
