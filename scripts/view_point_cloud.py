"""Open a persisted colored point cloud in an independent Open3D viewer."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from vggt_slam.open3d_focus import Open3DFocusController
from vggt_slam.ply_map import (
    PLYCameraPose,
    PLYObjectOBB,
    read_camera_poses_from_ply,
    read_object_obbs_from_ply,
)

_AXIS_MARKERS_PER_AXIS = 50

_CAMERA_FRUSTUM_SCALE = 0.075
_CAMERA_LINE_WIDTH = 2.0

# OBB edges are rendered as 3D cylinders rather than Open3D LineSets.
# Since the metricized map is in metres, 0.005 gives each edge a
# 5 mm radius / 10 mm diameter.
_OBJECT_OBB_RADIUS = 0.005
_OBJECT_OBB_COLOR = (1.0, 0.0, 0.0)

_BOX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int32)


def _with_axis_markers(
    point_cloud: "o3d.geometry.PointCloud",
    size: float = 1.0,
) -> "o3d.geometry.PointCloud":
    points = np.asarray(point_cloud.points)
    colors = (
        np.asarray(point_cloud.colors)
        if point_cloud.has_colors()
        else np.ones_like(points)
    )

    t = np.linspace(0.0, size, _AXIS_MARKERS_PER_AXIS)
    axis_directions = np.eye(3)
    axis_colors = np.eye(3)

    axis_points = np.concatenate(
        [np.outer(t, direction) for direction in axis_directions],
        axis=0,
    )
    axis_point_colors = np.concatenate(
        [
            np.tile(color, (_AXIS_MARKERS_PER_AXIS, 1))
            for color in axis_colors
        ],
        axis=0,
    )

    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(
        np.concatenate([points, axis_points], axis=0)
    )
    combined.colors = o3d.utility.Vector3dVector(
        np.concatenate([colors, axis_point_colors], axis=0)
    )
    return combined


def build_camera_frustums(
    camera_poses: list[PLYCameraPose],
    scale: float = _CAMERA_FRUSTUM_SCALE,
):
    """Build one LineSet of +Z-forward camera frustums in world coordinates."""
    if scale <= 0:
        raise ValueError("camera frustum scale must be positive")

    local = scale * np.array([
        [0, 0, 0],
        [-0.6, -0.4, 1],
        [0.6, -0.4, 1],
        [0.6, 0.4, 1],
        [-0.6, 0.4, 1],
    ])

    lines = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ], dtype=np.int32)

    all_points = []
    all_lines = []

    for index, pose in enumerate(camera_poses):
        points = (
            Rotation.from_quat(pose.quaternion_xyzw).apply(local)
            + np.asarray(pose.position_xyz)
        )
        all_points.append(points)
        all_lines.append(lines + 5 * index)

    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(
        np.concatenate(all_points)
        if all_points
        else np.empty((0, 3))
    )
    geometry.lines = o3d.utility.Vector2iVector(
        np.concatenate(all_lines)
        if all_lines
        else np.empty((0, 2), dtype=np.int32)
    )
    geometry.colors = o3d.utility.Vector3dVector(
        np.tile(
            [1.0, 0.5, 0.0],
            (len(lines) * len(camera_poses), 1),
        )
    )

    return geometry


def build_camera_trajectory(camera_poses: list[PLYCameraPose]):
    geometry = o3d.geometry.LineSet()

    centers = np.asarray(
        [pose.position_xyz for pose in camera_poses],
        dtype=float,
    ).reshape(-1, 3)

    geometry.points = o3d.utility.Vector3dVector(centers)

    geometry.lines = o3d.utility.Vector2iVector(
        np.array(
            [
                [i, i + 1]
                for i in range(max(0, len(centers) - 1))
            ],
            dtype=np.int32,
        ).reshape(-1, 2)
    )

    geometry.colors = o3d.utility.Vector3dVector(
        np.tile(
            [0.0, 1.0, 1.0],
            (max(0, len(centers) - 1), 1),
        )
    )

    return geometry


def _build_cylinder_between_points(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
):
    """Create a cylinder whose endpoints are approximately start and end."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction = end - start
    length = np.linalg.norm(direction)

    if length <= 1e-12:
        return None

    midpoint = 0.5 * (start + end)
    direction /= length

    # Open3D's create_cylinder() creates a cylinder along its local Z axis,
    # centered at the origin.
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=length,
        resolution=12,
        split=1,
    )

    cylinder.paint_uniform_color(color)

    # Rotate local +Z onto the desired edge direction.
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    cross_norm = np.linalg.norm(cross)
    dot = np.clip(np.dot(z_axis, direction), -1.0, 1.0)

    if cross_norm > 1e-12:
        rotation_axis = cross / cross_norm
        angle = np.arccos(dot)

        rotation_matrix = (
            o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * angle
            )
        )

        cylinder.rotate(
            rotation_matrix,
            center=(0.0, 0.0, 0.0),
        )

    elif dot < 0.0:
        # Desired direction is exactly opposite local +Z.
        rotation_matrix = (
            o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([np.pi, 0.0, 0.0])
            )
        )

        cylinder.rotate(
            rotation_matrix,
            center=(0.0, 0.0, 0.0),
        )

    cylinder.translate(midpoint)

    return cylinder


def build_object_obbs(
    object_obbs: list[PLYObjectOBB],
    radius: float = _OBJECT_OBB_RADIUS,
):
    """
    Build world-space OBB wireframes using cylinders.

    The stored OBB rotation matrix is interpreted as a row-major
    local-to-world rotation, matching the convention used when the
    OBB metadata was written to the PLY.
    """
    if radius <= 0:
        raise ValueError("OBB cylinder radius must be positive")

    geometries = []

    for obb in object_obbs:
        extent = np.asarray(
            obb.extent_xyz,
            dtype=float,
        )

        local_corners = 0.5 * extent * np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ], dtype=float)

        rotation = np.asarray(
            obb.rotation_matrix,
            dtype=float,
        ).reshape(3, 3)

        center = np.asarray(
            obb.center_xyz,
            dtype=float,
        )

        corners = (
            rotation @ local_corners.T
        ).T + center

        for start_index, end_index in _BOX_EDGES:
            cylinder = _build_cylinder_between_points(
                corners[start_index],
                corners[end_index],
                radius=radius,
                color=_OBJECT_OBB_COLOR,
            )

            if cylinder is not None:
                geometries.append(cylinder)

    return geometries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a saved point cloud with Open3D"
    )

    parser.add_argument(
        "--point_cloud",
        type=Path,
        help="Path to a point cloud, such as vggt_map.ply",
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Downsample with this voxel size before display",
    )

    parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Open3D render point size",
    )

    parser.add_argument(
        "--show_axes",
        action="store_true",
        help="Show coordinate axes at the origin",
    )

    parser.add_argument(
        "--show_camera_poses",
        action="store_true",
        help="Show embedded VGGT camera frustums and trajectory",
    )

    parser.add_argument(
        "--show_object_obbs",
        action="store_true",
        help="Show embedded open-set object OBBs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.point_size <= 0:
        raise ValueError("--point_size must be greater than zero")

    if args.voxel_size is not None and args.voxel_size <= 0:
        raise ValueError(
            "--voxel_size must be greater than zero when provided"
        )

    point_cloud = o3d.io.read_point_cloud(
        str(args.point_cloud)
    )

    original_count = len(point_cloud.points)

    if original_count == 0:
        raise ValueError(
            f"could not load a non-empty point cloud from "
            f"{args.point_cloud}"
        )

    if args.voxel_size is not None:
        point_cloud = point_cloud.voxel_down_sample(
            args.voxel_size
        )

        print(
            f"[PointCloudViewer] original_points="
            f"{original_count}"
        )
        print(
            f"[PointCloudViewer] downsampled_points="
            f"{len(point_cloud.points)}"
        )

        if len(point_cloud.points) == 0:
            raise ValueError(
                f"downsampling removed every point from "
                f"{args.point_cloud}"
            )

    points = np.asarray(point_cloud.points)

    print(f"[PointCloudViewer] file={args.point_cloud}")
    print(f"[PointCloudViewer] points={len(points)}")
    print(
        f"[PointCloudViewer] has_colors="
        f"{'yes' if point_cloud.has_colors() else 'no'}"
    )
    print(
        f"[PointCloudViewer] min_xyz="
        f"{tuple(np.min(points, axis=0))}"
    )
    print(
        f"[PointCloudViewer] max_xyz="
        f"{tuple(np.max(points, axis=0))}"
    )

    display_cloud = (
        _with_axis_markers(point_cloud)
        if args.show_axes
        else point_cloud
    )

    visualizer = o3d.visualization.VisualizerWithKeyCallback()

    visualizer.create_window(
        window_name=f"Point Cloud - {args.point_cloud.name}"
    )

    visualizer.add_geometry(display_cloud)

    render_option = visualizer.get_render_option()
    render_option.point_size = args.point_size

    if args.show_camera_poses:
        camera_poses = read_camera_poses_from_ply(
            args.point_cloud
        )

        if camera_poses:
            print(
                f"[PointCloudViewer] embedded_camera_poses="
                f"{len(camera_poses)}"
            )

            visualizer.add_geometry(
                build_camera_frustums(camera_poses)
            )
            visualizer.add_geometry(
                build_camera_trajectory(camera_poses)
            )

            render_option.line_width = _CAMERA_LINE_WIDTH

        else:
            print(
                "[PointCloudViewer] "
                "no embedded camera poses found"
            )

    if args.show_object_obbs:
        object_obbs = read_object_obbs_from_ply(
            args.point_cloud
        )

        if object_obbs:
            print(
                f"[PointCloudViewer] embedded_object_obbs="
                f"{len(object_obbs)}"
            )
            print(
                f"[PointCloudViewer] object_obb_radius="
                f"{_OBJECT_OBB_RADIUS}"
            )

            obb_geometries = build_object_obbs(
                object_obbs,
                radius=_OBJECT_OBB_RADIUS,
            )

            for geometry in obb_geometries:
                visualizer.add_geometry(geometry)

        else:
            print(
                "[PointCloudViewer] "
                "no embedded object OBBs found"
            )

    focus_controller = Open3DFocusController()
    focus_controller.register(visualizer)

    print(
        "[PointCloudViewer] Press F to focus/recenter on "
        "the visible surface at the view center."
    )

    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()