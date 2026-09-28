"""Open a persisted colored point cloud in an independent Open3D viewer."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from vggt_slam.open3d_focus import Open3DFocusController
from vggt_slam.ply_map import PLYCameraPose, read_camera_poses_from_ply

_AXIS_MARKERS_PER_AXIS = 50
_CAMERA_FRUSTUM_SCALE = 0.075
_CAMERA_LINE_WIDTH = 2.0


def _with_axis_markers(point_cloud: "o3d.geometry.PointCloud", size: float = 1.0) -> "o3d.geometry.PointCloud":
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else np.ones_like(points)

    t = np.linspace(0.0, size, _AXIS_MARKERS_PER_AXIS)
    axis_directions = np.eye(3)
    axis_colors = np.eye(3)
    axis_points = np.concatenate([np.outer(t, direction) for direction in axis_directions], axis=0)
    axis_point_colors = np.concatenate(
        [np.tile(color, (_AXIS_MARKERS_PER_AXIS, 1)) for color in axis_colors], axis=0
    )

    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(np.concatenate([points, axis_points], axis=0))
    combined.colors = o3d.utility.Vector3dVector(np.concatenate([colors, axis_point_colors], axis=0))
    return combined


def build_camera_frustums(camera_poses: list[PLYCameraPose], scale: float = _CAMERA_FRUSTUM_SCALE):
    """Build one LineSet of +Z-forward camera frustums in world coordinates."""
    if scale <= 0:
        raise ValueError("camera frustum scale must be positive")
    local = scale * np.array([[0, 0, 0], [-.6, -.4, 1], [.6, -.4, 1], [.6, .4, 1], [-.6, .4, 1]])
    lines = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int32)
    all_points, all_lines = [], []
    for index, pose in enumerate(camera_poses):
        points = Rotation.from_quat(pose.quaternion_xyzw).apply(local) + np.asarray(pose.position_xyz)
        all_points.append(points)
        all_lines.append(lines + 5 * index)
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(np.concatenate(all_points) if all_points else np.empty((0, 3)))
    geometry.lines = o3d.utility.Vector2iVector(np.concatenate(all_lines) if all_lines else np.empty((0, 2), dtype=np.int32))
    geometry.colors = o3d.utility.Vector3dVector(np.tile([1., .5, 0.], (len(lines) * len(camera_poses), 1)))
    return geometry


def build_camera_trajectory(camera_poses: list[PLYCameraPose]):
    geometry = o3d.geometry.LineSet()
    centers = np.asarray([pose.position_xyz for pose in camera_poses], dtype=float).reshape(-1, 3)
    geometry.points = o3d.utility.Vector3dVector(centers)
    geometry.lines = o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(max(0, len(centers) - 1))], dtype=np.int32).reshape(-1, 2))
    geometry.colors = o3d.utility.Vector3dVector(np.tile([0., 1., 1.], (max(0, len(centers) - 1), 1)))
    return geometry


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a saved point cloud with Open3D")
    parser.add_argument("--point_cloud", type=Path, help="Path to a point cloud, such as vggt_map.ply")
    parser.add_argument("--voxel_size", type=float, default=None, help="Downsample with this voxel size before display")
    parser.add_argument("--point_size", type=float, default=2.0, help="Open3D render point size")
    parser.add_argument("--show_axes", action="store_true", help="Show coordinate axes at the origin")
    parser.add_argument("--show_camera_poses", action="store_true", help="Show embedded VGGT camera frustums and trajectory")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.point_size <= 0:
        raise ValueError("--point_size must be greater than zero")
    if args.voxel_size is not None and args.voxel_size <= 0:
        raise ValueError("--voxel_size must be greater than zero when provided")

    point_cloud = o3d.io.read_point_cloud(str(args.point_cloud))
    original_count = len(point_cloud.points)
    if original_count == 0:
        raise ValueError(f"could not load a non-empty point cloud from {args.point_cloud}")
    if args.voxel_size is not None:
        point_cloud = point_cloud.voxel_down_sample(args.voxel_size)
        print(f"[PointCloudViewer] original_points={original_count}")
        print(f"[PointCloudViewer] downsampled_points={len(point_cloud.points)}")
        if len(point_cloud.points) == 0:
            raise ValueError(f"downsampling removed every point from {args.point_cloud}")

    points = np.asarray(point_cloud.points)
    print(f"[PointCloudViewer] file={args.point_cloud}")
    print(f"[PointCloudViewer] points={len(points)}")
    print(f"[PointCloudViewer] has_colors={'yes' if point_cloud.has_colors() else 'no'}")
    print(f"[PointCloudViewer] min_xyz={tuple(np.min(points, axis=0))}")
    print(f"[PointCloudViewer] max_xyz={tuple(np.max(points, axis=0))}")

    display_cloud = _with_axis_markers(point_cloud) if args.show_axes else point_cloud

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name=f"Point Cloud - {args.point_cloud.name}")
    visualizer.add_geometry(display_cloud)
    render_option = visualizer.get_render_option()
    render_option.point_size = args.point_size
    if args.show_camera_poses:
        camera_poses = read_camera_poses_from_ply(args.point_cloud)
        if camera_poses:
            print(f"[PointCloudViewer] embedded_camera_poses={len(camera_poses)}")
            visualizer.add_geometry(build_camera_frustums(camera_poses))
            visualizer.add_geometry(build_camera_trajectory(camera_poses))
            render_option.line_width = _CAMERA_LINE_WIDTH
        else:
            print("[PointCloudViewer] no embedded camera poses found")

    focus_controller = Open3DFocusController()
    focus_controller.register(visualizer)
    print("[PointCloudViewer] Press F to focus/recenter on the visible surface at the view center.")

    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
