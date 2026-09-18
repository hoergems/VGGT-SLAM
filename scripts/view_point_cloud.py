"""Open a persisted colored point cloud in an independent Open3D viewer."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from vggt_slam.open3d_focus import Open3DFocusController

_AXIS_MARKERS_PER_AXIS = 50


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


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a saved point cloud with Open3D")
    parser.add_argument("--point_cloud", type=Path, help="Path to a point cloud, such as vggt_map.ply")
    parser.add_argument("--voxel_size", type=float, default=None, help="Downsample with this voxel size before display")
    parser.add_argument("--point_size", type=float, default=2.0, help="Open3D render point size")
    parser.add_argument("--show_axes", action="store_true", help="Show coordinate axes at the origin")
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
    visualizer.get_render_option().point_size = args.point_size

    focus_controller = Open3DFocusController()
    focus_controller.register(visualizer)
    print("[PointCloudViewer] Press F to focus/recenter on the visible surface at the view center.")

    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
