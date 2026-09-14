"""Open a persisted colored point cloud in an independent Open3D viewer."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


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

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=f"Point Cloud - {args.point_cloud.name}")
    visualizer.add_geometry(point_cloud)
    visualizer.get_render_option().point_size = args.point_size
    if args.show_axes:
        visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
