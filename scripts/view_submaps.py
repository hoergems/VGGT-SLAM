"""Overlay persisted per-submap PLYs with distinct diagnostic colours."""

import argparse
import re
from pathlib import Path

import numpy as np
import open3d as o3d


PALETTE = np.asarray([
    [0.89, 0.10, 0.11], [0.22, 0.49, 0.72], [0.30, 0.69, 0.29],
    [0.60, 0.31, 0.64], [1.00, 0.50, 0.00], [0.65, 0.34, 0.16],
    [0.97, 0.51, 0.75], [0.50, 0.50, 0.50],
])


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect globally transformed submap PLY overlays")
    parser.add_argument("--directory", type=Path, required=True, help="Directory containing submap_*.ply files")
    parser.add_argument("--voxel_size", type=float, default=None, help="Downsample each submap independently")
    parser.add_argument("--point_size", type=float, default=2.0, help="Open3D render point size")
    parser.add_argument("--show_axes", action="store_true", help="Show coordinate axes at the origin")
    parser.add_argument("--use_original_colors", action="store_true", help="Keep stored PLY RGB instead of diagnostic submap colours")
    return parser.parse_args()


def submap_id(path):
    match = re.fullmatch(r"submap_(\d+)\.ply", path.name)
    if match is None:
        raise ValueError(f"invalid submap filename: {path.name}")
    return int(match.group(1))


def main():
    args = parse_args()
    if args.point_size <= 0:
        raise ValueError("--point_size must be greater than zero")
    if args.voxel_size is not None and args.voxel_size <= 0:
        raise ValueError("--voxel_size must be greater than zero when provided")
    if not args.directory.is_dir():
        raise ValueError(f"not a submap PLY directory: {args.directory}")
    paths = sorted(args.directory.glob("submap_*.ply"), key=submap_id)
    if not paths:
        raise ValueError(f"no submap_*.ply files found in {args.directory}")

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=f"Submap overlay - {args.directory}")
    for index, path in enumerate(paths):
        point_cloud = o3d.io.read_point_cloud(str(path))
        original_count = len(point_cloud.points)
        if original_count == 0:
            raise ValueError(f"could not load a non-empty point cloud from {path}")
        if args.voxel_size is not None:
            point_cloud = point_cloud.voxel_down_sample(args.voxel_size)
        if not args.use_original_colors:
            point_cloud.paint_uniform_color(PALETTE[index % len(PALETTE)])
        print(f"[SubmapViewer] id={submap_id(path)} file={path.name} points={len(point_cloud.points)} original_points={original_count}")
        visualizer.add_geometry(point_cloud)
    visualizer.get_render_option().point_size = args.point_size
    if args.show_axes:
        visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
