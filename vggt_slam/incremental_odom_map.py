"""Atomic, complete odom-frame planning-map snapshots for live SLAM."""

import os
from pathlib import Path
import time
import uuid

import numpy as np

from vggt_slam.odom_map_alignment import (
    build_first_frame_odom_alignment,
    pose_matrix_from_position_quaternion,
    rotation_angle_deg,
    validate_se3,
)


class IncrementalOdomMapSnapshot:
    """Regenerate one complete optimized map snapshot after each submap update."""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        if self.output_path.suffix.lower() != ".ply":
            raise ValueError("planning map output supports .ply files only")
        self.alignment = None
        self.update_count = 0

    @property
    def transform_odom_vggt(self):
        """The immutable first-frame map-to-odom transform, once initialized."""
        return None if self.alignment is None else self.alignment.transform_odom_vggt.copy()

    def _initialize_alignment(self, solver):
        metric_samples = solver.map.get_metric_camera_trajectory()
        vggt_samples = solver.map.get_vggt_camera_trajectory(solver.graph)
        self.alignment = build_first_frame_odom_alignment(metric_samples, vggt_samples)
        transform = validate_se3(self.alignment.transform_odom_vggt, "T_odom_vggt")
        p0_odom = np.asarray(self.alignment.metric_sample.position_xyz, dtype=float)
        first_vggt_pose = pose_matrix_from_position_quaternion(
            self.alignment.vggt_sample.position_xyz,
            self.alignment.vggt_sample.quaternion_xyzw,
        )
        print(f"[PlanningMap] initialized fixed T_odom_vggt from timestamp={self.alignment.vggt_sample.timestamp_ns}")
        print("[PlanningMap] first_go2_optical_camera_position_odom="
              f"({p0_odom[0]:.6f}, {p0_odom[1]:.6f}, {p0_odom[2]:.6f})")
        print(f"[PlanningMap] first_vggt_position_norm={np.linalg.norm(first_vggt_pose[:3, 3]):.9g}")
        print("[PlanningMap] first_vggt_rotation_from_identity_deg="
              f"{rotation_angle_deg(first_vggt_pose[:3, :3]):.9g}")
        return transform

    def update(self, solver):
        """Atomically replace the PLY with the current complete optimized map."""
        total_start = time.perf_counter()
        transform = self._initialize_alignment(solver) if self.alignment is None else self.alignment.transform_odom_vggt
        parent = self.output_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary_path = parent / f".{self.output_path.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex}.ply"
        write_start = time.perf_counter()
        try:
            ordinary_submaps, point_count = solver.map.write_odom_aligned_points_to_file(
                solver.graph,
                temporary_path,
                transform,
                verify_points=False,
                log_prefix="[PlanningMap]",
                log_summary=False,
            )
            os.replace(temporary_path, self.output_path)
        except Exception:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        snapshot_seconds = time.perf_counter() - write_start
        self.update_count += 1
        print(
            f"[PlanningMap] update={self.update_count} ordinary_submaps={ordinary_submaps} "
            f"points={point_count} snapshot_s={snapshot_seconds:.3f} total_s={time.perf_counter() - total_start:.3f} "
            f"path={self.output_path}"
        )
