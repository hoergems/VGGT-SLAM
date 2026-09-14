import csv
import os
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from vggt_slam.frame_metadata import MetricTrajectorySample, SubmapTrajectoryDiagnostic, VGGTTrajectorySample
from vggt_slam.slam_utils import decompose_camera, cosine_similarity
from vggt_slam.trajectory_diagnostics import fit_similarity_umeyama


def camera_pose_from_projection(projection_mat):
    """Return VGGT camera center and camera-to-world quaternion from ``P``.

    This intentionally goes through the established RQ decomposition path;
    neither a projection translation nor an SL(4) homography translation is a
    camera position in the optimized VGGT world.
    """
    projection_mat = np.asarray(projection_mat, dtype=float)
    if projection_mat.shape not in ((3, 4), (4, 4)):
        raise ValueError("global camera projection matrix must have shape (3, 4) or (4, 4)")
    if not np.isfinite(projection_mat).all():
        raise ValueError("global camera projection matrix contains non-finite values")
    if projection_mat.shape == (4, 4) and np.isclose(projection_mat[-1, -1], 0.0):
        raise ValueError("global camera projection matrix has zero normalization scale")

    try:
        _, rotation_matrix, position_xyz, _ = decompose_camera(projection_mat)
    except (AssertionError, np.linalg.LinAlgError, ValueError) as error:
        raise ValueError("could not decompose global camera projection matrix") from error

    if not np.isfinite(rotation_matrix).all() or not np.isfinite(position_xyz).all():
        raise ValueError("decomposed VGGT camera pose contains non-finite values")
    if not np.allclose(rotation_matrix.T @ rotation_matrix, np.eye(3), atol=1e-5):
        raise ValueError("decomposed VGGT camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-5):
        raise ValueError("decomposed VGGT camera rotation is not a proper rotation")

    quaternion_xyzw = R.from_matrix(rotation_matrix).as_quat()
    if not np.isfinite(quaternion_xyzw).all():
        raise ValueError("decomposed VGGT camera quaternion contains non-finite values")
    return tuple(float(value) for value in position_xyz), tuple(float(value) for value in quaternion_xyzw)


class GraphMap:
    def __init__(self):
        self.submaps = dict()
        self.rectifying_H_mats = []
        self.non_lc_submap_ids = []
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
        if not submap.get_lc_status():
            self.non_lc_submap_ids.append(submap_id)
    
    def get_largest_key(self, ignore_loop_closure_submaps=False):
        """
        Get the largest key of the first node of any submap.
        Return: The largest key, or None if the dictionary is empty.
        """
        if len(self.submaps) == 0:
            return None
        if ignore_loop_closure_submaps:
            non_lc_keys = [key for key, submap in self.submaps.items() if not submap.get_lc_status()]
            return max(non_lc_keys)
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self, ignore_loop_closure_submaps=False):
        return self.get_submap(self.get_largest_key(ignore_loop_closure_submaps))

    def retrieve_best_semantic_frame(self, query_text_vector):
        overall_best_score = 0.0
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            submap = self.submaps[submap_key]
            if submap.get_lc_status():
                continue
            submap_embeddings = submap.get_all_semantic_vectors()
            scores = []
            for index, embedding in enumerate(submap_embeddings):
                score = cosine_similarity(embedding, query_text_vector)
                scores.append(score)
            
            best_score_id = np.argmax(scores)
            best_score = scores[best_score_id]

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_submap_id = submap_key
                overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index
    
    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            if submap_key == current_submap_id:
                continue

            if self.non_lc_submap_ids and ignore_last_submap and submap_key == self.non_lc_submap_ids[-1]:
                continue

            else:
                submap = self.submaps[submap_key]
                if submap.get_lc_status():
                    continue
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for index, embedding in enumerate(submap_embeddings):
                    score = torch.linalg.norm(embedding-query_vector)
                    # score = embedding @ query_vector.t()
                    scores.append(score.item())

                # for now assume we can only have at most one loop closure per submap
                
                best_score_id = np.argmin(scores)
                best_score = scores[best_score_id]

                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
        return frames
    
    def get_submaps(self):
        return self.submaps.values()

    def get_keyframe_record_by_timestamp(self, timestamp_ns):
        """Return canonical Go2 metadata for an exact source timestamp."""
        if not isinstance(timestamp_ns, int):
            raise TypeError("timestamp_ns must be an exact Python int")
        matches = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            for record in submap.get_keyframe_records() or []:
                if record.timestamp_ns == timestamp_ns:
                    matches.append(record)
        if not matches:
            return None
        canonical = matches[0]
        if any(record != canonical for record in matches[1:]):
            raise ValueError(f"Conflicting metadata for timestamp {timestamp_ns}")
        return canonical

    @staticmethod
    def _metric_trajectory_sample_from_record(record):
        """Validate one Go2 keyframe record and convert it to a public sample."""
        if type(record.timestamp_ns) is not int:
            raise TypeError("metric trajectory timestamp_ns must be an exact Python int")
        if record.metric_pose is None:
            raise ValueError(
                f"Go2 keyframe record at timestamp {record.timestamp_ns} is missing its metric pose"
            )

        pose = record.metric_pose
        components = (
            ("position_xyz", pose.position_xyz, 3),
            ("quaternion_xyzw", pose.quaternion_xyzw, 4),
        )
        validated = []
        for name, values, expected_length in components:
            if not isinstance(values, tuple) or len(values) != expected_length:
                raise ValueError(
                    f"metric pose {name} must be a {expected_length}-element tuple "
                    f"for timestamp {record.timestamp_ns}"
                )
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
                raise ValueError(
                    f"metric pose {name} must contain numeric values for timestamp {record.timestamp_ns}"
                )
            validated.append(tuple(float(value) for value in values))

        position_xyz, quaternion_xyzw = validated
        return MetricTrajectorySample(
            timestamp_ns=record.timestamp_ns,
            frame_id=record.frame_id,
            position_xyz=position_xyz,
            quaternion_xyzw=quaternion_xyzw,
        )

    def get_metric_camera_trajectory(self) -> list[MetricTrajectorySample]:
        """Return unique ordinary-submap Go2 camera samples in map/frame order.

        Overlapping ordinary submaps retain the same source record.  The exact
        timestamp is therefore the identity used to deduplicate them; loop
        closure submaps are graph constraints and never trajectory samples.
        """
        samples = []
        canonical_records_by_timestamp = {}
        previous_timestamp = None

        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            for record in submap.get_keyframe_records() or []:
                if record.timestamp_ns is None:
                    continue

                timestamp_ns = record.timestamp_ns
                if type(timestamp_ns) is not int:
                    raise TypeError("metric trajectory timestamp_ns must be an exact Python int")
                canonical = canonical_records_by_timestamp.get(timestamp_ns)
                if canonical is not None:
                    if record != canonical:
                        raise ValueError(
                            f"Conflicting metadata for duplicate metric trajectory timestamp {timestamp_ns}"
                        )
                    continue

                sample = self._metric_trajectory_sample_from_record(record)
                if previous_timestamp is not None and timestamp_ns <= previous_timestamp:
                    raise ValueError(
                        "Non-monotonic metric trajectory timestamps: "
                        f"{timestamp_ns} follows {previous_timestamp}"
                    )
                canonical_records_by_timestamp[timestamp_ns] = record
                samples.append(sample)
                previous_timestamp = timestamp_ns

        return samples

    def write_metric_camera_trajectory_to_file(self, file_name: str) -> None:
        """Write canonical Go2 ``T_odom_camera`` samples, independent of VGGT poses."""
        samples = self.get_metric_camera_trajectory()
        with open(file_name, "w") as f:
            for sample in samples:
                f.write(
                    str(sample.timestamp_ns)
                    + " "
                    + " ".join(
                        f"{value:.8f}"
                        for value in (*sample.position_xyz, *sample.quaternion_xyzw)
                    )
                    + "\n"
                )

        print(f"[MetricTrajectory] unique_samples={len(samples)}")
        if samples:
            print(f"[MetricTrajectory] first_timestamp_ns={samples[0].timestamp_ns}")
            print(f"[MetricTrajectory] last_timestamp_ns={samples[-1].timestamp_ns}")
        print(f"[MetricTrajectory] wrote {file_name}")

    @staticmethod
    def _print_vggt_trajectory_summary(samples, overlap_position_deltas):
        """Print compact continuity diagnostics without imposing thresholds."""
        print(f"[VGGTTrajectory] unique_samples={len(samples)}")
        print(f"[VGGTTrajectory] overlap_duplicates={len(overlap_position_deltas)}")
        if samples:
            print(f"[VGGTTrajectory] first_timestamp_ns={samples[0].timestamp_ns}")
            print(f"[VGGTTrajectory] last_timestamp_ns={samples[-1].timestamp_ns}")
        if overlap_position_deltas:
            print(f"[VGGTTrajectory] max_overlap_position_delta={max(overlap_position_deltas):.8f}")
            print(f"[VGGTTrajectory] mean_overlap_position_delta={np.mean(overlap_position_deltas):.8f}")
        if len(samples) > 1:
            positions = np.asarray([sample.position_xyz for sample in samples])
            steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            max_index = int(np.argmax(steps)) + 1
            print(
                "[VGGTTrajectory] "
                f"min_step={np.min(steps):.8f} median_step={np.median(steps):.8f} "
                f"p95_step={np.percentile(steps, 95):.8f} max_step={np.max(steps):.8f} "
                f"max_step_timestamp_ns={samples[max_index].timestamp_ns}"
            )

    def get_vggt_camera_trajectory(self, graph) -> list[VGGTTrajectorySample]:
        """Return unique, latest-optimized VGGT cameras for Go2 keyframes.

        Ordinary submaps overlap by source timestamp.  The first local graph
        representation is canonical; a later overlap is only a continuity
        diagnostic because separately optimized graph nodes may differ.
        """
        samples = []
        samples_by_timestamp = {}
        overlap_position_deltas = []
        previous_timestamp = None

        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            records = submap.get_keyframe_records() or []
            if not any(record.timestamp_ns is not None for record in records):
                continue

            projection_mats = submap.get_all_poses_world(graph, give_camera_mat=True)
            frame_ids = submap.get_frame_ids()
            if len(projection_mats) != len(frame_ids) or len(records) != len(frame_ids):
                raise ValueError(
                    "VGGT trajectory submap length mismatch: "
                    f"projections={len(projection_mats)}, frame_ids={len(frame_ids)}, records={len(records)}"
                )

            for frame_index, (record, projection_mat) in enumerate(zip(records, projection_mats)):
                if record.timestamp_ns is None:
                    continue
                if type(record.timestamp_ns) is not int:
                    raise TypeError("VGGT trajectory timestamp_ns must be an exact Python int")

                position_xyz, quaternion_xyzw = camera_pose_from_projection(projection_mat)
                timestamp_ns = record.timestamp_ns
                existing = samples_by_timestamp.get(timestamp_ns)
                if existing is not None:
                    overlap_position_deltas.append(
                        float(np.linalg.norm(np.asarray(existing.position_xyz) - np.asarray(position_xyz)))
                    )
                    continue

                if previous_timestamp is not None and timestamp_ns <= previous_timestamp:
                    raise ValueError(
                        "Non-monotonic VGGT trajectory timestamps: "
                        f"{timestamp_ns} follows {previous_timestamp}"
                    )
                sample = VGGTTrajectorySample(
                    timestamp_ns=timestamp_ns,
                    frame_id=record.frame_id,
                    position_xyz=position_xyz,
                    quaternion_xyzw=quaternion_xyzw,
                    submap_id=submap.get_id(),
                    frame_index=frame_index,
                )
                samples_by_timestamp[timestamp_ns] = sample
                samples.append(sample)
                previous_timestamp = timestamp_ns

        metric_timestamps = [sample.timestamp_ns for sample in self.get_metric_camera_trajectory()]
        vggt_timestamps = [sample.timestamp_ns for sample in samples]
        if metric_timestamps != vggt_timestamps:
            mismatch_index = next(
                (index for index, pair in enumerate(zip(metric_timestamps, vggt_timestamps)) if pair[0] != pair[1]),
                min(len(metric_timestamps), len(vggt_timestamps)),
            )
            raise ValueError(
                "Metric and VGGT trajectory timestamp sequences differ at index "
                f"{mismatch_index}: metric={metric_timestamps[mismatch_index:mismatch_index + 1]}, "
                f"vggt={vggt_timestamps[mismatch_index:mismatch_index + 1]}"
            )

        self._print_vggt_trajectory_summary(samples, overlap_position_deltas)
        print("[VGGTTrajectory] timestamps_match_metric=yes")
        return samples

    @staticmethod
    def plot_vggt_camera_trajectory(samples, output_path) -> None:
        """Plot the unaligned optimized VGGT XY camera path for inspection."""
        if not samples:
            raise ValueError("cannot plot an empty VGGT trajectory")
        positions = np.asarray([sample.position_xyz for sample in samples])
        figure, axis = plt.subplots()
        axis.plot(positions[:, 0], positions[:, 1], "-", label="VGGT camera path")
        axis.scatter(*positions[0, :2], color="green", label="start", zorder=3)
        axis.scatter(*positions[-1, :2], color="red", label="end", zorder=3)
        axis.set_xlabel("VGGT X")
        axis.set_ylabel("VGGT Y")
        axis.set_title("Unaligned VGGT camera trajectory")
        axis.axis("equal")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        print(f"[VGGTTrajectory] wrote plot {output_path}")

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]

    def get_submap_trajectory_diagnostics(self, graph) -> list[SubmapTrajectoryDiagnostic]:
        """Measure each ordinary local VGGT window against its Go2 metadata."""
        diagnostics = []
        endpoint_epsilon = 1e-9
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            records = submap.get_keyframe_records()
            frame_ids = submap.get_frame_ids()
            local_xyz = submap.get_local_camera_centers()
            if records is None or frame_ids is None:
                raise ValueError(f"submap {submap.get_id()} lacks diagnostic keyframe metadata")
            if len(records) != len(frame_ids) or len(records) != len(local_xyz):
                raise ValueError(
                    f"submap {submap.get_id()} diagnostic record/frame/pose counts differ: "
                    f"{len(records)}/{len(frame_ids)}/{len(local_xyz)}"
                )
            if not records:
                raise ValueError(f"submap {submap.get_id()} has no diagnostic keyframes")
            metric_positions = []
            for record in records:
                if type(record.timestamp_ns) is not int or record.metric_pose is None:
                    raise ValueError(f"submap {submap.get_id()} has incomplete Go2 keyframe metadata")
                position = np.asarray(record.metric_pose.position_xyz, dtype=float)
                if position.shape != (3,) or not np.isfinite(position).all():
                    raise ValueError(f"submap {submap.get_id()} has invalid Go2 metric position")
                metric_positions.append(position)
            metric_xyz = np.asarray(metric_positions)
            projection_mats = submap.get_all_poses_world(graph, give_camera_mat=True)
            if len(projection_mats) != len(records):
                raise ValueError(f"submap {submap.get_id()} global pose count differs from records")
            global_xyz = np.asarray([camera_pose_from_projection(projection)[0] for projection in projection_mats])

            local_disp = float(np.linalg.norm(local_xyz[-1] - local_xyz[0]))
            metric_disp = float(np.linalg.norm(metric_xyz[-1] - metric_xyz[0]))
            global_disp = float(np.linalg.norm(global_xyz[-1] - global_xyz[0]))
            endpoint_scale = metric_disp / local_disp if local_disp > endpoint_epsilon else None
            global_scale = metric_disp / global_disp if global_disp > endpoint_epsilon else None

            umeyama_scale = rmse = median_error = max_error = None
            try:
                umeyama_scale, _, _, residuals = fit_similarity_umeyama(local_xyz, metric_xyz)
                rmse = float(np.sqrt(np.mean(residuals ** 2)))
                median_error = float(np.median(residuals))
                max_error = float(np.max(residuals))
            except ValueError as error:
                print(f"[SubmapDiag] warning id={submap.get_id()}: local Umeyama unavailable ({error})")

            diagnostics.append(SubmapTrajectoryDiagnostic(
                submap_id=submap.get_id(), num_frames=len(records),
                first_timestamp_ns=records[0].timestamp_ns, last_timestamp_ns=records[-1].timestamp_ns,
                incoming_stitch_scale=submap.get_incoming_scale_factor(),
                local_vggt_endpoint_displacement=local_disp,
                metric_endpoint_displacement_m=metric_disp,
                endpoint_implied_scale_m_per_vggt=endpoint_scale,
                local_umeyama_scale_m_per_vggt=umeyama_scale,
                local_umeyama_rmse_m=rmse, local_umeyama_median_error_m=median_error,
                local_umeyama_max_error_m=max_error,
                global_vggt_endpoint_displacement=global_disp,
                global_implied_scale_m_per_vggt=global_scale,
            ))
        return diagnostics

    @staticmethod
    def _diagnostic_value(value):
        return "n/a" if value is None else f"{value:.6g}"

    def print_submap_trajectory_diagnostics(self, graph) -> list[SubmapTrajectoryDiagnostic]:
        diagnostics = self.get_submap_trajectory_diagnostics(graph)
        for item in diagnostics:
            value = self._diagnostic_value
            print(
                f"[SubmapDiag] id={item.submap_id} frames={item.num_frames} "
                f"stitch_scale={value(item.incoming_stitch_scale)} metric_disp_m={item.metric_endpoint_displacement_m:.6g} "
                f"local_disp={item.local_vggt_endpoint_displacement:.6g} endpoint_scale={value(item.endpoint_implied_scale_m_per_vggt)}\n"
                f"             umeyama_scale={value(item.local_umeyama_scale_m_per_vggt)} rmse_m={value(item.local_umeyama_rmse_m)} "
                f"median_m={value(item.local_umeyama_median_error_m)} max_m={value(item.local_umeyama_max_error_m)} "
                f"global_disp={item.global_vggt_endpoint_displacement:.6g} global_scale={value(item.global_implied_scale_m_per_vggt)}"
            )
        scales = np.asarray([item.local_umeyama_scale_m_per_vggt for item in diagnostics
                             if item.local_umeyama_scale_m_per_vggt is not None], dtype=float)
        if len(scales):
            cv = float(np.std(scales) / np.mean(scales))
            worst = max((item for item in diagnostics if item.local_umeyama_rmse_m is not None),
                        key=lambda item: item.local_umeyama_rmse_m)
            print(f"[SubmapDiag] umeyama_scale min={scales.min():.6g} median={np.median(scales):.6g} max={scales.max():.6g} cv={cv:.6g}")
            print(f"[SubmapDiag] worst_local_rmse_submap={worst.submap_id} rmse_m={worst.local_umeyama_rmse_m:.6g}")
        return diagnostics

    @staticmethod
    def write_submap_trajectory_diagnostics_to_csv(diagnostics, output_path) -> None:
        fields = list(SubmapTrajectoryDiagnostic.__dataclass_fields__)
        with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fields)
            writer.writeheader()
            for item in diagnostics:
                row = {}
                for field in fields:
                    value = getattr(item, field)
                    row[field] = "" if value is None else (str(value) if field.endswith("timestamp_ns") else f"{value:.8f}" if isinstance(value, float) else value)
                writer.writerow(row)
        print(f"[SubmapDiag] wrote {output_path}")

    @staticmethod
    def plot_submap_scale_diagnostics(diagnostics, output_path) -> None:
        if not diagnostics:
            raise ValueError("cannot plot empty submap diagnostics")
        ids = np.asarray([item.submap_id for item in diagnostics])
        def values(name):
            return np.asarray([np.nan if getattr(item, name) is None else getattr(item, name) for item in diagnostics])
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
        axes[0].plot(ids, values("local_umeyama_scale_m_per_vggt"), "o-", label="local Umeyama [m/VGGT]")
        axes[0].plot(ids, values("endpoint_implied_scale_m_per_vggt"), "o-", label="endpoint [m/VGGT]")
        axes[0].plot(ids, values("incoming_stitch_scale"), "o-", label="internal incoming stitch scale")
        axes[0].set_ylabel("scale")
        axes[0].legend()
        axes[1].plot(ids, values("local_umeyama_rmse_m"), "o-", label="RMSE")
        axes[1].plot(ids, values("local_umeyama_median_error_m"), "o-", label="median")
        axes[1].plot(ids, values("local_umeyama_max_error_m"), "o-", label="max")
        axes[1].set_xlabel("submap ID")
        axes[1].set_ylabel("local error [m]")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        print(f"[SubmapDiag] wrote plot {output_path}")
    
    def get_all_homographies(self, graph):
        homographies = []
        for submap in self.ordered_submaps_by_key():
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                homographies.append(graph.get_homography(id))
        return np.stack(homographies)

    def get_all_cam_matricies(self, graph, give_camera_mat):
        cam_mats = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            poses = submap.get_all_poses_world(graph, give_camera_mat=give_camera_mat)
            cam_mats.append(poses)
        return np.vstack(cam_mats)

    def write_poses_to_file(self, file_name, graph, give_camera_mat=False, kitti_format=False):
        all_poses = self.get_all_cam_matricies(give_camera_mat=True, graph=graph)
        with open(file_name, "w") as f:

            if self.rectifying_H_mats:
                assert len(self.rectifying_H_mats) == len(all_poses), "Number of rectifying mats and number of poses do not match"
                print("Using rectifying homographies when writing poses to file.")
            count = 0
            for submap_index, submap in enumerate(self.ordered_submaps_by_key()):
                if submap.get_lc_status():
                    continue
                frame_ids = submap.get_frame_ids()
                print(frame_ids)
                for frame_index, frame_id in enumerate(frame_ids):
                    pose = all_poses[count]
                    K, rotation_matrix, t, scale = decompose_camera(pose)
                    # print("Decomposed K:\n", K)
                    count += 1
                    x, y, z = t
                    if kitti_format:
                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = rotation_matrix
                        pose_matrix[:3, 3] = t
                        output = pose_matrix.flatten()[:-4]
                        output = np.array([float(frame_id), *output])
                    else:
                        quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                        output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")

    def write_timestamped_poses_to_file(self, file_name, graph, give_camera_mat=False, samples=None):
        """Write canonical Go2 timestamp and optimized VGGT camera pose pairs."""
        if samples is None:
            samples = self.get_vggt_camera_trajectory(graph)
        with open(file_name, "w") as f:
            for sample in samples:
                f.write(
                    str(sample.timestamp_ns)
                    + " "
                    + " ".join(
                        f"{value:.8f}"
                        for value in (*sample.position_xyz, *sample.quaternion_xyzw)
                    )
                    + "\n"
                )
        print(f"[VGGTTrajectory] wrote {file_name}")

    def get_global_point_cloud(self, graph):
        """Build the final colored dense map from ordinary submaps only.

        Ordinary submaps intentionally retain their shared overlap frame, so a
        small number of duplicate dense points is expected.  Loop-closure
        submaps, however, are optimization helpers and are excluded.
        """
        points_by_submap = []
        colors_by_submap = []
        ordinary_submaps = 0

        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            points = np.asarray(submap.get_points_in_world_frame(graph))
            colors = np.asarray(submap.get_points_colors())
            try:
                points = points.reshape(-1, 3)
                colors = colors.reshape(-1, 3)
            except ValueError as error:
                raise ValueError("submap point and color arrays must each have three channels") from error
            points_by_submap.append(points)
            colors_by_submap.append(colors)
            if len(points) > 0:
                ordinary_submaps += 1

        if not points_by_submap:
            raise ValueError("cannot export point cloud: no ordinary submap points are available")

        points = np.concatenate(points_by_submap, axis=0)
        colors = np.concatenate(colors_by_submap, axis=0)
        if len(points) == 0:
            raise ValueError("cannot export point cloud: no ordinary submap points are available")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("exported point array must have shape (N, 3)")
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError("exported color array must have shape (N, 3)")
        if len(points) != len(colors):
            raise ValueError(
                "exported point and color counts differ: "
                f"points={len(points)}, colors={len(colors)}"
            )
        if not np.isfinite(points).all():
            raise ValueError("exported XYZ values contain non-finite values")
        if not np.isfinite(colors).all():
            raise ValueError("exported RGB values contain non-finite values")

        colors = colors.astype(float, copy=False)
        if colors.max() > 1.0:
            colors = colors / 255.0
        if (colors < 0.0).any() or (colors > 1.0).any():
            raise ValueError("exported RGB values must be in the [0, 1] or [0, 255] range")

        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud, ordinary_submaps

    def write_points_to_file(self, graph, file_name):
        """Write the final optimized colored map to any Open3D-supported format."""
        point_cloud, ordinary_submaps = self.get_global_point_cloud(graph)
        if not o3d.io.write_point_cloud(str(file_name), point_cloud):
            raise IOError(f"failed to write point cloud to {file_name}")

        points = np.asarray(point_cloud.points)
        print(f"[MapExport] ordinary_submaps={ordinary_submaps}")
        print(f"[MapExport] points={len(points)}")
        print(f"[MapExport] min_xyz={tuple(np.min(points, axis=0))}")
        print(f"[MapExport] max_xyz={tuple(np.max(points, axis=0))}")
        print(f"[MapExport] wrote {file_name}")

    def write_submap_point_clouds(self, graph, output_dir) -> None:
        """Write each ordinary submap in its current optimized global frame."""
        output_path = Path(output_dir)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"submap point-cloud output path is not a directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            points = np.asarray(submap.get_points_in_world_frame(graph))
            colors = np.asarray(submap.get_points_colors())
            try:
                points = points.reshape(-1, 3)
                colors = colors.reshape(-1, 3)
            except ValueError as error:
                raise ValueError(f"submap {submap.get_id()} points and colors must each have three channels") from error
            if len(points) != len(colors):
                raise ValueError(f"submap {submap.get_id()} point/color counts differ: {len(points)}/{len(colors)}")
            if len(points) == 0:
                raise ValueError(f"submap {submap.get_id()} has no points to export")
            if not np.isfinite(points).all() or not np.isfinite(colors).all():
                raise ValueError(f"submap {submap.get_id()} contains non-finite XYZ or RGB values")
            colors = colors.astype(float, copy=False)
            if colors.max() > 1.0:
                colors = colors / 255.0
            if (colors < 0.0).any() or (colors > 1.0).any():
                raise ValueError(f"submap {submap.get_id()} RGB must be in [0, 1] or [0, 255]")
            point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            filename = output_path / f"submap_{submap.get_id():06d}.ply"
            if not o3d.io.write_point_cloud(str(filename), point_cloud):
                raise IOError(f"failed to write submap point cloud {filename}")
            print(f"[SubmapPLY] id={submap.get_id()} points={len(points)} wrote {filename}")
            print(f"[SubmapPLY] id={submap.get_id()} min_xyz={tuple(np.min(points, axis=0))} max_xyz={tuple(np.max(points, axis=0))}")
