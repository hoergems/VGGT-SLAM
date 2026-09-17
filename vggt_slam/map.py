import csv
import json
import os
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from vggt_slam.frame_metadata import (
    KeyframeImageQuality, MetricTrajectorySample, SubmapImageQualityDiagnostic,
    SubmapMetricAlignment, SubmapMotionDiagnostic, SubmapMotionStep, SubmapTrajectoryDiagnostic,
    VGGTTrajectorySample, CameraAxisAlignment, KeyframeOrientationDiagnostic, SubmapOrientationDiagnostic,
)
from vggt_slam.image_quality import compute_laplacian_variance, normalize_image_size
from vggt_slam.slam_utils import decompose_camera, cosine_similarity
from vggt_slam.trajectory_diagnostics import apply_similarity, fit_similarity_umeyama
from vggt_slam.motion_diagnostics import relative_rotation_angle_deg, write_trajectory_comparison_plot
from vggt_slam.orientation_diagnostics import (
    mean_rotation, rotation_geodesic_angle_deg, rotation_matrix_from_xyzw, validate_rotation_matrix,
)
from vggt_slam.metric_pose_utils import metric_pose_to_optical_camera_to_odom


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

    def get_keyframe_image_quality_diagnostics(self, trajectory_diagnostics, blur_laplacian_threshold=None):
        """Measure exact ordinary-submap keyframe appearances without mutating SLAM state."""
        if trajectory_diagnostics is None:
            raise ValueError("trajectory_diagnostics must be supplied to avoid duplicate Umeyama work")
        if blur_laplacian_threshold is not None and (not np.isfinite(blur_laplacian_threshold) or blur_laplacian_threshold < 0):
            raise ValueError("blur_laplacian_threshold must be finite and non-negative")
        trajectory_by_id = {item.submap_id: item for item in trajectory_diagnostics}
        all_images = []
        submap_records = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            records = submap.get_keyframe_records()
            if not records:
                raise ValueError(f"submap {submap.get_id()} has no keyframe records for image quality diagnostics")
            frame_ids = submap.get_frame_ids()
            if frame_ids is None or len(records) != len(frame_ids):
                raise ValueError(
                    f"submap {submap.get_id()} image-quality record/frame counts differ: "
                    f"{len(records)}/{0 if frame_ids is None else len(frame_ids)}"
                )
            if any(record.frame_id != frame_id for record, frame_id in zip(records, frame_ids)):
                raise ValueError(f"submap {submap.get_id()} image-quality record frame IDs differ from submap frame IDs")
            if submap.get_id() not in trajectory_by_id:
                raise ValueError(f"submap {submap.get_id()} lacks matching trajectory diagnostics")
            loaded = []
            for index, record in enumerate(records):
                if type(record.timestamp_ns) is not int:
                    raise TypeError(f"submap {submap.get_id()} image-quality timestamp_ns must be an exact Python int")
                image = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"submap {submap.get_id()} cannot load keyframe image {record.image_path}")
                height, width = image.shape[:2]
                if height < 1 or width < 1:
                    raise ValueError(f"submap {submap.get_id()} has invalid image dimensions for {record.image_path}")
                loaded.append((index, record, image, width, height))
                all_images.append((width, height))
            submap_records.append((submap.get_id(), loaded))
        dimensions = set(all_images)
        if len(dimensions) > 1:
            print("[ImageQuality] warning: mixed keyframe resolutions; resizing to 640 px width for comparable scores")
        frame_quality, summaries = [], []
        for submap_id, loaded in submap_records:
            submap_frames = []
            for index, record, image, width, height in loaded:
                score = compute_laplacian_variance(normalize_image_size(image, dimensions))
                item = KeyframeImageQuality(submap_id, index, record.frame_id, record.timestamp_ns,
                                            record.image_path, width, height, score)
                frame_quality.append(item)
                submap_frames.append(item)
            scores = np.asarray([item.laplacian_variance for item in submap_frames], dtype=float)
            minimum = min(submap_frames, key=lambda item: item.laplacian_variance)
            trajectory = trajectory_by_id[submap_id]
            count_below = None if blur_laplacian_threshold is None else int(np.sum(scores < blur_laplacian_threshold))
            summaries.append(SubmapImageQualityDiagnostic(
                submap_id=submap_id, num_frames=len(submap_frames),
                first_timestamp_ns=submap_frames[0].timestamp_ns, last_timestamp_ns=submap_frames[-1].timestamp_ns,
                mean_laplacian_variance=float(np.mean(scores)), median_laplacian_variance=float(np.median(scores)),
                min_laplacian_variance=float(np.min(scores)), max_laplacian_variance=float(np.max(scores)),
                std_laplacian_variance=float(np.std(scores)),
                local_umeyama_rmse_m=trajectory.local_umeyama_rmse_m,
                local_umeyama_median_error_m=trajectory.local_umeyama_median_error_m,
                local_umeyama_max_error_m=trajectory.local_umeyama_max_error_m,
                min_sharpness_frame_index=minimum.frame_index, min_sharpness_timestamp_ns=minimum.timestamp_ns,
                num_below_blur_threshold=count_below,
                fraction_below_blur_threshold=None if count_below is None else count_below / len(submap_frames),
            ))
        return frame_quality, summaries

    @staticmethod
    def write_keyframe_image_quality_to_csv(frame_quality, output_path, blur_laplacian_threshold=None) -> None:
        fields = list(KeyframeImageQuality.__dataclass_fields__)
        if blur_laplacian_threshold is not None:
            fields.append("below_blur_threshold")
        with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fields)
            writer.writeheader()
            for item in frame_quality:
                row = {field: getattr(item, field) for field in KeyframeImageQuality.__dataclass_fields__}
                if blur_laplacian_threshold is not None:
                    row["below_blur_threshold"] = item.laplacian_variance < blur_laplacian_threshold
                writer.writerow(row)
        print(f"[ImageQuality] wrote {output_path}")

    @staticmethod
    def write_submap_image_quality_to_csv(diagnostics, output_path) -> None:
        fields = list(SubmapImageQualityDiagnostic.__dataclass_fields__)
        with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fields)
            writer.writeheader()
            for item in diagnostics:
                writer.writerow({field: "" if getattr(item, field) is None else getattr(item, field) for field in fields})
        print(f"[ImageQuality] wrote {output_path}")

    @staticmethod
    def print_image_quality_diagnostics(diagnostics, correlations) -> None:
        for item in diagnostics:
            value = GraphMap._diagnostic_value
            print(f"[ImageQuality] submap={item.submap_id} frames={item.num_frames} "
                  f"lap_mean={item.mean_laplacian_variance:.6g} lap_median={item.median_laplacian_variance:.6g} "
                  f"lap_min={item.min_laplacian_variance:.6g} lap_max={item.max_laplacian_variance:.6g}\n"
                  f"               local_rmse_m={value(item.local_umeyama_rmse_m)} local_max_m={value(item.local_umeyama_max_error_m)}\n"
                  f"               blurriest_frame_index={item.min_sharpness_frame_index} timestamp_ns={item.min_sharpness_timestamp_ns}")
        if diagnostics:
            print(f"[ImageQuality] lowest_mean_sharpness_submap={min(diagnostics, key=lambda item: item.mean_laplacian_variance).submap_id}")
            print(f"[ImageQuality] lowest_median_sharpness_submap={min(diagnostics, key=lambda item: item.median_laplacian_variance).submap_id}")
            with_rmse = [item for item in diagnostics if item.local_umeyama_rmse_m is not None]
            if with_rmse:
                print(f"[ImageQuality] worst_local_rmse_submap={max(with_rmse, key=lambda item: item.local_umeyama_rmse_m).submap_id}")
        print(f"[ImageQualityCorrelation] N={sum(item.local_umeyama_rmse_m is not None for item in diagnostics)}")
        for name, value in correlations.items():
            print(f"[ImageQualityCorrelation] {name}={'n/a' if value is None else f'{value:.6g}'}")

    def get_submap_metric_alignments(self) -> list[SubmapMetricAlignment]:
        """Fit every ordinary local submap independently into Go2 ``odom``."""
        alignments = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            submap_id = submap.get_id()
            try:
                records = submap.get_keyframe_records()
                local_xyz = np.asarray(submap.get_local_camera_centers(), dtype=float)
                if records is None:
                    raise ValueError("keyframe metadata is unavailable")
                if len(records) != len(local_xyz):
                    raise ValueError(f"record/local-pose counts differ: {len(records)}/{len(local_xyz)}")
                if not records:
                    raise ValueError("no keyframes are available")
                metric_xyz = []
                for record in records:
                    if type(record.timestamp_ns) is not int or record.metric_pose is None:
                        raise ValueError("incomplete Go2 keyframe metadata")
                    position = np.asarray(record.metric_pose.position_xyz, dtype=float)
                    if position.shape != (3,) or not np.isfinite(position).all():
                        raise ValueError("invalid Go2 metric position")
                    metric_xyz.append(position)
                scale, rotation, translation, residuals = fit_similarity_umeyama(
                    local_xyz, np.asarray(metric_xyz, dtype=float)
                )
            except ValueError as error:
                print(f"[MetricSubmapAlignment] warning id={submap_id} fit unavailable: {error}")
                raise ValueError(f"metric alignment unavailable for submap {submap_id}: {error}") from error
            alignments.append(SubmapMetricAlignment(
                submap_id=submap_id,
                num_frames=len(records),
                first_timestamp_ns=records[0].timestamp_ns,
                last_timestamp_ns=records[-1].timestamp_ns,
                scale_m_per_vggt=float(scale),
                rotation_matrix=tuple(tuple(float(value) for value in row) for row in rotation),
                translation_m=tuple(float(value) for value in translation),
                rmse_m=float(np.sqrt(np.mean(residuals ** 2))),
                median_error_m=float(np.median(residuals)),
                max_error_m=float(np.max(residuals)),
            ))
        return alignments

    @staticmethod
    def _orientation_candidates(submap, alignment):
        """Return R_v, R_go2, and C_i=R_v.T @ R_align.T @ R_go2 for exact local keyframes."""
        records, local_rotations = submap.get_keyframe_records(), submap.get_local_camera_orientations()
        if records is None or len(records) != len(local_rotations) or not records:
            raise ValueError(f"submap {submap.get_id()} lacks matching local poses and Go2 metadata")
        r_align = validate_rotation_matrix(alignment.rotation_matrix)
        go2_rotations = []
        for record in records:
            if type(record.timestamp_ns) is not int or record.metric_pose is None:
                raise ValueError(f"submap {submap.get_id()} has incomplete Go2 keyframe metadata")
            go2_rotations.append(rotation_matrix_from_xyzw(record.metric_pose.quaternion_xyzw))
        local_rotations = np.asarray(local_rotations, dtype=float)
        candidates = [local.T @ r_align.T @ go2 for local, go2 in zip(local_rotations, go2_rotations)]
        return records, local_rotations, np.asarray(go2_rotations), [validate_rotation_matrix(item) for item in candidates]

    @staticmethod
    def _axis_alignment(candidates) -> CameraAxisAlignment:
        axis = mean_rotation(candidates)
        deviations = np.asarray([rotation_geodesic_angle_deg(axis, item) for item in candidates], dtype=float)
        return CameraAxisAlignment(
            tuple(tuple(float(value) for value in row) for row in axis), tuple(float(value) for value in R.from_matrix(axis).as_quat()),
            len(candidates), float(np.median(deviations)), float(np.mean(deviations)), float(np.max(deviations)),
            float(np.sqrt(np.mean(deviations ** 2))),
        )

    def get_orientation_consistency_diagnostics(self, metric_alignments):
        """Measure local orientations only; never reads the graph or mutates mapping state."""
        if metric_alignments is None:
            raise ValueError("metric_alignments must be supplied")
        alignment_by_id = {item.submap_id: item for item in metric_alignments}
        data, all_candidates = {}, []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            submap_id = submap.get_id()
            if submap_id not in alignment_by_id:
                raise ValueError(f"submap {submap_id} lacks a metric alignment")
            values = self._orientation_candidates(submap, alignment_by_id[submap_id])
            data[submap_id] = (submap, alignment_by_id[submap_id], *values)
            all_candidates.extend(values[-1])
        if not all_candidates:
            raise ValueError("no ordinary submap orientation candidates are available")
        global_axis = self._axis_alignment(all_candidates)
        global_rotation = np.asarray(global_axis.rotation_matrix)
        keyframes, summaries = [], []
        for submap_id, (submap, alignment, records, local, go2, candidates) in data.items():
            loso_rotation = None
            if len(data) >= 2:
                held_in = [candidate for other_id, values in data.items() if other_id != submap_id for candidate in values[-1]]
                loso_rotation = np.asarray(self._axis_alignment(held_in).rotation_matrix)
            r_align = np.asarray(alignment.rotation_matrix)
            raw, global_errors, loso_errors, step_errors = [], [], [], []
            for index, (record, r_v, r_go2) in enumerate(zip(records, local, go2)):
                prediction = r_align @ r_v
                raw_error = rotation_geodesic_angle_deg(r_go2, prediction)
                global_error = rotation_geodesic_angle_deg(r_go2, prediction @ global_rotation)
                loso_error = None if loso_rotation is None else rotation_geodesic_angle_deg(r_go2, prediction @ loso_rotation)
                raw.append(raw_error); global_errors.append(global_error)
                if loso_error is not None: loso_errors.append(loso_error)
                go2_step = vggt_step = step_error = None
                if index:
                    go2_step = rotation_geodesic_angle_deg(go2[index - 1], r_go2)
                    vggt_step = rotation_geodesic_angle_deg(local[index - 1], r_v)
                    step_error = abs(go2_step - vggt_step); step_errors.append(step_error)
                keyframes.append(KeyframeOrientationDiagnostic(submap_id, index, record.frame_id, record.timestamp_ns,
                    raw_error, global_error, loso_error, go2_step, vggt_step, step_error))
            stats = lambda values: (float(np.sqrt(np.mean(np.square(values)))), float(np.median(values)), float(np.max(values)))
            raw_stats, global_stats = stats(raw), stats(global_errors)
            loso_stats = (None, None, None) if not loso_errors else stats(loso_errors)
            step_stats = (None, None, None) if not step_errors else (float(np.mean(step_errors)), *stats(step_errors)[::2])
            summaries.append(SubmapOrientationDiagnostic(submap_id, len(records), records[0].timestamp_ns, records[-1].timestamp_ns,
                alignment.rmse_m, *raw_stats, *global_stats, *loso_stats, *step_stats))
        return global_axis, keyframes, summaries

    @staticmethod
    def write_keyframe_orientation_diagnostics_to_csv(diagnostics, output_path) -> None:
        fields = list(KeyframeOrientationDiagnostic.__dataclass_fields__)
        with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fields); writer.writeheader()
            for item in diagnostics:
                writer.writerow({field: str(getattr(item, field)) if field.endswith("timestamp_ns") else ("" if getattr(item, field) is None else getattr(item, field)) for field in fields})
        print(f"[OrientationDiag] wrote {output_path}")

    @staticmethod
    def write_submap_orientation_diagnostics_to_csv(diagnostics, output_path) -> None:
        fields = list(SubmapOrientationDiagnostic.__dataclass_fields__)
        with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fields); writer.writeheader()
            for item in diagnostics:
                writer.writerow({field: str(getattr(item, field)) if field.endswith("timestamp_ns") else ("" if getattr(item, field) is None else getattr(item, field)) for field in fields})
        print(f"[OrientationDiag] wrote {output_path}")

    @staticmethod
    def write_camera_axis_alignment(axis, output_path) -> None:
        payload = {"convention": "C_vggtcam_from_go2cam", **{field: getattr(axis, field) for field in CameraAxisAlignment.__dataclass_fields__}}
        with open(output_path, "w", encoding="utf-8") as file_handle: json.dump(payload, file_handle, indent=2)
        print(f"[OrientationAxis] wrote {output_path}")

    @staticmethod
    def print_orientation_consistency_diagnostics(axis, diagnostics) -> None:
        print("[OrientationAxis] convention=C_vggtcam_from_go2cam")
        print(f"[OrientationAxis] samples={axis.num_samples} quaternion_xyzw={axis.quaternion_xyzw}")
        print(f"[OrientationAxis] matrix=\n" + "\n".join(f"    {row}" for row in axis.rotation_matrix))
        print(f"[OrientationAxis] candidate_mean_deviation_deg={axis.candidate_mean_deviation_deg:.6g} candidate_median_deviation_deg={axis.candidate_median_deviation_deg:.6g} candidate_max_deviation_deg={axis.candidate_max_deviation_deg:.6g}")
        for item in diagnostics:
            value = GraphMap._diagnostic_value
            print(f"[OrientationDiag] submap={item.submap_id} frames={item.num_frames} position_rmse_m={item.position_rmse_m:.6g}\n"
                  f"                  raw_rmse_deg={item.raw_orientation_rmse_deg:.6g} global_axis_rmse_deg={item.global_axis_orientation_rmse_deg:.6g} loso_rmse_deg={value(item.loso_orientation_rmse_deg)} loso_median_deg={value(item.loso_orientation_median_deg)} loso_max_deg={value(item.loso_orientation_max_deg)} relative_step_rot_rmse_deg={value(item.relative_step_rotation_rmse_deg)}")
        valid = [item for item in diagnostics if item.loso_orientation_rmse_deg is not None]
        if valid:
            print(f"[OrientationDiag] worst_loso_orientation_submap={max(valid, key=lambda item: item.loso_orientation_rmse_deg).submap_id}")
            print(f"[OrientationDiag] best_loso_orientation_submap={min(valid, key=lambda item: item.loso_orientation_rmse_deg).submap_id}")
            print(f"[OrientationDiag] median_loso_orientation_rmse_deg={np.median([item.loso_orientation_rmse_deg for item in valid]):.6g}")

    @staticmethod
    def write_submap_orientation_comparison_plots(keyframes, summaries, output_dir) -> None:
        """Render angular residuals and basis-invariant step magnitudes; never plot Euler angles."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary_by_id = {item.submap_id: item for item in summaries}
        for submap_id in sorted(summary_by_id):
            frames = [item for item in keyframes if item.submap_id == submap_id]
            figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            indices = np.asarray([item.frame_index for item in frames])
            axes[0].plot(indices, [item.raw_orientation_error_deg for item in frames], "o-", label="raw")
            axes[0].plot(indices, [item.global_axis_corrected_error_deg for item in frames], "o-", label="global axis")
            loso = np.asarray([np.nan if item.loso_axis_corrected_error_deg is None else item.loso_axis_corrected_error_deg for item in frames])
            if np.isfinite(loso).any(): axes[0].plot(indices, loso, "o-", label="LOSO axis")
            axes[0].set(xlabel="frame index", ylabel="angular residual [deg]", title=f"Submap {submap_id} orientation residual")
            axes[0].legend(loc="best")
            transitions = [item for item in frames if item.go2_step_rotation_deg is not None]
            if transitions:
                transition_indices = [item.frame_index for item in transitions]
                axes[1].plot(transition_indices, [item.go2_step_rotation_deg for item in transitions], "o-", label="Go2")
                axes[1].plot(transition_indices, [item.vggt_step_rotation_deg for item in transitions], "o-", label="VGGT local")
            axes[1].set(xlabel="transition ending at frame", ylabel="rotation magnitude [deg]", title="Inter-keyframe rotation magnitude")
            axes[1].legend(loc="best")
            summary = summary_by_id[submap_id]
            axes[0].text(.02, .98, f"position RMSE: {summary.position_rmse_m:.4g} m\nLOSO RMSE: {GraphMap._diagnostic_value(summary.loso_orientation_rmse_deg)} deg\nstep RMSE: {GraphMap._diagnostic_value(summary.relative_step_rotation_rmse_deg)} deg\nframes: {summary.num_frames}", transform=axes[0].transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": .8})
            figure.tight_layout(); figure.savefig(Path(output_dir) / f"submap_{submap_id:06d}.png", dpi=150); plt.close(figure)
        print(f"[OrientationDiag] wrote orientation comparisons to {output_dir}")

    @staticmethod
    def plot_orientation_vs_position(diagnostics, output_path) -> None:
        """Exploratory labeled position-RMSE versus held-out orientation-RMSE scatter plot."""
        valid = [item for item in diagnostics if item.loso_orientation_rmse_deg is not None]
        if not valid:
            raise ValueError("LOSO orientation diagnostics require at least two ordinary submaps")
        figure, axis = plt.subplots(figsize=(6, 5))
        x, y = [item.position_rmse_m for item in valid], [item.loso_orientation_rmse_deg for item in valid]
        axis.scatter(x, y)
        for item in valid: axis.annotate(str(item.submap_id), (item.position_rmse_m, item.loso_orientation_rmse_deg), xytext=(4, 4), textcoords="offset points")
        axis.set(xlabel="Position RMSE [m]", ylabel="LOSO orientation RMSE [deg]", title="Position versus held-out orientation error")
        figure.tight_layout(); figure.savefig(output_path, dpi=150); plt.close(figure)
        print(f"[OrientationDiag] wrote plot {output_path}")

    def get_submap_motion_diagnostics(self, metric_alignments, trajectory_diagnostics) -> tuple[
        list[SubmapMotionStep], list[SubmapMotionDiagnostic]
    ]:
        """Measure Go2 motion and aligned local-VGGT residuals without using the graph."""
        if metric_alignments is None or trajectory_diagnostics is None:
            raise ValueError("metric_alignments and trajectory_diagnostics must be supplied")
        alignment_by_id = {item.submap_id: item for item in metric_alignments}
        trajectory_by_id = {item.submap_id: item for item in trajectory_diagnostics}
        steps, summaries = [], []
        epsilon = np.finfo(float).eps
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            submap_id = submap.get_id()
            if submap_id not in alignment_by_id or submap_id not in trajectory_by_id:
                raise ValueError(f"submap {submap_id} lacks matching alignment or trajectory diagnostics")
            records = submap.get_keyframe_records()
            local_xyz = np.asarray(submap.get_local_camera_centers(), dtype=float)
            if records is None or len(records) < 2 or len(records) != len(local_xyz):
                raise ValueError(f"submap {submap_id} requires matching local poses and at least two keyframes")
            timestamps, positions, quaternions = [], [], []
            for record in records:
                if type(record.timestamp_ns) is not int:
                    raise TypeError(f"submap {submap_id} timestamp_ns must be an exact Python int")
                if record.metric_pose is None:
                    raise ValueError(f"submap {submap_id} has incomplete Go2 keyframe metadata")
                position = np.asarray(record.metric_pose.position_xyz, dtype=float)
                quaternion = np.asarray(record.metric_pose.quaternion_xyzw, dtype=float)
                if position.shape != (3,) or not np.isfinite(position).all():
                    raise ValueError(f"submap {submap_id} has invalid Go2 metric position")
                timestamps.append(record.timestamp_ns)
                positions.append(position)
                quaternions.append(quaternion)
            metric_xyz, metric_quaternions = np.asarray(positions), np.asarray(quaternions)
            alignment = alignment_by_id[submap_id]
            aligned_xyz = apply_similarity(local_xyz, alignment.scale_m_per_vggt,
                                           np.asarray(alignment.rotation_matrix), np.asarray(alignment.translation_m))
            residual_3d = np.linalg.norm(metric_xyz - aligned_xyz, axis=1)
            residual_xy = np.linalg.norm(metric_xyz[:, :2] - aligned_xyz[:, :2], axis=1)
            rmse = float(np.sqrt(np.mean(residual_3d ** 2)))
            if not np.isclose(rmse, alignment.rmse_m, rtol=1e-7, atol=1e-9):
                raise ValueError(f"submap {submap_id} aligned residual RMSE differs from metric alignment")
            cumulative_translation = cumulative_rotation = 0.0
            submap_steps = []
            for index in range(len(records) - 1):
                dt_s = (timestamps[index + 1] - timestamps[index]) * 1e-9
                if not np.isfinite(dt_s) or dt_s <= 0.0:
                    raise ValueError(f"submap {submap_id} keyframe timestamps must be strictly increasing")
                translation = float(np.linalg.norm(metric_xyz[index + 1] - metric_xyz[index]))
                rotation = relative_rotation_angle_deg(metric_quaternions[index], metric_quaternions[index + 1])
                if not np.isfinite(translation) or translation < 0.0:
                    raise ValueError(f"submap {submap_id} has invalid metric step translation")
                cumulative_translation += translation
                cumulative_rotation += rotation
                submap_steps.append(SubmapMotionStep(submap_id, index, index + 1, timestamps[index], timestamps[index + 1],
                    float(dt_s), translation, rotation, cumulative_translation, cumulative_rotation, float(residual_3d[index + 1])))
            step_translation = np.asarray([item.translation_m for item in submap_steps])
            step_rotation = np.asarray([item.rotation_deg for item in submap_steps])
            endpoint = float(np.linalg.norm(metric_xyz[-1] - metric_xyz[0]))
            total_rotation_rad = float(np.deg2rad(step_rotation).sum())
            window_metadata = submap.get_window_metadata() if hasattr(submap, "get_window_metadata") else None
            summaries.append(SubmapMotionDiagnostic(
                submap_id, len(records), len(submap_steps), timestamps[0], timestamps[-1],
                float((timestamps[-1] - timestamps[0]) * 1e-9), cumulative_translation, endpoint,
                None if cumulative_translation <= epsilon else endpoint / cumulative_translation,
                float(np.mean(step_translation)), float(np.median(step_translation)), float(np.max(step_translation)),
                cumulative_rotation, float(np.mean(step_rotation)), float(np.median(step_rotation)), float(np.max(step_rotation)),
                None if total_rotation_rad <= epsilon else cumulative_translation / total_rotation_rad,
                alignment.scale_m_per_vggt, alignment.rmse_m, alignment.median_error_m, alignment.max_error_m,
                float(np.sqrt(np.mean(residual_xy ** 2))), float(np.max(residual_xy)),
                None if window_metadata is None else window_metadata.policy,
                None if window_metadata is None else window_metadata.trigger_reason,
                None if window_metadata is None else window_metadata.num_keyframes,
                None if window_metadata is None else window_metadata.cumulative_translation_m,
                None if window_metadata is None else window_metadata.cumulative_rotation_deg,
            ))
            steps.extend(submap_steps)
        return steps, summaries

    @staticmethod
    def write_submap_motion_steps_to_csv(steps, output_path) -> None:
        fields = list(SubmapMotionStep.__dataclass_fields__)
        with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fields)
            writer.writeheader()
            for item in steps:
                writer.writerow({field: str(getattr(item, field)) if field.endswith("timestamp_ns") else getattr(item, field)
                                 for field in fields})
        print(f"[MotionDiag] wrote {output_path}")

    @staticmethod
    def write_submap_motion_diagnostics_to_csv(diagnostics, output_path) -> None:
        fields = list(SubmapMotionDiagnostic.__dataclass_fields__)
        with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fields)
            writer.writeheader()
            for item in diagnostics:
                writer.writerow({field: "" if getattr(item, field) is None else
                                 (str(getattr(item, field)) if field.endswith("timestamp_ns") else getattr(item, field))
                                 for field in fields})
        print(f"[MotionDiag] wrote {output_path}")

    @staticmethod
    def print_submap_motion_diagnostics(diagnostics, correlations) -> None:
        for item in diagnostics:
            value = GraphMap._diagnostic_value
            print(f"[MotionDiag] submap={item.submap_id} frames={item.num_frames} duration_s={item.duration_s:.6g}\n"
                  f"             path_m={item.metric_path_length_m:.6g} endpoint_m={item.metric_endpoint_displacement_m:.6g} path_efficiency={value(item.path_efficiency)}\n"
                  f"             total_rot_deg={item.total_rotation_deg:.6g} max_step_rot_deg={item.max_step_rotation_deg:.6g}\n"
                  f"             median_step_trans_m={item.median_step_translation_m:.6g} trans_per_rot_m_per_rad={value(item.translation_per_rotation_m_per_rad)}\n"
                  f"             local_rmse_m={item.local_umeyama_rmse_m:.6g} xy_rmse_m={item.aligned_vggt_xy_rmse_m:.6g}")
        if diagnostics:
            ratio = [item for item in diagnostics if item.translation_per_rotation_m_per_rad is not None]
            if ratio: print(f"[MotionDiag] lowest_translation_per_rotation_submap={min(ratio, key=lambda item: item.translation_per_rotation_m_per_rad).submap_id}")
            print(f"[MotionDiag] highest_total_rotation_submap={max(diagnostics, key=lambda item: item.total_rotation_deg).submap_id}")
            print(f"[MotionDiag] highest_max_step_rotation_submap={max(diagnostics, key=lambda item: item.max_step_rotation_deg).submap_id}")
            print(f"[MotionDiag] worst_local_rmse_submap={max(diagnostics, key=lambda item: item.local_umeyama_rmse_m).submap_id}")
        print(f"[MotionCorrelation] N={len(diagnostics)}")
        for name, value in correlations.items():
            print(f"[MotionCorrelation] {name}={'n/a' if value is None else f'{value:.6g}'}")

    @staticmethod
    def print_submap_window_policy_summary(diagnostics) -> None:
        """Print a compact, threshold-free summary for the A/B window experiment."""
        if not diagnostics:
            return
        def summary(values):
            values = np.asarray(values, dtype=float)
            return f"min={values.min():.6g} median={np.median(values):.6g} max={values.max():.6g}"
        print(f"[SubmapPolicySummary] ordinary_submaps={len(diagnostics)}")
        print(f"[SubmapPolicySummary] frames {summary([item.num_frames for item in diagnostics])}")
        print(f"[SubmapPolicySummary] path_m {summary([item.metric_path_length_m for item in diagnostics])}")
        print(f"[SubmapPolicySummary] rotation_deg {summary([item.total_rotation_deg for item in diagnostics])}")
        counts = {}
        for item in diagnostics:
            reason = item.window_trigger_reason or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
        print("[SubmapPolicySummary] trigger_counts: " + ", ".join(
            f"{reason}={count}" for reason, count in sorted(counts.items())
        ))

    def write_submap_trajectory_comparison_plots(self, metric_alignments, motion_diagnostics, output_dir) -> None:
        """Render local-only Sim(3)-aligned paths; deliberately never consults the graph."""
        alignment_by_id = {item.submap_id: item for item in metric_alignments}
        diagnostic_by_id = {item.submap_id: item for item in motion_diagnostics}
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            submap_id = submap.get_id()
            records = submap.get_keyframe_records()
            metric_xyz = np.asarray([record.metric_pose.position_xyz for record in records], dtype=float)
            alignment = alignment_by_id[submap_id]
            aligned_xyz = apply_similarity(np.asarray(submap.get_local_camera_centers(), dtype=float), alignment.scale_m_per_vggt,
                                           np.asarray(alignment.rotation_matrix), np.asarray(alignment.translation_m))
            residual_xyz = metric_xyz - aligned_xyz
            write_trajectory_comparison_plot(submap_id, metric_xyz, aligned_xyz, np.linalg.norm(residual_xyz, axis=1),
                                             np.linalg.norm(residual_xyz[:, :2], axis=1), diagnostic_by_id[submap_id], output_dir)
        print(f"[MotionDiag] wrote trajectory comparisons to {output_dir}")

    @staticmethod
    def _diagnostic_value(value):
        return "n/a" if value is None else f"{value:.6g}"

    def print_submap_trajectory_diagnostics(self, graph=None, diagnostics=None) -> list[SubmapTrajectoryDiagnostic]:
        """Print supplied diagnostics, or compute them once when not supplied."""
        if diagnostics is None:
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

    @staticmethod
    def _make_colored_point_cloud(points, colors, context):
        points = np.asarray(points).reshape(-1, 3)
        colors = np.asarray(colors).reshape(-1, 3)
        if len(points) != len(colors):
            raise ValueError(f"{context} point/color counts differ: {len(points)}/{len(colors)}")
        if len(points) == 0:
            raise ValueError(f"{context} has no points to export")
        if not np.isfinite(points).all() or not np.isfinite(colors).all():
            raise ValueError(f"{context} contains non-finite XYZ or RGB values")
        colors = colors.astype(float, copy=False)
        if colors.max() > 1.0:
            colors = colors / 255.0
        if (colors < 0.0).any() or (colors > 1.0).any():
            raise ValueError(f"{context} RGB must be in [0, 1] or [0, 255]")
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud

    def _metric_aligned_submap_clouds(self, alignments=None):
        if alignments is None:
            alignments = self.get_submap_metric_alignments()
        alignment_by_id = {item.submap_id: item for item in alignments}
        clouds = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            alignment = alignment_by_id[submap.get_id()]
            # This intentionally uses raw local dense geometry, never graph-world points.
            points = apply_similarity(
                submap.get_points_local(), alignment.scale_m_per_vggt,
                np.asarray(alignment.rotation_matrix), np.asarray(alignment.translation_m),
            )
            clouds.append((submap.get_id(), self._make_colored_point_cloud(
                points, submap.get_points_colors(), f"metric submap {submap.get_id()}"
            )))
        if not clouds:
            raise ValueError("cannot export metric map: no ordinary submaps are available")
        return clouds, alignments

    def get_metric_aligned_point_cloud(self, alignments=None):
        """Build an independent dense map in metres in the Go2 ``odom`` frame."""
        clouds, alignments = self._metric_aligned_submap_clouds(alignments)
        points = np.concatenate([np.asarray(cloud.points) for _, cloud in clouds], axis=0)
        colors = np.concatenate([np.asarray(cloud.colors) for _, cloud in clouds], axis=0)
        return self._make_colored_point_cloud(points, colors, "metric map"), alignments

    @staticmethod
    def _print_metric_alignments(alignments):
        for item in alignments:
            print(
                f"[MetricMap] submap={item.submap_id} scale={item.scale_m_per_vggt:.10g} "
                f"rmse_m={item.rmse_m:.10g} median_m={item.median_error_m:.10g} "
                f"max_m={item.max_error_m:.10g}"
            )

    def write_metric_aligned_map(self, file_name, alignments=None) -> list[SubmapMetricAlignment]:
        point_cloud, alignments = self.get_metric_aligned_point_cloud(alignments)
        if not o3d.io.write_point_cloud(str(file_name), point_cloud):
            raise IOError(f"failed to write metric-aligned point cloud to {file_name}")
        points = np.asarray(point_cloud.points)
        print(f"[MetricMap] ordinary_submaps={len(alignments)}")
        print(f"[MetricMap] points={len(points)}")
        print("[MetricMap] frame=odom")
        print("[MetricMap] units=meters")
        print(f"[MetricMap] min_xyz={tuple(np.min(points, axis=0))}")
        print(f"[MetricMap] max_xyz={tuple(np.max(points, axis=0))}")
        self._print_metric_alignments(alignments)
        print(f"[MetricMap] wrote {file_name}")
        return alignments

    def get_camera_space_roundtrip_diagnostics(self, tolerance=1e-5):
        """Validate retained ordinary-submap geometry before diagnostic exports."""
        diagnostics = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            records = submap.get_keyframe_records()
            for item in submap.get_camera_space_roundtrip_diagnostics(tolerance):
                index = item["frame_index"]
                record = records[index] if records is not None else None
                item = dict(item)
                item["frame_id"] = None if record is None else record.frame_id
                item["timestamp_ns"] = None if record is None else record.timestamp_ns
                diagnostics.append(item)
            values = [item for item in diagnostics if item["submap_id"] == submap.get_id()]
            print(
                f"[CameraSpaceRoundTrip] submap={submap.get_id()} frames={len(values)} "
                f"rmse_max={max(item['rmse_vggt_units'] for item in values):.8g} "
                f"max_error={max(item['max_error_vggt_units'] for item in values):.8g} status=pass"
            )
        if not diagnostics:
            raise ValueError("cannot validate camera-space geometry: no ordinary submap frames are available")
        return diagnostics

    @staticmethod
    def write_camera_space_roundtrip_to_csv(diagnostics, output_path) -> None:
        fields = ["submap_id", "frame_index", "frame_id", "timestamp_ns", "num_points",
                  "rmse_vggt_units", "median_error_vggt_units", "max_error_vggt_units"]
        with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fields)
            writer.writeheader()
            for item in diagnostics:
                row = dict(item)
                if row["timestamp_ns"] is not None:
                    row["timestamp_ns"] = str(row["timestamp_ns"])
                writer.writerow(row)
        print(f"[CameraSpaceRoundTrip] wrote {output_path}")

    def _camera_pose_diagnostic_point_cloud(self, mode, alignments=None):
        if mode not in {"orientation_corrected", "go2_pose"}:
            raise ValueError(f"unknown camera-pose diagnostic mode: {mode}")
        if alignments is None:
            alignments = self.get_submap_metric_alignments()
        alignment_by_id = {item.submap_id: item for item in alignments}
        points_by_frame, colors_by_frame, timestamps = [], [], []
        appearances = 0
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            submap_id = submap.get_id()
            if submap_id not in alignment_by_id:
                raise ValueError(f"submap {submap_id} lacks a metric alignment")
            alignment = alignment_by_id[submap_id]
            records = submap.get_keyframe_records()
            centers = submap.get_local_camera_centers()
            if records is None or len(records) != len(centers):
                raise ValueError(f"submap {submap_id} lacks matching keyframe records and local centers")
            for index, record in enumerate(records):
                if record.metric_pose is None or type(record.timestamp_ns) is not int:
                    raise ValueError(f"submap {submap_id} frame {index} lacks Go2 metric pose metadata")
                camera_points, colors = submap.get_frame_points_colors_camera(index)
                rotation = metric_pose_to_optical_camera_to_odom(record.metric_pose)
                scale = alignment.scale_m_per_vggt
                if mode == "orientation_corrected":
                    center = apply_similarity(
                        centers[index:index + 1], scale, np.asarray(alignment.rotation_matrix),
                        np.asarray(alignment.translation_m),
                    )[0]
                else:
                    center = np.asarray(record.metric_pose.position_xyz, dtype=float)
                if center.shape != (3,) or not np.isfinite(center).all():
                    raise ValueError(f"submap {submap_id} frame {index} has invalid metric camera origin")
                points_by_frame.append((rotation @ (scale * camera_points).T).T + center)
                colors_by_frame.append(colors)
                timestamps.append(record.timestamp_ns)
                appearances += 1
        if not points_by_frame:
            raise ValueError(f"cannot export {mode} map: no ordinary submap points are available")
        points = np.concatenate(points_by_frame, axis=0)
        colors = np.concatenate(colors_by_frame, axis=0)
        cloud = self._make_colored_point_cloud(points, colors, f"{mode} map")
        print(
            f"[{ 'OrientationCorrectedMap' if mode == 'orientation_corrected' else 'Go2PoseMap'}] "
            f"frame_appearances={appearances} unique_timestamps={len(set(timestamps))} "
            f"overlap_duplicates={appearances - len(set(timestamps))}"
        )
        return cloud, alignments

    def get_orientation_corrected_point_cloud(self, alignments=None):
        """Diagnostic B: Go2 optical orientation with aligned VGGT camera centers."""
        return self._camera_pose_diagnostic_point_cloud("orientation_corrected", alignments)

    def get_go2_pose_point_cloud(self, alignments=None):
        """Diagnostic C: exact Go2 metric camera pose with VGGT camera geometry."""
        return self._camera_pose_diagnostic_point_cloud("go2_pose", alignments)

    def _write_camera_pose_diagnostic_map(self, file_name, mode, alignments=None):
        cloud, alignments = self._camera_pose_diagnostic_point_cloud(mode, alignments)
        if not o3d.io.write_point_cloud(str(file_name), cloud):
            raise IOError(f"failed to write {mode} point cloud to {file_name}")
        label = "OrientationCorrectedMap" if mode == "orientation_corrected" else "Go2PoseMap"
        print(f"[{label}] frame=odom units=meters points={len(cloud.points)} wrote {file_name}")
        return alignments

    def write_orientation_corrected_map(self, file_name, alignments=None):
        return self._write_camera_pose_diagnostic_map(file_name, "orientation_corrected", alignments)

    def write_go2_pose_map(self, file_name, alignments=None):
        return self._write_camera_pose_diagnostic_map(file_name, "go2_pose", alignments)

    def write_metric_aligned_submap_point_clouds(self, output_dir, alignments=None) -> list[SubmapMetricAlignment]:
        output_path = Path(output_dir)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"metric submap point-cloud output path is not a directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        clouds, alignments = self._metric_aligned_submap_clouds(alignments)
        for submap_id, point_cloud in clouds:
            filename = output_path / f"submap_{submap_id:06d}.ply"
            if not o3d.io.write_point_cloud(str(filename), point_cloud):
                raise IOError(f"failed to write metric submap point cloud {filename}")
            print(f"[MetricSubmapPLY] id={submap_id} frame=odom units=meters points={len(point_cloud.points)} wrote {filename}")
        return alignments

    @staticmethod
    def write_submap_metric_alignments_to_csv(alignments, output_path) -> None:
        fields = [
            "submap_id", "num_frames", "first_timestamp_ns", "last_timestamp_ns", "scale_m_per_vggt",
            "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22",
            "tx_m", "ty_m", "tz_m", "rmse_m", "median_error_m", "max_error_m",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fields)
            writer.writeheader()
            for item in alignments:
                rotation = np.asarray(item.rotation_matrix).reshape(-1)
                writer.writerow({
                    "submap_id": item.submap_id, "num_frames": item.num_frames,
                    "first_timestamp_ns": str(item.first_timestamp_ns),
                    "last_timestamp_ns": str(item.last_timestamp_ns),
                    "scale_m_per_vggt": f"{item.scale_m_per_vggt:.10f}",
                    **{f"r{i}{j}": f"{rotation[3 * i + j]:.10f}" for i in range(3) for j in range(3)},
                    "tx_m": f"{item.translation_m[0]:.10f}", "ty_m": f"{item.translation_m[1]:.10f}",
                    "tz_m": f"{item.translation_m[2]:.10f}", "rmse_m": f"{item.rmse_m:.10f}",
                    "median_error_m": f"{item.median_error_m:.10f}", "max_error_m": f"{item.max_error_m:.10f}",
                })
        print(f"[MetricMap] wrote alignment CSV {output_path}")

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
