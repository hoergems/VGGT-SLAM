import time
import argparse

import numpy as np
import torch

from vggt_interface import CameraDisconnectedError, VGGTInterface

from vggt_slam.sam3_utils import run_sam3_text_query
from vggt_slam.cameras import BACKENDS
from vggt_slam.odom_map_alignment import (
    build_first_frame_odom_alignment,
    pose_matrix_from_position_quaternion,
    rotation_angle_deg,
    summarize_trajectory_alignment,
    trajectory_alignment_diagnostics,
    write_trajectory_alignment_csv,
)


parser = argparse.ArgumentParser(description="VGGT-SLAM RealSense live demo")
parser.add_argument("--keyframe_folder", type=str, default="keyframes", help="Folder to save captured keyframes")
parser.add_argument("--camera", type=str, default="realsense", choices=list(BACKENDS.keys()), help="Camera backend (default: realsense)")
parser.add_argument("--go2_host", type=str, default="192.168.123.24", help="Go2 protocol-v3 TCP host: the Jetson bridge IP for live operation, or 127.0.0.1 for camera_odom_replay.py")
parser.add_argument("--go2_port", type=int, default=5432, help="Go2 protocol-v3 TCP port")
parser.add_argument("--go2_receive_timeout_s", type=float, default=1.0, help="Go2 TCP receive timeout in seconds")
parser.add_argument("--go2_exit_on_disconnect", action="store_true", help="Exit the capture loop cleanly on Go2 TCP disconnect instead of reconnecting (for one-shot camera_odom_replay.py sessions)")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being built, otherwise only show the final map")
parser.add_argument("--vis_map_open3d", action="store_true", help="Visualize the current optimized colored map in a live native Open3D window as it is being built")
parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Applies to both the viser and Open3D live viewers. Default: no downsampling")
parser.add_argument("--vis_open3d_point_size", type=float, default=2.0, help="Open3D live-map render point size")
parser.add_argument("--no_live_display", action="store_true", help="Disable the OpenCV live camera window")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="Number of overlapping frames retained between fixed-policy submaps")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument(
    "--keyframe_min_sharpness",
    type=float,
    default=0.0,
    help=(
        "Minimum variance-of-Laplacian sharpness required for a disparity "
        "candidate to become a keyframe. 0.0 disables sharpness filtering."
    ),
)
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--metric_trajectory_path", type=str, default=None, help="Write unique Go2 metric keyframe trajectory as: timestamp_ns x y z qx qy qz qw")
parser.add_argument("--vggt_trajectory_path", type=str, default=None, help="Write unique timestamped optimized VGGT camera trajectory as: timestamp_ns x y z qx qy qz qw")
parser.add_argument("--vggt_submap_trajectory_path", type=str, default=None, help="Write overlap-preserving optimized VGGT camera poses with actual ordinary-submap membership for metric-scale diagnostics")
parser.add_argument("--vggt_scale_diagnostics_path", type=str, default=None, help="Write raw/local VGGT, final optimized VGGT, metric camera positions, and VGGT-SLAM incoming submap scale factors for offline scale diagnostics")
parser.add_argument("--metricize_submaps_from_go2", action="store_true", help="EXPERIMENTAL: scale each ordinary VGGT submap from synchronized Go2 translation only")
parser.add_argument("--go2_odom_translation_scale", type=float, default=1.20, help="Physical translation calibration applied to Go2 odometry-derived metric geometry (default: 1.20); used by Go2 submap metricization")
parser.add_argument("--metric_submap_diagnostics_path", type=str, default=None, help="Write per-ordinary-submap Go2 metricization diagnostics CSV")
parser.add_argument("--vggt_trajectory_plot_path", type=str, default=None, help="Write an XY diagnostic plot of the unaligned VGGT camera trajectory")
parser.add_argument("--map_output_path", type=str, default=None, help="Write the final optimized colored VGGT point cloud to this file (recommended: .ply)")
parser.add_argument("--odom_aligned_map_output_path", type=str, default=None, help="EXPERIMENTAL: export the final graph-optimized metric VGGT map rigidly aligned to Go2 odom using the first synchronized optical-camera pose (.ply only)")
parser.add_argument("--planning_map_output_path", type=str, default=None, help="Write/replace the current complete graph-optimized map in Go2 odom after every ordinary-submap optimization (.ply only; requires Go2 metricization)")


def run_semantic_query_loop(args, solver, clip_model, clip_tokenizer, processor):
    """Interactive open-set semantic query loop, run after capture ends."""
    from torchvision.transforms.functional import to_pil_image
    import vggt_slam.slam_utils as utils
    while True:
        query = input("\nEnter text query or q to quit: ").strip()
        if len(query) == 0:
            print("Empty query. Exiting.")
            return
        if query == "q":
            print("Exiting.")
            return

        text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
        overall_best_score, overall_best_submap_id, overall_best_frame_index = \
            solver.map.retrieve_best_semantic_frame(text_emb)

        found_submap = solver.map.get_submap(overall_best_submap_id)

        best_img = found_submap.get_frame_at_index(overall_best_frame_index)
        print("Score:", overall_best_score)
        with torch.no_grad():
            best_img = to_pil_image(best_img)
            output = run_sam3_text_query(processor, best_img, query)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
            print("Scores:", scores.float().cpu().numpy())

        masked_img = utils.overlay_masks(best_img, masks)
        masked_img.show()

        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy()
            points_in_mask = found_submap.get_points_in_mask(
                overall_best_frame_index, mask, solver.graph
            )
            obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(
                points_in_mask
            )
            print("3D points in mask:", points_in_mask.shape)
            print("3D min:", points_in_mask.min(axis=0))
            print("3D max:", points_in_mask.max(axis=0))
            print("OBB center:", obb_center)
            print("OBB extent:", obb_extent)
            object_id = solver.map.add_object_obb(
                center=obb_center, extent=obb_extent, rotation=obb_rotation
            )
            print(f"Stored object OBB id={object_id} for final PLY export")
            solver.viewer.visualize_obb(
                center=obb_center,
                extent=obb_extent,
                rotation=obb_rotation,
                color=(255, 0, 0),
                line_width=8.0,
            )


def main():
    args = parser.parse_args()

    try:
        slam = VGGTInterface(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        slam.resume_slam()
        try:
            while slam.is_alive and not slam.quit_requested:
                if slam.get_status().faulted:
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[Main] Shutting down...")
    finally:
        # Shutdown uses the same action barrier and commits the partial tail.
        try:
            slam.close()
        except CameraDisconnectedError as exc:
            # One-shot replay EOF still commits its accepted tail and exports.
            print(f"[Go2] {exc}")

    solver = slam.solver
    planning_map_snapshot = slam.planning_map_snapshot
    data_lock = slam.data_lock
    clip_model, clip_tokenizer = slam.clip_model, slam.clip_tokenizer

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.run_os:
        # Mapping is complete and no future submap can use VGGT. Release it
        # before SAM3 construction so the two large models never coexist.
        print("Releasing VGGT model before loading SAM3...")
        slam.release_vggt_model()

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print("Initializing and loading SAM3 model...")
        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        # Live queries are deliberately unavailable during capture. Register
        # the Viser panel now that SAM3 is present for post-capture queries.
        solver.viewer.add_object_query_gui(
            solver, clip_model, clip_tokenizer, processor, data_lock
        )
        run_semantic_query_loop(args, solver, clip_model, clip_tokenizer, processor)

    if args.metric_trajectory_path is not None:
        solver.map.write_metric_camera_trajectory_to_file(args.metric_trajectory_path)

    if args.vggt_trajectory_path is not None or args.vggt_trajectory_plot_path is not None:
        vggt_trajectory = solver.map.get_vggt_camera_trajectory(solver.graph)
        if args.vggt_trajectory_path is not None:
            solver.map.write_timestamped_poses_to_file(
                args.vggt_trajectory_path,
                solver.graph,
                samples=vggt_trajectory,
            )
        if args.vggt_trajectory_plot_path is not None:
            solver.map.plot_vggt_camera_trajectory(vggt_trajectory, args.vggt_trajectory_plot_path)

    if args.vggt_submap_trajectory_path is not None:
        solver.map.write_vggt_submap_trajectory_to_file(
            args.vggt_submap_trajectory_path,
            solver.graph,
        )

    if args.vggt_scale_diagnostics_path is not None:
        solver.map.write_vggt_scale_diagnostics_to_file(
            args.vggt_scale_diagnostics_path,
            solver.graph,
        )

    if args.metric_submap_diagnostics_path is not None:
        solver.map.write_metric_submap_diagnostics_to_file(args.metric_submap_diagnostics_path)

    if args.map_output_path is not None:
        solver.map.write_points_to_file(solver.graph, args.map_output_path)

    if args.odom_aligned_map_output_path is not None:
        metric_samples = solver.map.get_metric_camera_trajectory()
        vggt_samples = solver.map.get_vggt_camera_trajectory(solver.graph)
        alignment = build_first_frame_odom_alignment(metric_samples, vggt_samples)
        if planning_map_snapshot is not None:
            if not np.allclose(
                planning_map_snapshot.transform_odom_vggt,
                alignment.transform_odom_vggt,
                atol=1e-8,
            ):
                raise ValueError("planning-map and final odom-map first-frame transforms disagree")

        p0_odom = np.asarray(alignment.metric_sample.position_xyz, dtype=float)
        print(
            "[OdomMapAlign] first_go2_optical_camera_position_odom="
            f"({p0_odom[0]:.6f}, {p0_odom[1]:.6f}, {p0_odom[2]:.6f})"
        )

        first_vggt_pose = pose_matrix_from_position_quaternion(
            alignment.vggt_sample.position_xyz,
            alignment.vggt_sample.quaternion_xyzw,
        )
        print(f"[OdomMapAlign] first_timestamp_ns={alignment.vggt_sample.timestamp_ns}")
        print(f"[OdomMapAlign] metric_frame_id={alignment.metric_sample.frame_id} vggt_frame_id={alignment.vggt_sample.frame_id} vggt_submap_id={alignment.vggt_sample.submap_id} vggt_frame_index={alignment.vggt_sample.frame_index}")
        print(f"[OdomMapAlign] first_vggt_position_norm={np.linalg.norm(first_vggt_pose[:3, 3]):.9g}")
        print(f"[OdomMapAlign] first_vggt_rotation_from_identity_deg={rotation_angle_deg(first_vggt_pose[:3, :3]):.9g}")
        rows = trajectory_alignment_diagnostics(metric_samples, vggt_samples, alignment.transform_odom_vggt)
        summary = summarize_trajectory_alignment(rows)
        first_row = rows[0]
        print(f"[OdomMapAlign] first_frame_position_error_m={first_row['position_error_m']:.9g}")
        print(f"[OdomMapAlign] first_frame_orientation_error_deg={first_row['orientation_error_deg']:.9g}")
        print(f"[OdomMapAlign] trajectory_samples={summary['count']}")
        for label, values in (("position", summary["position_error_m"]), ("orientation", summary["orientation_error_deg"])):
            print(f"[OdomMapAlign] trajectory_{label}_error mean={values['mean']:.9g} median={values['median']:.9g} p95={values['p95']:.9g} max={values['max']:.9g}")
        solver.map.write_odom_aligned_points_to_file(
            solver.graph, args.odom_aligned_map_output_path, alignment.transform_odom_vggt
        )
        csv_path = write_trajectory_alignment_csv(args.odom_aligned_map_output_path, rows)
        print(f"[OdomMapAlign] wrote {csv_path}")

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)
        if not args.skip_dense_log:
            solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))


if __name__ == "__main__":
    main()
