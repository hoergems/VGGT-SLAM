import os
import time
import threading
import argparse

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.cameras import BACKENDS, CameraFrame, Go2Camera, Go2ConnectionError
from vggt_slam.frame_metadata import KeyframeRecord
from vggt_slam.frame_overlap import compute_image_sharpness
from vggt_slam.open3d_viewer import Open3DMapViewer
from vggt_slam.submap_window import MetricMotionWindowState, SubmapWindowMetadata

from vggt.models.vggt import VGGT

# --- Thread Safety Primitives ---
solver_lock = threading.Lock()  # Ensures only one solver thread runs at a time
data_lock = threading.Lock()    # Protects shared SLAM state (solver)
map_updated_event = threading.Event()  # Signals the main thread that a new global map snapshot is ready

parser = argparse.ArgumentParser(description="VGGT-SLAM RealSense live demo")
parser.add_argument("--keyframe_folder", type=str, default="keyframes", help="Folder to save captured keyframes")
parser.add_argument("--camera", type=str, default="realsense", choices=list(BACKENDS.keys()), help="Camera backend (default: realsense)")
parser.add_argument("--go2_host", type=str, default="192.168.123.24", help="Go2 protocol-v2 TCP host: the Jetson bridge IP for live operation, or 127.0.0.1 for camera_odom_replay.py")
parser.add_argument("--go2_port", type=int, default=5432, help="Go2 protocol-v2 TCP port")
parser.add_argument("--go2_receive_timeout_s", type=float, default=1.0, help="Go2 TCP receive timeout in seconds")
parser.add_argument("--go2_exit_on_disconnect", action="store_true", help="Exit the capture loop cleanly on Go2 TCP disconnect instead of reconnecting (for one-shot camera_odom_replay.py sessions)")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being built, otherwise only show the final map")
parser.add_argument("--vis_map_open3d", action="store_true", help="Visualize the current optimized colored map in a live native Open3D window as it is being built")
parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Applies to both the viser and Open3D live viewers. Default: no downsampling")
parser.add_argument("--vis_open3d_point_size", type=float, default=2.0, help="Open3D live-map render point size")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--submap_policy", choices=("fixed", "metric_motion"), default="fixed", help="Submap window policy. 'fixed' preserves the existing submap_size + overlap behavior. 'metric_motion' closes a Go2 submap when cumulative metric translation, cumulative rotation, or the maximum keyframe count is reached.")
parser.add_argument("--metric_motion_max_translation_m", type=float, default=1.5, help="Metric-motion policy cumulative camera-path limit in metres")
parser.add_argument("--metric_motion_max_rotation_deg", type=float, default=90.0, help="Metric-motion policy cumulative camera-rotation limit in degrees")
parser.add_argument("--metric_motion_max_keyframes", type=int, default=17, help="Metric-motion policy maximum VGGT keyframes, including overlap")
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
parser.add_argument("--vggt_trajectory_plot_path", type=str, default=None, help="Write an XY diagnostic plot of the unaligned VGGT camera trajectory")
parser.add_argument("--map_output_path", type=str, default=None, help="Write the final optimized colored VGGT point cloud to this file (recommended: .ply)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_keyframe(frame: CameraFrame, folder: str, frame_count: int) -> KeyframeRecord:
    """Persist one keyframe while retaining its source identity and metadata."""
    if frame.timestamp_ns is not None:
        if not isinstance(frame.timestamp_ns, int):
            raise TypeError("Go2 timestamp_ns must be an exact Python int")
        filename = os.path.join(folder, f"{frame.timestamp_ns}.png")
    else:
        filename = os.path.join(folder, f"frame_{frame_count:06d}.png")
    if not cv2.imwrite(filename, frame.image):
        raise IOError(f"Failed to save keyframe to {filename}")
    record = KeyframeRecord(
        image_path=filename,
        # This is the pre-existing local VGGT realtime identity, deliberately
        # separate from the Go2 source timestamp used in the filename.
        frame_id=frame_count,
        timestamp_ns=frame.timestamp_ns,
        metric_pose=frame.metric_pose,
        sequence_id=frame.sequence_id,
    )
    if record.timestamp_ns is not None:
        if record.metric_pose is None:
            raise ValueError("Go2 keyframe metadata is incomplete or inconsistent")
    return record


def create_camera(args):
    """Construct the configured camera backend, wiring Go2 TCP endpoint options."""
    if args.camera == "go2":
        return Go2Camera(
            host=args.go2_host,
            port=args.go2_port,
            receive_timeout_s=args.go2_receive_timeout_s,
        )
    return BACKENDS[args.camera]()


def restart_camera(camera, stop_event=None, retry_delay: float = 2.0):
    """Stop and re-start the camera, retrying until a device is streaming again.

    Used to recover from a mid-session disconnect (e.g. a RealSense USB drop),
    where ``capture()`` raises. Honors ``stop_event`` so the user can still
    cancel while a camera is unplugged. Returns the working camera object, or
    None if aborted via stop_event.
    """
    try:
        camera.stop()
    except Exception:
        pass  # device may already be gone; ignore teardown errors

    attempt = 0
    while stop_event is None or not stop_event.is_set():
        attempt += 1
        try:
            camera.start()
            print(f"[Camera] Reconnected after {attempt} attempt(s).")
            return camera
        except Exception as e:
            print(f"[Camera] Reconnect attempt {attempt} failed: {e}.")
            print(f"[Camera] Retrying in {retry_delay:.0f}s...")
            time.sleep(retry_delay)
    print("[Camera] Reconnection aborted (session cancelled).")
    return None


def threaded_process_submap(keyframe_records, solver, model, args, clip_model, clip_preprocess, window_metadata=None):
    """Background thread: run VGGT inference + graph optimisation for one submap."""
    try:
        image_names = [record.image_path for record in keyframe_records]
        print(f"[SLAM] Processing submap ({len(keyframe_records)} frames)...")
        predictions = solver.run_predictions(
            image_names, model, args.max_loops, clip_model, clip_preprocess,
            keyframe_records=keyframe_records,
        )
        if window_metadata is not None:
            solver.current_working_submap.set_window_metadata(window_metadata)
        with data_lock:
            solver.add_points(predictions)
            solver.graph.optimize()
            if args.vis_map:
                if len(predictions.get("detected_loops", [])) > 0:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
        if args.vis_map_open3d:
            map_updated_event.set()
        print("[SLAM] Submap done.")
    except Exception as e:
        import traceback
        print(f"[SLAM ERROR] {e}")
        traceback.print_exc()
    finally:
        if solver_lock.locked():
            solver_lock.release()


def run_semantic_query_loop(args, solver, clip_model, clip_tokenizer, processor):
    """Interactive open-set semantic query loop, run after capture ends."""
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
            inference_state = processor.set_image(best_img)
            output = processor.set_text_prompt(state=inference_state, prompt=query)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
            print("Scores:", scores.cpu().numpy())

        masked_img = utils.overlay_masks(best_img, masks)
        masked_img.show()

        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy()
            obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(
                found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph)
            )
            solver.viewer.visualize_obb(
                center=obb_center,
                extent=obb_extent,
                rotation=obb_rotation,
                color=(255, 0, 0),
                line_width=8.0,
            )


def main():
    args = parser.parse_args()

    if args.overlapping_window_size != 1:
        parser.error("only --overlapping_window_size 1 is currently supported")
    if args.vis_open3d_point_size <= 0:
        parser.error("--vis_open3d_point_size must be greater than zero")
    if args.vis_voxel_size is not None and args.vis_voxel_size <= 0:
        parser.error("--vis_voxel_size must be greater than zero when provided")
    if not np.isfinite(args.keyframe_min_sharpness) or args.keyframe_min_sharpness < 0:
        parser.error("--keyframe_min_sharpness must be finite and non-negative")
    if args.submap_policy == "metric_motion":
        if not np.isfinite(args.metric_motion_max_translation_m) or args.metric_motion_max_translation_m <= 0:
            parser.error("--metric_motion_max_translation_m must be finite and positive")
        if not np.isfinite(args.metric_motion_max_rotation_deg) or args.metric_motion_max_rotation_deg <= 0:
            parser.error("--metric_motion_max_rotation_deg must be finite and positive")
        if args.metric_motion_max_keyframes < args.overlapping_window_size + 2:
            parser.error("--metric_motion_max_keyframes must be at least overlap + 2")
        print("[SubmapPolicy] policy=metric_motion")
        print(f"[SubmapPolicy] max_translation_m={args.metric_motion_max_translation_m}")
        print(f"[SubmapPolicy] max_rotation_deg={args.metric_motion_max_rotation_deg}")
        print(f"[SubmapPolicy] max_keyframes={args.metric_motion_max_keyframes}")
        print(f"[SubmapPolicy] overlap={args.overlapping_window_size}")
    else:
        print(f"[SubmapPolicy] policy=fixed new_frames={args.submap_size} overlap={args.overlapping_window_size} target_frames={args.submap_size + args.overlapping_window_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # When --run_os is set, SAM3/decord loads its own libxcb which poisons
    # OpenCV's XCB state and makes cv2.waitKey() hang. Skip cv2 display in that
    # case and print periodic status to the console instead.
    use_display = not args.run_os

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size,
        vis_imgs=args.vis_imgs,
    )

    open3d_viewer = None
    if args.vis_map_open3d:
        open3d_viewer = Open3DMapViewer(
            point_size=args.vis_open3d_point_size,
            voxel_size=args.vis_voxel_size,
        )
        open3d_viewer.start()

    print("Initializing and loading VGGT model...")

    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer, processor = None, None

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(torch.bfloat16)  # use half precision
    model = model.to(device)
    print("All models loaded. Starting SLAM loop.")

    # Register the viser object-query panel so the user can search for objects
    # live (and after capture) without using the terminal.
    if args.run_os:
        solver.viewer.add_object_query_gui(solver, clip_model, clip_tokenizer, processor, data_lock)

    # --- Camera setup ---
    camera = create_camera(args)
    print(f"Initializing {args.camera} camera...")
    os.makedirs(args.keyframe_folder, exist_ok=True)
    camera.start()

    # Warm up the camera — first few frames can be None
    print("Waiting for first camera frame...")
    first_frame = None
    go2_disconnected_before_first_frame = False
    while first_frame is None:
        try:
            first_frame = camera.capture()
        except Go2ConnectionError as e:
            if args.go2_exit_on_disconnect:
                print(f"[Go2] Disconnected before first frame ({e}). Exiting (--go2_exit_on_disconnect).")
                go2_disconnected_before_first_frame = True
                break
            print(f"[Camera] Error during warm-up ({e}). Reconnecting...")
            camera = restart_camera(camera)
        except Exception as e:
            print(f"[Camera] Error during warm-up ({e}). Reconnecting...")
            camera = restart_camera(camera)

    if go2_disconnected_before_first_frame:
        camera.stop()
        if open3d_viewer is not None:
            open3d_viewer.close()
        return

    if use_display:
        cv2.imshow("VGGT-SLAM Live", first_frame.image)
        cv2.waitKey(1)
    print("Camera ready.")

    frame_count = 0
    keyframe_records = []
    target_size = args.submap_size + args.overlapping_window_size
    metric_window = None
    if args.submap_policy == "metric_motion":
        metric_window = MetricMotionWindowState(
            args.metric_motion_max_translation_m,
            args.metric_motion_max_rotation_deg,
            args.metric_motion_max_keyframes,
        )
    submap_count = 0
    last_status_frame = 0  # for console status throttle when display is off
    stop_event = threading.Event()

    try:
        while True:
            if stop_event.is_set():
                break

            try:
                frame = camera.capture()
            except Go2ConnectionError as e:
                if args.go2_exit_on_disconnect:
                    print(f"[Go2] Disconnected ({e}). Exiting capture loop (--go2_exit_on_disconnect).")
                    break
                print(f"[Camera] Lost connection ({e}). Attempting to reconnect...")
                camera = restart_camera(camera, stop_event)
                if camera is None:  # session cancelled while reconnecting
                    break
                continue
            except Exception as e:
                print(f"[Camera] Lost connection ({e}). Attempting to reconnect...")
                camera = restart_camera(camera, stop_event)
                if camera is None:  # session cancelled while reconnecting
                    break
                continue

            if frame is None:
                continue

            img = frame.image

            frame_count += 1

            selected_record = None
            is_candidate = solver.flow_tracker.compute_disparity_candidate(
                img, args.min_disparity, args.vis_flow
            )
            if is_candidate:
                accepted = True
                if args.keyframe_min_sharpness > 0.0:
                    sharpness = compute_image_sharpness(img)
                    accepted = sharpness >= args.keyframe_min_sharpness
                    if not accepted:
                        print(
                            "\033[93m[Keyframe] Rejected blurry candidate: "
                            f"sharpness={sharpness:.2f} < "
                            f"threshold={args.keyframe_min_sharpness:.2f}\033[0m"
                        )
                if accepted:
                    solver.flow_tracker.accept_keyframe(img)
                    selected_record = save_keyframe(frame, args.keyframe_folder, frame_count)

            if args.submap_policy == "fixed":
                if selected_record is not None:
                    keyframe_records.append(selected_record)
                if len(keyframe_records) >= target_size:
                    if solver_lock.acquire(blocking=False):
                        submap_count += 1
                        print(f"[Main] Launching submap {submap_count} (frame {frame_count})...")
                        go2_records = [record for record in keyframe_records if record.timestamp_ns is not None]
                        if go2_records:
                            print(
                                f"[Main]   frames={len(keyframe_records)} "
                                f"first_timestamp_ns={go2_records[0].timestamp_ns} "
                                f"last_timestamp_ns={go2_records[-1].timestamp_ns} "
                                f"metric_poses={sum(record.metric_pose is not None for record in go2_records)}/{len(go2_records)}"
                            )
                        metadata = SubmapWindowMetadata("fixed", "max_keyframes", None, None, len(keyframe_records))
                        t = threading.Thread(target=threaded_process_submap,
                            args=(list(keyframe_records), solver, model, args, clip_model, clip_preprocess, metadata), daemon=True)
                        t.start()
                        keyframe_records = keyframe_records[-args.overlapping_window_size:]
                    else:
                        # SLAM still busy; cap the backlog so we don't grow unbounded.
                        if len(keyframe_records) > target_size * 2:
                            num_dropped = len(keyframe_records) - target_size
                            print(
                                f"\033[93m[WARNING] Dropping {num_dropped} pending keyframe records "
                                f"(backlog={len(keyframe_records)}, keeping={target_size})\033[0m"
                            )
                            keyframe_records = keyframe_records[-target_size:]
            elif selected_record is not None:
                metric_window.add_selected(selected_record)

            if args.submap_policy == "metric_motion" and metric_window.ready_records is not None:
                if solver_lock.acquire(blocking=False):
                    launched_records, status = metric_window.launch_ready(args.overlapping_window_size)
                    submap_count += 1
                    print(f"[SubmapPolicy] ready policy=metric_motion frames={status.num_keyframes}")
                    print(f"[SubmapPolicy] path_m={status.cumulative_translation_m:.6g} total_rot_deg={status.cumulative_rotation_deg:.6g}")
                    print(f"[SubmapPolicy] trigger={status.trigger_reason}")
                    print(f"[Main] Launching submap {submap_count} (frame {frame_count})...")
                    print(f"[Main]   frames={len(launched_records)} first_timestamp_ns={launched_records[0].timestamp_ns} last_timestamp_ns={launched_records[-1].timestamp_ns} metric_poses={len(launched_records)}/{len(launched_records)}")
                    metadata = SubmapWindowMetadata("metric_motion", status.trigger_reason,
                                                     status.cumulative_translation_m,
                                                     status.cumulative_rotation_deg, status.num_keyframes)
                    t = threading.Thread(
                        target=threaded_process_submap,
                        args=(launched_records, solver, model, args, clip_model, clip_preprocess, metadata),
                        daemon=True,
                    )
                    t.start()

            kf = len(keyframe_records) if args.submap_policy == "fixed" else len(metric_window.active_records)
            slam_busy = solver_lock.locked()

            if use_display:
                display = img.copy()
                if args.submap_policy == "fixed":
                    status = f"KFs: {kf}/{target_size}  Submaps: {submap_count}"
                elif metric_window.ready_records is not None:
                    status = f"KFs: READY  pending: {len(metric_window.pending_records)}  Submaps: {submap_count}"
                else:
                    window_status = metric_window.evaluate_active()
                    status = (f"KFs: {kf}/{args.metric_motion_max_keyframes}  "
                              f"path: {window_status.cumulative_translation_m:.2f}/{args.metric_motion_max_translation_m:.2f}m  "
                              f"rot: {window_status.cumulative_rotation_deg:.0f}/{args.metric_motion_max_rotation_deg:.0f}deg  Submaps: {submap_count}")
                if slam_busy:
                    status += "  [SLAM running]"
                cv2.putText(display, status, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("VGGT-SLAM Live", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if frame_count - last_status_frame >= 30:
                    busy_str = "  [SLAM running]" if slam_busy else ""
                    if args.submap_policy == "fixed":
                        print(f"[Camera] frame={frame_count}  KFs={kf}/{target_size} submaps={submap_count}{busy_str}")
                    elif metric_window.ready_records is not None:
                        print(f"[Camera] frame={frame_count} KFs=READY pending={len(metric_window.pending_records)} submaps={submap_count}{busy_str}")
                    else:
                        window_status = metric_window.evaluate_active()
                        print(f"[Camera] frame={frame_count} KFs={kf}/{args.metric_motion_max_keyframes} path_m={window_status.cumulative_translation_m:.6g} total_rot_deg={window_status.cumulative_rotation_deg:.6g} pending={len(metric_window.pending_records)} submaps={submap_count}{busy_str}")
                    last_status_frame = frame_count

            if open3d_viewer is not None and open3d_viewer.is_active:
                if map_updated_event.is_set():
                    map_updated_event.clear()
                    with data_lock:
                        point_cloud = solver.get_global_point_cloud()
                    open3d_viewer.update(point_cloud)
                else:
                    open3d_viewer.poll()

    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
    finally:
        if camera is not None:
            camera.stop()
        if use_display:
            cv2.destroyAllWindows()

    # Wait for any in-flight submap to finish before final visualization/logging.
    with solver_lock:
        pass

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if open3d_viewer is not None and open3d_viewer.is_active:
        with data_lock:
            point_cloud = solver.get_global_point_cloud()
        open3d_viewer.update(point_cloud)

    if args.run_os:
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

    if args.map_output_path is not None:
        solver.map.write_points_to_file(solver.graph, args.map_output_path)

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)
        if not args.skip_dense_log:
            solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))

    if open3d_viewer is not None:
        open3d_viewer.close()


if __name__ == "__main__":
    main()
