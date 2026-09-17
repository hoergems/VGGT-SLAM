import os
import time
import threading
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.cameras import BACKENDS, CameraFrame, Go2Camera, Go2ConnectionError
from vggt_slam.frame_metadata import KeyframeRecord
from vggt_slam.image_quality import (
    compute_quality_error_correlations, plot_quality_vs_rmse,
    write_submap_contact_sheets, write_worst_keyframes_contact_sheet,
)
from vggt_slam.motion_diagnostics import compute_motion_error_correlations, plot_motion_vs_rmse
from vggt_slam.submap_window import MetricMotionWindowState, SubmapWindowMetadata

from vggt.models.vggt import VGGT

# --- Thread Safety Primitives ---
solver_lock = threading.Lock()  # Ensures only one solver thread runs at a time
data_lock = threading.Lock()    # Protects shared SLAM state (solver)

parser = argparse.ArgumentParser(description="VGGT-SLAM RealSense live demo")
parser.add_argument("--keyframe_folder", type=str, default="keyframes", help="Folder to save captured keyframes")
parser.add_argument("--camera", type=str, default="realsense", choices=list(BACKENDS.keys()), help="Camera backend (default: realsense)")
parser.add_argument("--go2_host", type=str, default="192.168.123.24", help="Go2 protocol-v2 TCP host: the Jetson bridge IP for live operation, or 127.0.0.1 for camera_odom_replay.py")
parser.add_argument("--go2_port", type=int, default=5432, help="Go2 protocol-v2 TCP port")
parser.add_argument("--go2_receive_timeout_s", type=float, default=1.0, help="Go2 TCP receive timeout in seconds")
parser.add_argument("--go2_exit_on_disconnect", action="store_true", help="Exit the capture loop cleanly on Go2 TCP disconnect instead of reconnecting (for one-shot camera_odom_replay.py sessions)")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being built, otherwise only show the final map")
parser.add_argument("--vis_imgs", action="store_true", help="Show camera images in the viser frustums. By default only the frustums are shown (faster visualization)")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
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
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--phase4_debug", action="store_true", help="Print temporary Phase 4 Go2 metadata integrity diagnostics for newly added submaps")
parser.add_argument("--metric_trajectory_path", type=str, default=None, help="Write unique Go2 metric keyframe trajectory as: timestamp_ns x y z qx qy qz qw")
parser.add_argument("--vggt_trajectory_path", type=str, default=None, help="Write unique timestamped optimized VGGT camera trajectory as: timestamp_ns x y z qx qy qz qw")
parser.add_argument("--vggt_trajectory_plot_path", type=str, default=None, help="Write an XY diagnostic plot of the unaligned VGGT camera trajectory")
parser.add_argument("--map_output_path", type=str, default=None, help="Write the final optimized colored VGGT point cloud to this file (recommended: .ply)")
parser.add_argument("--submap_alignment_mode", choices=("vggt", "metric"), default="vggt", help="Point-cloud alignment mode. 'vggt' keeps the graph-transformed map; 'metric' independently aligns ordinary submaps to Go2 odom.")
parser.add_argument("--metric_map_output_path", type=str, default=None, help="Write the independently Go2-odom-aligned metric submap map to this file")
parser.add_argument("--camera_space_roundtrip_path", type=str, default=None, help="Write per-frame retained-geometry camera-space round-trip validation as CSV")
parser.add_argument("--orientation_corrected_map_output_path", type=str, default=None, help="Write diagnostic B: Go2 optical orientation plus aligned VGGT camera positions as PLY")
parser.add_argument("--go2_pose_map_output_path", type=str, default=None, help="Write diagnostic C: full Go2 optical poses with VGGT camera geometry as PLY")
parser.add_argument("--metric_submap_point_cloud_dir", type=str, default=None, help="Write independently Go2-odom-aligned ordinary submaps as PLY files in this directory")
parser.add_argument("--metric_submap_alignment_path", type=str, default=None, help="Write independent local-VGGT-to-Go2-odom submap fits as CSV")
parser.add_argument("--submap_diagnostics_path", type=str, default=None, help="Write per-submap VGGT-versus-Go2 diagnostics as CSV")
parser.add_argument("--submap_diagnostics_plot_path", type=str, default=None, help="Write a per-submap scale and error diagnostic plot")
parser.add_argument("--submap_point_cloud_dir", type=str, default=None, help="Write each ordinary optimized submap as a PLY in this directory")
parser.add_argument("--keyframe_quality_path", type=str, default=None, help="Write one Laplacian-variance sharpness row per ordinary submap keyframe appearance")
parser.add_argument("--submap_image_quality_path", type=str, default=None, help="Write per-submap keyframe sharpness and local trajectory-error diagnostics as CSV")
parser.add_argument("--image_quality_plot_path", type=str, default=None, help="Write sharpness-versus-local-Umeyama-RMSE scatter plots")
parser.add_argument("--submap_contact_sheet_dir", type=str, default=None, help="Write one temporal keyframe sharpness contact sheet per ordinary submap")
parser.add_argument("--worst_keyframes_contact_sheet_path", type=str, default=None, help="Write a deduplicated global sheet of the lowest-sharpness keyframes")
parser.add_argument("--blur_laplacian_threshold", type=float, default=None, help="Optional diagnostic-only Laplacian-variance threshold; no default blur cutoff is assumed")
parser.add_argument("--submap_motion_steps_path", type=str, default=None, help="Write per-transition Go2 camera motion and aligned local-VGGT residuals as CSV")
parser.add_argument("--submap_motion_diagnostics_path", type=str, default=None, help="Write per-submap Go2 motion geometry and local-VGGT residual summaries as CSV")
parser.add_argument("--motion_quality_plot_path", type=str, default=None, help="Write motion-geometry versus local-Umeyama-RMSE scatter plots")
parser.add_argument("--submap_trajectory_comparison_dir", type=str, default=None, help="Write aligned local-VGGT versus Go2 trajectory comparison PNGs")
parser.add_argument("--keyframe_orientation_diagnostics_path", type=str, default=None, help="Write per-keyframe VGGT-versus-Go2 orientation diagnostics as CSV")
parser.add_argument("--submap_orientation_diagnostics_path", type=str, default=None, help="Write per-submap orientation diagnostic summaries as CSV")
parser.add_argument("--camera_axis_alignment_path", type=str, default=None, help="Write estimated C_vggtcam_from_go2cam diagnostic calibration as JSON")
parser.add_argument("--submap_orientation_plot_dir", type=str, default=None, help="Write one orientation comparison PNG per ordinary submap")
parser.add_argument("--orientation_vs_position_plot_path", type=str, default=None, help="Write position-RMSE versus LOSO-orientation-RMSE scatter plot")


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


def _phase4_record_errors(frame_id, record, image_name, lookup_record):
    """Return metadata consistency errors for one Go2 keyframe record."""
    errors = []
    if record.frame_id != frame_id:
        errors.append("frame ID mismatch")
    if type(record.timestamp_ns) is not int:
        errors.append("timestamp type mismatch")
    if image_name is None or Path(image_name).stem != str(record.timestamp_ns):
        errors.append("filename/timestamp mismatch")
    if type(record.timestamp_ns) is int:
        if lookup_record != record:
            errors.append("timestamp lookup mismatch")
    if record.image_path != image_name:
        errors.append("record/image mismatch")
    if record.metric_pose is None:
        errors.append("missing metric pose")
    return errors


def _print_phase4_submap_records(solver, submap, label):
    """Print and validate the metadata records belonging to one submap."""
    frame_ids = submap.get_frame_ids()
    records = submap.get_keyframe_records()

    if not records or not any(record.timestamp_ns is not None for record in records):
        print("[Phase4Check]   no keyframe metadata; skipping Go2 checks")
        return

    errors = []
    if frame_ids is None or len(frame_ids) != len(records):
        frame_count = 0 if frame_ids is None else len(frame_ids)
        errors.append(f"frame ID/record count mismatch ({frame_count}/{len(records)})")

    image_names = []
    for index in range(len(records)):
        try:
            image_names.append(submap.get_img_names_at_index(index))
        except IndexError:
            errors.append(f"index {index}: missing image name")
            image_names.append(None)
    image_count = sum(image_name is not None for image_name in image_names)
    if image_count != len(records):
        errors.append(f"image/record count mismatch ({image_count}/{len(records)})")

    for index, record in enumerate(records):
        if record.timestamp_ns is None:
            print(f"[Phase4Check]   i={index:02d} no Go2 metadata; skipping")
            continue

        frame_id = frame_ids[index] if frame_ids is not None and index < len(frame_ids) else None
        image_name = image_names[index]
        lookup_record = None
        if type(record.timestamp_ns) is int:
            lookup_record = solver.map.get_keyframe_record_by_timestamp(record.timestamp_ns)
        record_errors = _phase4_record_errors(frame_id, record, image_name, lookup_record)
        errors.extend(f"index {index}: {error}" for error in record_errors)
        image_basename = Path(image_name).name if image_name is not None else "<missing>"
        metric_pose = "yes" if record.metric_pose is not None else "no"
        lookup = "ok" if lookup_record == record else "mismatch"
        print(
            f"[Phase4Check]   i={index:02d} frame_id={frame_id} "
            f"timestamp_ns={record.timestamp_ns} image={image_basename} "
            f"metric_pose={metric_pose} lookup={lookup}"
        )

    if errors:
        print(f"[Phase4Check]   FAIL: {len(errors)} metadata inconsistencies")
        for error in errors:
            print(f"[Phase4Check]     - {error}")
    else:
        if label == "loop-closure":
            print("[Phase4Check]   PASS: loop-closure metadata preserved")
        else:
            print(f"[Phase4Check]   PASS: {len(records)}/{len(records)} records consistent")


def _print_phase4_overlap_diagnostics(solver, current_submap):
    """Validate the configured one-frame overlap with the preceding normal submap."""
    normal_submaps = [
        submap for submap in solver.map.ordered_submaps_by_key()
        if not submap.get_lc_status()
    ]
    try:
        current_index = normal_submaps.index(current_submap)
    except ValueError:
        return
    if current_index == 0:
        return

    previous = normal_submaps[current_index - 1]
    previous_records = previous.get_keyframe_records()
    current_records = current_submap.get_keyframe_records()
    if not previous_records or not current_records:
        print("[Phase4Check] Overlap metadata unavailable; skipping Go2 check")
        return

    previous_last = previous_records[-1]
    current_first = current_records[0]
    print(
        f"[Phase4Check] Overlap previous={previous.get_id()} "
        f"current={current_submap.get_id()}:"
    )
    print(f"[Phase4Check]   prev_last_timestamp_ns={previous_last.timestamp_ns}")
    print(f"[Phase4Check]   curr_first_timestamp_ns={current_first.timestamp_ns}")
    if (
        type(previous_last.timestamp_ns) is int
        and previous_last.timestamp_ns == current_first.timestamp_ns
        and previous_last == current_first
    ):
        print("[Phase4Check]   PASS: overlap metadata identical")
    else:
        print("[Phase4Check]   FAIL: overlap metadata mismatch")


def print_phase4_submap_diagnostics(solver, new_submap_ids):
    """Print temporary Phase 4 metadata diagnostics for submaps just added to the map."""
    for submap_id in new_submap_ids:
        submap = solver.map.get_submap(submap_id)
        is_loop_closure = submap.get_lc_status()
        frame_ids = submap.get_frame_ids()
        frame_count = 0 if frame_ids is None else len(frame_ids)
        if is_loop_closure:
            print(f"[Phase4Check] Loop-closure submap id={submap_id} frames={frame_count}")
            _print_phase4_submap_records(solver, submap, "loop-closure")
        else:
            print(f"[Phase4Check] Submap id={submap_id} lc=False frames={frame_count}")
            _print_phase4_submap_records(solver, submap, "normal")
            _print_phase4_overlap_diagnostics(solver, submap)


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
            before_ids = {
                submap.get_id()
                for submap in solver.map.get_submaps()
            }
            solver.add_points(predictions)
            solver.graph.optimize()
            after_ids = {
                submap.get_id()
                for submap in solver.map.get_submaps()
            }
            new_submap_ids = sorted(after_ids - before_ids)
            if args.phase4_debug:
                print_phase4_submap_diagnostics(solver, new_submap_ids)
            if args.vis_map:
                if len(predictions.get("detected_loops", [])) > 0:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
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
            if solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow):
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

    orientation_requested = any((args.keyframe_orientation_diagnostics_path, args.submap_orientation_diagnostics_path,
                                 args.camera_axis_alignment_path, args.submap_orientation_plot_dir,
                                 args.orientation_vs_position_plot_path))
    motion_requested = any((args.submap_motion_steps_path, args.submap_motion_diagnostics_path,
                            args.motion_quality_plot_path, args.submap_trajectory_comparison_dir))
    camera_space_requested = any((args.camera_space_roundtrip_path,
                                  args.orientation_corrected_map_output_path,
                                  args.go2_pose_map_output_path))
    metric_requested = any((args.metric_map_output_path, args.metric_submap_point_cloud_dir,
                            args.metric_submap_alignment_path, motion_requested, orientation_requested,
                            args.orientation_corrected_map_output_path, args.go2_pose_map_output_path))
    metric_alignments = solver.map.get_submap_metric_alignments() if metric_requested else None
    if args.metric_map_output_path is not None:
        solver.map.write_metric_aligned_map(args.metric_map_output_path, metric_alignments)
    if args.metric_submap_point_cloud_dir is not None:
        solver.map.write_metric_aligned_submap_point_clouds(args.metric_submap_point_cloud_dir, metric_alignments)
    if args.metric_submap_alignment_path is not None:
        solver.map.write_submap_metric_alignments_to_csv(metric_alignments, args.metric_submap_alignment_path)
    if camera_space_requested:
        roundtrip = solver.map.get_camera_space_roundtrip_diagnostics()
        if args.camera_space_roundtrip_path is not None:
            solver.map.write_camera_space_roundtrip_to_csv(roundtrip, args.camera_space_roundtrip_path)
        if args.orientation_corrected_map_output_path is not None:
            solver.map.write_orientation_corrected_map(args.orientation_corrected_map_output_path, metric_alignments)
        if args.go2_pose_map_output_path is not None:
            solver.map.write_go2_pose_map(args.go2_pose_map_output_path, metric_alignments)
    if args.submap_alignment_mode == "metric" and args.vis_map:
        print("[MetricMap] live Viser still shows the native VGGT-SLAM graph map; final metric output is written separately.")

    quality_requested = any((args.keyframe_quality_path, args.submap_image_quality_path,
                             args.image_quality_plot_path, args.submap_contact_sheet_dir,
                             args.worst_keyframes_contact_sheet_path))
    diagnostics_requested = args.submap_diagnostics_path is not None or args.submap_diagnostics_plot_path is not None
    trajectory_diagnostics = None
    if diagnostics_requested or quality_requested or motion_requested:
        trajectory_diagnostics = solver.map.get_submap_trajectory_diagnostics(solver.graph)
    if diagnostics_requested:
        diagnostics = trajectory_diagnostics
        solver.map.print_submap_trajectory_diagnostics(diagnostics=diagnostics)
        if args.submap_diagnostics_path is not None:
            solver.map.write_submap_trajectory_diagnostics_to_csv(diagnostics, args.submap_diagnostics_path)
        if args.submap_diagnostics_plot_path is not None:
            solver.map.plot_submap_scale_diagnostics(diagnostics, args.submap_diagnostics_plot_path)

    if quality_requested:
        frame_quality, submap_quality = solver.map.get_keyframe_image_quality_diagnostics(
            trajectory_diagnostics, blur_laplacian_threshold=args.blur_laplacian_threshold,
        )
        correlations = compute_quality_error_correlations(submap_quality)
        solver.map.print_image_quality_diagnostics(submap_quality, correlations)
        if args.keyframe_quality_path is not None:
            solver.map.write_keyframe_image_quality_to_csv(
                frame_quality, args.keyframe_quality_path, args.blur_laplacian_threshold,
            )
        if args.submap_image_quality_path is not None:
            solver.map.write_submap_image_quality_to_csv(submap_quality, args.submap_image_quality_path)
        if args.image_quality_plot_path is not None:
            plot_quality_vs_rmse(submap_quality, args.image_quality_plot_path)
        if args.submap_contact_sheet_dir is not None:
            write_submap_contact_sheets(frame_quality, submap_quality, args.submap_contact_sheet_dir,
                                        args.blur_laplacian_threshold)
        if args.worst_keyframes_contact_sheet_path is not None:
            write_worst_keyframes_contact_sheet(frame_quality, args.worst_keyframes_contact_sheet_path)

    if motion_requested:
        motion_steps, motion_diagnostics = solver.map.get_submap_motion_diagnostics(
            metric_alignments, trajectory_diagnostics,
        )
        correlations = compute_motion_error_correlations(motion_diagnostics)
        solver.map.print_submap_motion_diagnostics(motion_diagnostics, correlations)
        solver.map.print_submap_window_policy_summary(motion_diagnostics)
        if args.submap_motion_steps_path is not None:
            solver.map.write_submap_motion_steps_to_csv(motion_steps, args.submap_motion_steps_path)
        if args.submap_motion_diagnostics_path is not None:
            solver.map.write_submap_motion_diagnostics_to_csv(motion_diagnostics, args.submap_motion_diagnostics_path)
        if args.motion_quality_plot_path is not None:
            plot_motion_vs_rmse(motion_diagnostics, args.motion_quality_plot_path)
        if args.submap_trajectory_comparison_dir is not None:
            solver.map.write_submap_trajectory_comparison_plots(
                metric_alignments, motion_diagnostics, args.submap_trajectory_comparison_dir,
            )

    if orientation_requested:
        axis_alignment, keyframe_orientation, submap_orientation = solver.map.get_orientation_consistency_diagnostics(metric_alignments)
        solver.map.print_orientation_consistency_diagnostics(axis_alignment, submap_orientation)
        if args.keyframe_orientation_diagnostics_path is not None:
            solver.map.write_keyframe_orientation_diagnostics_to_csv(keyframe_orientation, args.keyframe_orientation_diagnostics_path)
        if args.submap_orientation_diagnostics_path is not None:
            solver.map.write_submap_orientation_diagnostics_to_csv(submap_orientation, args.submap_orientation_diagnostics_path)
        if args.camera_axis_alignment_path is not None:
            solver.map.write_camera_axis_alignment(axis_alignment, args.camera_axis_alignment_path)
        if args.submap_orientation_plot_dir is not None:
            solver.map.write_submap_orientation_comparison_plots(keyframe_orientation, submap_orientation, args.submap_orientation_plot_dir)
        if args.orientation_vs_position_plot_path is not None:
            solver.map.plot_orientation_vs_position(submap_orientation, args.orientation_vs_position_plot_path)

    if args.submap_point_cloud_dir is not None:
        solver.map.write_submap_point_clouds(solver.graph, args.submap_point_cloud_dir)

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)
        if not args.skip_dense_log:
            solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))


if __name__ == "__main__":
    main()
