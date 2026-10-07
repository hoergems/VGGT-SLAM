"""Owner/facade of a persistent, action-synchronized realtime SLAM session."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import threading

from .cameras import Go2ConnectionError
from .realtime_io import (
    create_camera, reset_keyframe_folder, restart_camera, save_keyframe,
)

from .realtime_processing import (
    process_submap,
    retain_fixed_window_overlap,
    should_flush_final_fixed_window,
)


class CameraDisconnectedError(RuntimeError):
    """Terminal Go2 disconnect requested by the one-shot capture policy."""


def validate_realtime_args(args):
    """Validate configuration independently of any CLI parser."""
    if args.submap_size <= 0:
        raise ValueError("--submap_size must be positive")
    if args.overlapping_window_size < 0:
        raise ValueError("--overlapping_window_size must be non-negative")
    if args.vis_open3d_point_size <= 0:
        raise ValueError("--vis_open3d_point_size must be greater than zero")
    if args.vis_voxel_size is not None and args.vis_voxel_size <= 0:
        raise ValueError("--vis_voxel_size must be greater than zero when provided")
    if not math.isfinite(args.keyframe_min_sharpness) or args.keyframe_min_sharpness < 0:
        raise ValueError("--keyframe_min_sharpness must be finite and non-negative")
    if not math.isfinite(args.go2_odom_translation_scale) or args.go2_odom_translation_scale <= 0:
        raise ValueError("--go2_odom_translation_scale must be finite and positive")
    if args.planning_map_output_path is not None:
        if not args.metricize_submaps_from_go2:
            raise ValueError("--planning_map_output_path requires --metricize_submaps_from_go2")
        if args.camera != "go2":
            raise ValueError("--planning_map_output_path requires --camera go2")
        if not args.planning_map_output_path.lower().endswith(".ply"):
            raise ValueError("--planning_map_output_path supports .ply files only")


def _create_solver(**kwargs):
    from .solver import Solver
    return Solver(**kwargs)


def _load_model(args, device):
    import torch
    from vggt.models.vggt import VGGT

    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    model.eval()
    return model.to(torch.bfloat16).to(device)


def _create_snapshot(path):
    from .incremental_odom_map import IncrementalOdomMapSnapshot
    return IncrementalOdomMapSnapshot(path)


def _load_clip(args):
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True).cuda()
    return (model, transforms.get_image_transform(model.image_size),
            transforms.get_text_tokenizer(model.context_length))


def _create_viewer(**kwargs):
    from .open3d_viewer import Open3DMapViewer
    return Open3DMapViewer(**kwargs)


@dataclass(frozen=True)
class VGGTStatus:
    phase: str
    frame_count: int
    buffered_keyframes: int
    submap_count: int
    slam_busy: bool
    faulted: bool


@dataclass
class RealtimeSlamState:
    keyframe_records: list = field(default_factory=list)
    submap_count: int = 0
    retained_overlap_size: int = 0


class VGGTInterface:
    def __init__(self, args, *, solver_factory=_create_solver,
                 model_factory=_load_model, camera_factory=create_camera,
                 planning_map_snapshot_factory=_create_snapshot,
                 clip_factory=_load_clip, viewer_factory=_create_viewer):
        """Create a ready, initially paused session.

        Factories are explicit testing seams; production callers supply only args.
        The camera readiness frame is discarded before admission starts.
        """
        validate_realtime_args(args)
        reset_keyframe_folder(args.keyframe_folder)
        self.args = args
        self.solver_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.map_updated_event = threading.Event()
        self.state = RealtimeSlamState()
        self._accept_frames = threading.Event()
        self._state_lock = threading.Lock()
        self._state_changed = threading.Condition(self._state_lock)
        self._close_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_ready = threading.Event()
        self._quit_requested = threading.Event()
        self._phase = "INITIALIZING"
        self._background_failure = None
        self._capture_failure = None
        self._capture_thread = None
        self._submap_worker = None
        self._solver = None
        self._planning_map_snapshot = None
        self._camera = None
        self._viewer = None
        self.frame_count = 0
        self.model = None
        self.clip_model = self.clip_preprocess = self.clip_tokenizer = None
        self._use_display = not args.run_os and not getattr(args, "no_live_display", False)
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._solver = solver_factory(
                init_conf_threshold=args.conf_threshold, lc_thres=args.lc_thres,
                vis_voxel_size=args.vis_voxel_size, vis_imgs=args.vis_imgs,
                metricize_submaps_from_go2=args.metricize_submaps_from_go2,
                go2_odom_translation_scale=args.go2_odom_translation_scale,
            )
            self._planning_map_snapshot = (
                planning_map_snapshot_factory(args.planning_map_output_path)
                if args.planning_map_output_path is not None else None
            )
            if args.run_os:
                self.clip_model, self.clip_preprocess, self.clip_tokenizer = clip_factory(args)
            self.model = model_factory(args, self.device)
            self._camera = camera_factory(args)
            try:
                self._camera.start()
            except Exception as exc:
                self._recover_camera(exc)
            while self._capture_frame() is None:
                self._stop_event.wait(0.01)
            if args.vis_map_open3d:
                self._viewer = viewer_factory(
                    point_size=args.vis_open3d_point_size, voxel_size=args.vis_voxel_size,
                )
            self._phase = "PAUSED"
            self._capture_thread = threading.Thread(
                target=self._capture_loop, name="vggt-capture", daemon=True,
            )
            try:
                self._capture_thread.start()
            except BaseException:
                self._capture_thread = None
                raise
            self._capture_ready.wait()
            self._raise_if_background_failed()
        except BaseException:
            # Construction must never leak a started camera or viewer.
            try:
                self.close()
            except BaseException:
                pass
            raise

    @property
    def solver(self):
        """Completed-map diagnostics/standalone compatibility, not a planner API."""
        return self._solver

    @property
    def planning_map_snapshot(self):
        """Standalone diagnostic access to the publisher."""
        return self._planning_map_snapshot

    @property
    def phase(self):
        with self._state_lock:
            return self._phase

    @property
    def is_alive(self):
        return self._capture_thread is not None and self._capture_thread.is_alive()

    @property
    def quit_requested(self):
        return self._quit_requested.is_set()

    def get_status(self):
        """Return diagnostic state, including sticky capture/processing faults."""
        # Read the buffer before the lifecycle lock, as in submap launch.
        with self._buffer_lock, self._state_lock:
            return VGGTStatus(
                self._phase, self.frame_count, len(self.state.keyframe_records),
                self.state.submap_count, self.solver_lock.locked(),
                self._background_failure is not None or self._capture_failure is not None,
            )

    def _recover_camera(self, exc):
        if self.args.go2_exit_on_disconnect and isinstance(exc, (Go2ConnectionError, ConnectionError)):
            raise CameraDisconnectedError(
                "SLAM capture failed: camera disconnected (--go2_exit_on_disconnect)"
            ) from exc
        print(f"[Camera] Lost connection ({exc}). Attempting to reconnect...")
        restart_camera(self._camera, self._stop_event)

    def _capture_frame(self):
        while not self._stop_event.is_set():
            try:
                return self._camera.capture()
            except Exception as exc:
                if self._stop_event.is_set():
                    return None
                self._recover_camera(exc)
        return None

    def _capture_loop(self):
        try:
            if self._viewer is not None:
                self._viewer.start()
            self._capture_ready.set()
            last_status_frame = 0
            while not self._stop_event.is_set():
                frame = self._capture_frame()
                if self._stop_event.is_set():
                    break
                if frame is None:
                    self._poll_viewer()
                    self._stop_event.wait(0.01)
                    continue
                self.frame_count += 1
                self._process_frame(frame)
                if self._use_display:
                    import cv2
                    status = self.get_status()
                    display = frame.image.copy()
                    target_size = self.args.submap_size + self.args.overlapping_window_size
                    text = f"KFs: {status.buffered_keyframes}/{target_size}  Submaps: {status.submap_count}  {status.phase}"
                    if status.slam_busy:
                        text += "  [SLAM running]"
                    cv2.putText(display, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    cv2.imshow("VGGT-SLAM Live", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self._quit_requested.set()
                        break
                elif self.frame_count - last_status_frame >= 30:
                    status = self.get_status()
                    print(f"[Camera] frame={status.frame_count} KFs={status.buffered_keyframes} submaps={status.submap_count} {status.phase}")
                    last_status_frame = self.frame_count
                self._poll_viewer()
        except BaseException as exc:
            with self._state_lock:
                self._capture_failure = exc
                self._accept_frames.clear()
        finally:
            self._capture_ready.set()
            try:
                self._close_live_viewers()
            except BaseException as exc:
                with self._state_lock:
                    if self._capture_failure is None:
                        self._capture_failure = exc
                    self._accept_frames.clear()

    def _process_frame(self, frame):
        with self._frame_processing() as admitted:
            if not admitted:
                return
            img = frame.image
            tracker = self.solver.flow_tracker
            if tracker.compute_disparity_candidate(img, self.args.min_disparity, self.args.vis_flow):
                accepted = True
                if self.args.keyframe_min_sharpness > 0:
                    from .frame_overlap import compute_image_sharpness
                    sharpness = compute_image_sharpness(img)
                    accepted = sharpness >= self.args.keyframe_min_sharpness
                    if not accepted:
                        print(f"[Keyframe] Rejected blurry candidate: sharpness={sharpness:.2f} < threshold={self.args.keyframe_min_sharpness:.2f}")
                if accepted:
                    tracker.accept_keyframe(img)
                    self.state.keyframe_records.append(
                        save_keyframe(frame, self.args.keyframe_folder, self.frame_count)
                    )
            self._launch_full_submap()

    def _poll_viewer(self):
        if self._viewer is not None and self._viewer.is_active:
            if self.map_updated_event.is_set():
                self.map_updated_event.clear()
                with self.data_lock:
                    points = self.solver.get_global_point_cloud()
                self._viewer.update(points)
            else:
                self._viewer.poll()

    def _close_live_viewers(self):
        if self._use_display:
            import cv2
            # Only this session's window; external OpenCV windows are untouched.
            try:
                cv2.destroyWindow("VGGT-SLAM Live")
            except cv2.error:
                pass
        if self._viewer is not None:
            self._viewer.close()

    @property
    def accepting_frames(self):
        return self._accept_frames.is_set()

    def resume_slam(self):
        """Enable frame admission without resetting any SLAM state."""
        with self._state_lock:
            if self._phase in ("CLOSED", "CLOSING") or self._stop_event.is_set():
                raise RuntimeError(f"Cannot resume SLAM while {self._phase.lower()}")
            self._raise_if_background_failed()
            if self._phase != "PAUSED":
                raise RuntimeError(f"Cannot resume SLAM while {self._phase.lower()}")
            if self.model is None:
                raise RuntimeError("VGGT model has been released; recreate the session")
            self._phase = "RUNNING"
            self._accept_frames.set()
        print("[VGGTInterface] SLAM resumed; accepting action-execution frames.")

    @contextmanager
    def _frame_processing(self):
        """Serialize capture admission with flushing; never drop accepted work."""
        with self._buffer_lock:
            target_size = self.args.submap_size + self.args.overlapping_window_size
            admitted = self.accepting_frames
            # Apply backpressure before flow selection/saving rather than deleting
            # accepted keyframes. The next admitted frame can launch pending work.
            if self.solver_lock.locked() and len(self.state.keyframe_records) >= target_size * 2:
                admitted = False
            yield admitted

    def _process_submap(self, records):
        process_submap(
            records, self.solver, self.model, self.args, self.clip_model,
            self.clip_preprocess, self.planning_map_snapshot, self.data_lock,
            self.map_updated_event,
        )

    def _threaded_process_submap(self, records):
        try:
            self._process_submap(records)
        except BaseException as exc:
            with self._state_lock:
                self._background_failure = exc
                self._accept_frames.clear()
            print(f"[SLAM ERROR] {exc}")
        finally:
            self.solver_lock.release()

    def _raise_if_background_failed(self):
        # Failure is sticky: a partially mutated solver cannot safely resume.
        if self._background_failure is not None:
            raise RuntimeError("SLAM submap processing failed; recreate the session") from self._background_failure
        if self._capture_failure is not None:
            if isinstance(self._capture_failure, CameraDisconnectedError):
                raise self._capture_failure
            raise RuntimeError("SLAM capture failed; recreate the session") from self._capture_failure

    def _launch_full_submap(self):
        """Called inside the capture admission transaction."""
        records = self.state.keyframe_records
        target_size = self.args.submap_size + self.args.overlapping_window_size
        if len(records) < target_size or not self.solver_lock.acquire(blocking=False):
            return
        try:
            worker = threading.Thread(
                target=self._threaded_process_submap, args=(list(records),), daemon=True,
            )
            worker.start()
            self._submap_worker = worker
        except BaseException:
            self.solver_lock.release()
            raise
        self.state.submap_count += 1
        records[:] = retain_fixed_window_overlap(records, self.args.overlapping_window_size)
        self.state.retained_overlap_size = len(records)

    def update_slam(self):
        """Pause, drain accepted work and publish before returning.

        Calling while paused is allowed for EOF cleanup and empty intervals.
        Concurrent lifecycle calls raise. Failures leave the session paused and
        permanently faulted because graph mutations may already have occurred.
        """
        with self._state_lock:
            if self._phase not in ("PAUSED", "RUNNING") or self._stop_event.is_set():
                raise RuntimeError(f"Cannot update SLAM while {self._phase.lower()}")
            self._accept_frames.clear()
            self._phase = "UPDATING"
        print("[VGGTInterface] SLAM update requested; frame acceptance paused.")
        try:
            self._drain_and_flush()
            updates = getattr(self.planning_map_snapshot, "update_count", None)
            print(f"[VGGTInterface] SLAM update complete; submaps={self.state.submap_count} planning_map_update={updates}.")
        except BaseException as exc:
            with self._state_lock:
                if self._background_failure is None:
                    self._background_failure = exc
            raise
        finally:
            with self._state_lock:
                self._phase = "PAUSED"
                self._state_changed.notify_all()
        self._raise_if_background_failed()

    def _drain_and_flush(self):
        with self._buffer_lock:
            print("[VGGTInterface] Waiting for in-flight submap...")
            with self.solver_lock:
                if self._background_failure is not None:
                    raise RuntimeError("SLAM submap processing failed; recreate the session") from self._background_failure
                records = self.state.keyframe_records
                if should_flush_final_fixed_window(
                    len(records), self.state.submap_count, self.state.retained_overlap_size,
                ):
                    pending = list(records)
                    print(f"[VGGTInterface] Flushing partial submap ({len(pending)} frames).")
                    self._process_submap(pending)
                    self.state.submap_count += 1
                    records[:] = retain_fixed_window_overlap(pending, self.args.overlapping_window_size)
                    self.state.retained_overlap_size = len(records)

    def close(self):
        """Stop capture and commit accepted work. Idempotent even after faults.

        A terminal camera fault is reported by lifecycle calls/status; close can
        still flush its accepted tail. Graph-mutation failures prohibit flushing.
        """
        with self._close_lock:
            with self._state_changed:
                if self._phase == "CLOSED":
                    return
                self._accept_frames.clear()
                self._stop_event.set()
                while self._phase == "UPDATING":
                    self._state_changed.wait()
                self._accept_frames.clear()
                self._phase = "CLOSING"
            error = None
            try:
                if self._camera is not None:
                    try:
                        self._camera.stop()  # Interrupt a blocked backend read.
                    except Exception as exc:
                        error = exc
                if self._capture_thread is not None:
                    self._capture_thread.join()
                if self._background_failure is None:
                    self._drain_and_flush()
                else:
                    with self.solver_lock:
                        pass
            except BaseException as exc:
                error = exc
                if self._background_failure is None:
                    self._background_failure = exc
            finally:
                try:
                    if self._submap_worker is not None:
                        self._submap_worker.join()
                    if self._camera is not None:
                        self._camera.stop()  # Also covers a reconnect racing stop.
                    if self._capture_thread is None:
                        self._close_live_viewers()
                finally:
                    with self._state_changed:
                        self._phase = "CLOSED"
                        self._state_changed.notify_all()
            if error is not None:
                raise error
            self._raise_if_background_failed()

    def release_vggt_model(self):
        """Release VGGT after shutdown, before standalone SAM3 queries."""
        with self._state_lock:
            if self._phase != "CLOSED":
                raise RuntimeError("Close SLAM before releasing the VGGT model")
            self.model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise
        return False
