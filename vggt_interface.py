"""Action boundaries for a persistent realtime VGGT-SLAM session.

Capture callers use ``_frame_processing`` around selection, saving and
``_launch_full_submap``. This admission transaction lets update_slam stop
new work immediately and then drain a frame already being selected.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading

from vggt_slam.realtime_processing import (
    process_submap,
    retain_fixed_window_overlap,
    should_flush_final_fixed_window,
)


@dataclass
class RealtimeSlamState:
    keyframe_records: list = field(default_factory=list)
    submap_count: int = 0
    retained_overlap_size: int = 0


class VGGTInterface:
    def __init__(self, solver, model, args, clip_model=None, clip_preprocess=None,
                 planning_map_snapshot=None, solver_lock=None, data_lock=None,
                 map_updated_event=None):
        self.solver = solver
        self.model = model
        self.args = args
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.planning_map_snapshot = planning_map_snapshot
        self.solver_lock = solver_lock if solver_lock is not None else threading.Lock()
        self.data_lock = data_lock if data_lock is not None else threading.Lock()
        self.map_updated_event = map_updated_event if map_updated_event is not None else threading.Event()
        if args.submap_size <= 0 or args.overlapping_window_size < 0:
            raise ValueError("submap_size must be positive and overlap non-negative")
        self.state = RealtimeSlamState()
        self._accept_frames = threading.Event()
        self._state_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._phase = "PAUSED"
        self._background_failure = None

    @property
    def accepting_frames(self):
        return self._accept_frames.is_set()

    def resume_slam(self):
        """Enable frame admission without resetting any SLAM state."""
        with self._state_lock:
            if self._phase != "PAUSED":
                raise RuntimeError(f"Cannot resume SLAM while {self._phase.lower()}")
            self._raise_if_background_failed()
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
            if self._phase == "UPDATING":
                raise RuntimeError("SLAM update already in progress")
            self._accept_frames.clear()
            self._phase = "UPDATING"
        print("[VGGTInterface] SLAM update requested; frame acceptance paused.")
        try:
            with self._buffer_lock:
                print("[VGGTInterface] Waiting for in-flight submap...")
                with self.solver_lock:
                    self._raise_if_background_failed()
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
