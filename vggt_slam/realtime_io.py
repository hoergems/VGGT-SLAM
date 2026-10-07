"""Camera lifecycle and keyframe persistence for realtime sessions."""

import os
import shutil
import time

import cv2

from vggt_slam.cameras import BACKENDS, CameraFrame, Go2Camera
from vggt_slam.frame_metadata import KeyframeRecord


def reset_keyframe_folder(keyframe_folder: str) -> None:
    """Clear and recreate the configured keyframe output directory."""
    if not keyframe_folder.strip():
        raise ValueError("Keyframe folder path must not be empty")
    if os.path.exists(keyframe_folder):
        if not os.path.isdir(keyframe_folder):
            raise ValueError(
                f"Keyframe folder path exists but is not a directory: {keyframe_folder}"
            )
        shutil.rmtree(keyframe_folder)

    os.makedirs(keyframe_folder, exist_ok=True)


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
        imu_samples=frame.imu_samples,
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
            if stop_event is None:
                time.sleep(retry_delay)
            else:
                stop_event.wait(retry_delay)
    print("[Camera] Reconnection aborted (session cancelled).")
    return None



