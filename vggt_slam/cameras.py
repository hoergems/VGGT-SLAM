"""Camera backends for real-time SLAM."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
from typing import Optional

import cv2
import numpy as np

from vggt_slam.go2_ipc import BridgeFrame, ProtocolError, decode_frame
from vggt_slam.frame_metadata import MetricCameraPose


@dataclass(frozen=True)
class CameraFrame:
    """A BGR camera observation with optional source metadata."""

    image: np.ndarray
    timestamp_ns: int | None = None
    metric_pose: MetricCameraPose | None = None
    sequence_id: int | None = None


class Camera(ABC):
    """Abstract camera that produces complete BGR uint8 observations."""

    @abstractmethod
    def start(self) -> None:
        """Open the device and begin streaming."""
        ...

    @abstractmethod
    def capture(self) -> Optional[CameraFrame]:
        """Return a BGR uint8 camera frame, or None if unavailable."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release the device."""
        ...


class RealSenseCamera(Camera):
    """Intel RealSense color stream via pyrealsense2."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = None

    def start(self) -> None:
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        self._pipeline.start(config)

    def capture(self) -> Optional[CameraFrame]:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if color is None:
            return None
        return CameraFrame(image=np.asanyarray(color.get_data()))

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


def _bridge_frame_to_camera_frame(bridge_frame: BridgeFrame) -> CameraFrame | None:
    """Decode a bridge frame JPEG into a camera frame."""
    image = cv2.imdecode(
        np.frombuffer(bridge_frame.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        print(
            f"[Go2IPC] JPEG decode warning: seq={bridge_frame.sequence_id}",
            file=sys.stderr,
            flush=True,
        )
        return None

    return CameraFrame(
        image=image,
        timestamp_ns=bridge_frame.timestamp_ns,
        metric_pose=MetricCameraPose(
            position_xyz=bridge_frame.position_xyz,
            quaternion_xyzw=bridge_frame.quaternion_xyzw,
        ),
        sequence_id=bridge_frame.sequence_id,
    )


def _decode_packet_to_camera_frame(packet: bytes) -> CameraFrame | None:
    """Decode one Go2 packet into a camera frame, discarding bad JPEG data."""
    return _bridge_frame_to_camera_frame(decode_frame(packet))


class Go2IPCCamera(Camera):
    """ROS-independent camera backend receiving Go2 frames over ZeroMQ."""

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        receive_timeout_ms: int = 1000,
    ):
        if receive_timeout_ms < 0:
            raise ValueError("receive_timeout_ms must be non-negative")
        self.endpoint = endpoint
        self.receive_timeout_ms = receive_timeout_ms
        self.context = None
        self.socket = None
        self.previous_sequence_id: int | None = None
        self.previous_timestamp_ns: int | None = None

    def start(self) -> None:
        if self.socket is not None:
            return

        import zmq

        context = zmq.Context()
        socket = None
        try:
            socket = context.socket(zmq.PULL)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, self.receive_timeout_ms)
            socket.connect(self.endpoint)
        except Exception:
            if socket is not None:
                socket.close(linger=0)
            context.term()
            raise

        self.context = context
        self.socket = socket
        self.previous_sequence_id = None
        self.previous_timestamp_ns = None
        print(f"[Go2IPC] Connected to {self.endpoint}", flush=True)

    def _validate_progression(self, sequence_id: int, timestamp_ns: int) -> None:
        if self.previous_sequence_id is not None:
            if sequence_id > self.previous_sequence_id + 1:
                missing = sequence_id - self.previous_sequence_id - 1
                print(
                    f"[Go2IPC] sequence gap: expected {self.previous_sequence_id + 1}, "
                    f"received {sequence_id}, missing {missing} frame(s)",
                    file=sys.stderr,
                    flush=True,
                )
            elif sequence_id <= self.previous_sequence_id:
                print(
                    f"[Go2IPC] non-monotonic sequence: "
                    f"previous={self.previous_sequence_id} current={sequence_id}",
                    file=sys.stderr,
                    flush=True,
                )
        self.previous_sequence_id = sequence_id

        if (
            self.previous_timestamp_ns is not None
            and timestamp_ns <= self.previous_timestamp_ns
        ):
            print(
                f"[Go2IPC] timestamp warning: previous {self.previous_timestamp_ns}, "
                f"received {timestamp_ns}",
                file=sys.stderr,
                flush=True,
            )
        self.previous_timestamp_ns = timestamp_ns

    def capture(self) -> Optional[CameraFrame]:
        if self.socket is None:
            raise RuntimeError("Go2 IPC camera has not been started")
        import zmq

        try:
            packet = self.socket.recv()
        except zmq.Again:
            return None

        try:
            bridge_frame = decode_frame(packet)
        except ProtocolError as error:
            print(f"[Go2IPC] protocol warning: {error}", file=sys.stderr, flush=True)
            return None

        self._validate_progression(bridge_frame.sequence_id, bridge_frame.timestamp_ns)
        return _bridge_frame_to_camera_frame(bridge_frame)

    def stop(self) -> None:
        socket, context = self.socket, self.context
        self.socket = None
        self.context = None
        self.previous_sequence_id = None
        self.previous_timestamp_ns = None
        if socket is not None:
            socket.close(linger=0)
        if context is not None:
            context.term()


BACKENDS = {
    "realsense": RealSenseCamera,
    "go2": Go2IPCCamera,
}
