"""Camera backends for real-time SLAM."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import socket as socket_module
import sys
from typing import Optional

import cv2
import numpy as np

from vggt_slam.go2_protocol import (
    HEADER_STRUCT,
    IMU_STRUCT,
    Go2Packet,
    ProtocolError,
    decode_header,
    decode_packet,
)
from vggt_slam.frame_metadata import ImuSample, MetricCameraPose


@dataclass(frozen=True)
class CameraFrame:
    """A BGR camera observation with optional source metadata."""

    image: np.ndarray
    timestamp_ns: int | None = None
    metric_pose: MetricCameraPose | None = None
    sequence_id: int | None = None
    odom_before_timestamp_ns: int | None = None
    odom_after_timestamp_ns: int | None = None
    imu_samples: tuple[ImuSample, ...] = ()


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


def _packet_to_camera_frame(packet: Go2Packet) -> CameraFrame | None:
    """Decode a protocol-v3 packet's JPEG into a camera frame."""
    image = cv2.imdecode(
        np.frombuffer(packet.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        print(
            f"[Go2] JPEG decode warning: seq={packet.sequence_id}",
            file=sys.stderr,
            flush=True,
        )
        return None

    return CameraFrame(
        image=image,
        timestamp_ns=packet.camera_timestamp_ns,
        metric_pose=MetricCameraPose(
            position_xyz=packet.position_xyz,
            quaternion_xyzw=packet.quaternion_xyzw,
        ),
        sequence_id=packet.sequence_id,
        odom_before_timestamp_ns=packet.odom_before_timestamp_ns,
        odom_after_timestamp_ns=packet.odom_after_timestamp_ns,
        imu_samples=packet.imu_samples,
    )


def _decode_packet_to_camera_frame(packet_bytes: bytes) -> CameraFrame | None:
    """Decode one complete Go2 protocol-v3 packet into a camera frame."""
    return _packet_to_camera_frame(decode_packet(packet_bytes))


class Go2ConnectionError(ConnectionError):
    """Raised when the Go2 TCP connection is lost (EOF, reset, or broken pipe)."""


class Go2Camera(Camera):
    """Camera backend reading a direct TCP protocol-v3 stream from Go2CameraBridge.

    Works identically against a live Jetson bridge (``192.168.123.24:5432``)
    and a ``camera_odom_replay.py`` server (``127.0.0.1:5432``) since both
    speak the same ``G2CO`` protocol-v3 wire format over plain TCP.
    """

    def __init__(
        self,
        host: str = "192.168.123.24",
        port: int = 5432,
        receive_timeout_s: float = 1.0,
    ):
        if not host:
            raise ValueError("host must be non-empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be in range 1..65535")
        if receive_timeout_s <= 0:
            raise ValueError("receive_timeout_s must be positive")
        self.host = host
        self.port = port
        self.receive_timeout_s = receive_timeout_s
        self._socket: socket_module.socket | None = None
        self.previous_sequence_id: int | None = None
        self.previous_timestamp_ns: int | None = None

    # Corrupt headers must not turn into unbounded socket reads. These limits
    # remain far above normal Go2 operation (about 15 IMU samples/frame).
    MAX_IMU_SAMPLES = 1_000_000
    MAX_JPEG_BYTES = 64 * 1024 * 1024
    MAX_PAYLOAD_BYTES = 128 * 1024 * 1024

    def start(self) -> None:
        if self._socket is not None:
            return

        sock = socket_module.create_connection(
            (self.host, self.port), timeout=self.receive_timeout_s
        )
        sock.settimeout(self.receive_timeout_s)

        self._socket = sock
        self.previous_sequence_id = None
        self.previous_timestamp_ns = None
        print(f"[Go2] Connected to {self.host}:{self.port}", flush=True)

    def _read_exact(self, num_bytes: int, allow_idle_timeout: bool = False) -> bytes:
        """Read exactly ``num_bytes`` from the socket, or raise on EOF/timeout.

        ``allow_idle_timeout`` is only appropriate at a frame boundary (before
        any byte of the next frame has been consumed): there, a timeout with
        zero bytes read so far is a normal "no frame available yet" idle
        condition and is reported by returning ``b""``. Once any byte of a
        frame has been consumed — including this call itself, when
        ``allow_idle_timeout`` is False, as is always the case once framing is
        already committed to reading a declared-length payload — a timeout is
        a framing failure, not a resumable idle state, and is raised as a
        connection error rather than silently discarding the partial frame.
        """
        assert self._socket is not None
        chunks = bytearray()
        while len(chunks) < num_bytes:
            try:
                chunk = self._socket.recv(num_bytes - len(chunks))
            except socket_module.timeout:
                if len(chunks) == 0 and allow_idle_timeout:
                    return b""
                raise Go2ConnectionError(
                    f"timed out after reading {len(chunks)}/{num_bytes} bytes mid-frame"
                )
            except OSError as error:
                raise Go2ConnectionError(f"socket error while reading: {error}") from error

            if chunk == b"":
                if len(chunks) == 0:
                    raise Go2ConnectionError("connection closed (EOF)")
                raise Go2ConnectionError(
                    f"connection closed (EOF) after reading {len(chunks)}/{num_bytes} bytes mid-frame"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def _validate_progression(self, sequence_id: int, timestamp_ns: int) -> None:
        if self.previous_sequence_id is not None:
            if sequence_id > self.previous_sequence_id + 1:
                missing = sequence_id - self.previous_sequence_id - 1
                print(
                    f"[Go2] sequence gap: expected {self.previous_sequence_id + 1}, "
                    f"received {sequence_id}, missing {missing} frame(s)",
                    file=sys.stderr,
                    flush=True,
                )
            elif sequence_id <= self.previous_sequence_id:
                print(
                    f"[Go2] non-monotonic sequence: "
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
                f"[Go2] timestamp warning: previous {self.previous_timestamp_ns}, "
                f"received {timestamp_ns}",
                file=sys.stderr,
                flush=True,
            )
        self.previous_timestamp_ns = timestamp_ns

    def capture(self) -> Optional[CameraFrame]:
        if self._socket is None:
            raise RuntimeError("Go2 camera has not been started")

        header_bytes = self._read_exact(HEADER_STRUCT.size, allow_idle_timeout=True)
        if header_bytes == b"":
            return None

        try:
            header = decode_header(header_bytes)
        except ProtocolError as error:
            raise Go2ConnectionError(
                f"invalid Go2 protocol header: {error}"
            ) from error

        imu_bytes_length = header.imu_sample_count * IMU_STRUCT.size
        total_payload_bytes = imu_bytes_length + header.jpeg_length
        if (header.imu_sample_count > self.MAX_IMU_SAMPLES
                or header.jpeg_length > self.MAX_JPEG_BYTES
                or total_payload_bytes > self.MAX_PAYLOAD_BYTES):
            raise Go2ConnectionError(
                "declared Go2 protocol payload exceeds safety limits: "
                f"imu_samples={header.imu_sample_count}, jpeg_bytes={header.jpeg_length}"
            )
        imu_bytes = self._read_exact(imu_bytes_length)
        jpeg_bytes = self._read_exact(header.jpeg_length)
        try:
            packet = decode_packet(header_bytes + imu_bytes + jpeg_bytes)
        except ProtocolError as error:
            raise Go2ConnectionError(f"invalid Go2 protocol packet: {error}") from error

        self._validate_progression(packet.sequence_id, packet.camera_timestamp_ns)
        return _packet_to_camera_frame(packet)

    def stop(self) -> None:
        sock = self._socket
        self._socket = None
        self.previous_sequence_id = None
        self.previous_timestamp_ns = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


BACKENDS = {
    "realsense": RealSenseCamera,
    "go2": Go2Camera,
}
