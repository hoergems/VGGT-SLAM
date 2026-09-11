"""Wire protocol helpers for frames published by the Go2 IPC bridge.

This module deliberately has no ROS, NumPy, OpenCV, or ZeroMQ dependency so
the protocol can be validated independently from camera transport.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct


MAGIC = b"G2VG"
PROTOCOL_VERSION = 1
HEADER_STRUCT = struct.Struct("!4sBQQ7dI")


class ProtocolError(ValueError):
    """Raised when a packet does not conform to the Go2 bridge protocol."""


@dataclass(frozen=True)
class BridgeFrame:
    """One encoded frame received from the Go2 bridge."""

    sequence_id: int
    timestamp_ns: int
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    jpeg_bytes: bytes


def _validate_frame(frame: BridgeFrame) -> None:
    if not 0 <= frame.sequence_id <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ProtocolError("sequence_id must fit in uint64")
    if not 0 <= frame.timestamp_ns <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ProtocolError("timestamp_ns must fit in uint64")
    if len(frame.position_xyz) != 3:
        raise ProtocolError("position_xyz must contain exactly three values")
    if len(frame.quaternion_xyzw) != 4:
        raise ProtocolError("quaternion_xyzw must contain exactly four values")

    pose_values = (*frame.position_xyz, *frame.quaternion_xyzw)
    if not all(math.isfinite(value) for value in pose_values):
        raise ProtocolError("camera pose contains a non-finite value")
    if not any(frame.quaternion_xyzw):
        raise ProtocolError("camera quaternion must not be all zero")
    if not isinstance(frame.jpeg_bytes, bytes):
        raise ProtocolError("jpeg_bytes must be bytes")
    if len(frame.jpeg_bytes) > 0xFFFF_FFFF:
        raise ProtocolError("JPEG payload exceeds uint32 length")


def encode_frame(frame: BridgeFrame) -> bytes:
    """Encode a frame for tests or compatible Go2 bridge producers."""
    _validate_frame(frame)
    return HEADER_STRUCT.pack(
        MAGIC,
        PROTOCOL_VERSION,
        frame.sequence_id,
        frame.timestamp_ns,
        *frame.position_xyz,
        *frame.quaternion_xyzw,
        len(frame.jpeg_bytes),
    ) + frame.jpeg_bytes


def decode_frame(packet: bytes) -> BridgeFrame:
    """Decode and validate one complete ZeroMQ frame packet."""
    if len(packet) < HEADER_STRUCT.size:
        raise ProtocolError(
            f"packet is {len(packet)} bytes, shorter than {HEADER_STRUCT.size}-byte header"
        )

    magic, version, sequence_id, timestamp_ns, *pose_and_length = HEADER_STRUCT.unpack_from(packet)
    *pose, jpeg_length = pose_and_length

    if magic != MAGIC:
        raise ProtocolError(f"invalid magic {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version {version}; expected {PROTOCOL_VERSION}"
        )

    actual_length = len(packet) - HEADER_STRUCT.size
    if jpeg_length != actual_length:
        raise ProtocolError(
            f"JPEG length is {jpeg_length}, but packet contains {actual_length} payload bytes"
        )

    frame = BridgeFrame(
        sequence_id=sequence_id,
        timestamp_ns=timestamp_ns,
        position_xyz=(pose[0], pose[1], pose[2]),
        quaternion_xyzw=(pose[3], pose[4], pose[5], pose[6]),
        jpeg_bytes=packet[HEADER_STRUCT.size :],
    )
    _validate_frame(frame)
    return frame
