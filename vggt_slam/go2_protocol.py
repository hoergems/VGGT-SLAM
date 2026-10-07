"""Wire protocol helpers for frames published by ``go2_camera_bridge`` (v3).

This module deliberately has no ROS, NumPy, OpenCV, or solver dependency so
the protocol can be validated independently from camera transport.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from vggt_slam.frame_metadata import ImuSample


MAGIC = b"G2CO"
PROTOCOL_VERSION = 3
HEADER_STRUCT = struct.Struct("!4sB3xQQ7dQQII")
IMU_STRUCT = struct.Struct("!Q10d")

assert HEADER_STRUCT.size == 104, f"unexpected v3 header size: {HEADER_STRUCT.size}"
assert IMU_STRUCT.size == 88, f"unexpected v3 IMU record size: {IMU_STRUCT.size}"


class ProtocolError(ValueError):
    """Raised when a packet does not conform to the Go2 protocol-v3 format."""


def _are_finite(values) -> bool:
    """Return false for non-numeric values so validation stays ProtocolError-only."""
    try:
        return all(math.isfinite(value) for value in values)
    except TypeError:
        return False


@dataclass(frozen=True)
class Go2PacketHeader:
    """The fixed protocol-v3 header, without IMU and JPEG payloads."""

    sequence_id: int
    camera_timestamp_ns: int
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    odom_before_timestamp_ns: int
    odom_after_timestamp_ns: int
    imu_sample_count: int
    jpeg_length: int


@dataclass(frozen=True)
class Go2Packet:
    """One complete protocol-v3 frame with passive IMU metadata and JPEG.

    The transmitted pose is the interpolated Go2 body/base odometry pose
    (``/utlidar/robot_odom``), not a physically calibrated optical camera
    pose. Body-to-optical axis conversion and base-to-camera extrinsic
    calibration remain separate downstream/future concerns; this module
    preserves both pose and IMU values exactly as transmitted.
    """

    sequence_id: int
    camera_timestamp_ns: int
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    odom_before_timestamp_ns: int
    odom_after_timestamp_ns: int
    imu_samples: tuple[ImuSample, ...] = ()
    jpeg_bytes: bytes = b""


def _validate_header_fields(sequence_id, camera_timestamp_ns, position_xyz,
                            quaternion_xyzw, odom_before_timestamp_ns,
                            odom_after_timestamp_ns, imu_sample_count,
                            jpeg_length) -> None:
    for name, value in (("sequence_id", sequence_id), ("camera_timestamp_ns", camera_timestamp_ns),
                        ("odom_before_timestamp_ns", odom_before_timestamp_ns),
                        ("odom_after_timestamp_ns", odom_after_timestamp_ns)):
        if not isinstance(value, int) or not 0 <= value <= 0xFFFF_FFFF_FFFF_FFFF:
            raise ProtocolError(f"{name} must fit in uint64")
    if len(position_xyz) != 3:
        raise ProtocolError("position_xyz must contain exactly three values")
    if len(quaternion_xyzw) != 4:
        raise ProtocolError("quaternion_xyzw must contain exactly four values")
    if not _are_finite((*position_xyz, *quaternion_xyzw)):
        raise ProtocolError("camera pose contains a non-finite value")
    if not any(quaternion_xyzw):
        raise ProtocolError("camera quaternion must not be all zero")
    for name, value in (("imu_sample_count", imu_sample_count), ("jpeg_length", jpeg_length)):
        if not isinstance(value, int) or not 0 <= value <= 0xFFFF_FFFF:
            raise ProtocolError(f"{name} must fit in uint32")
    if not odom_before_timestamp_ns <= camera_timestamp_ns <= odom_after_timestamp_ns:
        raise ProtocolError(
            "invalid odometry synchronization bracket: expected "
            f"odom_before ({odom_before_timestamp_ns}) <= camera "
            f"({camera_timestamp_ns}) <= odom_after ({odom_after_timestamp_ns})"
        )


def _validate_imu_samples(imu_samples: tuple[ImuSample, ...], camera_timestamp_ns: int) -> None:
    previous_timestamp_ns: int | None = None
    for index, sample in enumerate(imu_samples):
        if not isinstance(sample, ImuSample):
            raise ProtocolError(f"imu_samples[{index}] must be an ImuSample")
        if not isinstance(sample.timestamp_ns, int) or not 0 <= sample.timestamp_ns <= 0xFFFF_FFFF_FFFF_FFFF:
            raise ProtocolError(f"imu_samples[{index}].timestamp_ns must fit in uint64")
        values = (*sample.orientation_xyzw, *sample.angular_velocity_xyz,
                  *sample.linear_acceleration_xyz)
        if len(values) != 10 or not _are_finite(values):
            raise ProtocolError(f"imu_samples[{index}] must contain ten finite values")
        if previous_timestamp_ns is not None and sample.timestamp_ns < previous_timestamp_ns:
            raise ProtocolError("IMU timestamps must be nondecreasing within a packet")
        if sample.timestamp_ns > camera_timestamp_ns:
            raise ProtocolError("IMU timestamp must not be after camera timestamp")
        previous_timestamp_ns = sample.timestamp_ns


def _validate_packet(packet: Go2Packet) -> None:
    if not isinstance(packet.jpeg_bytes, bytes):
        raise ProtocolError("jpeg_bytes must be bytes")
    if not isinstance(packet.imu_samples, tuple):
        raise ProtocolError("imu_samples must be an immutable tuple")
    _validate_header_fields(packet.sequence_id, packet.camera_timestamp_ns,
                            packet.position_xyz, packet.quaternion_xyzw,
                            packet.odom_before_timestamp_ns, packet.odom_after_timestamp_ns,
                            len(packet.imu_samples), len(packet.jpeg_bytes))
    _validate_imu_samples(packet.imu_samples, packet.camera_timestamp_ns)


def encode_packet(packet: Go2Packet) -> bytes:
    """Encode a complete packet for tests or compatible protocol-v3 producers."""
    _validate_packet(packet)
    header = HEADER_STRUCT.pack(
        MAGIC, PROTOCOL_VERSION, packet.sequence_id, packet.camera_timestamp_ns,
        *packet.position_xyz, *packet.quaternion_xyzw,
        packet.odom_before_timestamp_ns, packet.odom_after_timestamp_ns,
        len(packet.imu_samples), len(packet.jpeg_bytes),
    )
    imu_bytes = b"".join(
        IMU_STRUCT.pack(sample.timestamp_ns, *sample.orientation_xyzw,
                        *sample.angular_velocity_xyz, *sample.linear_acceleration_xyz)
        for sample in packet.imu_samples
    )
    return header + imu_bytes + packet.jpeg_bytes


def decode_header(header_bytes: bytes) -> Go2PacketHeader:
    """Decode and validate the fixed protocol-v3 header."""
    if len(header_bytes) != HEADER_STRUCT.size:
        raise ProtocolError(f"header is {len(header_bytes)} bytes, expected exactly {HEADER_STRUCT.size}")
    (magic, version, sequence_id, camera_timestamp_ns, px, py, pz, qx, qy, qz, qw,
     odom_before_timestamp_ns, odom_after_timestamp_ns, imu_sample_count,
     jpeg_length) = HEADER_STRUCT.unpack(header_bytes)
    if magic != MAGIC:
        raise ProtocolError(f"invalid magic {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}; expected {PROTOCOL_VERSION}")
    position_xyz = (px, py, pz)
    quaternion_xyzw = (qx, qy, qz, qw)
    _validate_header_fields(sequence_id, camera_timestamp_ns, position_xyz, quaternion_xyzw,
                            odom_before_timestamp_ns, odom_after_timestamp_ns,
                            imu_sample_count, jpeg_length)
    return Go2PacketHeader(
        sequence_id=sequence_id, camera_timestamp_ns=camera_timestamp_ns,
        position_xyz=position_xyz, quaternion_xyzw=quaternion_xyzw,
        odom_before_timestamp_ns=odom_before_timestamp_ns,
        odom_after_timestamp_ns=odom_after_timestamp_ns,
        imu_sample_count=imu_sample_count, jpeg_length=jpeg_length,
    )


def decode_packet(packet_bytes: bytes) -> Go2Packet:
    """Decode and validate one complete protocol-v3 packet."""
    if len(packet_bytes) < HEADER_STRUCT.size:
        raise ProtocolError(f"packet is {len(packet_bytes)} bytes, shorter than {HEADER_STRUCT.size}-byte header")
    header = decode_header(packet_bytes[:HEADER_STRUCT.size])
    imu_length = header.imu_sample_count * IMU_STRUCT.size
    expected_length = HEADER_STRUCT.size + imu_length + header.jpeg_length
    if len(packet_bytes) != expected_length:
        if len(packet_bytes) < expected_length:
            raise ProtocolError(f"packet is truncated: expected {expected_length} bytes, got {len(packet_bytes)}")
        raise ProtocolError(f"packet has {len(packet_bytes) - expected_length} unexpected trailing bytes")
    imu_start = HEADER_STRUCT.size
    imu_samples = tuple(
        ImuSample(values[0], tuple(values[1:5]), tuple(values[5:8]), tuple(values[8:11]))
        for values in (IMU_STRUCT.unpack_from(packet_bytes, imu_start + i * IMU_STRUCT.size)
                       for i in range(header.imu_sample_count))
    )
    packet = Go2Packet(
        sequence_id=header.sequence_id, camera_timestamp_ns=header.camera_timestamp_ns,
        position_xyz=header.position_xyz, quaternion_xyzw=header.quaternion_xyzw,
        odom_before_timestamp_ns=header.odom_before_timestamp_ns,
        odom_after_timestamp_ns=header.odom_after_timestamp_ns, imu_samples=imu_samples,
        jpeg_bytes=packet_bytes[imu_start + imu_length:],
    )
    _validate_packet(packet)
    return packet
