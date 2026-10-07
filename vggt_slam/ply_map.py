"""VGGT-SLAM PLY extensions for optimized-camera metadata.

The dense map remains a conventional PLY ``vertex`` element.  Camera poses
are kept in a separate, optional element so normal point-cloud readers can
ignore them.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PLYCameraPose:
    frame_id: int
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class PLYObjectOBB:
    """An object OBB whose row-major rotation maps local axes into world space."""

    object_id: int
    center_xyz: tuple[float, float, float]
    extent_xyz: tuple[float, float, float]
    rotation_matrix: tuple[float, float, float, float, float, float, float, float, float]


_VERTEX_DTYPE = np.dtype([
    ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
_CAMERA_DTYPE = np.dtype([
    ("frame_id", "<i4"), ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
    ("qx", "<f8"), ("qy", "<f8"), ("qz", "<f8"), ("qw", "<f8"),
])
_OBJECT_OBB_DTYPE = np.dtype([
    ("object_id", "<i4"),
    ("center_x", "<f8"), ("center_y", "<f8"), ("center_z", "<f8"),
    ("extent_x", "<f8"), ("extent_y", "<f8"), ("extent_z", "<f8"),
    ("rotation_00", "<f8"), ("rotation_01", "<f8"), ("rotation_02", "<f8"),
    ("rotation_10", "<f8"), ("rotation_11", "<f8"), ("rotation_12", "<f8"),
    ("rotation_20", "<f8"), ("rotation_21", "<f8"), ("rotation_22", "<f8"),
])
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_REQUIRED_CAMERA_PROPERTIES = ("frame_id", "x", "y", "z", "qx", "qy", "qz", "qw")
_REQUIRED_OBJECT_OBB_PROPERTIES = tuple(_OBJECT_OBB_DTYPE.names)


def write_map_ply(file_name, points, colors, camera_poses, object_obbs=None) -> None:
    """Bulk-write vertices, camera records, and optional object OBB metadata."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=float).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("PLY points and colors must have the same length")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError("PLY points and colors must be finite")
    if (colors < 0).any() or (colors > 1).any():
        raise ValueError("PLY colors must be in [0, 1]")

    vertices = np.empty(len(points), dtype=_VERTEX_DTYPE)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    rgb = np.rint(colors * 255).clip(0, 255).astype(np.uint8)
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    cameras = np.empty(len(camera_poses), dtype=_CAMERA_DTYPE)
    for index, pose in enumerate(camera_poses):
        if not np.isfinite((*pose.position_xyz, *pose.quaternion_xyzw)).all():
            raise ValueError("PLY camera poses must be finite")
        if not np.iinfo(np.int32).min <= pose.frame_id <= np.iinfo(np.int32).max:
            raise ValueError(f"PLY camera frame_id is outside int32 range: {pose.frame_id}")
        cameras[index] = (pose.frame_id, *pose.position_xyz, *pose.quaternion_xyzw)

    object_obbs = () if object_obbs is None else object_obbs
    obbs = np.empty(len(object_obbs), dtype=_OBJECT_OBB_DTYPE)
    for index, obb in enumerate(object_obbs):
        if isinstance(obb.object_id, bool) or not isinstance(obb.object_id, (int, np.integer)):
            raise ValueError("PLY object OBB id must be an integer")
        if not np.iinfo(np.int32).min <= obb.object_id <= np.iinfo(np.int32).max:
            raise ValueError(f"PLY object OBB id is outside int32 range: {obb.object_id}")
        center = np.asarray(obb.center_xyz, dtype=float)
        extent = np.asarray(obb.extent_xyz, dtype=float)
        rotation = np.asarray(obb.rotation_matrix, dtype=float)
        if center.shape != (3,) or extent.shape != (3,) or rotation.shape != (9,):
            raise ValueError("PLY object OBB must have center (3), extent (3), and rotation (9)")
        if not np.isfinite(np.concatenate((center, extent, rotation))).all():
            raise ValueError("PLY object OBB values must be finite")
        if (extent <= 0).any():
            raise ValueError("PLY object OBB extents must be strictly positive")
        obbs[index] = (int(obb.object_id), *center, *extent, *rotation)

    header = "\n".join((
        "ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}",
        "property double x", "property double y", "property double z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element camera {len(cameras)}", "property int frame_id",
        "property double x", "property double y", "property double z",
        "property double qx", "property double qy", "property double qz", "property double qw",
        f"element object_obb {len(obbs)}", "property int object_id",
        "property double center_x", "property double center_y", "property double center_z",
        "property double extent_x", "property double extent_y", "property double extent_z",
        "property double rotation_00", "property double rotation_01", "property double rotation_02",
        "property double rotation_10", "property double rotation_11", "property double rotation_12",
        "property double rotation_20", "property double rotation_21", "property double rotation_22",
        "end_header", "",
    )).encode("ascii")
    with Path(file_name).open("wb") as output:
        output.write(header)
        output.write(vertices.tobytes())
        output.write(cameras.tobytes())
        output.write(obbs.tobytes())


def _read_header(input_file):
    lines, elements, current = [], [], None
    while True:
        raw = input_file.readline()
        if not raw:
            raise ValueError("PLY header ended before end_header")
        try:
            line = raw.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("PLY header must be ASCII") from error
        lines.append(line)
        if line == "end_header":
            break
        fields = line.split()
        if fields[:1] == ["format"] and len(fields) >= 2:
            file_format = fields[1]
        elif fields[:1] == ["element"] and len(fields) == 3:
            current = {"name": fields[1], "count": int(fields[2]), "properties": []}
            elements.append(current)
        elif fields[:1] == ["property"] and current is not None:
            if fields[1:2] == ["list"]:
                current["properties"].append((None, None))
            elif len(fields) == 3:
                current["properties"].append((fields[2], fields[1]))
            else:
                raise ValueError("malformed PLY property declaration")
    if not lines or lines[0] != "ply" or "file_format" not in locals():
        raise ValueError("not a valid PLY file")
    return file_format, elements


def _read_binary_elements(input_file, elements):
    """Yield header-described binary elements, including ones to be skipped."""
    for element in elements:
        if any(type_name is None for _, type_name in element["properties"]):
            raise ValueError("PLY metadata cannot follow list-valued elements")
        try:
            dtype = np.dtype([
                (name, "<" + _PLY_TYPES[type_name])
                for name, type_name in element["properties"]
            ])
        except KeyError as error:
            raise ValueError(f"unsupported PLY property type: {error.args[0]}") from error
        data = np.fromfile(input_file, dtype=dtype, count=element["count"])
        if len(data) != element["count"]:
            raise ValueError(f"PLY data ended while reading {element['name']} element")
        yield element, data


def read_camera_poses_from_ply(file_name) -> list[PLYCameraPose]:
    """Read optional camera metadata while deriving record sizes from the header."""
    with Path(file_name).open("rb") as input_file:
        file_format, elements = _read_header(input_file)
        camera = next((element for element in elements if element["name"] == "camera"), None)
        if camera is None:
            return []
        names = [name for name, _ in camera["properties"]]
        missing = set(_REQUIRED_CAMERA_PROPERTIES) - set(names)
        if missing:
            raise ValueError(f"camera PLY element is missing properties: {', '.join(sorted(missing))}")
        if file_format != "binary_little_endian":
            raise ValueError("camera PLY metadata currently requires binary_little_endian format")
        for element, data in _read_binary_elements(input_file, elements):
            if element is camera:
                return [PLYCameraPose(int(row["frame_id"]), tuple(float(row[name]) for name in ("x", "y", "z")),
                                      tuple(float(row[name]) for name in ("qx", "qy", "qz", "qw"))) for row in data]
    return []


def read_object_obbs_from_ply(file_name) -> list[PLYObjectOBB]:
    """Read optional object OBB metadata from a binary little-endian PLY."""
    with Path(file_name).open("rb") as input_file:
        file_format, elements = _read_header(input_file)
        object_obb = next((element for element in elements if element["name"] == "object_obb"), None)
        if object_obb is None:
            return []
        names = [name for name, _ in object_obb["properties"]]
        missing = set(_REQUIRED_OBJECT_OBB_PROPERTIES) - set(names)
        if missing:
            raise ValueError(
                "object_obb PLY element is missing properties: "
                + ", ".join(sorted(missing))
            )
        if file_format != "binary_little_endian":
            raise ValueError("object OBB PLY metadata currently requires binary_little_endian format")
        for element, data in _read_binary_elements(input_file, elements):
            if element is object_obb:
                return [
                    PLYObjectOBB(
                        int(row["object_id"]),
                        tuple(float(row[name]) for name in ("center_x", "center_y", "center_z")),
                        tuple(float(row[name]) for name in ("extent_x", "extent_y", "extent_z")),
                        tuple(float(row[name]) for name in (
                            "rotation_00", "rotation_01", "rotation_02",
                            "rotation_10", "rotation_11", "rotation_12",
                            "rotation_20", "rotation_21", "rotation_22",
                        )),
                    )
                    for row in data
                ]
    return []
