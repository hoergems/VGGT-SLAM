"""Shared `F`-to-focus interaction for classic Open3D viewers.

Both ``vggt_slam/open3d_viewer.py`` (the live map viewer) and
``scripts/view_point_cloud.py`` (the standalone viewer) register an
``Open3DFocusController`` with their ``VisualizerWithKeyCallback`` window so
that pressing `F` moves the camera's look-at pivot to the visible surface at
the center of the current view. The controller only registers a keyboard
callback; it never touches mouse events and therefore never displaces
Open3D's native rotate/pan/roll/zoom navigation. This module has no
dependency on VGGT, GraphMap, Solver, Go2, ROS, or Viser.
"""

import numpy as np


def find_nearest_valid_depth_pixel(depth: np.ndarray, x: float, y: float, max_radius: int = 5):
    """Return the ``(row, col)`` of the valid depth pixel nearest to ``(x, y)``.

    ``x``/``y`` are column/row coordinates. A pixel is valid when its depth is
    finite and strictly positive. Searches a square neighborhood of side
    ``2 * max_radius + 1`` centered on the rounded ``(x, y)`` and returns the
    candidate with the smallest Euclidean screen-space distance to ``(x, y)``.
    Returns ``None`` when no valid pixel is found within that neighborhood.
    """
    if depth.ndim != 2:
        return None

    height, width = depth.shape
    if height == 0 or width == 0:
        return None

    center_col = int(round(x))
    center_row = int(round(y))

    best = None
    best_dist_sq = None
    for row in range(max(0, center_row - max_radius), min(height, center_row + max_radius + 1)):
        for col in range(max(0, center_col - max_radius), min(width, center_col + max_radius + 1)):
            value = depth[row, col]
            if not np.isfinite(value) or value <= 0:
                continue
            dist_sq = (col - x) ** 2 + (row - y) ** 2
            if best_dist_sq is None or dist_sq < best_dist_sq:
                best = (row, col)
                best_dist_sq = dist_sq

    return best


def depth_pixel_to_world(u: float, v: float, depth: float, intrinsic_matrix: np.ndarray, extrinsic: np.ndarray):
    """Back-project pixel ``(u, v)`` at ``depth`` to a world-space point.

    ``intrinsic_matrix`` is the 3x3 pinhole intrinsic matrix. ``extrinsic`` is
    the 4x4 world-to-camera matrix, as returned by Open3D's
    ``PinholeCameraParameters.extrinsic``. Returns ``None`` when any input is
    invalid (non-finite matrices, non-positive depth, non-positive focal
    lengths, a singular extrinsic, or a degenerate homogeneous divide).
    """
    if not np.isfinite(depth) or depth <= 0:
        return None

    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if intrinsic_matrix.shape != (3, 3) or extrinsic.shape != (4, 4):
        return None
    if not np.all(np.isfinite(intrinsic_matrix)) or not np.all(np.isfinite(extrinsic)):
        return None

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    if not np.isfinite(fx) or not np.isfinite(fy) or fx <= 0 or fy <= 0:
        return None

    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    camera_point = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)

    try:
        camera_to_world = np.linalg.inv(extrinsic)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(camera_to_world)):
        return None

    world_h = camera_to_world @ camera_point
    if not np.isfinite(world_h[3]) or world_h[3] == 0:
        return None

    world_point = world_h[:3] / world_h[3]
    if not np.all(np.isfinite(world_point)):
        return None

    return world_point


class Open3DFocusController:
    """Adds `F`-to-focus/recenter to a classic Open3D visualizer.

    Register once per visualizer via :meth:`register`; keep the controller
    instance alive for as long as the visualizer exists.
    """

    def __init__(self, max_search_radius: int = 5):
        self._visualizer = None
        self._max_search_radius = max_search_radius

    def register(self, visualizer) -> None:
        self._visualizer = visualizer
        visualizer.register_key_callback(ord("F"), self.on_focus_key)

    def on_focus_key(self, visualizer) -> bool:
        depth_image = visualizer.capture_depth_float_buffer(do_render=False)
        depth = np.asarray(depth_image)
        if depth.ndim != 2 or depth.shape[0] == 0 or depth.shape[1] == 0:
            self._no_surface_found()
            return False

        height, width = depth.shape
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0

        pixel = find_nearest_valid_depth_pixel(depth, center_x, center_y, self._max_search_radius)
        if pixel is None:
            self._no_surface_found()
            return False

        row, col = pixel
        depth_value = float(depth[row, col])

        view_control = visualizer.get_view_control()
        params = view_control.convert_to_pinhole_camera_parameters()
        intrinsic_matrix = np.asarray(params.intrinsic.intrinsic_matrix)
        extrinsic = np.asarray(params.extrinsic)

        world_point = depth_pixel_to_world(col, row, depth_value, intrinsic_matrix, extrinsic)
        if world_point is None:
            self._no_surface_found()
            return False

        view_control.set_lookat(world_point)
        visualizer.update_renderer()
        return False

    @staticmethod
    def _no_surface_found() -> None:
        print("[Open3DViewer] No surface near view center; focus unchanged.")
