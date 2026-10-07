"""Live, non-blocking Open3D window that displays the current optimized map.

Mirrors the classic visualizer pattern used by ``scripts/view_point_cloud.py``,
but keeps a single viewer-owned geometry that is updated in place (rather than
re-added) so the map refreshes incrementally without resetting the user's
camera viewpoint.
"""

from typing import Callable, Optional

import open3d as o3d

from vggt_slam.open3d_focus import Open3DFocusController


class Open3DMapViewer:
    def __init__(
        self,
        point_size: float = 2.0,
        voxel_size: Optional[float] = None,
        window_name: str = "VGGT-SLAM Live Map",
        visualizer_factory: Optional[Callable[[], object]] = None,
    ):
        if point_size <= 0:
            raise ValueError("point_size must be greater than zero")
        if voxel_size is not None and voxel_size <= 0:
            raise ValueError("voxel_size must be greater than zero when provided")

        self._point_size = point_size
        self._voxel_size = voxel_size
        self._window_name = window_name
        self._visualizer_factory = visualizer_factory or o3d.visualization.VisualizerWithKeyCallback

        self._visualizer = None
        self._display_cloud = None  # viewer-owned geometry; never replaced
        self._geometry_added = False
        self._active = False
        self._focus_controller = None

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self) -> None:
        self._visualizer = self._visualizer_factory()
        self._visualizer.create_window(window_name=self._window_name)
        self._visualizer.get_render_option().point_size = self._point_size
        self._display_cloud = o3d.geometry.PointCloud()
        self._geometry_added = False
        self._active = True

        self._focus_controller = Open3DFocusController()
        self._focus_controller.register(self._visualizer)
        print("[Open3DViewer] Press F to focus/recenter on the visible surface at the view center.")

    def update(self, point_cloud: "o3d.geometry.PointCloud") -> bool:
        """Refresh the displayed map from the current global point cloud.

        Returns whether the viewer is still active (window open).
        """
        if not self._active or self._visualizer is None:
            return False

        display_source = point_cloud
        if self._voxel_size is not None:
            display_source = point_cloud.voxel_down_sample(self._voxel_size)

        self._display_cloud.points = display_source.points
        self._display_cloud.colors = display_source.colors

        if not self._geometry_added:
            self._visualizer.add_geometry(self._display_cloud, reset_bounding_box=True)
            self._geometry_added = True
        else:
            self._visualizer.update_geometry(self._display_cloud)

        still_open = self._visualizer.poll_events()
        if not still_open:
            self._deactivate()
            return False

        self._visualizer.update_renderer()
        return True

    def poll(self) -> bool:
        """Service GUI events/rendering without rebuilding the displayed geometry.

        Cheap to call every frame so the window stays responsive to user
        rotate/zoom/pan between the (much rarer) map data refreshes done by
        ``update()``.
        """
        if not self._active or self._visualizer is None:
            return False

        still_open = self._visualizer.poll_events()
        if not still_open:
            self._deactivate()
            return False

        self._visualizer.update_renderer()
        return True

    def close(self) -> None:
        if self._visualizer is not None:
            try:
                self._visualizer.destroy_window()
            except Exception:
                pass
        self._active = False
        self._visualizer = None
        self._geometry_added = False
        self._focus_controller = None

    def _deactivate(self) -> None:
        if self._active:
            print("[Open3DViewer] Window closed; continuing SLAM without Open3D visualization.")
        self._active = False
