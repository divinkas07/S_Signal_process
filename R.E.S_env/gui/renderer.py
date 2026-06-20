"""
2D Scene Renderer – Visualizes satellite orbits, ground stations, and links.
Uses pyqtgraph for high-performance real-time rendering on dark background.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class Renderer(QWidget):
    """
    2D scene renderer showing satellite trajectory, ground station,
    and the current link status.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_PYQTGRAPH:
            layout.addWidget(QLabel("⚠️ pyqtgraph not installed"))
            self._plot = None
            return

        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True, background="#0a0e1a", foreground="#c0c8e0")

        self._plot = pg.PlotWidget(title="🛰️ Orbital View (ECEF Projection)")
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("bottom", "X (km)")
        self._plot.setLabel("left", "Y (km)")
        layout.addWidget(self._plot)

        # Draw Earth circle
        theta = np.linspace(0, 2 * np.pi, 200)
        R = 6371  # Earth radius in km
        self._plot.plot(R * np.cos(theta), R * np.sin(theta),
                       pen=pg.mkPen("#1e3a5f", width=2), name="Earth")

        # Satellite trail
        self._sat_trail = self._plot.plot([], [],
                                          pen=pg.mkPen("#00d4ff", width=1.5, style=Qt.PenStyle.DotLine))
        self._sat_trail_x = []
        self._sat_trail_y = []

        # Satellite current position
        self._sat_marker = self._plot.plot([], [],
                                           pen=None,
                                           symbol="o",
                                           symbolBrush="#00ff88",
                                           symbolSize=10)

        # Station marker
        self._sta_marker = self._plot.plot([], [],
                                           pen=None,
                                           symbol="t",  # triangle
                                           symbolBrush="#ff6644",
                                           symbolSize=12)

        # Link line
        self._link_line = self._plot.plot([], [],
                                          pen=pg.mkPen("#ffcc00", width=1.5, style=Qt.PenStyle.DashLine))

        # Device type colors
        self.COLORS = {
            "satellite": "#0099ff",  # Blue
            "station": "#00ff88",    # Green
            "modem": "#ff9900"       # Orange
        }

        self._max_trail = 2000
        
        # FPS Limiter (30 FPS = ~33ms)
        self._last_data = None
        from PySide6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._render_frame)
        self._timer.start()

    def update_view(self, data: dict):
        """Buffer new simulation data. Rendering is handled by QTimer."""
        self._last_data = data

    def _render_frame(self):
        """Perform the actual pyqtgraph update at a capped framerate."""
        if self._plot is None or self._last_data is None:
            return

        data = self._last_data
        sat_ecef = data.get("sat_ecef", [0, 0, 0])
        sta_ecef = data.get("sta_ecef", [0, 0, 0])
        visible = data.get("visible", False)

        # Convert to km
        sx, sy = sat_ecef[0] / 1e3, sat_ecef[1] / 1e3
        gx, gy = sta_ecef[0] / 1e3, sta_ecef[1] / 1e3

        # Update satellite trail
        # Only append if position changed significantly to save memory/processing
        if not self._sat_trail_x or abs(self._sat_trail_x[-1] - sx) > 0.1 or abs(self._sat_trail_y[-1] - sy) > 0.1:
            self._sat_trail_x.append(sx)
            self._sat_trail_y.append(sy)
            if len(self._sat_trail_x) > self._max_trail:
                self._sat_trail_x.pop(0)
                self._sat_trail_y.pop(0)
            self._sat_trail.setData(self._sat_trail_x, self._sat_trail_y)

        # Apply specific colors (demonstrating color by point type)
        self._sat_marker.setSymbolBrush(self.COLORS["satellite"])
        self._sat_marker.setData([sx], [sy])
        
        self._sta_marker.setSymbolBrush(self.COLORS["station"])
        self._sta_marker.setData([gx], [gy])

        # Link line (only when visible)
        if visible:
            self._link_line.setData([gx, sx], [gy, sy])
        else:
            self._link_line.setData([], [])

    def clear_trail(self):
        """Clear the satellite trail."""
        self._sat_trail_x.clear()
        self._sat_trail_y.clear()
        self._last_data = None
        if self._sat_trail:
            self._sat_trail.setData([], [])
