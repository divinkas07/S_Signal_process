"""
Metrics Dashboard – Real-time charts for simulation metrics.
Displays AOA error, SNR, RMSE vs CRLB, elevation, and Doppler shift.
Uses pyqtgraph for high-performance plotting.
"""

import numpy as np
from collections import deque
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout
from PySide6.QtCore import Qt

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class MetricCard(QWidget):
    """Single metric display card with label and value."""

    def __init__(self, title: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        self.value_label = QLabel("—")
        self.value_label.setObjectName("metricValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)

        self.unit = unit

    def set_value(self, value, fmt=".1f"):
        if value is None:
            self.value_label.setText("—")
        else:
            self.value_label.setText(f"{value:{fmt}} {self.unit}")

    def set_status(self, ok: bool):
        color = "#00cc66" if ok else "#ff4444"
        self.value_label.setStyleSheet(f"color: {color};")


class MetricsDashboard(QWidget):
    """
    Real-time metrics dashboard with charts and metric cards.
    """

    MAX_POINTS = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── Metric Cards Row ───────────────────────────────────────
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(6)

        self.snr_card = MetricCard("SNR", "dB")
        self.aoa_card = MetricCard("AOA Error", "°")
        self.rmse_card = MetricCard("RMSE", "°")
        self.elev_card = MetricCard("Elevation", "°")
        self.doppler_card = MetricCard("Doppler", "kHz")
        self.range_card = MetricCard("Range", "km")

        for card in [self.snr_card, self.aoa_card, self.rmse_card,
                     self.elev_card, self.doppler_card, self.range_card]:
            cards_layout.addWidget(card)

        layout.addLayout(cards_layout)

        if not HAS_PYQTGRAPH:
            layout.addWidget(QLabel("⚠️ pyqtgraph not installed – no charts"))
            self._plots_available = False
            return

        self._plots_available = True
        pg.setConfigOptions(antialias=True, background="#0a0e1a", foreground="#c0c8e0")

        # ── Charts ─────────────────────────────────────────────────
        charts_layout = QGridLayout()
        charts_layout.setSpacing(4)

        # SNR plot
        self.snr_plot = pg.PlotWidget(title="SNR (dB)")
        self.snr_plot.showGrid(x=True, y=True, alpha=0.15)
        self.snr_plot.setLabel("bottom", "Time (s)")
        self.snr_curve = self.snr_plot.plot(pen=pg.mkPen("#00d4ff", width=2))
        charts_layout.addWidget(self.snr_plot, 0, 0)

        # AOA Error plot
        self.aoa_plot = pg.PlotWidget(title="AOA Error (°)")
        self.aoa_plot.showGrid(x=True, y=True, alpha=0.15)
        self.aoa_plot.setLabel("bottom", "Time (s)")
        self.aoa_curve = self.aoa_plot.plot(pen=pg.mkPen("#ff6644", width=2), name="Error")
        self.crlb_curve = self.aoa_plot.plot(pen=pg.mkPen("#ffcc00", width=1.5,
                                             style=Qt.PenStyle.DashLine), name="CRLB")
        charts_layout.addWidget(self.aoa_plot, 0, 1)

        # Elevation plot
        self.elev_plot = pg.PlotWidget(title="Elevation (°)")
        self.elev_plot.showGrid(x=True, y=True, alpha=0.15)
        self.elev_plot.setLabel("bottom", "Time (s)")
        self.elev_curve = self.elev_plot.plot(pen=pg.mkPen("#00ff88", width=2))
        charts_layout.addWidget(self.elev_plot, 1, 0)

        # Doppler plot
        self.dop_plot = pg.PlotWidget(title="Doppler (kHz)")
        self.dop_plot.showGrid(x=True, y=True, alpha=0.15)
        self.dop_plot.setLabel("bottom", "Time (s)")
        self.dop_curve = self.dop_plot.plot(pen=pg.mkPen("#cc66ff", width=2))
        charts_layout.addWidget(self.dop_plot, 1, 1)

        # RMSE Histogram
        self.rmse_hist = pg.PlotWidget(title="RMSE Histogram")
        self.rmse_hist.showGrid(x=True, y=True, alpha=0.15)
        self.rmse_hist.setLabel("bottom", "RMSE (°)")
        self.rmse_bg = pg.BarGraphItem(x=[0], height=[0], width=0.5, brush="#ff6644")
        self.rmse_hist.addItem(self.rmse_bg)
        charts_layout.addWidget(self.rmse_hist, 2, 0)

        # SNR Histogram
        self.snr_hist = pg.PlotWidget(title="SNR Histogram")
        self.snr_hist.showGrid(x=True, y=True, alpha=0.15)
        self.snr_hist.setLabel("bottom", "SNR (dB)")
        self.snr_bg = pg.BarGraphItem(x=[0], height=[0], width=0.5, brush="#00d4ff")
        self.snr_hist.addItem(self.snr_bg)
        charts_layout.addWidget(self.snr_hist, 2, 1)

        layout.addLayout(charts_layout)

        # Data buffers
        self._time = deque(maxlen=self.MAX_POINTS)
        self._snr = deque(maxlen=self.MAX_POINTS)
        self._aoa_err = deque(maxlen=self.MAX_POINTS)
        self._crlb = deque(maxlen=self.MAX_POINTS)
        self._elev = deque(maxlen=self.MAX_POINTS)
        self._dop = deque(maxlen=self.MAX_POINTS)
        
        # Full history for histograms
        self._full_rmse = []
        self._full_snr = []
        self._hist_update_counter = 0

    def update_data(self, data: dict):
        """Update charts and cards with new simulation data."""
        t = data.get("time", 0)
        visible = data.get("visible", False)

        if not visible:
            self.elev_card.set_value(data.get("elevation_deg"), ".1f")
            self.range_card.set_value(data.get("range_km"), ".0f")
            return

        # Update cards
        snr = data.get("snr_db")
        self.snr_card.set_value(snr)
        if snr is not None:
            self.snr_card.set_status(snr > 10)

        aoa_est = data.get("aoa_estimate_deg", [])
        true_aoa = data.get("true_aoa_deg")
        aoa_err = None
        if aoa_est and true_aoa is not None:
            est = aoa_est[0] if isinstance(aoa_est, list) else aoa_est
            aoa_err = abs(est - true_aoa)
        self.aoa_card.set_value(aoa_err, ".3f")

        elev = data.get("elevation_deg")
        self.elev_card.set_value(elev)

        doppler = data.get("doppler_hz")
        if doppler is not None:
            self.doppler_card.set_value(doppler / 1e3, ".1f")

        range_km = data.get("range_km")
        self.range_card.set_value(range_km, ".0f")

        # RMSE
        rmse = data.get("crlb_deg")
        self.rmse_card.set_value(rmse, ".4f")

        # Update charts
        if not self._plots_available:
            return

        self._time.append(t)

        if snr is not None:
            self._snr.append(snr)
            self.snr_curve.setData(list(self._time)[-len(self._snr):], list(self._snr))

        if aoa_err is not None:
            self._aoa_err.append(aoa_err)
            self.aoa_curve.setData(list(self._time)[-len(self._aoa_err):], list(self._aoa_err))

        crlb = data.get("crlb_deg")
        if crlb is not None:
            self._crlb.append(crlb)
            self.crlb_curve.setData(list(self._time)[-len(self._crlb):], list(self._crlb))

        if elev is not None:
            self._elev.append(elev)
            self.elev_curve.setData(list(self._time)[-len(self._elev):], list(self._elev))

        if doppler is not None:
            self._dop.append(doppler / 1e3)
            self.dop_curve.setData(list(self._time)[-len(self._dop):], list(self._dop))
            
        # Update histograms periodically (every 10 frames)
        if aoa_err is not None:
            self._full_rmse.append(aoa_err)
        if snr is not None:
            self._full_snr.append(snr)
            
        self._hist_update_counter += 1
        if self._hist_update_counter >= 10:
            self._hist_update_counter = 0
            if len(self._full_rmse) > 5:
                y, x = np.histogram(self._full_rmse, bins=20)
                self.rmse_bg.setOpts(x=x[:-1], height=y, width=(x[1]-x[0])*0.8)
            if len(self._full_snr) > 5:
                y, x = np.histogram(self._full_snr, bins=20)
                self.snr_bg.setOpts(x=x[:-1], height=y, width=(x[1]-x[0])*0.8)

    def clear_plots(self):
        """Clear all chart data."""
        for buf in [self._time, self._snr, self._aoa_err, self._crlb, self._elev, self._dop]:
            buf.clear()
        
        self._full_rmse.clear()
        self._full_snr.clear()
        
        if self._plots_available:
            for curve in [self.snr_curve, self.aoa_curve, self.crlb_curve,
                         self.elev_curve, self.dop_curve]:
                curve.setData([], [])
            self.rmse_bg.setOpts(x=[0], height=[0], width=0.5)
            self.snr_bg.setOpts(x=[0], height=[0], width=0.5)
