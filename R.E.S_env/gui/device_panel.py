"""
Device Panel – UI for adding and configuring simulation devices.
Provides forms for satellite, ground station, and terminal parameters.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QPushButton, QLabel, QDoubleSpinBox, QSpinBox,
                               QComboBox, QFormLayout, QFrame, QScrollArea)
from PySide6.QtCore import Qt, Signal


class DevicePanel(QWidget):
    """Panel for configuring simulation devices and scenario parameters."""

    # Signals
    config_changed = Signal(dict)
    start_requested = Signal()
    stop_requested = Signal()
    reset_requested = Signal()
    report_requested = Signal()
    add_satellite_requested = Signal(dict)
    add_station_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.setMaximumWidth(380)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)

        # ── Simulation Controls ────────────────────────────────────
        ctrl_group = self._create_group("⚡ Simulation Control")
        ctrl_layout = QVBoxLayout(ctrl_group)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("startBtn")
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.setObjectName("resetBtn")

        for btn in [self.start_btn, self.stop_btn, self.reset_btn]:
            btn.setMinimumHeight(36)
            btn_layout.addWidget(btn)

        ctrl_layout.addLayout(btn_layout)

        # Duration
        form = QFormLayout()
        self.duration = QDoubleSpinBox()
        self.duration.setRange(10, 86400)
        self.duration.setValue(600)
        self.duration.setSuffix(" s")
        form.addRow("Duration:", self.duration)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 1.0)
        self.dt_spin.setValue(0.01)
        self.dt_spin.setDecimals(3)
        self.dt_spin.setSuffix(" s")
        form.addRow("Time Step (dt):", self.dt_spin)
        ctrl_layout.addLayout(form)

        layout.addWidget(ctrl_group)

        # ── Satellite Config ───────────────────────────────────────
        sat_group = self._create_group("🛰️ Satellite")
        sat_form = QFormLayout()

        self.altitude = QDoubleSpinBox()
        self.altitude.setRange(200, 2000)
        self.altitude.setValue(550)
        self.altitude.setSuffix(" km")
        sat_form.addRow("Altitude:", self.altitude)

        self.inclination = QDoubleSpinBox()
        self.inclination.setRange(0, 180)
        self.inclination.setValue(53.0)
        self.inclination.setSuffix("°")
        sat_form.addRow("Inclination:", self.inclination)

        self.tx_power = QDoubleSpinBox()
        self.tx_power.setRange(10, 50)
        self.tx_power.setValue(30)
        self.tx_power.setSuffix(" dBW")
        sat_form.addRow("Tx Power:", self.tx_power)

        self.add_sat_btn = QPushButton("➕ Add Satellite")
        
        sat_layout = QVBoxLayout(sat_group)
        sat_layout.addLayout(sat_form)
        sat_layout.addWidget(self.add_sat_btn)
        
        layout.addWidget(sat_group)

        # ── Ground Station Config ──────────────────────────────────
        sta_group = self._create_group("📡 Ground Station")
        sta_form = QFormLayout()

        self.latitude = QDoubleSpinBox()
        self.latitude.setRange(-90, 90)
        self.latitude.setValue(36.75)
        self.latitude.setDecimals(4)
        self.latitude.setSuffix("°")
        sta_form.addRow("Latitude:", self.latitude)

        self.longitude = QDoubleSpinBox()
        self.longitude.setRange(-180, 180)
        self.longitude.setValue(3.05)
        self.longitude.setDecimals(4)
        self.longitude.setSuffix("°")
        sta_form.addRow("Longitude:", self.longitude)

        self.n_antennas = QSpinBox()
        self.n_antennas.setRange(2, 64)
        self.n_antennas.setValue(8)
        sta_form.addRow("Antennas:", self.n_antennas)

        self.add_sta_btn = QPushButton("➕ Add Station")

        sta_layout = QVBoxLayout(sta_group)
        sta_layout.addLayout(sta_form)
        sta_layout.addWidget(self.add_sta_btn)

        layout.addWidget(sta_group)

        # ── RF Parameters ──────────────────────────────────────────
        rf_group = self._create_group("📻 RF Parameters")
        rf_form = QFormLayout(rf_group)

        self.modulation = QComboBox()
        self.modulation.addItems(["QPSK", "16QAM", "64QAM"])
        rf_form.addRow("Modulation:", self.modulation)

        self.carrier_freq = QDoubleSpinBox()
        self.carrier_freq.setRange(10, 20)
        self.carrier_freq.setValue(14.0)
        self.carrier_freq.setSuffix(" GHz")
        rf_form.addRow("Frequency:", self.carrier_freq)

        layout.addWidget(rf_group)

        # ── Channel Conditions ─────────────────────────────────────
        ch_group = self._create_group("🌧️ Channel")
        ch_form = QFormLayout(ch_group)

        self.rain_rate = QDoubleSpinBox()
        self.rain_rate.setRange(0, 150)
        self.rain_rate.setValue(0)
        self.rain_rate.setSuffix(" mm/h")
        ch_form.addRow("Rain Rate:", self.rain_rate)

        self.k_factor = QDoubleSpinBox()
        self.k_factor.setRange(-10, 30)
        self.k_factor.setValue(10)
        self.k_factor.setSuffix(" dB")
        ch_form.addRow("Rician K:", self.k_factor)

        self.scenario_preset = QComboBox()
        self.scenario_preset.addItems(["Custom", "Clear Sky", "Rain Fade",
                                       "Urban", "Vehicular", "High SNR"])
        ch_form.addRow("Preset:", self.scenario_preset)

        layout.addWidget(ch_group)

        # Spacer
        layout.addStretch()

        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        # Connect signals
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        
        # New Feature: Generate Report
        self.report_btn = QPushButton("📑 Generate TXT Report")
        layout.insertWidget(layout.count() - 2, self.report_btn) # Add before stretch
        self.report_btn.clicked.connect(self.report_requested.emit)
        
        # Add Device signals
        self.add_sat_btn.clicked.connect(self._emit_add_satellite)
        self.add_sta_btn.clicked.connect(self._emit_add_station)

    def _emit_add_satellite(self):
        self.add_satellite_requested.emit({
            "altitude_km": self.altitude.value(),
            "inclination_deg": self.inclination.value(),
            "tx_power_dbw": self.tx_power.value()
        })
        
    def _emit_add_station(self):
        self.add_station_requested.emit({
            "latitude_deg": self.latitude.value(),
            "longitude_deg": self.longitude.value(),
            "n_antennas": self.n_antennas.value()
        })

    def get_config(self) -> dict:
        """Return current configuration from all widgets."""
        return {
            "duration": self.duration.value(),
            "dt": self.dt_spin.value(),
            "altitude_km": self.altitude.value(),
            "inclination_deg": self.inclination.value(),
            "tx_power_dbw": self.tx_power.value(),
            "latitude_deg": self.latitude.value(),
            "longitude_deg": self.longitude.value(),
            "n_antennas": self.n_antennas.value(),
            "modulation": self.modulation.currentText(),
            "fc_ghz": self.carrier_freq.value(),
            "rain_rate_mmh": self.rain_rate.value(),
            "rician_k_db": self.k_factor.value(),
        }

    @staticmethod
    def _create_group(title: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setObjectName("deviceGroup")
        return group
