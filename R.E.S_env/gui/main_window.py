"""
Main Window – Master layout for the LEO Ku-Band Dynamic Reality Simulator.
Assembles all GUI panels with a premium dark theme.
"""

import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QSplitter, QTextEdit, QLabel,
                               QFrame, QStatusBar)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from gui.device_panel import DevicePanel
from gui.renderer import Renderer
from gui.pipeline_view import PipelineView
from gui.metrics_dashboard import MetricsDashboard
from gui.controller import SimulationThread


# ─── Dark Theme Stylesheet ───────────────────────────────────────────────
DARK_STYLE = """
QMainWindow {
    background-color: #0d1117;
}
QWidget {
    color: #c9d1d9;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 8px 8px 8px;
    background-color: #161b22;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    color: #58a6ff;
    font-weight: bold;
    font-size: 13px;
}
QPushButton {
    background-color: #0d419d;
    color: #ffffff;
    border: 1px solid #1f6feb;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1f6feb;
    border-color: #58a6ff;
}
QPushButton:pressed {
    background-color: #0a3069;
}
QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #30363d;
}
/* Removed specialized button colors to keep them all uniform dark blue */
QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 4px 8px;
    color: #c9d1d9;
}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #58a6ff;
}
QComboBox::drop-down {
    border: none;
    background-color: #21262d;
    width: 24px;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}
QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    color: #c9d1d9;
    selection-background-color: #1f6feb;
}
QTextEdit {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    color: #8b949e;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
    padding: 6px;
}
QLabel {
    color: #c9d1d9;
}
QLabel#metricTitle {
    color: #8b949e;
    font-size: 11px;
}
QLabel#metricValue {
    color: #58a6ff;
    font-size: 18px;
    font-weight: bold;
}
QFrame#pipelineBlock {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
}
QLabel#blockTitle {
    color: #58a6ff;
    font-weight: bold;
    font-size: 11px;
}
QLabel#blockMetric {
    color: #c9d1d9;
    font-size: 10px;
}
QWidget#metricCard {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
}
QSplitter::handle {
    background-color: #21262d;
    width: 3px;
    height: 3px;
}
QSplitter::handle:hover {
    background-color: #58a6ff;
}
QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    border-top: 1px solid #21262d;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    background-color: #0d1117;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #30363d;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #484f58;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


class MainWindow(QMainWindow):
    """
    Master window assembling all simulator panels.
    Layout: Left (config) | Center (viz + charts) | Pipeline + logs at bottom.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛰️ LEO Ku-Band Dynamic Reality Simulator – Control Center")
        self.resize(1440, 950)
        self.setMinimumSize(1024, 700)

        # Apply dark theme
        self.setStyleSheet(DARK_STYLE)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # ── Top: Config + Viz + Metrics ────────────────────────────
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Device panel
        self.device_panel = DevicePanel()
        top_splitter.addWidget(self.device_panel)

        # Center: Renderer (orbital view)
        self.renderer = Renderer()
        top_splitter.addWidget(self.renderer)

        # Right: Metrics dashboard
        self.dashboard = MetricsDashboard()
        top_splitter.addWidget(self.dashboard)

        top_splitter.setSizes([300, 500, 640])
        main_layout.addWidget(top_splitter, stretch=5)

        # ── Middle: Pipeline View ──────────────────────────────────
        self.pipeline_view = PipelineView()
        main_layout.addWidget(self.pipeline_view, stretch=0)

        # ── Bottom: Log Output ─────────────────────────────────────
        log_frame = QFrame()
        log_frame.setFrameShape(QFrame.Shape.NoFrame)
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(0, 0, 0, 0)

        log_header = QLabel("📋 Event Log")
        log_header.setStyleSheet("color: #8b949e; font-weight: bold; padding: 2px;")
        log_layout.addWidget(log_header)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(140)
        log_layout.addWidget(self.log_output)

        main_layout.addWidget(log_frame, stretch=1)

        # ── Status Bar ─────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready – Configure parameters and click Start")

        # ── Connections ────────────────────────────────────────────
        self.device_panel.start_requested.connect(self.start_simulation)
        self.device_panel.stop_requested.connect(self.stop_simulation)
        self.device_panel.reset_requested.connect(self.reset_simulation)
        self.device_panel.report_requested.connect(self.generate_report)
        self.device_panel.add_satellite_requested.connect(self.add_satellite)
        self.device_panel.add_station_requested.connect(self.add_station)

        self.sim_thread = None
        self.append_log("🛰️ LEO Ku-Band Dynamic Reality Simulator initialized.")
        self.append_log("   Configure parameters on the left panel and click Start.")

    # ── Simulation Control ─────────────────────────────────────────

    def start_simulation(self):
        """Start a new simulation run."""
        # If already running, do nothing
        if self.sim_thread and self.sim_thread.isRunning():
            return
            
        config = self.device_panel.get_config()
        self.sim_thread = SimulationThread(config)
        self.sim_thread.update_data.connect(self.on_data_received)
        self.sim_thread.update_metrics.connect(self.on_metrics_received)
        self.sim_thread.log_message.connect(self.append_log)
        self.sim_thread.finished_sig.connect(self.on_sim_finished)

        self.device_panel.start_btn.setEnabled(False)
        self.device_panel.stop_btn.setEnabled(True)
        self.status_bar.showMessage("Simulation running...")
        self.sim_thread.start()

    def stop_simulation(self):
        """Stop the running simulation."""
        if self.sim_thread:
            self.sim_thread.stop()
            self.append_log("⏹ Stopping simulation...")
            self.status_bar.showMessage("Stopping...")

    def reset_simulation(self):
        """Reset the GUI state."""
        self.log_output.clear()
        self.renderer.clear_trail()
        self.dashboard.clear_plots()
        self.status_bar.showMessage("Reset – Ready")
        self.append_log("↺ Simulation reset.")

    # ── Data Handlers ──────────────────────────────────────────────

    def on_data_received(self, data: dict):
        """Handle incoming simulation data."""
        self.renderer.update_view(data)
        self.dashboard.update_data(data)
        self.pipeline_view.update_pipeline(data.get("pipeline_metrics", {}))

        # Update status bar
        t = data.get("time", 0)
        vis = "🟢" if data.get("visible") else "🔴"
        self.status_bar.showMessage(
            f"t={t:.1f}s | {vis} | "
            f"El={data.get('elevation_deg', 0):.1f}° | "
            f"SNR={data.get('snr_db', 0):.1f} dB"
        )

    def on_metrics_received(self, data: dict):
        """Handle metrics summary updates."""
        pass  # Dashboard already updated via on_data_received

    def on_sim_finished(self):
        """Handle simulation completion."""
        self.device_panel.start_btn.setEnabled(True)
        self.device_panel.stop_btn.setEnabled(False)
        self.status_bar.showMessage("Simulation complete")

    def append_log(self, text: str):
        """Append message to the log panel."""
        self.log_output.append(text)

    def add_satellite(self, params: dict):
        self.append_log(f"➕ Adding Satellite: Alt={params['altitude_km']}km, Inc={params['inclination_deg']}°")
        if self.sim_thread and self.sim_thread.isRunning():
            self.sim_thread.add_satellite(params)
        else:
            self.start_simulation()
            
    def add_station(self, params: dict):
        self.append_log(f"➕ Adding Station: Lat={params['latitude_deg']}°, Lon={params['longitude_deg']}°")
        if self.sim_thread and self.sim_thread.isRunning():
            self.sim_thread.add_station(params)
        else:
            self.start_simulation()

    def generate_report(self):
        """Generate a TXT report of the simulation parameters and scores."""
        import datetime
        import os
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SSN_Lab_Simulation_Report_{timestamp}.txt"
        config = self.device_panel.get_config()
        
        metrics = "N/A"
        if self.sim_thread and hasattr(self.sim_thread, 'metrics') and self.sim_thread.metrics:
            metrics = self.sim_thread.metrics.get_summary()
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write("      SSN LAB - SIMULATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated: {datetime.datetime.now()}\n\n")
            f.write("--- CONFIGURATION ---\n")
            for k, v in config.items():
                f.write(f"{k}: {v}\n")
            
            f.write("\n--- SCORES & METRICS ---\n")
            f.write(str(metrics) + "\n\n")
            f.write("="*50 + "\n")
            
        self.append_log(f"📑 Report generated: {filename}")
        self.status_bar.showMessage(f"Report saved to {os.path.abspath(filename)}")


def main():
    app = QApplication(sys.argv)

    # Set application-wide font
    font = QFont("Segoe UI", 11)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
