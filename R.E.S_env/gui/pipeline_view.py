"""
Pipeline View – Visualizes the signal processing pipeline with live metrics.
Shows block diagram: Signal → ICA → MUSIC → IMM → PINN → Consensus
Each block displays real-time SNR, BER, and confidence values.
"""

from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel,
                               QFrame, QSizePolicy)
from PySide6.QtCore import Qt


class PipelineBlockWidget(QFrame):
    """Visual representation of a single pipeline processing block."""

    STATUS_COLORS = {
        "active": "#00cc66",
        "inactive": "#555566",
        "error": "#ff4444",
        "warning": "#ffaa00",
    }

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.block_name = name
        self.setObjectName("pipelineBlock")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(110)
        self.setMaximumWidth(160)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        # Block title
        self.title = QLabel(name)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setObjectName("blockTitle")
        layout.addWidget(self.title)

        # Status indicator
        self.status_label = QLabel("—")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setObjectName("blockStatus")
        layout.addWidget(self.status_label)

        # Metric display
        self.metric_label = QLabel("")
        self.metric_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metric_label.setObjectName("blockMetric")
        layout.addWidget(self.metric_label)

        self._set_status("inactive")

    def update_metrics(self, metrics: dict):
        """Update displayed metrics from pipeline output."""
        if not metrics:
            self._set_status("inactive")
            return

        status = metrics.get("status", "active")
        self._set_status("active" if status == "success" else "error" if "error" in str(status) else "active")

        # Display key metric
        metric_text = ""
        if "aoa_deg" in metrics:
            aoa = metrics["aoa_deg"]
            if isinstance(aoa, list) and aoa:
                metric_text = f"AOA: {aoa[0]:.1f}°"
        elif "filtered_aoa" in metrics:
            metric_text = f"AOA: {metrics['filtered_aoa']:.1f}°"
        elif "fused_aoa" in metrics:
            metric_text = f"Fused: {metrics['fused_aoa']:.1f}°"
        elif "confidence" in metrics:
            metric_text = f"Conf: {metrics['confidence']:.1%}"
        elif "n_sources" in metrics:
            metric_text = f"Sources: {metrics['n_sources']}"
        elif "alert" in metrics:
            metric_text = "🚨 SPOOF" if metrics["alert"] else "✅ OK"

        self.metric_label.setText(metric_text)
        self.status_label.setText("●" if status == "success" else "○")

    def _set_status(self, status: str):
        color = self.STATUS_COLORS.get(status, "#555566")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 14px;")


class ArrowLabel(QLabel):
    """Arrow connector between pipeline blocks."""
    def __init__(self, parent=None):
        super().__init__("→", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("color: #00d4ff; font-size: 20px; font-weight: bold;")
        self.setFixedWidth(24)


class PipelineView(QWidget):
    """
    Horizontal pipeline view showing signal flow through algorithm blocks.
    """

    BLOCK_NAMES = ["ICA", "MUSIC", "IMM", "PINN", "Consensus", "Spoof"]
    BLOCK_KEYS = ["ICA", "MUSIC_AOA", "IMM_Tracker", "PINN_Predictor",
                  "Mesh_Consensus", "Spoofing_Detector"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pipelineView")
        self.setFixedHeight(120)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(0)

        # Signal input indicator
        sig_label = QLabel("📶")
        sig_label.setStyleSheet("font-size: 22px;")
        sig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sig_label.setFixedWidth(30)
        layout.addWidget(sig_label)

        layout.addWidget(ArrowLabel())

        # Create blocks
        self.blocks: dict[str, PipelineBlockWidget] = {}
        for i, (name, key) in enumerate(zip(self.BLOCK_NAMES, self.BLOCK_KEYS)):
            block_widget = PipelineBlockWidget(name)
            self.blocks[key] = block_widget
            layout.addWidget(block_widget)

            if i < len(self.BLOCK_NAMES) - 1:
                layout.addWidget(ArrowLabel())

        # Output indicator
        layout.addWidget(ArrowLabel())
        out_label = QLabel("📊")
        out_label.setStyleSheet("font-size: 22px;")
        out_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        out_label.setFixedWidth(30)
        layout.addWidget(out_label)

    def update_pipeline(self, pipeline_metrics: dict):
        """Update all pipeline blocks with new metrics."""
        for key, block_widget in self.blocks.items():
            metrics = pipeline_metrics.get(key, {})
            block_widget.update_metrics(metrics)
