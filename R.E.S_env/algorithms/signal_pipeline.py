"""
Signal Processing Pipeline – Orchestrates the chain of algorithm blocks.

Pipeline flow:
    RF signal → ICA → MUSIC AOA → IMM Tracker → PINN Prediction → Mesh Consensus
    
Each block implements: process(data) → data, get_metrics() → dict
"""

from typing import Optional
from core.plugin_loader import PluginBlock


class PipelineBlock:
    """
    Wrapper around a processing block with metrics tracking.
    Each block in the pipeline inherits from this or PluginBlock.
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._metrics: dict = {}
        self._processor = None

    def set_processor(self, processor):
        """Attach a processing object (must have .process() method)."""
        self._processor = processor

    def process(self, data: dict) -> dict:
        """Process data through this block."""
        if not self.enabled or self._processor is None:
            return data
        try:
            result = self._processor.process(data)
            self._metrics = self._processor.get_metrics() if hasattr(self._processor, 'get_metrics') else {}
            return result
        except Exception as e:
            self._metrics = {"error": str(e)}
            return data

    @property
    def metrics(self) -> dict:
        return self._metrics


class SignalPipeline:
    """
    Manages the full signal processing chain.
    Blocks can be added, removed, enabled/disabled, or replaced.
    """

    def __init__(self):
        self.blocks: list[PipelineBlock] = []
        self._pipeline_metrics: dict = {}

    def add_block(self, block: PipelineBlock):
        """Add a processing block to the end of the pipeline."""
        self.blocks.append(block)

    def insert_block(self, index: int, block: PipelineBlock):
        """Insert a block at a specific position."""
        self.blocks.insert(index, block)

    def remove_block(self, name: str):
        """Remove a block by name."""
        self.blocks = [b for b in self.blocks if b.name != name]

    def enable_block(self, name: str, enabled: bool = True):
        """Enable or disable a specific block."""
        for b in self.blocks:
            if b.name == name:
                b.enabled = enabled
                break

    def process(self, data: dict) -> dict:
        """
        Run data through the complete pipeline.
        Each block receives the output of the previous block.
        """
        current = data.copy()
        stage_metrics = {}

        for block in self.blocks:
            if block.enabled:
                current = block.process(current)
                stage_metrics[block.name] = block.metrics.copy()

        self._pipeline_metrics = stage_metrics
        current["pipeline_metrics"] = stage_metrics
        return current

    def get_pipeline_metrics(self) -> dict:
        """Return metrics from all pipeline stages."""
        return self._pipeline_metrics

    def get_block_names(self) -> list[str]:
        """Return ordered list of block names with enabled status."""
        return [(b.name, b.enabled) for b in self.blocks]

    def __repr__(self):
        blocks = " -> ".join(
            f"{b.name}[ON]" if b.enabled else f"{b.name}[OFF]"
            for b in self.blocks
        )
        return f"Pipeline[{blocks}]"
