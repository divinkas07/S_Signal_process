"""
Pipeline Manager – Configures and wires the signal processing pipeline.
Creates the default pipeline and allows runtime reconfiguration.
"""

from algorithms.signal_pipeline import SignalPipeline, PipelineBlock
from algorithms.ica import ICAProcessor
from algorithms.music_aoa import MUSICAoA
from algorithms.imm_tracker import IMMTracker
from algorithms.pinn_predictor import PINNPredictor
from algorithms.mesh_consensus import MeshConsensus
from algorithms.spoofing_detector import SpoofingDetector


class PipelineManager:
    """
    Factory for constructing and configuring the signal processing pipeline.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.pipeline = SignalPipeline()
        self._blocks = {}

    def build_default_pipeline(self) -> SignalPipeline:
        """
        Build the default full processing pipeline:
        ICA → MUSIC AOA → IMM Tracker → PINN Predictor → Mesh Consensus → Spoofing Detector
        """
        self.pipeline = SignalPipeline()
        self._blocks.clear()

        # 1. ICA – Blind source separation
        ica_block = PipelineBlock("ICA")
        ica_block.set_processor(ICAProcessor(
            n_components=self.config.get("ica_components", None),
        ))
        self.pipeline.add_block(ica_block)
        self._blocks["ICA"] = ica_block

        # 2. MUSIC AOA – Angle of arrival estimation
        music_block = PipelineBlock("MUSIC_AOA")
        music_block.set_processor(MUSICAoA(
            n_sources=self.config.get("n_sources", 1),
            scan_resolution=self.config.get("scan_resolution", 0.5),
        ))
        self.pipeline.add_block(music_block)
        self._blocks["MUSIC_AOA"] = music_block

        # 3. IMM Tracker
        imm_block = PipelineBlock("IMM_Tracker")
        imm_block.set_processor(IMMTracker(
            dt=self.config.get("dt", 0.01),
        ))
        self.pipeline.add_block(imm_block)
        self._blocks["IMM_Tracker"] = imm_block

        # 4. PINN Predictor
        pinn_block = PipelineBlock("PINN_Predictor")
        pinn_block.set_processor(PINNPredictor(
            dt=self.config.get("dt", 0.01),
        ))
        self.pipeline.add_block(pinn_block)
        self._blocks["PINN_Predictor"] = pinn_block

        # 5. Mesh Consensus
        mesh_block = PipelineBlock("Mesh_Consensus")
        mesh_block.set_processor(MeshConsensus(
            n_nodes=self.config.get("n_mesh_nodes", 4),
        ))
        self.pipeline.add_block(mesh_block)
        self._blocks["Mesh_Consensus"] = mesh_block

        # 6. Spoofing Detector
        spoof_block = PipelineBlock("Spoofing_Detector")
        spoof_block.set_processor(SpoofingDetector())
        self.pipeline.add_block(spoof_block)
        self._blocks["Spoofing_Detector"] = spoof_block

        return self.pipeline

    def build_minimal_pipeline(self) -> SignalPipeline:
        """Build a minimal pipeline with only MUSIC AOA."""
        self.pipeline = SignalPipeline()
        self._blocks.clear()

        music_block = PipelineBlock("MUSIC_AOA")
        music_block.set_processor(MUSICAoA(n_sources=1))
        self.pipeline.add_block(music_block)
        self._blocks["MUSIC_AOA"] = music_block

        return self.pipeline

    def enable_block(self, name: str, enabled: bool = True):
        """Enable or disable a pipeline block by name."""
        self.pipeline.enable_block(name, enabled)

    def get_pipeline(self) -> SignalPipeline:
        return self.pipeline

    def get_block_status(self) -> list:
        """Return status of all blocks."""
        return self.pipeline.get_block_names()

    def __repr__(self):
        return f"PipelineManager({self.pipeline})"
