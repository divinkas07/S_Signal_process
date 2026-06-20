"""
Scenario Manager – Creates, configures, loads and saves simulation scenarios.
Orchestrates the full simulation lifecycle.
"""

import json
import os
import numpy as np
from core.config import SIM_CONFIG, SATELLITE_PARAMS, STATION_PARAMS, RF_PARAMS, CHANNEL_PARAMS


class ScenarioManager:
    """
    Manages simulation scenarios: parameter sets, device configurations,
    and pipeline settings that define a complete simulation run.
    """

    def __init__(self):
        self.name = "Default LEO Ku-Band Scenario"
        self.description = "Standard LEO satellite Ku-band link simulation"
        self.sim_config = SIM_CONFIG.copy()
        self.satellite_params = SATELLITE_PARAMS.copy()
        self.station_params = STATION_PARAMS.copy()
        self.rf_params = RF_PARAMS.copy()
        self.channel_params = CHANNEL_PARAMS.copy()
        self._is_running = False

    # ── Scenario Configuration ─────────────────────────────────────────

    def set_param(self, section: str, key: str, value):
        """Set a parameter in a specific config section."""
        sections = {
            "sim": self.sim_config,
            "satellite": self.satellite_params,
            "station": self.station_params,
            "rf": self.rf_params,
            "channel": self.channel_params,
        }
        cfg = sections.get(section)
        if cfg and key in cfg:
            cfg[key] = value
        else:
            raise KeyError(f"Unknown config section '{section}' or key '{key}'")

    def get_all_params(self) -> dict:
        """Return all configuration parameters as a single dict."""
        return {
            "name": self.name,
            "description": self.description,
            "sim": self.sim_config.copy(),
            "satellite": self.satellite_params.copy(),
            "station": self.station_params.copy(),
            "rf": self.rf_params.copy(),
            "channel": self.channel_params.copy(),
        }

    # ── Preset Scenarios ───────────────────────────────────────────────

    def load_preset(self, preset_name: str):
        """Load a preset scenario configuration."""
        presets = {
            "clear_sky": {
                "channel": {"rain_rate_mmh": 0.0, "shadowing_sigma_db": 1.0},
                "name": "Clear Sky Leo Link",
            },
            "rain_fade": {
                "channel": {"rain_rate_mmh": 25.0, "shadowing_sigma_db": 4.0},
                "name": "Heavy Rain Ku-Band Link",
            },
            "urban": {
                "channel": {"rician_k_db": 3.0, "num_multipath_taps": 5, "shadowing_sigma_db": 8.0},
                "station": {"mode": "pedestrian", "speed_mps": 1.5},
                "name": "Urban Mobile Terminal",
            },
            "vehicular": {
                "channel": {"rician_k_db": 5.0, "num_multipath_taps": 4},
                "station": {"mode": "vehicle", "speed_mps": 30.0},
                "name": "Vehicular LEO Terminal",
            },
            "high_snr": {
                "channel": {"rain_rate_mmh": 0.0, "shadowing_sigma_db": 0.5, "phase_noise_var": 1e-6},
                "satellite": {"tx_power_dbw": 35.0},
                "name": "High SNR Test",
            },
        }

        preset = presets.get(preset_name)
        if not preset:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

        self.name = preset.get("name", self.name)
        for section, params in preset.items():
            if section == "name":
                continue
            for key, value in params.items():
                try:
                    self.set_param(section, key, value)
                except KeyError:
                    pass

    def list_presets(self) -> list[str]:
        return ["clear_sky", "rain_fade", "urban", "vehicular", "high_snr"]

    # ── Save / Load ────────────────────────────────────────────────────

    def save(self, filepath: str):
        """Save scenario to JSON file."""
        data = self.get_all_params()
        # Convert numpy types for JSON serialization
        data = self._sanitize_for_json(data)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load scenario from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scenario file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        self.name = data.get("name", self.name)
        self.description = data.get("description", self.description)
        for section in ["sim", "satellite", "station", "rf", "channel"]:
            if section in data:
                for key, value in data[section].items():
                    try:
                        self.set_param(section, key, value)
                    except KeyError:
                        pass

    @staticmethod
    def _sanitize_for_json(obj):
        """Convert numpy types to Python native types for JSON."""
        if isinstance(obj, dict):
            return {k: ScenarioManager._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def __repr__(self):
        return f"Scenario('{self.name}', duration={self.sim_config['duration']}s)"
