"""
Environment Model – Weather, terrain effects, and interference.
Manages dynamic environmental conditions during simulation.
"""

import numpy as np


class Environment:
    """
    Simulates environmental conditions affecting the RF link.
    Provides time-varying weather, terrain obstruction, and interference.
    """

    def __init__(self):
        self.weather_state = "clear"   # "clear", "light_rain", "heavy_rain", "storm"
        self.rain_rate_mmh = 0.0       # Current rain rate (mm/h)
        self.wind_speed_mps = 0.0
        self.interference_level_dbm = -120.0  # Background interference

        # Weather transition probabilities (per step)
        self._weather_states = {
            "clear":      {"clear": 0.995, "light_rain": 0.004, "heavy_rain": 0.001},
            "light_rain": {"clear": 0.01,  "light_rain": 0.98,  "heavy_rain": 0.01},
            "heavy_rain": {"clear": 0.002, "light_rain": 0.018, "heavy_rain": 0.98},
        }

        # Rain rate mapping
        self._rain_rates = {
            "clear": 0.0,
            "light_rain": 5.0,
            "heavy_rain": 25.0,
            "storm": 80.0,
        }

    def update(self, t: float, dt: float = 0.01):
        """
        Update environment state. May trigger weather transitions.
        """
        # Markov chain weather transition
        transitions = self._weather_states.get(self.weather_state, {})
        if transitions:
            states = list(transitions.keys())
            probs = [transitions[s] for s in states]
            self.weather_state = np.random.choice(states, p=probs)
            self.rain_rate_mmh = self._rain_rates.get(self.weather_state, 0.0)

    def get_rain_attenuation_factor(self) -> float:
        """Get current rain rate for channel model."""
        return self.rain_rate_mmh

    def get_interference(self) -> float:
        """Get current interference level (dBm)."""
        return self.interference_level_dbm

    def set_weather(self, state: str):
        """Manually set weather state."""
        if state in self._rain_rates:
            self.weather_state = state
            self.rain_rate_mmh = self._rain_rates[state]

    def get_state(self) -> dict:
        """Return current environment state."""
        return {
            "weather": self.weather_state,
            "rain_rate_mmh": self.rain_rate_mmh,
            "interference_dbm": self.interference_level_dbm,
        }

    def __repr__(self):
        return f"Environment(weather={self.weather_state}, rain={self.rain_rate_mmh}mm/h)"
