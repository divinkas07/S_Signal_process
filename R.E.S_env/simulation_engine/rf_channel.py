"""
RF Channel Model – Complete Ku-band channel simulation.
Implements: FSPL, Rician fading, multipath (tapped delay line),
            rain attenuation (ITU-R P.838), log-normal shadowing.

Channel model:
    h(t) = Σᵢ αᵢ · e^(jφᵢ) · δ(t - τᵢ)
    y(t) = (s * h)(t) + n(t)
"""

import numpy as np
from core.config import CONSTANTS, RF_PARAMS, CHANNEL_PARAMS


class RFChannel:
    """
    Complete RF channel model for Ku-band satellite links.
    Applies all channel impairments to a baseband signal.
    """

    def __init__(self, fc: float = None, config: dict = None):
        cfg = config or CHANNEL_PARAMS
        self.fc = fc or RF_PARAMS["fc"]
        self.c = CONSTANTS["c"]
        self.wavelength = self.c / self.fc

        # Rician fading
        self.K_db = cfg.get("rician_k_db", 10.0)
        self.K = 10 ** (self.K_db / 10)

        # Multipath
        self.num_taps = cfg.get("num_multipath_taps", 3)

        # Shadowing
        self.shadow_sigma = cfg.get("shadowing_sigma_db", 3.0)

        # Rain
        self.rain_rate = cfg.get("rain_rate_mmh", 0.0)

        # Phase noise
        self.phase_noise_var = cfg.get("phase_noise_var", 1e-4)

        # Internal state
        self._phase_state = 0.0  # Wiener process accumulator

    # ── Complete Channel Application ───────────────────────────────────

    def apply(self, signal: np.ndarray, distance_m: float,
              snr_db: float = 20.0, elevation_deg: float = 45.0) -> dict:
        """
        Apply complete channel model to a signal.
        
        Args:
            signal: Complex baseband signal array
            distance_m: Slant range to satellite (m)
            snr_db: Target SNR (dB) after channel
            elevation_deg: Elevation angle (degrees)
        
        Returns:
            dict with: signal, fspl_db, total_loss_db, channel_response, snr_out
        """
        n_samples = len(signal)

        # 1. Free-space path loss
        fspl_db = self.compute_fspl(distance_m)

        # 2. Rician fading
        h_rician = self.generate_rician_fading(n_samples)

        # 3. Multipath channel
        h_multipath, delays = self.generate_multipath()

        # 4. Shadowing
        shadow_db = self.generate_shadowing()

        # 5. Rain attenuation
        rain_db = self.compute_rain_attenuation(elevation_deg)

        # 6. Phase noise (Wiener process)
        phase_noise = self.generate_phase_noise(n_samples)

        # --- Apply impairments ---
        # Apply Rician fading
        sig = signal * h_rician

        # Apply multipath convolution
        sig = self._apply_multipath(sig, h_multipath, delays)

        # Compute total loss
        total_loss_db = fspl_db + shadow_db + rain_db

        # Apply path loss as amplitude scaling
        loss_linear = 10 ** (-total_loss_db / 20)
        sig = sig * loss_linear

        # Apply phase noise
        sig = sig * np.exp(1j * phase_noise)

        # Add AWGN at target SNR
        sig_power = np.mean(np.abs(sig) ** 2)
        if sig_power > 0:
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
            )
            sig_noisy = sig + noise
            snr_out = 10 * np.log10(sig_power / noise_power)
        else:
            sig_noisy = sig
            snr_out = -np.inf

        return {
            "signal": sig_noisy,
            "fspl_db": fspl_db,
            "shadow_db": shadow_db,
            "rain_db": rain_db,
            "total_loss_db": total_loss_db,
            "snr_out_db": snr_out,
            "channel_response": h_rician,
        }

    # ── Individual Channel Components ──────────────────────────────────

    def compute_fspl(self, distance_m: float) -> float:
        """
        Free-Space Path Loss (dB).
        L_fs = 20·log10(d) + 20·log10(f) + 20·log10(4π/c)
             = 20·log10(d_km) + 20·log10(f_MHz) + 32.44
        """
        if distance_m <= 0:
            return 0.0
        d_km = distance_m / 1e3
        f_mhz = self.fc / 1e6
        return 20 * np.log10(d_km) + 20 * np.log10(f_mhz) + 32.44

    def generate_rician_fading(self, n_samples: int) -> np.ndarray:
        """
        Generate Rician fading coefficients.
        h = √(K/(K+1)) · e^(jθ_LOS) + √(1/(K+1)) · CN(0,1)
        """
        K = self.K
        los = np.sqrt(K / (K + 1))
        nlos_scale = np.sqrt(1 / (2 * (K + 1)))
        nlos = nlos_scale * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
        h = los + nlos
        return h

    def generate_multipath(self) -> tuple:
        """
        Generate multipath channel taps.
        h(t) = Σ αᵢ · e^(jφᵢ) · δ(t - τᵢ)
        
        Returns:
            (amplitudes: complex array, delays: int array in samples)
        """
        if self.num_taps <= 1:
            return np.array([1.0 + 0j]), np.array([0])

        # Exponentially decaying power profile
        powers = np.exp(-np.arange(self.num_taps) * 0.5)
        powers /= np.sum(powers)  # Normalize

        amplitudes = np.sqrt(powers) * np.exp(1j * 2 * np.pi * np.random.rand(self.num_taps))

        # Delays in samples (first tap at 0)
        delays = np.arange(self.num_taps)

        return amplitudes, delays

    def generate_shadowing(self) -> float:
        """
        Log-normal shadowing attenuation (dB).
        X_σ ~ N(0, σ²)
        """
        return np.random.normal(0, self.shadow_sigma)

    def compute_rain_attenuation(self, elevation_deg: float) -> float:
        """
        Rain attenuation based on simplified ITU-R P.838.
        Uses specific attenuation γ_R and effective path length.
        
        A_rain = γ_R · L_eff / sin(elevation)
        """
        if self.rain_rate <= 0:
            return 0.0

        f_ghz = self.fc / 1e9
        # Simplified ITU-R coefficients for Ku-band
        k = 0.0101 * f_ghz ** 1.276
        alpha = 1.21
        gamma_r = k * self.rain_rate ** alpha  # dB/km

        # Effective path through rain (simplified)
        elev = max(np.radians(elevation_deg), np.radians(5.0))
        rain_height_km = 4.0  # Typical freezing layer
        L_eff = rain_height_km / np.sin(elev)

        return gamma_r * L_eff

    def generate_phase_noise(self, n_samples: int) -> np.ndarray:
        """
        Phase noise as a Wiener process.
        φ(t) = φ(t-1) + N(0, σ²_φ)
        """
        increments = np.random.normal(0, np.sqrt(self.phase_noise_var), n_samples)
        phase = np.cumsum(increments) + self._phase_state
        self._phase_state = phase[-1] if len(phase) > 0 else 0.0
        return phase

    def set_rain_rate(self, rate_mmh: float):
        """Update rain rate dynamically."""
        self.rain_rate = max(0, rate_mmh)

    # ── Private ────────────────────────────────────────────────────────

    @staticmethod
    def _apply_multipath(signal: np.ndarray, amplitudes: np.ndarray,
                         delays: np.ndarray) -> np.ndarray:
        """Apply multipath by summing delayed copies of the signal."""
        output = np.zeros_like(signal)
        for amp, delay in zip(amplitudes, delays):
            d = int(delay)
            if d == 0:
                output += amp * signal
            elif d < len(signal):
                output[d:] += amp * signal[:-d]
        return output

    def reset(self):
        """Reset internal state."""
        self._phase_state = 0.0
