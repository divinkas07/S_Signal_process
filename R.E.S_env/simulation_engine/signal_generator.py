"""
Signal Generator – OFDM Ku-band signal generation.
Implements the complete signal model:

    s(t) = Σ aₙ · p(t − nTₛ)                      (baseband)
    s_RF(t) = Re{ s(t) · e^(j2πf_c·t) }            (RF)
    s_D(t) = s(t) · e^(j2π(f_c + f_D)t)            (Doppler)
    x(t) = a(θ) · s(t) + n(t)                       (array)
    aₖ(θ) = e^(-j2π(k-1)(d/λ)sinθ)                 (steering vector)
"""

import numpy as np
from core.config import RF_PARAMS, ANTENNA_PARAMS, CONSTANTS, DERIVED


class SignalGenerator:
    """
    Generates OFDM Ku-band signals with realistic physical parameters.
    Supports QPSK/16QAM/64QAM modulation, Doppler shift, and antenna
    array steering vector generation.
    """

    # Modulation constellation maps
    CONSTELLATIONS = {
        "QPSK": np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2),
        "16QAM": None,  # Generated on demand
        "64QAM": None,  # Generated on demand
    }

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.fc = cfg.get("fc", RF_PARAMS["fc"])
        self.bandwidth = cfg.get("bandwidth", RF_PARAMS["bandwidth"])
        self.fs = cfg.get("sampling_rate", RF_PARAMS["sampling_rate"])
        self.n_fft = cfg.get("n_fft", RF_PARAMS["n_fft"])
        self.cp_len = cfg.get("cp_len", RF_PARAMS["cp_len"])
        self.modulation = cfg.get("modulation", RF_PARAMS["modulation"])
        self.rolloff = cfg.get("rolloff", RF_PARAMS["rolloff"])
        self.n_antennas = cfg.get("n_antennas", ANTENNA_PARAMS["n_antennas"])
        self.d_lambda = cfg.get("antenna_spacing_lambda", ANTENNA_PARAMS["antenna_spacing_lambda"])

        self.c = CONSTANTS["c"]
        self.wavelength = self.c / self.fc
        self.d = self.d_lambda * self.wavelength  # Physical antenna spacing

    # ── Signal Generation ──────────────────────────────────────────────

    def generate_ofdm_signal(self, n_symbols: int = 1) -> np.ndarray:
        """
        Generate OFDM baseband signal: IFFT + cyclic prefix.
        
        Args:
            n_symbols: Number of OFDM symbols to generate.
            
        Returns:
            Complex baseband signal array.
        """
        signals = []
        for _ in range(n_symbols):
            # Generate random data symbols
            data = self._modulate(self.n_fft)
            
            # OFDM: IFFT
            time_domain = np.fft.ifft(data, self.n_fft) * np.sqrt(self.n_fft)
            
            # Add cyclic prefix
            cp = time_domain[-self.cp_len:]
            ofdm_symbol = np.concatenate([cp, time_domain])
            signals.append(ofdm_symbol)
        
        return np.concatenate(signals)

    def generate_baseband(self, n_samples: int) -> np.ndarray:
        """
        Generate continuous baseband signal of specified length.
        Uses OFDM symbols to fill the required number of samples.
        """
        symbol_len = self.n_fft + self.cp_len
        n_symbols = max(1, int(np.ceil(n_samples / symbol_len)))
        signal = self.generate_ofdm_signal(n_symbols)
        return signal[:n_samples]

    # ── Doppler Application ────────────────────────────────────────────

    def apply_doppler(self, signal: np.ndarray, radial_velocity_mps: float) -> np.ndarray:
        """
        Apply Doppler frequency shift.
        
        f_D = (v/c) · f_c
        s_D(t) = s(t) · e^(j2π·f_D·t)
        """
        f_doppler = (radial_velocity_mps / self.c) * self.fc
        n = len(signal)
        t = np.arange(n) / self.fs
        return signal * np.exp(1j * 2 * np.pi * f_doppler * t)

    def compute_doppler_shift(self, radial_velocity_mps: float) -> float:
        """Compute Doppler frequency shift in Hz."""
        return (radial_velocity_mps / self.c) * self.fc

    # ── Antenna Array ──────────────────────────────────────────────────

    def steering_vector(self, theta_deg: float) -> np.ndarray:
        """
        Compute ULA steering vector for a given angle.
        
        aₖ(θ) = e^(-j·2π·(k-1)·(d/λ)·sin(θ))
        
        Args:
            theta_deg: Angle of arrival (degrees)
            
        Returns:
            Complex steering vector of shape (n_antennas,)
        """
        theta = np.radians(theta_deg)
        k = np.arange(self.n_antennas)
        return np.exp(-1j * 2 * np.pi * k * self.d_lambda * np.sin(theta))

    def generate_array_signal(self, signal: np.ndarray, theta_deg: float,
                              snr_db: float = 20.0) -> np.ndarray:
        """
        Generate multi-antenna received signal.
        
        x(t) = a(θ)·s(t) + n(t)
        
        Args:
            signal: 1D baseband signal
            theta_deg: Angle of arrival (degrees)
            snr_db: Signal-to-noise ratio (dB)
            
        Returns:
            Array signal of shape (n_antennas, n_samples)
        """
        a = self.steering_vector(theta_deg)  # (n_antennas,)
        n_samples = len(signal)

        # Signal component: outer product a(θ) ⊗ s(t)
        X = np.outer(a, signal)  # (n_antennas, n_samples)

        # Add noise
        sig_power = np.mean(np.abs(signal) ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(self.n_antennas, n_samples) +
            1j * np.random.randn(self.n_antennas, n_samples)
        )

        return X + noise

    # ── Modulation ─────────────────────────────────────────────────────

    def _modulate(self, n_symbols: int) -> np.ndarray:
        """Generate random modulated symbols."""
        if self.modulation == "QPSK":
            bits = np.random.randint(0, 4, n_symbols)
            constellation = self.CONSTELLATIONS["QPSK"]
            return constellation[bits]

        elif self.modulation == "16QAM":
            constellation = self._get_qam_constellation(16)
            bits = np.random.randint(0, 16, n_symbols)
            return constellation[bits]

        elif self.modulation == "64QAM":
            constellation = self._get_qam_constellation(64)
            bits = np.random.randint(0, 64, n_symbols)
            return constellation[bits]

        else:
            # Default to QPSK
            bits = np.random.randint(0, 4, n_symbols)
            return self.CONSTELLATIONS["QPSK"][bits]

    @staticmethod
    def _get_qam_constellation(M: int) -> np.ndarray:
        """Generate square QAM constellation."""
        m = int(np.sqrt(M))
        real = np.arange(m) - (m - 1) / 2
        imag = np.arange(m) - (m - 1) / 2
        constellation = (real[:, None] + 1j * imag[None, :]).flatten()
        # Normalize to unit average power
        constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))
        return constellation

    def generate_rrc_pulse(self, n_taps: int = 65) -> np.ndarray:
        """
        Generate Root Raised Cosine pulse shape.
        """
        alpha = self.rolloff
        T = 1.0  # Symbol period (normalized)
        t = np.arange(-(n_taps // 2), n_taps // 2 + 1) / 4  # 4 samples/symbol

        h = np.zeros_like(t, dtype=float)
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = (1 + alpha * (4 / np.pi - 1))
            elif abs(abs(ti) - T / (4 * alpha)) < 1e-10 and alpha > 0:
                h[i] = (alpha / np.sqrt(2)) * (
                    (1 + 2/np.pi) * np.sin(np.pi / (4*alpha)) +
                    (1 - 2/np.pi) * np.cos(np.pi / (4*alpha))
                )
            else:
                num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
                den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
                if abs(den) > 1e-15:
                    h[i] = num / den
                else:
                    h[i] = 0.0

        h /= np.sqrt(np.sum(h ** 2))  # Normalize
        return h
