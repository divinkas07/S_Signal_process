"""
MUSIC AOA Estimation – MUltiple SIgnal Classification.
Estimates Angle of Arrival using eigendecomposition of the
spatial covariance matrix.

    R_x = E[x·x^H] = A·R_s·A^H + σ²I
    P(θ) = 1 / (a^H(θ)·E_n·E_n^H·a(θ))

where E_n is the noise subspace eigenvectors.
"""

import numpy as np
from core.config import ANTENNA_PARAMS


class MUSICAoA:
    """
    MUSIC (MUltiple SIgnal Classification) algorithm for AOA estimation.
    Works with ULA (Uniform Linear Array) steering vectors.
    """

    def __init__(self, n_antennas: int = None, d_lambda: float = None,
                 n_sources: int = 1, scan_range: tuple = (-90, 90),
                 scan_resolution: float = 0.1):
        self.n_antennas = n_antennas or ANTENNA_PARAMS["n_antennas"]
        self.d_lambda = d_lambda or ANTENNA_PARAMS["antenna_spacing_lambda"]
        self.n_sources = n_sources
        self.scan_min, self.scan_max = scan_range
        self.scan_resolution = scan_resolution
        self._metrics = {}
        self._last_spectrum = None
        self._last_estimate = None

    def process(self, data: dict) -> dict:
        """
        Pipeline interface: estimate AOA from array signal.
        
        Expects:
            'array_signal': ndarray (n_antennas, n_samples)
            
        Adds:
            'aoa_estimate_deg': estimated angle(s) of arrival
            'music_spectrum': (angles, power) tuple
        """
        X = data.get("array_signal")
        if X is None:
            self._metrics = {"status": "no_array_signal"}
            return data

        # Estimate covariance matrix
        R = self.estimate_covariance(X)

        # MUSIC spectrum
        angles, spectrum = self.music_spectrum(R)

        # Peak detection
        aoa_estimates = self.find_peaks(angles, spectrum, self.n_sources)

        data["aoa_estimate_deg"] = aoa_estimates
        data["music_spectrum"] = (angles, spectrum)
        data["covariance_matrix"] = R

        self._last_spectrum = (angles, spectrum)
        self._last_estimate = aoa_estimates

        # Compute CRLB for validation
        if "snr_db" in data:
            n_samples = X.shape[1]
            crlb = self.compute_crlb(data["snr_db"], n_samples, aoa_estimates[0] if aoa_estimates else 0)
            data["crlb_deg"] = crlb
            self._metrics["crlb_deg"] = float(crlb)

        self._metrics.update({
            "aoa_deg": [float(a) for a in aoa_estimates],
            "n_sources_detected": len(aoa_estimates),
            "status": "success",
        })
        return data

    # ── Core MUSIC Algorithm ───────────────────────────────────────────

    @staticmethod
    def estimate_covariance(X: np.ndarray) -> np.ndarray:
        """
        Estimate spatial covariance matrix.
        R_x = (1/N) · X · X^H
        """
        n_antennas, n_samples = X.shape
        R = (X @ X.conj().T) / n_samples
        return R

    def music_spectrum(self, R: np.ndarray) -> tuple:
        """
        Compute MUSIC pseudo-spectrum.
        
        P(θ) = 1 / (a^H(θ) · E_n · E_n^H · a(θ))
        
        Returns:
            (angles, spectrum_db): scan angles and power spectrum
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)

        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Noise subspace: eigenvectors corresponding to smallest eigenvalues
        En = eigenvectors[:, self.n_sources:]  # (n_antennas, n_antennas - n_sources)

        # Scan angles
        angles = np.arange(self.scan_min, self.scan_max + self.scan_resolution,
                          self.scan_resolution)
        spectrum = np.zeros(len(angles))

        for i, theta in enumerate(angles):
            a = self._steering_vector(theta)
            # MUSIC denominator
            proj = a.conj() @ En
            denom = np.real(proj @ proj.conj())
            spectrum[i] = 1.0 / (denom + 1e-15)

        # Convert to dB
        spectrum_db = 10 * np.log10(spectrum / np.max(spectrum) + 1e-15)

        return angles, spectrum_db

    def find_peaks(self, angles: np.ndarray, spectrum_db: np.ndarray,
                   n_peaks: int = 1) -> list:
        """
        Find AOA estimates from MUSIC spectrum peaks.
        """
        # Simple peak detection: find local maxima
        peaks = []
        for i in range(1, len(spectrum_db) - 1):
            if spectrum_db[i] > spectrum_db[i-1] and spectrum_db[i] > spectrum_db[i+1]:
                if spectrum_db[i] > -20:  # Threshold
                    peaks.append((spectrum_db[i], angles[i]))

        # Sort by power (descending) and take top n
        peaks.sort(reverse=True)
        return [p[1] for p in peaks[:n_peaks]]

    # ── CRLB Computation ───────────────────────────────────────────────

    def compute_crlb(self, snr_db: float, n_samples: int, theta_deg: float = 0.0) -> float:
        """
        Compute Cramér-Rao Lower Bound for AOA estimation.
        
        CRLB(θ) = 1 / (2·N·SNR · ||d·a(θ)/dθ||²)
        
        For a ULA:
        CRLB(θ) ≈ 3λ² / (2π²·d²·M·(M²-1)·N·SNR·cos²θ)
        """
        snr_linear = 10 ** (snr_db / 10)
        theta = np.radians(theta_deg)
        M = self.n_antennas
        d = self.d_lambda

        # Analytical CRLB for ULA
        numerator = 3
        denominator = (2 * np.pi**2 * d**2 * M * (M**2 - 1) *
                       n_samples * snr_linear * np.cos(theta)**2)

        if denominator == 0:
            return float("inf")

        crlb_rad2 = numerator / denominator
        crlb_deg = np.degrees(np.sqrt(crlb_rad2))
        return crlb_deg

    # ── Steering Vector ────────────────────────────────────────────────

    def _steering_vector(self, theta_deg: float) -> np.ndarray:
        """
        ULA steering vector.
        a_k(θ) = e^(-j·2π·(k-1)·(d/λ)·sin(θ))
        """
        theta = np.radians(theta_deg)
        k = np.arange(self.n_antennas)
        return np.exp(-1j * 2 * np.pi * k * self.d_lambda * np.sin(theta))

    def get_metrics(self) -> dict:
        return self._metrics
