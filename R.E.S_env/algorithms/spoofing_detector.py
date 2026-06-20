"""
Spoofing Detector – Detects signal spoofing and jamming attacks.
Uses consistency checks between expected and observed signal parameters.
"""

import numpy as np


class SpoofingDetector:
    """
    Multi-layer spoofing detection combining:
    - AOA consistency check (expected vs observed direction)
    - Power anomaly detection (unexpected signal strength)
    - Doppler consistency check (expected vs observed Doppler shift)
    """

    def __init__(self, aoa_threshold_deg: float = 10.0,
                 power_threshold_db: float = 6.0,
                 doppler_threshold_hz: float = 5000.0):
        self.aoa_threshold = aoa_threshold_deg
        self.power_threshold = power_threshold_db
        self.doppler_threshold = doppler_threshold_hz
        self._metrics = {}
        self._alert_history: list = []

    def process(self, data: dict) -> dict:
        """
        Pipeline interface: check for spoofing indicators.
        
        Expects (optional):
            'aoa_estimate_deg': measured AOA
            'expected_aoa_deg': expected AOA from orbital prediction
            'snr_out_db': measured SNR
            'expected_snr_db': expected SNR from link budget
            'doppler_hz': measured Doppler
            'expected_doppler_hz': expected Doppler from orbit

        Adds:
            'spoofing_alert': bool
            'spoofing_confidence': 0.0 to 1.0
            'spoofing_details': dict of individual checks
        """
        checks = {}
        alert_score = 0.0
        total_checks = 0

        # 1. AOA consistency
        aoa_est = data.get("aoa_estimate_deg", [])
        aoa_exp = data.get("expected_aoa_deg")
        if aoa_est and aoa_exp is not None:
            measured = aoa_est[0] if isinstance(aoa_est, list) else aoa_est
            aoa_error = abs(measured - aoa_exp)
            aoa_suspicious = aoa_error > self.aoa_threshold
            checks["aoa"] = {
                "error_deg": float(aoa_error),
                "threshold": self.aoa_threshold,
                "suspicious": aoa_suspicious,
            }
            if aoa_suspicious:
                alert_score += min(aoa_error / self.aoa_threshold, 3.0) / 3.0
            total_checks += 1

        # 2. Power anomaly
        snr_meas = data.get("snr_out_db")
        snr_exp = data.get("expected_snr_db")
        if snr_meas is not None and snr_exp is not None:
            power_diff = abs(snr_meas - snr_exp)
            power_suspicious = power_diff > self.power_threshold
            checks["power"] = {
                "diff_db": float(power_diff),
                "threshold": self.power_threshold,
                "suspicious": power_suspicious,
            }
            if power_suspicious:
                alert_score += min(power_diff / self.power_threshold, 3.0) / 3.0
            total_checks += 1

        # 3. Doppler consistency
        dop_meas = data.get("doppler_hz")
        dop_exp = data.get("expected_doppler_hz")
        if dop_meas is not None and dop_exp is not None:
            dop_diff = abs(dop_meas - dop_exp)
            dop_suspicious = dop_diff > self.doppler_threshold
            checks["doppler"] = {
                "diff_hz": float(dop_diff),
                "threshold": self.doppler_threshold,
                "suspicious": dop_suspicious,
            }
            if dop_suspicious:
                alert_score += min(dop_diff / self.doppler_threshold, 3.0) / 3.0
            total_checks += 1

        # Overall assessment
        confidence = alert_score / max(total_checks, 1)
        is_spoofed = confidence > 0.5

        alert = {
            "spoofing_alert": is_spoofed,
            "spoofing_confidence": float(confidence),
            "spoofing_details": checks,
        }

        if is_spoofed:
            self._alert_history.append(alert)

        data.update(alert)

        self._metrics = {
            "alert": is_spoofed,
            "confidence": float(confidence),
            "checks_run": total_checks,
            "status": "success",
        }
        return data

    def get_alert_history(self) -> list:
        return self._alert_history

    def reset(self):
        self._alert_history.clear()

    def get_metrics(self) -> dict:
        return self._metrics
