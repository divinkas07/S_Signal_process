"""
Metrics Collector – Real-time aggregation of simulation metrics.
Stores time-series data for SNR, RMSE, BER, AOA error, CRLB, etc.
"""

import numpy as np
from collections import deque


class MetricsCollector:
    """
    Collects and aggregates simulation metrics in real-time.
    Provides history buffers for plotting and summary statistics.
    """

    def __init__(self, max_history: int = 5000):
        self.max_history = max_history

        # Time-series buffers
        self.time_history = deque(maxlen=max_history)
        self.snr_history = deque(maxlen=max_history)
        self.aoa_error_history = deque(maxlen=max_history)
        self.aoa_estimate_history = deque(maxlen=max_history)
        self.aoa_true_history = deque(maxlen=max_history)
        self.rmse_history = deque(maxlen=max_history)
        self.crlb_history = deque(maxlen=max_history)
        self.doppler_history = deque(maxlen=max_history)
        self.ber_history = deque(maxlen=max_history)
        self.elevation_history = deque(maxlen=max_history)
        self.range_history = deque(maxlen=max_history)
        self.link_budget_history = deque(maxlen=max_history)

        # Counters
        self.total_steps = 0
        self._running_sum_error2 = 0.0

    def update(self, t: float, data: dict):
        """
        Record metrics from a simulation step.
        
        Args:
            t: Current simulation time
            data: Dictionary containing metric values
        """
        self.total_steps += 1
        self.time_history.append(t)

        # SNR
        snr = data.get("snr_out_db", data.get("snr_db"))
        if snr is not None:
            self.snr_history.append(float(snr))

        # AOA
        aoa_est = data.get("aoa_estimate_deg", [])
        aoa_true = data.get("true_aoa_deg")
        if aoa_est:
            est = aoa_est[0] if isinstance(aoa_est, list) else aoa_est
            self.aoa_estimate_history.append(float(est))
            if aoa_true is not None:
                error = abs(float(est) - float(aoa_true))
                self.aoa_error_history.append(error)
                self.aoa_true_history.append(float(aoa_true))
                self._running_sum_error2 += error**2

        # CRLB
        crlb = data.get("crlb_deg")
        if crlb is not None:
            self.crlb_history.append(float(crlb))

        # RMSE (running)
        if self.aoa_error_history:
            rmse = np.sqrt(self._running_sum_error2 / len(self.aoa_error_history))
            self.rmse_history.append(rmse)

        # Doppler
        doppler = data.get("doppler_hz")
        if doppler is not None:
            self.doppler_history.append(float(doppler))

        # Elevation & Range
        elevation = data.get("elevation_deg")
        if elevation is not None:
            self.elevation_history.append(float(elevation))

        rng = data.get("range_m")
        if rng is not None:
            self.range_history.append(float(rng))

        # Link budget
        rx_power = data.get("rx_power_dbw")
        if rx_power is not None:
            self.link_budget_history.append(float(rx_power))

    def get_summary(self) -> dict:
        """Return summary statistics of collected metrics."""
        summary = {"total_steps": self.total_steps}

        if self.snr_history:
            snr = np.array(self.snr_history)
            summary["snr"] = {
                "mean": float(np.mean(snr)),
                "min": float(np.min(snr)),
                "max": float(np.max(snr)),
                "current": float(snr[-1]),
            }

        if self.aoa_error_history:
            errors = np.array(self.aoa_error_history)
            summary["aoa_error"] = {
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mean": float(np.mean(errors)),
                "max": float(np.max(errors)),
                "current": float(errors[-1]),
            }

        if self.crlb_history:
            summary["crlb_current"] = float(self.crlb_history[-1])

        if self.rmse_history and self.crlb_history:
            summary["rmse_to_crlb_ratio"] = float(
                self.rmse_history[-1] / max(self.crlb_history[-1], 1e-10)
            )

        if self.doppler_history:
            summary["doppler_current_hz"] = float(self.doppler_history[-1])

        return summary

    def get_plot_data(self) -> dict:
        """Return data suitable for real-time plotting."""
        return {
            "time": list(self.time_history),
            "snr": list(self.snr_history),
            "aoa_error": list(self.aoa_error_history),
            "aoa_estimate": list(self.aoa_estimate_history),
            "aoa_true": list(self.aoa_true_history),
            "rmse": list(self.rmse_history),
            "crlb": list(self.crlb_history),
            "doppler": list(self.doppler_history),
            "elevation": list(self.elevation_history),
            "range": list(self.range_history),
        }

    def get_current_metrics(self) -> dict:
        """Return the latest metric values."""
        return {
            "snr_db": float(self.snr_history[-1]) if self.snr_history else None,
            "aoa_error_deg": float(self.aoa_error_history[-1]) if self.aoa_error_history else None,
            "rmse_deg": float(self.rmse_history[-1]) if self.rmse_history else None,
            "crlb_deg": float(self.crlb_history[-1]) if self.crlb_history else None,
            "doppler_hz": float(self.doppler_history[-1]) if self.doppler_history else None,
            "elevation_deg": float(self.elevation_history[-1]) if self.elevation_history else None,
        }

    def reset(self):
        """Clear all collected metrics."""
        self.time_history.clear()
        self.snr_history.clear()
        self.aoa_error_history.clear()
        self.aoa_estimate_history.clear()
        self.aoa_true_history.clear()
        self.rmse_history.clear()
        self.crlb_history.clear()
        self.doppler_history.clear()
        self.ber_history.clear()
        self.elevation_history.clear()
        self.range_history.clear()
        self.link_budget_history.clear()
        self.total_steps = 0
        self._running_sum_error2 = 0.0

    def export_csv(self, filepath: str):
        """Export all collected time-series metrics to a CSV file."""
        import csv
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                "Time_s", "SNR_dB", "AOA_Error_deg", "AOA_Estimate_deg",
                "AOA_True_deg", "RMSE_deg", "CRLB_deg", "Doppler_Hz",
                "Elevation_deg", "Range_m", "Rx_Power_dBW"
            ])
            
            # Write data rows
            n_rows = len(self.time_history)
            for i in range(n_rows):
                writer.writerow([
                    _safe_get(self.time_history, i),
                    _safe_get(self.snr_history, i),
                    _safe_get(self.aoa_error_history, i),
                    _safe_get(self.aoa_estimate_history, i),
                    _safe_get(self.aoa_true_history, i),
                    _safe_get(self.rmse_history, i),
                    _safe_get(self.crlb_history, i),
                    _safe_get(self.doppler_history, i),
                    _safe_get(self.elevation_history, i),
                    _safe_get(self.range_history, i),
                    _safe_get(self.link_budget_history, i)
                ])
                
    def export_json(self, filepath: str):
        """Export all collected metrics to a JSON file."""
        import json
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = self.get_plot_data()
        data["summary"] = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)


def _safe_get(dq: deque, idx: int):
    """Safely get from a deque, returning empty string if index out of bounds."""
    try:
        return dq[idx]
    except IndexError:
        return ""
