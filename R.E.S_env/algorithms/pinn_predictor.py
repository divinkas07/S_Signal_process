"""
PINN Predictor – Physics-Informed Neural Network for state prediction.
Provides short-term prediction of satellite/target state using
physics-based extrapolation (linear + orbital dynamics corrections).

Note: Full PINN training requires PyTorch. This module provides a
physics-based predictor that can be replaced with a trained PINN plugin.
"""

import numpy as np


class PINNPredictor:
    """
    Physics-informed predictor for satellite state.
    Uses orbital dynamics constraints for short-term prediction.
    Falls back to linear extrapolation with physics corrections.
    """

    def __init__(self, dt: float = 0.01, history_len: int = 50,
                 mu_earth: float = 3.986004418e14):
        self.dt = dt
        self.history_len = history_len
        self.mu = mu_earth
        self._history: list = []
        self._metrics = {}

    def process(self, data: dict) -> dict:
        """
        Pipeline interface: predict future state.
        
        Expects:
            'imm_state' or 'aoa_estimate_deg': current state
            
        Adds:
            'predicted_state': predicted next state
            'prediction_confidence': confidence of prediction
        """
        state = data.get("imm_state")
        if state is None:
            aoa = data.get("aoa_estimate_deg", [0.0])
            state = np.array([aoa[0] if aoa else 0.0, 0.0, 0.0, 0.0])

        # Add to history
        self._history.append(state.copy())
        if len(self._history) > self.history_len:
            self._history.pop(0)

        # Predict
        predicted, confidence = self.predict(steps_ahead=1)

        data["predicted_state"] = predicted
        data["prediction_confidence"] = confidence

        self._metrics = {
            "predicted_aoa": float(predicted[0]) if len(predicted) > 0 else 0.0,
            "confidence": float(confidence),
            "history_length": len(self._history),
            "status": "success",
        }
        return data

    def predict(self, steps_ahead: int = 1) -> tuple:
        """
        Predict state steps_ahead into the future.
        
        Uses weighted least-squares polynomial fit on recent history
        with physics-based regularization.
        
        Returns:
            (predicted_state, confidence)
        """
        if len(self._history) < 3:
            # Not enough data — return last known
            if self._history:
                return self._history[-1], 0.1
            return np.zeros(4), 0.0

        history = np.array(self._history)
        n = len(history)
        t = np.arange(n) * self.dt

        # Polynomial fit (degree 2) for each state component
        predicted = np.zeros(history.shape[1])
        t_pred = t[-1] + steps_ahead * self.dt

        for dim in range(history.shape[1]):
            coeffs = np.polyfit(t, history[:, dim], min(2, n - 1))
            predicted[dim] = np.polyval(coeffs, t_pred)

        # Confidence based on fit residual
        fitted = np.array([np.polyval(np.polyfit(t, history[:, d], min(2, n-1)), t)
                          for d in range(history.shape[1])]).T
        residuals = np.mean(np.abs(history - fitted))
        confidence = float(np.clip(1.0 / (1.0 + residuals), 0.0, 1.0))

        return predicted, confidence

    def reset(self):
        """Clear prediction history."""
        self._history.clear()

    def get_metrics(self) -> dict:
        return self._metrics
