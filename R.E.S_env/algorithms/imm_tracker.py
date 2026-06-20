"""
Interacting Multiple Model (IMM) Tracker – Multi-model state estimator.
Combines multiple Kalman filters (constant velocity + constant turn rate)
to track a maneuvering satellite/target.

Each model has its own state transition, and mode probabilities
are updated at each step via Bayesian mixing.
"""

import numpy as np


class KalmanFilter:
    """Standard Kalman filter for a single motion model."""

    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        Args:
            F: State transition matrix
            H: Observation matrix
            Q: Process noise covariance
            R: Measurement noise covariance
        """
        self.F = F  # State transition
        self.H = H  # Observation
        self.Q = Q  # Process noise
        self.R = R  # Measurement noise

        n = F.shape[0]
        self.x = np.zeros(n)            # State estimate
        self.P = np.eye(n) * 100.0      # State covariance

    def predict(self):
        """Predict step: x̂ = F·x, P̂ = F·P·F^T + Q"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> float:
        """
        Update step with measurement z.
        Returns the measurement likelihood.
        """
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P

        # Likelihood
        n = len(y)
        det_S = np.linalg.det(S)
        if det_S <= 0:
            det_S = 1e-10
        exponent = -0.5 * y.T @ np.linalg.inv(S) @ y
        likelihood = np.exp(exponent) / np.sqrt((2 * np.pi)**n * det_S)

        return float(np.real(likelihood))


class IMMTracker:
    """
    IMM estimator with 2 models:
    - Model 1: Constant Velocity (CV)
    - Model 2: Constant Turn Rate (CT)
    
    State: [x, y, vx, vy] for AOA tracking in 2D.
    """

    def __init__(self, dt: float = 0.01, process_noise: float = 1.0,
                 meas_noise: float = 0.5):
        self.dt = dt
        self._metrics = {}

        # Model transition probability matrix
        self.TPM = np.array([
            [0.95, 0.05],
            [0.05, 0.95],
        ])

        # Mode probabilities
        self.mu = np.array([0.5, 0.5])

        # Create filters
        self.filters = [
            self._create_cv_filter(dt, process_noise, meas_noise),
            self._create_ct_filter(dt, process_noise, meas_noise),
        ]

    # ── Pipeline Interface ─────────────────────────────────────────────

    def process(self, data: dict) -> dict:
        """
        Pipeline interface: track the AOA over time.
        
        Expects:
            'aoa_estimate_deg': list of AOA estimates
            
        Adds:
            'imm_state': filtered state estimate
            'imm_mode_probs': mode probabilities [p_cv, p_ct]
        """
        aoa = data.get("aoa_estimate_deg", [])
        if not aoa:
            self._metrics = {"status": "no_aoa"}
            return data

        z = np.array([aoa[0], 0.0])  # Measurement: [azimuth, elevation_placeholder]
        state, mode_probs = self.update(z)

        data["imm_state"] = state
        data["imm_mode_probs"] = mode_probs.tolist()
        data["imm_filtered_aoa"] = float(state[0])

        self._metrics = {
            "filtered_aoa": float(state[0]),
            "cv_prob": float(mode_probs[0]),
            "ct_prob": float(mode_probs[1]),
            "status": "success",
        }
        return data

    # ── IMM Algorithm ──────────────────────────────────────────────────

    def update(self, z: np.ndarray) -> tuple:
        """
        Full IMM cycle: mix → predict → update → combine.
        
        Args:
            z: Measurement vector [az, el]
            
        Returns:
            (state_estimate, mode_probabilities)
        """
        n_models = len(self.filters)

        # 1. Mixing probabilities
        c = self.TPM.T @ self.mu  # Normalization constant per model
        c = np.maximum(c, 1e-15)

        mu_mix = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                mu_mix[i, j] = self.TPM[i, j] * self.mu[i] / c[j]

        # 2. State mixing
        mixed_states = []
        mixed_covs = []
        for j in range(n_models):
            x_mix = np.zeros_like(self.filters[0].x)
            for i in range(n_models):
                x_mix += mu_mix[i, j] * self.filters[i].x
            mixed_states.append(x_mix)

            P_mix = np.zeros_like(self.filters[0].P)
            for i in range(n_models):
                diff = self.filters[i].x - x_mix
                P_mix += mu_mix[i, j] * (self.filters[i].P + np.outer(diff, diff))
            mixed_covs.append(P_mix)

        # Set mixed initial conditions
        for j in range(n_models):
            self.filters[j].x = mixed_states[j]
            self.filters[j].P = mixed_covs[j]

        # 3. Predict + Update each filter
        likelihoods = np.zeros(n_models)
        for j in range(n_models):
            self.filters[j].predict()
            likelihoods[j] = self.filters[j].update(z) + 1e-300

        # 4. Mode probability update
        self.mu = c * likelihoods
        total = np.sum(self.mu)
        if total > 0:
            self.mu /= total
        else:
            self.mu = np.ones(n_models) / n_models

        # 5. Combined state estimate
        x_combined = np.zeros_like(self.filters[0].x)
        for j in range(n_models):
            x_combined += self.mu[j] * self.filters[j].x

        return x_combined, self.mu.copy()

    def reset(self):
        """Reset tracker state."""
        self.mu = np.array([0.5, 0.5])
        for f in self.filters:
            f.x = np.zeros_like(f.x)
            f.P = np.eye(len(f.x)) * 100.0

    def get_metrics(self) -> dict:
        return self._metrics

    # ── Filter Creation ────────────────────────────────────────────────

    @staticmethod
    def _create_cv_filter(dt: float, q: float, r: float) -> KalmanFilter:
        """Constant Velocity model: state = [x, y, vx, vy]"""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        Q = q * np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ],
        ])
        R = r * np.eye(2)
        return KalmanFilter(F, H, Q, R)

    @staticmethod
    def _create_ct_filter(dt: float, q: float, r: float,
                          omega: float = 0.1) -> KalmanFilter:
        """
        Constant Turn Rate model (linearized).
        Uses angular rate omega for curved motion.
        """
        cos_w = np.cos(omega * dt)
        sin_w = np.sin(omega * dt)
        
        # Avoid division by zero for small omega
        if abs(omega) < 1e-6:
            return IMMTracker._create_cv_filter(dt, q, r)
        
        F = np.array([
            [1, 0,  sin_w/omega,       -(1-cos_w)/omega],
            [0, 1,  (1-cos_w)/omega,    sin_w/omega     ],
            [0, 0,  cos_w,             -sin_w           ],
            [0, 0,  sin_w,              cos_w            ],
        ])
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        Q = q * np.eye(4) * dt**2
        R = r * np.eye(2)
        return KalmanFilter(F, H, Q, R)
