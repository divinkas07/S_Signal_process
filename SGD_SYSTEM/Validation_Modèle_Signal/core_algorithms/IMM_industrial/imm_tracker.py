import numpy as np
from filterpy.kalman import KalmanFilter, IMMEstimator

class IndustrialIMMTracker:
    def __init__(self, dt=0.1, R_hz=25.0, sigma_drift=1.0, sigma_maneuver=50.0):
        """
        Industrial IMM Tracker for Doppler tracking.
        
        Args:
            dt (float): Sampling interval.
            R_hz (float): Measurement noise variance (R).
            sigma_drift (float): Process noise for drift model.
            sigma_maneuver (float): Process noise for maneuver model.
        """
        self.dt = dt
        
        # All filters share a state [f, f_dot]
        # x = [frequency, frequency_rate]
        
        # 1. Constant Frequency Model (very low Q)
        kf1 = self._create_kf(q_std=1e-4) # Almost no change
        
        # 2. Drift Model (standard dynamic)
        kf2 = self._create_kf(q_std=sigma_drift)
        
        # 3. Maneuver Model (high Q)
        kf3 = self._create_kf(q_std=sigma_maneuver)
        
        filters = [kf1, kf2, kf3]
        
        # Initial probabilities (equal or biased towards stability)
        mu = [0.8, 0.1, 0.1] 
        
        # Transition Matrix (Highly stable as recommended)
        # Prob of staying in current model is 0.98
        trans = np.array([
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98]
        ])
        
        self.imm = IMMEstimator(filters, mu, trans)
        
    def _create_kf(self, q_std):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # State transition F = [1 dt; 0 1]
        kf.F = np.array([
            [1, self.dt],
            [0, 1]
        ])
        
        # Measurement matrix H = [1 0] (we only measure frequency)
        kf.H = np.array([[1, 0]])
        
        # Measurement noise R
        kf.R = np.eye(1) * 25.0 # User recommended 25 Hz^2 for 5Hz RMS
        
        # Process noise Q (Discrete constant white noise acceleration model)
        # Recommended: Q = [dt^3/3 dt^2/2; dt^2/2 dt] * sigma^2
        q_elements = np.array([
            [(self.dt**3)/3, (self.dt**2)/2],
            [(self.dt**2)/2, self.dt]
        ])
        kf.Q = q_elements * (q_std**2)
        
        # Initial state and covariance
        kf.x = np.array([0, 0])
        kf.P *= 1000 # High initial uncertainty
        
        return kf

    def predict(self):
        self.imm.predict()
        
    def update(self, z):
        # z should be a float or [float]
        self.imm.update(np.array([z]))
        
    @property
    def x(self):
        return self.imm.x.copy()
    
    @property
    def mu(self):
        return self.imm.mu.copy()

if __name__ == "__main__":
    # Quick test
    tracker = IndustrialIMMTracker(dt=0.1)
    tracker.update(500.0)
    tracker.predict()
    print(f"Initial state: {tracker.x}")
    print(f"Initial probabilities: {tracker.mu}")
    print("Industrial IMM Tracker initialized.")
