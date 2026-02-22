import numpy as np
from filterpy.kalman import KalmanFilter, IMMEstimator

class IMMFilter:
    def __init__(self, dt=0.1):
        self.dt = dt
        
        # Define 4 dynamic models
        # Model 1: Static (very low Q)
        # Model 2: Walking (low Q)
        # Model 3: Running (medium Q)
        # Model 4: Erratic (high Q)
        
        kf_static = self._create_kf(q_scale=0.0001)
        kf_walk = self._create_kf(q_scale=0.01)
        kf_run = self._create_kf(q_scale=0.1)
        kf_erratic = self._create_kf(q_scale=1.0)
        
        filters = [kf_static, kf_walk, kf_run, kf_erratic]
        
        # Initial probabilities
        mu = [0.25, 0.25, 0.25, 0.25]
        
        # Transition matrix
        # High probability of staying in the current mode
        trans = np.array([
            [0.90, 0.03, 0.03, 0.04],
            [0.03, 0.90, 0.03, 0.04],
            [0.03, 0.03, 0.90, 0.04],
            [0.03, 0.03, 0.03, 0.91]
        ])
        
        self.imm = IMMEstimator(filters, mu, trans)
        
    def _create_kf(self, q_scale):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix F
        kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix H
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Initial covariance P
        kf.P *= 10
        
        # Measurement noise R
        kf.R = np.eye(2) * 0.5
        
        # Process noise Q
        q = np.array([
            [(self.dt**4)/4, 0, (self.dt**3)/2, 0],
            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
            [(self.dt**3)/2, 0, self.dt**2, 0],
            [0, (self.dt**3)/2, 0, self.dt**2]
        ]) * q_scale
        kf.Q = q
        
        return kf

    def predict(self):
        self.imm.predict()
        
    def update(self, z):
        self.imm.update(z)
        
    @property
    def x(self):
        return self.imm.x.copy()
    
    @property
    def mu(self):
        return self.imm.mu.copy()
