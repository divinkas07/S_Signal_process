import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from leo_doppler_sim import generate_leo_trajectory, add_maneuver
from imm_tracker import IndustrialIMMTracker

def create_simple_kf(dt, R_hz=25.0):
    """Constant frequency model (Model 1)."""
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.array([[1]])
    kf.H = np.array([[1]])
    kf.R = np.eye(1) * R_hz
    kf.Q = np.eye(1) * 0.01 # Very low process noise
    kf.x = np.array([0])
    kf.P *= 1000
    return kf

def create_drift_kf(dt, R_hz=25.0, sigma_drift=1.0):
    """Constant acceleration/drift model (Model 2). Represents 'EKF' for this linear system."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1, dt], [0, 1]])
    kf.H = np.array([[1, 0]])
    kf.R = np.eye(1) * R_hz
    q_elements = np.array([[(dt**3)/3, (dt**2)/2], [(dt**2)/2, dt]])
    kf.Q = q_elements * (sigma_drift**2)
    kf.x = np.array([0, 0])
    kf.P *= 1000
    return kf

def run_comparison():
    # 1. Simulation Setup
    duration = 600
    dt = 0.1
    t, fd_true, fd_meas = generate_leo_trajectory(duration=duration, dt=dt)
    
    # Add a maneuver at t=400s
    fd_truth_maneuver = add_maneuver(t, fd_true, maneuver_start=400, maneuver_duration=50, accel_hz_s2=2.0)
    # Regenerate measurements with maneuver
    fd_meas_maneuver = fd_truth_maneuver + np.random.normal(0, 5.0, size=len(t))
    
    # 2. Filter Initializations
    kf_simple = create_simple_kf(dt)
    kf_drift = create_drift_kf(dt)
    imm = IndustrialIMMTracker(dt=dt)
    
    # 3. Tracking Loop
    n = len(t)
    results_simple = np.zeros(n)
    results_drift = np.zeros(n)
    results_imm = np.zeros(n)
    probs_imm = np.zeros((n, 3))
    
    print("Tracking in progress...")
    for i in range(n):
        z = fd_meas_maneuver[i]
        
        # Simple KF
        kf_simple.predict()
        kf_simple.update(np.array([z]))
        results_simple[i] = kf_simple.x[0]
        
        # Drift KF
        kf_drift.predict()
        kf_drift.update(np.array([z]))
        results_drift[i] = kf_drift.x[0]
        
        # IMM
        imm.predict()
        imm.update(z)
        results_imm[i] = imm.x[0]
        probs_imm[i] = imm.mu
        
    # 4. Metrics
    rmse_simple = np.sqrt(np.mean((results_simple - fd_truth_maneuver)**2))
    rmse_drift = np.sqrt(np.mean((results_drift - fd_truth_maneuver)**2))
    rmse_imm = np.sqrt(np.mean((results_imm - fd_truth_maneuver)**2))
    
    print(f"RMSE Results (Hz):")
    print(f"  Simple KF: {rmse_simple:.2f}")
    print(f"  Drift KF : {rmse_drift:.2f}")
    print(f"  IMM Tracker: {rmse_imm:.2f}")
    
    # 5. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Tracking Comparison
    axes[0].plot(t, fd_truth_maneuver, 'k', label='Ground Truth', linewidth=2)
    axes[0].plot(t, results_simple, '--', label='Simple KF (Const Freq)', alpha=0.7)
    axes[0].plot(t, results_drift, '--', label='Drift KF (EKF proxy)', alpha=0.7)
    axes[0].plot(t, results_imm, 'r', label='IMM Tracker', linewidth=1.5)
    axes[0].set_ylabel('Doppler (Hz)')
    axes[0].set_title('Doppler Tracking Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Error Comparison
    axes[1].plot(t, results_simple - fd_truth_maneuver, label='Simple KF Error', alpha=0.5)
    axes[1].plot(t, results_drift - fd_truth_maneuver, label='Drift KF Error', alpha=0.5)
    axes[1].plot(t, results_imm - fd_truth_maneuver, 'r', label='IMM Error')
    axes[1].set_ylabel('Error (Hz)')
    axes[1].set_title('Tracking Error')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: IMM Model Probabilities
    axes[2].plot(t, probs_imm[:, 0], label='Model 1 (Const Freq)')
    axes[2].plot(t, probs_imm[:, 1], label='Model 2 (Drift)')
    axes[2].plot(t, probs_imm[:, 2], label='Model 3 (Maneuver)')
    axes[2].set_ylabel('Probability')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('IMM Model Weights over Time')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('doppler_tracking_benchmark.png', dpi=150)
    # plt.show()
    
    print("Benchmark complete. Results saved to 'doppler_tracking_benchmark.png'")

if __name__ == "__main__":
    run_comparison()
