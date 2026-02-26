import numpy as np
import matplotlib.pyplot as plt

def generate_leo_trajectory(duration=600, dt=0.1, altitude_km=700, v_km_s=7.5, f_carrier_hz=2e9, noise_std_hz=5.0):
    """
    Simulates a LEO satellite Doppler trajectory.
    
    Parameters:
        duration (float): Total duration in seconds.
        dt (float): Time step in seconds.
        altitude_km (float): Satellite altitude.
        v_km_s (float): Orbital velocity.
        f_carrier_hz (float): Carrier frequency.
        noise_std_hz (float): RMS measurement noise in Hz.
        
    Returns:
        t (ndarray): Time vector.
        fd_true (ndarray): Ground truth Doppler frequency.
        fd_meas (ndarray): Measured Doppler frequency (with noise).
    """
    t = np.arange(0, duration, dt)
    n_samples = len(t)
    
    # Constants
    c = 3e8 # Speed of light m/s
    R_earth = 6371 # km
    r = R_earth + altitude_km # Distance from center of Earth
    
    # Angular velocity omega
    # In a simplified model, the satellite passes over the station.
    # theta(t) = omega * (t - t_mid)
    # At t_mid, theta = 0 (zenith)
    t_mid = duration / 2
    
    # mu = G * M_earth
    mu = 3.986e5 # km^3 / s^2
    omega = np.sqrt(mu / (r**3)) # rad/s
    
    theta = omega * (t - t_mid)
    
    # Radial velocity v_r = v * sin(theta)
    v_r = (v_km_s * 1000) * np.sin(theta)
    
    # Doppler fd = (v_r / c) * f_c
    fd_true = (v_r / c) * f_carrier_hz
    
    # Add noise
    fd_meas = fd_true + np.random.normal(0, noise_std_hz, size=n_samples)
    
    return t, fd_true, fd_meas

def add_maneuver(t, fd, maneuver_start=400, maneuver_duration=50, accel_hz_s2=5.0):
    """Adds a sudden maneuver (acceleration change) to the Doppler signal."""
    fd_maneuver = fd.copy()
    mask = (t >= maneuver_start) & (t < maneuver_start + maneuver_duration)
    # Acceleration effect: integration of accel over time
    t_m = t[mask] - maneuver_start
    # Integral of constant accel is linear frequency ramp
    # But here we want a 'robustness' test, so maybe a step in acceleration
    # resulting in a parabolic change or just a stronger ramp than orbital one.
    fd_maneuver[t >= maneuver_start] += 0.5 * accel_hz_s2 * (t[t >= maneuver_start] - maneuver_start)**2
    return fd_maneuver

if __name__ == "__main__":
    t, fd_true, fd_meas = generate_leo_trajectory()
    fd_maneuver = add_maneuver(t, fd_true)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, fd_true, label='True Doppler (Orbit)')
    plt.plot(t, fd_maneuver, '--', label='Doppler with Maneuver')
    plt.scatter(t[::100], fd_meas[::100], color='red', alpha=0.5, label='Measurements (sampled)')
    plt.xlabel('Time (s)')
    plt.ylabel('Doppler Frequency (Hz)')
    plt.title('Simulated LEO Doppler Trajectory')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("LEO Trajectory Simulation complete.")
