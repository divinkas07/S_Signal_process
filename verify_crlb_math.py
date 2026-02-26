import numpy as np
import scipy.linalg

def verify_math():
    fs = 1000.0
    N = 1000
    M = 8
    snr_db = 20.0
    snr_lin = 10**(snr_db / 10.0)
    theta_deg = 30.0
    theta_rad = np.deg2rad(theta_deg)
    d_lam = 0.5
    
    # 1. CRLB AOA
    # beta = 2*pi*d*cos(theta)/lambda
    beta = 2 * np.pi * d_lam * np.cos(theta_rad)
    # var = 6 / (beta^2 * M * (M^2-1) * SNR * L)
    crlb_var_rad = 6 / (beta**2 * M * (M**2 - 1) * snr_lin * N)
    crlb_std_deg = np.rad2deg(np.sqrt(crlb_var_rad))
    
    print(f"CRLB AOA (deg): {crlb_std_deg:.6f}")
    
    # 2. Simulation verification
    num_runs = 1000
    errors = []
    
    for _ in range(num_runs):
        t = np.arange(N) / fs
        # Single source
        s = np.exp(1j * 2 * np.pi * 50 * t) 
        p_s = np.mean(np.abs(s)**2) # Should be 1.0
        
        # Array signal
        n = np.arange(M)
        a = np.exp(1j * 2 * np.pi * d_lam * n * np.sin(theta_rad)).reshape(-1, 1)
        X = a @ s.reshape(1, -1)
        
        # Noise
        sigma2 = p_s / snr_lin
        noise = np.sqrt(sigma2/2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
        X_noisy = X + noise
        
        # Realized SNR check
        p_sig_real = np.mean(np.abs(X)**2)
        p_noise_real = np.mean(np.abs(noise)**2)
        snr_real = p_sig_real / p_noise_real
        # print(f"Realized SNR: {10*np.log10(snr_real):.2f} dB")
        
        # MUSIC or simple beamformer
        angles = np.linspace(theta_deg-1, theta_deg+1, 1000)
        spectrum = []
        # Beamformer for simplicity (should match CRLB at high SNR)
        for ang in angles:
            a_search = np.exp(1j * 2 * np.pi * d_lam * n * np.sin(np.deg2rad(ang))).reshape(-1, 1)
            # bf = a^H Rxx a
            # For 1 snapshot: |a^H x|^2
            # Here we average over snapshots
            val = np.mean(np.abs(a_search.conj().T @ X_noisy)**2)
            spectrum.append(val)
            
        est_angle = angles[np.argmax(spectrum)]
        errors.append(est_angle - theta_deg)
        
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    print(f"Measured RMSE (deg): {rmse:.6f}")
    print(f"Ratio RMSE/CRLB: {rmse/crlb_std_deg:.4f}")

if __name__ == "__main__":
    verify_math()
