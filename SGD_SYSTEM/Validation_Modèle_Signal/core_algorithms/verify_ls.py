import numpy as np

def verify_with_ls():
    # Parameters
    M = 8
    N = 1000
    snr_db = 20 # Using 20dB for clarity
    snr_lin = 10**(snr_db/10)
    theta_true = 30.0
    theta_rad = np.deg2rad(theta_true)
    d_lambda = 0.5
    beta = 2 * np.pi * d_lambda
    
    # Bound calculation (Analytical)
    beta_sq = beta**2
    cos_sq = np.cos(theta_rad)**2
    crlb_var_rad = 6 / (beta_sq * cos_sq * M * (M**2 - 1) * snr_lin * N)
    crlb_std_deg = np.rad2deg(np.sqrt(crlb_var_rad))
    
    print(f"--- LS Verification (SNR={snr_db}dB, M={M}, N={N}) ---")
    print(f"Théorique CRLB (std) : {crlb_std_deg:.8f} °")
    
    n_runs = 50
    errors = []
    
    for _ in range(n_runs):
        # 1. Generate signal (One snapshot for speed, then scale variance)
        # Or better, full N snapshots
        s = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)
        P_s = np.mean(np.abs(s)**2)
        
        # 2. Array manifold
        n = np.arange(M)
        a_true = np.exp(1j * beta * n * np.sin(theta_rad)).reshape(-1, 1)
        
        # 3. Received signal X = a*s + noise
        X = a_true @ s.reshape(1, -1)
        sigma2 = P_s / snr_lin
        noise = np.sqrt(sigma2/2) * (np.random.randn(M, N) + 1j*np.random.randn(M, N))
        X_noisy = X + noise
        
        # 4. LS Estimate: Scan theta to maximize |a(theta)^H * X|
        # For single source, MLE is peak of beamformer
        scan_angles = np.arange(theta_true - 1.0, theta_true + 1.0, 0.001)
        vals = []
        for ang in scan_angles:
            a_scan = np.exp(1j * beta * n * np.sin(np.deg2rad(ang))).reshape(-1, 1)
            # Power in direction a: sum_n |a^H x_n|^2 = a^H Rxx a
            # MLE for deterministic signal: max over theta of trace(P_a(theta) Rxx)
            # which is equivalent to max a^H Rxx a / (a^H a)
            metric = np.sum(np.abs(a_scan.conj().T @ X_noisy)**2)
            vals.append(metric)
        
        est_angle = scan_angles[np.argmax(vals)]
        errors.append(est_angle - theta_true)
        
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    ratio = rmse / crlb_std_deg
    
    print(f"RMSE LS ({n_runs} runs)  : {rmse:.8f} °")
    print(f"Ratio RMSE/CRLB        : {ratio:.4f}")

if __name__ == "__main__":
    verify_with_ls()
