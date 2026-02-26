import numpy as np

def numerical_crlb():
    M = 8
    N = 1000
    snr_db = 30
    snr_lin = 10**(snr_db/10)
    theta_deg = 30
    theta_rad = np.deg2rad(theta_deg)
    d_lambda = 0.5
    beta = 2 * np.pi * d_lambda
    
    # Per-element signal power P=1, noise power sigma2 = 1/snr_lin
    sigma2 = 1.0 / snr_lin
    
    # Steering vector derivative da/dtheta
    n = np.arange(M)
    a = np.exp(1j * beta * n * np.sin(theta_rad))
    da = 1j * beta * n * np.cos(theta_rad) * a
    
    # Fisher Information Matrix (FIM) for deterministic model with unknown complex amplitude
    # We estimate theta and complex amplitude s = s_re + j*s_im
    # Actually, for a block of N snapshots, we have N independent s_n.
    # Total FIM is sum of FIMs for each snapshot.
    # For one snapshot:
    # J_theta_theta = (2/sigma2) * Re( (da*s)^H * (da*s) ) = (2*|s|^2/sigma2) * ||da||^2
    # J_theta_sre = (2/sigma2) * Re( (da*s)^H * a )
    # J_theta_sim = (2/sigma2) * Re( (da*s)^H * (j*a) )
    # Let s = 1.0 for simplicity (Power = 1)
    
    norm_da_sq = np.sum(np.abs(da)**2)
    J_theta_theta = (2.0 / sigma2) * norm_da_sq
    
    # Cross terms for unknown complex amplitude
    # Term 1: (da^H * a)
    dot_da_a = np.vdot(da, a)
    J_theta_sre = (2.0 / sigma2) * np.real(np.conj(dot_da_a))
    J_theta_sim = (2.0 / sigma2) * np.real(np.conj(dot_da_a) * 1j)
    
    # J_s_s = (2/sigma2) * Re( [a j*a]^H * [a j*a] )
    J_sre_sre = (2.0 / sigma2) * np.real(np.vdot(a, a))
    J_sim_sim = (2.0 / sigma2) * np.real(np.vdot(ja := 1j*a, ja))
    J_sre_sim = (2.0 / sigma2) * np.real(np.vdot(a, 1j*a))
    
    # Block FIM for one snapshot
    FIM_1 = np.array([
        [J_theta_theta, J_theta_sre, J_theta_sim],
        [J_theta_sre, J_sre_sre, J_sre_sim],
        [J_theta_sim, J_sre_sim, J_sim_sim]
    ])
    
    # Total FIM over N snapshots (assuming same power = 1)
    FIM_N = FIM_1 * N
    
    # CRLB is the inverse
    CRLB_mat = np.linalg.inv(FIM_N)
    crlb_var_rad = CRLB_mat[0, 0]
    crlb_std_deg = np.rad2deg(np.sqrt(crlb_var_rad))
    
    # Compare with formula: 6 / (beta_sq * cos^2 * M * (M^2-1) * SNR * N)
    beta_sq = beta**2
    cos_sq = np.cos(theta_rad)**2
    formula_var_rad = 6 / (beta_sq * cos_sq * M * (M**2 - 1) * snr_lin * N)
    formula_std_deg = np.rad2deg(np.sqrt(formula_var_rad))
    
    print(f"--- FIM Numerical Verification ---")
    print(f"Numerical CRLB (std deg) : {crlb_std_deg:.8f}")
    print(f"Formula CRLB (std deg)   : {formula_std_deg:.8f}")
    print(f"Ratio Numerical/Formula  : {crlb_std_deg / formula_std_deg:.4f}")

if __name__ == "__main__":
    numerical_crlb()
