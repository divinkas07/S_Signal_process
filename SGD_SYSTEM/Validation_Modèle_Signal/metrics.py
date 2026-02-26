"""
Ir divinkas — SYNAPTIC Lab
SGD System — Métriques de performance ICA

Fonctions pour évaluer la qualité de la séparation de sources :
- Résolution des ambiguïtés (permutation, échelle, phase)
- Calcul du SIR (Signal-to-Interference Ratio)
"""

import numpy as np
import scipy.optimize
from scipy.signal import find_peaks

def detect_spectral_peaks(s: np.ndarray, fs: float, n_peaks: int) -> np.ndarray:
    """
    Detecte les n_peaks frequences dominantes dans le spectre avec interpolation.
    """
    N = len(s)
    # Zero-padding pour meilleure résolution (interpolation plus fine)
    N_fft = max(N, 8192)
    S = np.fft.fftshift(np.abs(np.fft.fft(s, n=N_fft)))
    freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / fs))
    df = freqs[1] - freqs[0]
    
    # Trouver les indices des n_peaks plus grands
    # Mais on veut éviter les bins adjacents au même pic
    from scipy.signal import find_peaks as sp_find_peaks
    peaks_idx, props = sp_find_peaks(S, distance=N//100) # Séparation minimale
    
    if len(peaks_idx) < n_peaks:
        # Fallback si find_peaks rate
        peaks_idx = np.argsort(S)[-n_peaks:]
    else:
        # Prendre les n_peaks les plus hauts parmi les pics detectés
        heights = S[peaks_idx]
        top_peaks = np.argsort(heights)[-n_peaks:]
        peaks_idx = peaks_idx[top_peaks]

    interpolated_freqs = []
    df = freqs[1] - freqs[0]
    
    for idx in peaks_idx:
        if 0 < idx < len(S) - 1:
            # On utilise l'interpolation parabolique sur le LOG-magnitude 
            # (bien plus précis pour un pic de type sinc)
            y1, y2, y3 = np.log(S[idx-1] + 1e-12), np.log(S[idx] + 1e-12), np.log(S[idx+1] + 1e-12)
            denom = 2 * (y1 - 2*y2 + y3)
            if abs(denom) > 1e-12:
                d_idx = (y1 - y3) / denom
                f_est = freqs[idx] + d_idx * df
            else:
                f_est = freqs[idx]
        else:
            f_est = freqs[idx]
        interpolated_freqs.append(f_est)
        
    return np.sort(np.array(interpolated_freqs))

def compute_spectral_error(expected_freqs, detected_freqs):
    """
    Calcule l'erreur spectrale relative en pourcentage.
    """
    expected = np.sort(expected_freqs)
    detected = np.sort(detected_freqs)
    errors_pct = np.abs(detected - expected) / expected * 100
    return errors_pct

def compute_snr_error(target_snr_db, measured_snr_db):
    """
    Calcule l'erreur absolue de SNR en dB.
    """
    return np.abs(measured_snr_db - target_snr_db)

def compute_doppler_error(target_fd, shift_measured):
    """
    Calcule l'erreur relative de décalage Doppler en pourcentage.
    """
    return np.abs(shift_measured - target_fd) / np.abs(target_fd) * 100

def compute_doppler_rate_error(target_rate, estimated_rate):
    """
    Calcule l'erreur relative de dérive Doppler en pourcentage.
    """
    if abs(target_rate) < 1e-12:
        return 0.0 if abs(estimated_rate) < 1e-12 else 100.0
    return np.abs(estimated_rate - target_rate) / np.abs(target_rate) * 100

def compute_crlb_efficiency(rmse, crlb_std):
    """
    Calcule le ratio d'efficacité (RMSE / sqrt(CRLB)).
    Idéalement proche de 1.
    """
    return rmse / crlb_std

def rmse(est, true):
    """
    Calcule la Root Mean Square Error (RMSE).
    est: tableau des estimations
    true: tableau des valeurs vraies (ou scalaire)
    """
    return np.sqrt(np.mean(np.abs(est - true)**2))

def mae(est, true):
    """
    Calcule la Mean Absolute Error (MAE).
    est: tableau des estimations
    true: tableau des valeurs vraies (ou scalaire)
    """
    return np.mean(np.abs(est - true))

def compute_correlation_matrix(S1, S2):
    """
    Calcule la matrice de corrélation croisée entre les lignes de S1 et S2.
    S1: (n_sources, n_samples)
    S2: (n_sources, n_samples)
    Returns: C (n_sources, n_sources) où C[i, j] = corr(S1[i], S2[j])
    """
    n_s1 = S1.shape[0]
    n_s2 = S2.shape[0]
    C = np.zeros((n_s1, n_s2))
    
    # Normalisation
    S1_norm = S1 / (np.linalg.norm(S1, axis=1, keepdims=True) + 1e-12)
    S2_norm = S2 / (np.linalg.norm(S2, axis=1, keepdims=True) + 1e-12)
    
    # Produit scalaire = Corrélation (pour signaux centrés unit-variance)
    # C = | <u, v> |
    # Attention: S1 @ S2.conj().T
    C_complex = S1_norm @ S2_norm.conj().T
    return np.abs(C_complex)

def align_sources(S_true, S_est):
    """
    Trouve la meilleure permutation pour aligner S_est sur S_true.
    Utilise l'algorithme Hongrois (linear sum assignment) sur la matrice de corrélation.
    
    Returns:
    S_est_aligned: Sources estimées permutées pour correspondre à S_true.
    indices: (row_ind, col_ind) indices de correspondance.
    """
    C = compute_correlation_matrix(S_true, S_est)
    
    # On veut maximiser la corrélation -> minimiser (-C)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-C)
    
    # Réorganiser S_est
    S_est_aligned = S_est[col_ind]
    
    return S_est_aligned, (row_ind, col_ind)

def compute_sir(S_true, S_est):
    """
    Calcule le SIR (Signal-to-Interference Ratio) en dB pour chaque source.
    Gère l'alignement et la correction de phase/échelle.
    
    SIR = 10 log10 ( ||s_target||^2 / ||s_target - alpha * s_est||^2 )
    
    Returns:
    sirs: np.array (n_sources,) en dB
    permutations: indices de permutation utilisés
    """
    n_sources = S_true.shape[0]
    
    # 1. Aligner les sources
    S_est_aligned, (row_ind, col_ind) = align_sources(S_true, S_est)
    
    sirs = np.zeros(n_sources)
    
    for i in range(n_sources):
        s_t = S_true[i]
        s_e = S_est_aligned[i]
        
        # 2. Estimation du facteur d'échelle alpha optimal (moindres carrés)
        # min || s_t - alpha * s_e ||^2
        # alpha = <s_t, s_e> / ||s_e||^2
        
        num = np.vdot(s_e, s_t) # s_e.conj * s_t
        den = np.vdot(s_e, s_e).real
        
        if den < 1e-12:
            sirs[i] = 0.0
            continue
            
        alpha = num / den
        
        error = s_t - alpha * s_e
        p_signal = np.vdot(s_t, s_t).real
        p_error = np.vdot(error, error).real
        
        if p_error < 1e-12:
            sirs[i] = 100.0 # Perfect separation
        else:
            sirs[i] = 10 * np.log10(p_signal / p_error)
            
    return sirs, col_ind

if __name__ == "__main__":
    # Test unitaire
    print("Testing Metrics...")
    
    N = 1000
    s1 = np.exp(1j * 2 * np.pi * 0.01 * np.arange(N))
    s2 = np.random.randn(N) + 1j * np.random.randn(N)
    
    S_true = np.vstack([s1, s2])
    
    # Cas parfait (permuté + scale + phase)
    S_est = np.vstack([s2 * 0.5j, s1 * 2])
    
    sirs, perms = compute_sir(S_true, S_est)
    print(f"SIRs for perfect case: {sirs}")
    print(f"Permutations: {perms}")
    
    # Cas bruité
    S_est_noisy = S_est + 0.1 * (np.random.randn(*S_est.shape) + 1j * np.random.randn(*S_est.shape))
    sirs_noisy, _ = compute_sir(S_true, S_est_noisy)
    print(f"SIRs for noisy case: {sirs_noisy}")
    print("✅ Metrics validation PASSED" if np.all(sirs > 80) else "❌ Metrics validation FAILED")
