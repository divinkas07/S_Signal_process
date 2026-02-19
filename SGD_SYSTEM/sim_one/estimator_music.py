"""
Ir divinkas — SYNAPTIC Lab
SGD System — Estimateur MUSIC (MUltiple SIgnal Classification)

Stub fonctionnel :
  1. Calcul de la matrice de covariance spatiale
  2. Décomposition en valeurs propres
  3. Séparation sous-espace signal / bruit
  4. Balayage du pseudo-spectre MUSIC
  5. Détection des pics → estimation AOA
"""

import numpy as np


def music_spectrum(
    X: np.ndarray,
    n_sources: int,
    d_lambda: float,
    angle_range: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule le pseudo-spectre MUSIC.

    Parameters
    ----------
    X : np.ndarray, shape (M, N)
        Signal reçu sur M antennes, N échantillons.
    n_sources : int
        Nombre de sources à estimer.
    d_lambda : float
        Espacement inter-antennes (fraction de λ).
    angle_range : np.ndarray or None
        Angles à balayer (degrés). Défaut: -90 à 90, pas 0.1.

    Returns
    -------
    angles : np.ndarray
        Angles balayés (degrés).
    spectrum : np.ndarray
        Pseudo-spectre MUSIC (dB).
    """
    M, N = X.shape

    if angle_range is None:
        angle_range = np.arange(-90, 90.1, 0.1)

    # ── Matrice de covariance spatiale ──
    Rxx = (X @ X.conj().T) / N

    # ── Décomposition en valeurs propres ──
    eigenvalues, eigenvectors = np.linalg.eigh(Rxx)

    # Tri décroissant
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # ── Sous-espace bruit ──
    En = eigenvectors[:, n_sources:]

    # ── Balayage pseudo-spectre ──
    spectrum = np.zeros(len(angle_range))
    for i, theta in enumerate(angle_range):
        a = np.exp(1j * 2 * np.pi * d_lambda * np.arange(M) * np.sin(np.deg2rad(theta)))
        a = a.reshape(-1, 1)
        denom = a.conj().T @ En @ En.conj().T @ a
        spectrum[i] = 1.0 / np.abs(denom[0, 0])

    spectrum_db = 10 * np.log10(spectrum / np.max(spectrum) + 1e-12)

    return angle_range, spectrum_db


def estimate_aoa_music(
    X: np.ndarray,
    n_sources: int,
    d_lambda: float,
    angle_range: np.ndarray = None,
) -> np.ndarray:
    """
    Estime les AOA via MUSIC.

    Returns
    -------
    aoa_estimates : np.ndarray, shape (n_sources,)
        Angles estimés (degrés), triés croissant.
    """
    angles, spectrum = music_spectrum(X, n_sources, d_lambda, angle_range)

    # Détection des n_sources plus grands pics
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(spectrum, height=-30)

    if len(peaks) == 0:
        return np.array([np.nan] * n_sources)

    peak_heights = spectrum[peaks]
    top_indices = np.argsort(peak_heights)[-n_sources:]
    aoa_estimates = np.sort(angles[peaks[top_indices]])

    return aoa_estimates


if __name__ == "__main__":
    from signal_model import load_config, generate_signal_from_config
    from channel_model import apply_channel

    cfg = load_config()
    t, s, X = generate_signal_from_config(cfg)
    X_noisy, _ = apply_channel(X, cfg)

    aoa_true = cfg["array"]["aoa_deg"]
    aoa_est = estimate_aoa_music(X_noisy, n_sources=1, d_lambda=cfg["array"]["d_lambda"])

    print(f"AOA vrai    : {aoa_true}°")
    print(f"AOA estimé  : {aoa_est[0]:.2f}°")
    print(f"Erreur      : {abs(aoa_est[0] - aoa_true):.4f}°")
