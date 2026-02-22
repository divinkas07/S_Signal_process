"""
Ir divinkas — SYNAPTIC Lab
SGD System — Estimateur MUSIC (MUltiple SIgnal Classification)

Ce module implémente l'algorithme MUSIC pour l'estimation de l'angle d'arrivée (AOA).
"""

import numpy as np
from scipy.signal import find_peaks

def music_spectrum(X, n_sources, d_lambda, angle_range=None):
    """
    Calcule le pseudo-spectre MUSIC pour un réseau linéaire uniforme (ULA).
    
    X: (n_antennas, n_samples)
    n_sources: Nombre de sources attendues
    d_lambda: Espacement inter-antennes en fraction de λ
    angle_range: Tableau d'angles à scanner (degrés)
    """
    M, N = X.shape
    if angle_range is None:
        angle_range = np.linspace(-90, 90, 1801) # Résolution 0.1° por défaut

    # 1. Matrice de covariance spatiale
    R = (X @ X.conj().T) / N

    # 2. Décomposition en valeurs propres
    eigenvals, eigenvecs = np.linalg.eigh(R)
    
    # 3. Tri et séparation du sous-espace bruit
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Le sous-espace bruit correspond aux plus petites valeurs propres
    En = eigenvecs[:, n_sources:]

    # 4. Calcul du pseudo-spectre
    spectrum = np.zeros(len(angle_range))
    EnEnH = En @ En.conj().T
    
    for i, angle in enumerate(angle_range):
        theta = np.deg2rad(angle)
        # Steering vector a(theta) = [1, e^(j*2*pi*d*sin(theta)/lam), ...]
        steering = np.exp(1j * 2 * np.pi * d_lambda * np.arange(M) * np.sin(theta)).reshape(-1, 1)
        
        # P_music = 1 / (a(theta)^H * En * En^H * a(theta))
        denom = steering.conj().T @ EnEnH @ steering
        spectrum[i] = 1.0 / np.abs(denom[0, 0])

    # Normalisation en dB
    spectrum_db = 10 * np.log10(spectrum / np.max(spectrum) + 1e-12)
    
    return angle_range, spectrum_db

def estimate_aoa(X, n_sources, d_lambda, angle_range=None):
    """
    Estime les angles d'arrivée en cherchant les pics du pseudo-spectre.
    """
    angles, spectrum = music_spectrum(X, n_sources, d_lambda, angle_range)
    
    # Recherche des pics
    peaks, properties = find_peaks(spectrum, height=-30) # Seuil relatif à 30dB sous le max
    
    if len(peaks) == 0:
        return np.array([np.nan] * n_sources)

    # Sélection des n_sources plus hauts pics
    peak_heights = spectrum[peaks]
    top_indices = np.argsort(peak_heights)[-n_sources:]
    aoa_estimates = np.sort(angles[peaks[top_indices]])
    
    return aoa_estimates
