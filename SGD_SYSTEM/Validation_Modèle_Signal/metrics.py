"""
Ir divinkas — SYNAPTIC Lab
SGD System — Métriques d'évaluation

Fonctions de calcul :
  - RMSE  (Root Mean Square Error)
  - SIR   (Signal-to-Interference Ratio)
  - MAE   (Mean Absolute Error)
"""

import numpy as np


def rmse(estimated: np.ndarray, true: np.ndarray) -> float:
    """
    Root Mean Square Error.

    Parameters
    ----------
    estimated : np.ndarray
        Valeurs estimées.
    true : np.ndarray
        Valeurs de référence (ground truth).

    Returns
    -------
    float
        RMSE.
    """
    return np.sqrt(np.mean(np.abs(estimated - true) ** 2))


def mae(estimated: np.ndarray, true: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(estimated - true))


def sir(signal: np.ndarray, interference: np.ndarray) -> float:
    """
    Signal-to-Interference Ratio en dB.

    Parameters
    ----------
    signal : np.ndarray
        Signal désiré.
    interference : np.ndarray
        Signal interférent.

    Returns
    -------
    float
        SIR en dB.
    """
    p_signal = np.mean(np.abs(signal) ** 2)
    p_interference = np.mean(np.abs(interference) ** 2)

    if p_interference == 0:
        return np.inf

    return 10 * np.log10(p_signal / p_interference)


def snr_measured(signal: np.ndarray, noise: np.ndarray) -> float:
    """SNR mesuré en dB à partir du signal propre et du bruit."""
    p_sig = np.mean(np.abs(signal) ** 2)
    p_noise = np.mean(np.abs(noise) ** 2)
    if p_noise == 0:
        return np.inf
    return 10 * np.log10(p_sig / p_noise)
