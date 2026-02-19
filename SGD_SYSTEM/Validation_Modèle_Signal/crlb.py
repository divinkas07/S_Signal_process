"""
Ir divinkas — SYNAPTIC Lab
SGD System — Cramér-Rao Lower Bound (CRLB)

Calcule la borne inférieure de Cramér-Rao pour :
  - Estimation fréquentielle d'un signal multi-ton
  - Estimation de l'angle d'arrivée (AOA) sur ULA

Références :
  - Kay, S.M., "Fundamentals of Statistical Signal Processing: Estimation Theory"
  - Stoica & Moses, "Spectral Analysis of Signals"
"""

import numpy as np


# ──────────────────────────────────────────────
# CRLB — Estimation Fréquentielle
# ──────────────────────────────────────────────

def crlb_frequency(
    fs: float,
    N: int,
    snr_db: float,
    amplitude: float = 1.0,
) -> float:
    """
    CRLB pour l'estimation de la fréquence d'un signal mono-ton complexe.

    Pour un signal A·exp(j·2π·f·n/fs + φ) dans du bruit CN(0, σ²) :
        CRLB(f) = 12·fs² / (4π²·A²·N·(N²-1)·SNR_lin)

    On retourne ici sqrt(CRLB) = la borne sur le RMSE en Hz.

    Parameters
    ----------
    fs : float
        Fréquence d'échantillonnage.
    N : int
        Nombre d'échantillons.
    snr_db : float
        SNR en dB.
    amplitude : float
        Amplitude du signal.

    Returns
    -------
    float
        sqrt(CRLB) en Hz — borne inférieure sur le RMSE fréquentiel.
    """
    snr_lin = 10 ** (snr_db / 10)
    # Variance de la fréquence normalisée
    # CRLB(f_norm) = 12 / (4π² · SNR · N · (N²-1))
    # Conversion en Hz : f = f_norm · fs
    crlb_var = (12 * fs**2) / (4 * np.pi**2 * amplitude**2 * N * (N**2 - 1) * snr_lin)
    return np.sqrt(crlb_var)


# ──────────────────────────────────────────────
# CRLB — Estimation AOA
# ──────────────────────────────────────────────

def crlb_aoa(
    n_elements: int,
    d_lambda: float,
    snr_db: float,
    aoa_deg: float,
    n_snapshots: int = 1,
) -> float:
    """
    CRLB pour l'estimation de l'angle d'arrivée sur un ULA.

    Pour un ULA à M éléments, espacement d :
        CRLB(θ) = 6 / (π²·cos²(θ)·(2π·d/λ)²·M·(M²-1)·SNR·L)

    On retourne sqrt(CRLB) en degrés.

    Parameters
    ----------
    n_elements : int
        Nombre d'antennes M.
    d_lambda : float
        Espacement inter-antennes (fraction de λ).
    snr_db : float
        SNR en dB.
    aoa_deg : float
        Angle d'arrivée vrai (degrés).
    n_snapshots : int
        Nombre de snapshots (observations indépendantes).

    Returns
    -------
    float
        sqrt(CRLB) en degrés — borne sur le RMSE angulaire.
    """
    snr_lin = 10 ** (snr_db / 10)
    aoa_rad = np.deg2rad(aoa_deg)
    M = n_elements

    # Facteur de réseau
    beta_sq = (2 * np.pi * d_lambda) ** 2

    crlb_var_rad = 6 / (beta_sq * np.cos(aoa_rad)**2 * M * (M**2 - 1) * snr_lin * n_snapshots)
    crlb_std_deg = np.rad2deg(np.sqrt(crlb_var_rad))

    return crlb_std_deg


# ──────────────────────────────────────────────
# Calcul CRLB depuis la config
# ──────────────────────────────────────────────

def compute_crlb_from_config(config: dict) -> dict:
    """
    Calcule toutes les CRLB à partir de la configuration.

    Returns
    -------
    dict
        {'freq_rmse_hz': ..., 'aoa_rmse_deg': ...}
    """
    sig = config["signal"]
    ch = config["channel"]
    arr = config["array"]

    N = int(sig["fs"] * sig["duration"])

    # CRLB fréquentielle (pour chaque ton)
    freq_crlbs = []
    for amp in sig["amplitudes"]:
        crlb_f = crlb_frequency(sig["fs"], N, ch["snr_db"], amp)
        freq_crlbs.append(crlb_f)

    # CRLB angulaire
    crlb_a = crlb_aoa(
        arr["n_elements"], arr["d_lambda"],
        ch["snr_db"], arr["aoa_deg"], n_snapshots=N,
    )

    return {
        "freq_crlb_hz": freq_crlbs,
        "aoa_crlb_deg": crlb_a,
    }


# ──────────────────────────────────────────────
# Test rapide
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from signal_model import load_config

    cfg = load_config()
    crlbs = compute_crlb_from_config(cfg)

    print("═" * 50)
    print("  CRAMÉR-RAO LOWER BOUNDS")
    print("═" * 50)
    for i, crlb_f in enumerate(crlbs["freq_crlb_hz"]):
        print(f"  Ton {i+1} — CRLB fréquence : {crlb_f:.6f} Hz")
    print(f"  AOA  — CRLB angle     : {crlbs['aoa_crlb_deg']:.6f} degrés")
    print(f"  SNR  : {cfg['channel']['snr_db']} dB")
    print("═" * 50)
