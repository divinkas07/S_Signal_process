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
    total_power: float = 1.0,
    doppler_rate_hz_per_sec: float = 0.0,
) -> float:
    """
    CRLB MSE pour estimation frequence sous Doppler LEO lineaire.
    Modele misspecifie (estimateur ton pur sur signal chirp).

    Parameters
    ----------
    fs : float
        Frequence d'echantillonnage.
    N : int
        Nombre d'echantillons.
    snr_db : float
        SNR TOTAL en dB (P_totale / sigma^2).
    amplitude : float
        Amplitude A du ton cible.
    total_power : float
        Puissance totale du signal multi-ton (Sigma A_k^2).
    doppler_rate_hz_per_sec : float
        Derive Doppler (Hz/s) -> alpha.

    Returns
    -------
    float
        RMSE bound en Hz : sqrt(var_thermal + bias^2).
    """
    snr_total_lin = 10 ** (snr_db / 10)
    # SNR_ton = SNR_total * (A^2 / P_tot)
    snr_ton_lin = snr_total_lin * (amplitude**2 / total_power)
    
    # 1. Variance thermique (modele Kay - ton pur)
    var_thermal = (6 * fs**2) / (4 * np.pi**2 * snr_ton_lin * N * (N**2 - 1))
    
    # 2. Biais physique LEO (moyenne frequentielle approximee)
    T_obs = N / fs
    bias = 0.5 * doppler_rate_hz_per_sec * T_obs
    var_bias = bias**2
    
    # 3. MSE totale (physique)
    rmse_bound = np.sqrt(var_thermal + var_bias)
    return rmse_bound


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
    CRLB (borne deterministe) pour l'estimation de l'angle d'arrivee sur un ULA.

    Parameters
    ----------
    n_elements : int
        Nombre d'antennes M.
    d_lambda : float
        Espacement inter-antenne (d/lambda).
    snr_db : float
        SNR effectif par antenne.
    aoa_deg : float
        Angle d'incidence en degres.
    n_snapshots : int
        Nombre d'echantillons temporels.

    Returns
    -------
    float
        sqrt(CRLB) en degres.
    """
    snr_lin = 10 ** (snr_db / 10)
    aoa_rad = np.deg2rad(aoa_deg)
    M = n_elements
    beta_sq = (2 * np.pi * d_lambda) ** 2
    
    # CRLB Deterministe (Stoica & Moses): 6 / (SNR * L * beta^2 * cos^2(theta) * M(M^2-1))
    # Attention : beta ici est 2*pi*d/lambda. cos(theta) est separe.
    crlb_var_rad = 6 / (beta_sq * np.cos(aoa_rad)**2 * M * (M**2 - 1) * snr_lin * n_snapshots)
    
    return np.rad2deg(np.sqrt(crlb_var_rad))


# ──────────────────────────────────────────────
# Calcul CRLB depuis la config
# ──────────────────────────────────────────────

def compute_crlb_from_config(config: dict) -> dict:
    """
    Calcule toutes les CRLB a partir de la configuration.
    """
    sig = config["signal"]
    ch = config["channel"]
    arr = config["array"]
    dop = config.get("doppler", {})

    N = int(sig["fs"] * sig["duration"])

    # Doppler rate pour le biais LEO
    # On le recupere soit directement soit par defaut (ex: config LEO active)
    alpha = dop.get("fd_rate", 0.0)
    if alpha == 0 and config.get("leo_scenario", {}).get("enabled"):
        # Importation locale pour eviter dependance circulaire
        from signal_model import get_leo_doppler_at_time
        _, alpha = get_leo_doppler_at_time(0, config["leo_scenario"])

    # Puissance totale theoretical
    total_power = np.sum(np.array(sig["amplitudes"])**2)

    # CRLB frequentielle
    freq_crlbs = []
    for amp in sig["amplitudes"]:
        rmse_f = crlb_frequency(sig["fs"], N, ch["snr_db"], amp, total_power, doppler_rate_hz_per_sec=alpha)
        freq_crlbs.append(rmse_f)

    # CRLB angulaire (standard)
    rmse_a = crlb_aoa(
        arr["n_elements"], arr["d_lambda"],
        ch["snr_db"], arr["aoa_deg"], n_snapshots=N
    )

    return {
        "freq_crlb_hz": freq_crlbs,
        "aoa_crlb_deg": rmse_a,
    }


# ──────────────────────────────────────────────
# Test rapide
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from signal_model import load_config

    cfg = load_config()
    crlbs = compute_crlb_from_config(cfg)

    print("=" * 50)
    print("  CRAMER-RAO LOWER BOUNDS")
    print("=" * 50)
    for i, crlb_f in enumerate(crlbs["freq_crlb_hz"]):
        print(f"  Ton {i+1} -- CRLB frequence : {crlb_f:.6f} Hz")
    print(f"  AOA  -- CRLB angle     : {crlbs['aoa_crlb_deg']:.6f} degres")
    print(f"  SNR  : {cfg['channel']['snr_db']} dB")
    print("=" * 50)
