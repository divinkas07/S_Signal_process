"""
Ir divinkas — SYNAPTIC Lab
SGD System — Modèle de Canal

Applique les effets de canal au signal :
  - AWGN (bruit gaussien blanc additif complexe)
  - Multipath (retards + gains complexes)

Formulation :
    r(t) = Σ_i  α_i · s(t - τ_i) + n(t)
    n(t) ~ CN(0, σ²)
    σ² = P_signal / 10^(SNR_dB / 10)
"""

import numpy as np


# ──────────────────────────────────────────────
# Puissance du signal
# ──────────────────────────────────────────────

def signal_power(s: np.ndarray) -> float:
    """Calcule la puissance moyenne d'un signal complexe."""
    return np.mean(np.abs(s) ** 2)


# ──────────────────────────────────────────────
# Bruit AWGN complexe
# ──────────────────────────────────────────────

def generate_awgn(shape: tuple, snr_db: float, p_signal: float) -> np.ndarray:
    """
    Génère un bruit gaussien blanc complexe calibré sur le SNR cible.

    Parameters
    ----------
    shape : tuple
        Forme du tableau de bruit (identique au signal).
    snr_db : float
        SNR cible en dB.
    p_signal : float
        Puissance du signal de référence.

    Returns
    -------
    noise : np.ndarray
        Bruit complexe CN(0, σ²).
    """
    snr_lin = 10 ** (snr_db / 10)
    sigma2 = p_signal / snr_lin
    # Bruit circulaire complexe : Re ~ N(0, σ²/2), Im ~ N(0, σ²/2)
    sigma = np.sqrt(sigma2 / 2)
    noise = sigma * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
    return noise


# ──────────────────────────────────────────────
# Multipath
# ──────────────────────────────────────────────

def apply_multipath(
    s: np.ndarray,
    fs: float,
    paths: list[dict],
) -> np.ndarray:
    """
    Applique un modèle multipath au signal.

    Chaque trajet est défini par un gain complexe et un retard.

    Parameters
    ----------
    s : np.ndarray, shape (N,) or (M, N)
        Signal d'entrée.
    fs : float
        Fréquence d'échantillonnage.
    paths : list[dict]
        Liste de chemins avec clés 'gain' et 'delay_s'.

    Returns
    -------
    r : np.ndarray
        Signal avec multipath appliqué (même forme que s).
    """
    r = np.copy(s).astype(complex)

    for path in paths:
        gain = path["gain"]
        delay_samples = int(round(path["delay_s"] * fs))

        if delay_samples == 0:
            r += gain * s
        else:
            if s.ndim == 1:
                delayed = np.zeros_like(s)
                delayed[delay_samples:] = s[:-delay_samples]
                r += gain * delayed
            else:
                # Cas multi-antenne (M, N)
                delayed = np.zeros_like(s)
                delayed[:, delay_samples:] = s[:, :-delay_samples]
                r += gain * delayed

    return r


# ──────────────────────────────────────────────
# Application complète du canal
# ──────────────────────────────────────────────

def apply_channel(signal: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique le modèle de canal complet au signal.

    Parameters
    ----------
    signal : np.ndarray
        Signal d'entrée, shape (N,) ou (M, N).
    config : dict
        Configuration complète (contient 'channel' et 'signal').

    Returns
    -------
    received : np.ndarray
        Signal reçu après canal.
    noise : np.ndarray
        Bruit ajouté (pour analyse séparée).
    """
    ch = config["channel"]
    sig = config["signal"]
    snr_db = ch["snr_db"]

    # ── Multipath (optionnel) ──
    if ch["multipath"]["enabled"]:
        signal_mp = apply_multipath(signal, sig["fs"], ch["multipath"]["paths"])
    else:
        signal_mp = signal.copy()

    # ── AWGN ──
    p_sig = signal_power(signal_mp)
    noise = generate_awgn(signal_mp.shape, snr_db, p_sig)
    received = signal_mp + noise

    return received, noise


# ──────────────────────────────────────────────
# Point d'entrée pour test rapide
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from signal_model import load_config, generate_signal_from_config

    cfg = load_config()
    t, s, X = generate_signal_from_config(cfg)

    # Appliquer le canal sur le signal mono
    received, noise = apply_channel(s, cfg)

    # Appliquer le canal sur le signal multi-antenne
    received_array, noise_array = apply_channel(X, cfg)

    print(f"Signal mono    — P_signal: {signal_power(s):.4f}")
    print(f"Bruit mono     — P_noise : {signal_power(noise):.4f}")
    print(f"SNR mesuré     : {10 * np.log10(signal_power(s) / signal_power(noise)):.2f} dB")
    print(f"SNR cible      : {cfg['channel']['snr_db']} dB")
    print(f"Array received : {received_array.shape}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(t * 1000, np.real(s), linewidth=0.5, label="Signal")
    axes[0].set_title("Signal original")
    axes[0].set_xlabel("Temps (ms)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t * 1000, np.real(noise), linewidth=0.5, color="red", alpha=0.5)
    axes[1].set_title(f"Bruit AWGN (SNR = {cfg['channel']['snr_db']} dB)")
    axes[1].set_xlabel("Temps (ms)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t * 1000, np.real(received), linewidth=0.5, color="green")
    axes[2].set_title("Signal reçu (signal + bruit)")
    axes[2].set_xlabel("Temps (ms)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("channel_preview.png", dpi=150)
    plt.show()
    print("✅ Canal appliqué avec succès")
