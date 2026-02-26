"""
Ir divinkas — SYNAPTIC Lab
SGD System — Modèle de Signal Multi-Ton

Génère un signal multi-ton complexe avec décalage Doppler et
support antenne ULA (Uniform Linear Array).

Formulation :
    s(t) = Σ_k  A_k · exp(j·(2π·(f_k + f_d)·t + φ_k))

Signal reçu sur ULA :
    x(t) = a(θ) ⊗ s(t)
    a_n(θ) = exp(j·2π·d·n·sin(θ))   pour n = 0, ..., N-1
"""

import numpy as np
import yaml
from pathlib import Path

# ──────────────────────────────────────────────
# Chargement de la configuration
# ──────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """Charge le fichier config.yaml et retourne un dict."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Génération du vecteur temps
# ──────────────────────────────────────────────

def generate_time_vector(fs: float, duration: float) -> np.ndarray:
    """Retourne un vecteur temps échantillonné à fs Hz sur `duration` secondes."""
    N = int(fs * duration)
    return np.arange(N) / fs


# ──────────────────────────────────────────────
# Signal multi-ton avec Doppler
# ──────────────────────────────────────────────

def generate_multitone(
    t: np.ndarray,
    frequencies: list[float],
    amplitudes: list[float],
    phases: list[float] | None = None,
    fd: float = 0.0,
    fd_rate: float = 0.0,
) -> np.ndarray:
    """
    Génère un signal multi-ton complexe.

    Parameters
    ----------
    t : np.ndarray
        Vecteur temps (s).
    frequencies : list[float]
        Fréquences des K tons (Hz).
    amplitudes : list[float]
        Amplitudes linéaires des K tons.
    phases : list[float] or None
        Phases initiales (rad). Si None → phases aléatoires uniformes [0, 2π).
    fd : float
        Décalage Doppler commun (Hz).

    Returns
    -------
    s : np.ndarray, shape (N,)
        Signal multi-ton complexe avec décalage Doppler f(t) = fd + fd_rate * t
    """
    K = len(frequencies)
    assert len(amplitudes) == K, "Nombre d'amplitudes ≠ nombre de fréquences"

    if phases is None:
        phases = np.random.uniform(0, 2 * np.pi, size=K)
    else:
        phases = np.asarray(phases, dtype=float)
        assert len(phases) == K, "Nombre de phases ≠ nombre de fréquences"

    s = np.zeros(len(t), dtype=complex)
    for k in range(K):
        # f(t) = frequencies[k] + fd + fd_rate * t
        # Phase = 2π * ∫ f(τ) dτ = 2π * ( (frequencies[k] + fd) * t + 0.5 * fd_rate * t^2 )
        phase_t = 2 * np.pi * ((frequencies[k] + fd) * t + 0.5 * fd_rate * t**2) + phases[k]
        s += amplitudes[k] * np.exp(1j * phase_t)

    return s


# ──────────────────────────────────────────────
# Steering vector ULA
# ──────────────────────────────────────────────

def steering_vector(n_elements: int, d_lambda: float, aoa_deg: float) -> np.ndarray:
    """
    Calcule le steering vector d'un ULA.

    Parameters
    ----------
    n_elements : int
        Nombre d'antennes.
    d_lambda : float
        Espacement inter-antennes en fraction de λ.
    aoa_deg : float
        Angle d'arrivée (degrés).

    Returns
    -------
    a : np.ndarray, shape (n_elements, 1)
        Steering vector colonne.
    """
    aoa_rad = np.deg2rad(aoa_deg)
    n = np.arange(n_elements)
    a = np.exp(1j * 2 * np.pi * d_lambda * n * np.sin(aoa_rad))
    return a.reshape(-1, 1)


# ──────────────────────────────────────────────
# Signal reçu sur le réseau d'antennes
# ──────────────────────────────────────────────

def generate_array_signal(
    s: np.ndarray,
    n_elements: int,
    d_lambda: float,
    aoa_deg: float,
) -> np.ndarray:
    """
    Projette le signal sur un ULA via le steering vector.

    Parameters
    ----------
    s : np.ndarray, shape (N,)
        Signal temporel mono-canal.
    n_elements : int
        Nombre d'antennes.
    d_lambda : float
        Espacement inter-antennes (fraction de λ).
    aoa_deg : float
        Angle d'arrivée (degrés).

    Returns
    -------
    X : np.ndarray, shape (n_elements, N)
        Signal reçu sur chaque antenne.
    """
    a = steering_vector(n_elements, d_lambda, aoa_deg)
    s_row = s.reshape(1, -1)
    return a @ s_row


def get_leo_doppler_at_time(t: float, leo_cfg: dict) -> tuple[float, float]:
    """
    Calcule fd et fd_rate à un instant t à partir des paramètres LEO.
    """
    v = leo_cfg["v_km_s"] * 1000
    h = leo_cfg["altitude_km"]
    fc = leo_cfg["f_carrier_hz"]
    c = 3e8
    R_earth = 6371000 # m
    r = R_earth + h * 1000 if h < 10000 else h # Gestion km vs m
    mu = 3.986e14
    omega = np.sqrt(mu / (r**3))
    
    t_mid = leo_cfg.get("duration_s", 600) / 2
    theta = omega * (t - t_mid)
    
    # fd = (v/c) * fc * sin(theta)
    fd = (v / c) * fc * np.sin(theta)
    
    # fd_rate = (v/c) * fc * omega * cos(theta)
    fd_rate = (v / c) * fc * omega * np.cos(theta)
    
    return fd, fd_rate


# ──────────────────────────────────────────────
# Wrapper haut niveau depuis config
# ──────────────────────────────────────────────

def generate_signal_from_config(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Génère le signal complet à partir de la configuration.
    Si leo_scenario est activé, surcharge fd et fd_rate.
    
    Returns
    -------
    t : np.ndarray — vecteur temps
    s : np.ndarray — signal mono-canal (N,)
    X : np.ndarray — signal sur ULA (n_elements, N)
    fd : float — Doppler effectif utilisé
    fd_rate : float — Dérive Doppler effective utilisée
    """
    sig = config["signal"]
    dop = config["doppler"]
    arr = config["array"]
    leo = config.get("leo_scenario", {"enabled": False})

    t = generate_time_vector(sig["fs"], sig["duration"])

    fd = dop["fd"]
    fd_rate = dop.get("fd_rate", 0.0) if dop.get("dynamic", False) else 0.0

    if leo.get("enabled", False):
        # On calcule le Doppler à t=0 pour le bloc de signal
        # (Ou on pourrait passer t_start dans le config si on voulait simuler un autre moment du pass)
        fd, fd_rate = get_leo_doppler_at_time(0, leo)

    phases = sig.get("phases", None)

    s = generate_multitone(
        t=t,
        frequencies=sig["frequencies"],
        amplitudes=sig["amplitudes"],
        phases=phases,
        fd=fd,
        fd_rate=fd_rate,
    )

    X = generate_array_signal(
        s=s,
        n_elements=arr["n_elements"],
        d_lambda=arr["d_lambda"],
        aoa_deg=arr["aoa_deg"],
    )

    return t, s, X, fd, fd_rate


# ──────────────────────────────────────────────
# Point d'entrée pour test rapide
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cfg = load_config()
    t, s, X = generate_signal_from_config(cfg)

    print(f"Signal shape  : {s.shape}")
    print(f"Array shape   : {X.shape}")
    print(f"Duration      : {t[-1]:.4f} s")
    print(f"Samples       : {len(t)}")

    # Spectre du signal mono-canal
    N = len(s)
    freqs = np.fft.fftfreq(N, d=1 / cfg["signal"]["fs"])
    S = np.fft.fft(s) / N

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Amplitude temporelle
    axes[0].plot(t * 1000, np.real(s), linewidth=0.5)
    axes[0].set_xlabel("Temps (ms)")
    axes[0].set_ylabel("Amplitude (Re)")
    axes[0].set_title("Signal multi-ton — domaine temporel")
    axes[0].grid(True, alpha=0.3)

    # Spectre
    axes[1].stem(freqs[:N // 2], np.abs(S[:N // 2]) * 2, markerfmt=".", basefmt=" ")
    axes[1].set_xlabel("Fréquence (Hz)")
    axes[1].set_ylabel("|S(f)|")
    axes[1].set_title("Spectre du signal multi-ton")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("signal_preview.png", dpi=150)
    plt.show()
    print("✅ Signal généré avec succès")
