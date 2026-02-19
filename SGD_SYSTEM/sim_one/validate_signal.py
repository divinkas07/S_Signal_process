"""
Ir divinkas — SYNAPTIC Lab
SGD System — Validation du Modèle de Signal

Script exécutant les 5 tests de validation définis dans le plan :
  1. Fidélité spectrale      (Δf < 1%)
  2. Précision SNR            (ΔSNR < 0.5 dB)
  3. Précision Doppler        (Δf_d < 1%)
  4. Gaussianité du bruit     (Shapiro p-value > 0.05)
  5. Stabilité numérique      (1000 runs, 0 crash)

Usage :
    python validate_signal.py
"""

import numpy as np
from scipy import stats
import sys
import traceback

from signal_model import load_config, generate_time_vector, generate_multitone
from channel_model import apply_channel, signal_power


# ══════════════════════════════════════════════
# Utilitaires
# ══════════════════════════════════════════════

def detect_spectral_peaks(s: np.ndarray, fs: float, n_peaks: int) -> np.ndarray:
    """
    Détecte les n_peaks fréquences dominantes dans le spectre du signal.

    Returns
    -------
    peak_freqs : np.ndarray, shape (n_peaks,)
        Fréquences des pics triées par ordre croissant.
    """
    N = len(s)
    S = np.abs(np.fft.fft(s))[:N // 2]
    freqs = np.fft.fftfreq(N, d=1 / fs)[:N // 2]

    # Indices des n_peaks plus grands pics
    peak_indices = np.argsort(S)[-n_peaks:]
    peak_freqs = np.sort(freqs[peak_indices])
    return peak_freqs


def print_result(test_name: str, passed: bool, details: str):
    """Affiche le résultat formaté d'un test."""
    icon = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{'═' * 60}")
    print(f"  {icon}  │  {test_name}")
    print(f"{'─' * 60}")
    print(f"  {details}")
    print(f"{'═' * 60}")


# ══════════════════════════════════════════════
# TEST 1 : Fidélité Spectrale
# ══════════════════════════════════════════════

def test_spectral_fidelity(config: dict) -> bool:
    """
    Vérifie que les fréquences détectées correspondent aux
    fréquences injectées (f_k + f_d), avec Δf < 1%.
    """
    sig = config["signal"]
    dop = config["doppler"]
    THRESHOLD_PCT = 1.0

    t = generate_time_vector(sig["fs"], sig["duration"])
    s = generate_multitone(
        t, sig["frequencies"], sig["amplitudes"],
        phases=[0, 0, 0],  # phases fixes pour reproductibilité
        fd=dop["fd"],
    )

    expected_freqs = np.array(sig["frequencies"]) + dop["fd"]
    detected_freqs = detect_spectral_peaks(s, sig["fs"], len(sig["frequencies"]))

    errors_pct = np.abs(detected_freqs - expected_freqs) / expected_freqs * 100
    max_error = np.max(errors_pct)
    passed = max_error < THRESHOLD_PCT

    details_lines = []
    for i, (exp, det, err) in enumerate(zip(expected_freqs, detected_freqs, errors_pct)):
        details_lines.append(f"    Ton {i+1}: attendu={exp:.1f} Hz, mesuré={det:.1f} Hz, erreur={err:.4f}%")
    details = "\n".join(details_lines) + f"\n    Max erreur: {max_error:.4f}% (seuil: {THRESHOLD_PCT}%)"

    print_result("Test 1 — Fidélité Spectrale", passed, details)
    return passed


# ══════════════════════════════════════════════
# TEST 2 : Précision SNR
# ══════════════════════════════════════════════

def test_snr_accuracy(config: dict) -> bool:
    """
    Vérifie que le SNR mesuré est cohérent avec le SNR cible (ΔSNR < 0.5 dB).
    Moyenne sur 100 réalisations pour réduire la variance.
    """
    THRESHOLD_DB = 0.5
    N_TRIALS = 100
    snr_target = config["channel"]["snr_db"]

    sig = config["signal"]
    dop = config["doppler"]
    t = generate_time_vector(sig["fs"], sig["duration"])

    snr_measured_list = []
    for _ in range(N_TRIALS):
        s = generate_multitone(t, sig["frequencies"], sig["amplitudes"], fd=dop["fd"])
        received, noise = apply_channel(s, config)
        p_sig = signal_power(s)
        p_noise = signal_power(noise)
        snr_meas = 10 * np.log10(p_sig / p_noise)
        snr_measured_list.append(snr_meas)

    snr_mean = np.mean(snr_measured_list)
    snr_std = np.std(snr_measured_list)
    delta_snr = abs(snr_mean - snr_target)
    passed = delta_snr < THRESHOLD_DB

    details = (
        f"    SNR cible   : {snr_target:.1f} dB\n"
        f"    SNR mesuré  : {snr_mean:.2f} ± {snr_std:.2f} dB (sur {N_TRIALS} runs)\n"
        f"    ΔSNR        : {delta_snr:.3f} dB (seuil: {THRESHOLD_DB} dB)"
    )
    print_result("Test 2 — Précision SNR", passed, details)
    return passed


# ══════════════════════════════════════════════
# TEST 3 : Précision Doppler
# ══════════════════════════════════════════════

def test_doppler_accuracy(config: dict) -> bool:
    """
    Vérifie que le décalage Doppler est correctement appliqué (Δf_d < 1%).
    On génère un signal avec et sans Doppler et mesure le décalage spectral.
    """
    THRESHOLD_PCT = 1.0
    sig = config["signal"]
    fd_target = config["doppler"]["fd"]

    t = generate_time_vector(sig["fs"], sig["duration"])

    # Signal sans Doppler
    s_no_doppler = generate_multitone(
        t, sig["frequencies"], sig["amplitudes"],
        phases=[0, 0, 0], fd=0.0,
    )
    # Signal avec Doppler
    s_with_doppler = generate_multitone(
        t, sig["frequencies"], sig["amplitudes"],
        phases=[0, 0, 0], fd=fd_target,
    )

    freqs_no_d = detect_spectral_peaks(s_no_doppler, sig["fs"], len(sig["frequencies"]))
    freqs_with_d = detect_spectral_peaks(s_with_doppler, sig["fs"], len(sig["frequencies"]))

    measured_shifts = freqs_with_d - freqs_no_d
    errors_pct = np.abs(measured_shifts - fd_target) / fd_target * 100
    max_error = np.max(errors_pct)
    passed = max_error < THRESHOLD_PCT

    details_lines = []
    for i, (shift, err) in enumerate(zip(measured_shifts, errors_pct)):
        details_lines.append(f"    Ton {i+1}: décalage mesuré={shift:.1f} Hz, erreur={err:.4f}%")
    details = (
        "\n".join(details_lines) +
        f"\n    fd cible: {fd_target} Hz, max erreur: {max_error:.4f}% (seuil: {THRESHOLD_PCT}%)"
    )
    print_result("Test 3 — Précision Doppler", passed, details)
    return passed


# ══════════════════════════════════════════════
# TEST 4 : Gaussianité du Bruit
# ══════════════════════════════════════════════

def test_noise_gaussianity(config: dict) -> bool:
    """
    Vérifie que le bruit ajouté suit une distribution gaussienne.
    Test de Shapiro-Wilk sur les parties réelle et imaginaire.
    """
    P_VALUE_THRESHOLD = 0.05
    sig = config["signal"]
    dop = config["doppler"]

    t = generate_time_vector(sig["fs"], sig["duration"])
    s = generate_multitone(t, sig["frequencies"], sig["amplitudes"], fd=dop["fd"])
    _, noise = apply_channel(s, config)

    # Shapiro-Wilk (max 5000 échantillons)
    n_samples = min(len(noise), 5000)
    noise_sample = noise[:n_samples]

    _, p_real = stats.shapiro(np.real(noise_sample))
    _, p_imag = stats.shapiro(np.imag(noise_sample))

    # Statistiques descriptives
    mean_real = np.mean(np.real(noise))
    mean_imag = np.mean(np.imag(noise))
    std_real = np.std(np.real(noise))
    std_imag = np.std(np.imag(noise))

    passed = p_real > P_VALUE_THRESHOLD and p_imag > P_VALUE_THRESHOLD

    details = (
        f"    Partie réelle  : μ={mean_real:.6f}, σ={std_real:.4f}, p-value={p_real:.4f}\n"
        f"    Partie imag.   : μ={mean_imag:.6f}, σ={std_imag:.4f}, p-value={p_imag:.4f}\n"
        f"    Seuil p-value  : {P_VALUE_THRESHOLD}"
    )
    print_result("Test 4 — Gaussianité du Bruit", passed, details)
    return passed


# ══════════════════════════════════════════════
# TEST 5 : Stabilité Numérique
# ══════════════════════════════════════════════

def test_numerical_stability(config: dict) -> bool:
    """
    Exécute 1000 runs et vérifie qu'aucun ne produit NaN, Inf ou exception.
    Teste aussi à des SNR extrêmes (-20 dB, 60 dB).
    """
    N_RUNS = 1000
    sig = config["signal"]
    dop = config["doppler"]

    crashes = 0
    nan_count = 0
    inf_count = 0

    snr_values = [-20, -10, 0, 10, 20, 40, 60]

    for run in range(N_RUNS):
        try:
            t = generate_time_vector(sig["fs"], sig["duration"])
            s = generate_multitone(t, sig["frequencies"], sig["amplitudes"], fd=dop["fd"])

            # Varier le SNR
            test_config = config.copy()
            test_config["channel"] = config["channel"].copy()
            test_config["channel"]["snr_db"] = snr_values[run % len(snr_values)]

            received, noise = apply_channel(s, test_config)

            if np.any(np.isnan(received)):
                nan_count += 1
            if np.any(np.isinf(received)):
                inf_count += 1

        except Exception:
            crashes += 1
            if crashes <= 3:
                traceback.print_exc()

    total_issues = crashes + nan_count + inf_count
    passed = total_issues == 0

    details = (
        f"    Runs totaux   : {N_RUNS}\n"
        f"    Crashes       : {crashes}\n"
        f"    NaN détectés  : {nan_count}\n"
        f"    Inf détectés  : {inf_count}\n"
        f"    SNR testés    : {snr_values}\n"
        f"    Résultat      : {'Aucune instabilité' if passed else f'{total_issues} problèmes détectés'}"
    )
    print_result("Test 5 — Stabilité Numérique", passed, details)
    return passed


# ══════════════════════════════════════════════
# MAIN — Exécution Séquentielle
# ══════════════════════════════════════════════

def main():
    print("\n" + "█" * 60)
    print("  SGD SYSTEM — VALIDATION DU MODÈLE DE SIGNAL")
    print("  SYNAPTIC Lab — Ir divinkas")
    print("█" * 60)

    config = load_config()

    results = {}
    results["spectral_fidelity"] = test_spectral_fidelity(config)
    results["snr_accuracy"] = test_snr_accuracy(config)
    results["doppler_accuracy"] = test_doppler_accuracy(config)
    results["noise_gaussianity"] = test_noise_gaussianity(config)
    results["numerical_stability"] = test_numerical_stability(config)

    # ── Bilan ──
    all_pass = all(results.values())
    n_pass = sum(results.values())
    n_total = len(results)

    print("\n" + "█" * 60)
    print(f"  BILAN : {n_pass}/{n_total} tests réussis")
    print("█" * 60)

    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {name}")

    if all_pass:
        print("\n  🚦 GO — Tous les critères sont satisfaits.")
        print("  ➡️  Le modèle de signal est validé pour le Monte Carlo.")
    else:
        print("\n  🚦 NO-GO — Certains critères ne sont pas satisfaits.")
        print("  ⚠️  Corriger les problèmes avant de lancer la simulation massive.")

    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
