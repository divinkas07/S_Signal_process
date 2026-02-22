"""
Ir divinkas — SYNAPTIC Lab
SGD System — Validation du Modèle de Signal

Refactored class-based implementation for signal model validation.
"""

import numpy as np
from scipy import stats
import sys
import traceback

from signal_model import load_config, generate_time_vector, generate_multitone
from channel_model import apply_channel, signal_power
from validation_report import get_report
from metrics import detect_spectral_peaks, compute_spectral_error, compute_snr_error, compute_doppler_error, compute_doppler_rate_error

class SignalValidator:
    """
    Encapsule les tests de validation du modèle de signal.
    """
    def __init__(self, config: dict):
        self.config = config
        self.report = get_report("SGD SYSTEM — VALIDATION DU MODÈLE DE SIGNAL")

    # Peak detection logic moved to metrics.py

    def test_spectral_fidelity(self) -> bool:
        """Test 1: Fidélité Spectrale (Δf < 1%)"""
        sig = self.config["signal"]
        dop = self.config["doppler"]
        THRESHOLD_PCT = 1.0

        t = generate_time_vector(sig["fs"], sig["duration"])
        s = generate_multitone(
            t, sig["frequencies"], sig["amplitudes"],
            phases=[0] * len(sig["frequencies"]),
            fd=dop["fd"],
        )

        expected_freqs = np.array(sig["frequencies"]) + dop["fd"]
        detected_freqs = detect_spectral_peaks(s, sig["fs"], len(sig["frequencies"]))

        errors_pct = compute_spectral_error(expected_freqs, detected_freqs)
        max_error = np.max(errors_pct)
        passed = max_error < THRESHOLD_PCT

        details = "\n".join([
            f"Ton {i+1}: attendu={exp:.1f} Hz, mesure={det:.1f} Hz, erreur={err:.4f}%"
            for i, (exp, det, err) in enumerate(zip(expected_freqs, detected_freqs, errors_pct))
        ]) + f"\nMax erreur: {max_error:.4f}% (seuil: {THRESHOLD_PCT}%)"

        self.report.add_result("Fidélité Spectrale", passed, details)
        return passed

    def test_snr_accuracy(self) -> bool:
        """Test 2: Précision SNR (ΔSNR < 0.5 dB)"""
        THRESHOLD_DB = 0.5
        N_TRIALS = 100
        snr_target = self.config["channel"]["snr_db"]
        sig = self.config["signal"]
        dop = self.config["doppler"]
        t = generate_time_vector(sig["fs"], sig["duration"])

        snr_measured_list = []
        for _ in range(N_TRIALS):
            s = generate_multitone(t, sig["frequencies"], sig["amplitudes"], fd=dop["fd"])
            received, noise = apply_channel(s, self.config)
            p_sig = signal_power(s)
            p_noise = signal_power(noise)
            snr_meas = 10 * np.log10(p_sig / p_noise)
            snr_measured_list.append(snr_meas)

        snr_mean = np.mean(snr_measured_list)
        snr_std = np.std(snr_measured_list)
        delta_snr = compute_snr_error(snr_target, snr_mean)
        passed = delta_snr < THRESHOLD_DB

        details = (
            f"SNR cible   : {snr_target:.1f} dB\n"
            f"SNR mesure  : {snr_mean:.2f} +/- {snr_std:.2f} dB (sur {N_TRIALS} runs)\n"
            f"Delta SNR   : {delta_snr:.3f} dB (seuil: {THRESHOLD_DB} dB)"
        )
        self.report.add_result("Précision SNR", passed, details)
        return passed

    def test_doppler_accuracy(self) -> bool:
        """Test 3: Précision Doppler (Δf_d < 1%)"""
        THRESHOLD_PCT = 1.0
        sig = self.config["signal"]
        fd_target = self.config["doppler"]["fd"]
        t = generate_time_vector(sig["fs"], sig["duration"])

        # Sans Doppler
        s0 = generate_multitone(t, sig["frequencies"], sig["amplitudes"], phases=[0]*3, fd=0.0)
        # Avec Doppler
        s1 = generate_multitone(t, sig["frequencies"], sig["amplitudes"], phases=[0]*3, fd=fd_target)

        f0 = detect_spectral_peaks(s0, sig["fs"], len(sig["frequencies"]))
        f1 = detect_spectral_peaks(s1, sig["fs"], len(sig["frequencies"]))

        shifts = f1 - f0
        # On calcule l'erreur sur le premier décalage détecté (ou la moyenne)
        avg_shift = np.mean(shifts)
        max_err = compute_doppler_error(fd_target, avg_shift)
        passed = max_err < THRESHOLD_PCT

        # Calculer les erreurs individuelles pour l'affichage
        errors = np.abs(shifts - fd_target) / np.abs(fd_target) * 100

        details = "\n".join([
            f"Ton {i+1}: decalage={s:.1f} Hz, erreur={e:.4f}%"
            for i, (s, e) in enumerate(zip(shifts, errors))
        ]) + f"\nfd cible: {fd_target} Hz, moyenne erreur: {max_err:.4f}%"

        self.report.add_result("Précision Doppler", passed, details)
        return passed

    def test_noise_gaussianity(self) -> bool:
        """Test 4: Gaussianité du Bruit (p-value > 0.05)"""
        P_VALUE_THRESHOLD = 0.05
        sig = self.config["signal"]
        t = generate_time_vector(sig["fs"], sig["duration"])
        s = generate_multitone(t, sig["frequencies"], sig["amplitudes"])
        _, noise = apply_channel(s, self.config)

        # Shapiro-Wilk
        sample = noise[:min(len(noise), 5000)]
        _, p_real = stats.shapiro(np.real(sample))
        _, p_imag = stats.shapiro(np.imag(sample))

        passed = p_real > P_VALUE_THRESHOLD and p_imag > P_VALUE_THRESHOLD
        details = (
            f"Partie reelle  : mean={np.mean(np.real(noise)):.6f}, p-value={p_real:.4f}\n"
            f"Partie imag.   : mean={np.mean(np.imag(noise)):.6f}, p-value={p_imag:.4f}\n"
            f"Seuil p-value  : {P_VALUE_THRESHOLD}"
        )
        self.report.add_result("Gaussianité du Bruit", passed, details)
        return passed

    def test_numerical_stability(self) -> bool:
        """Test 5: Stabilité Numérique (1000 runs, 0 crash)"""
        N_RUNS = 1000
        sig = self.config["signal"]
        dop = self.config["doppler"]
        issues = {"crash": 0, "nan": 0, "inf": 0}
        snr_values = [-20, -10, 0, 10, 20, 40, 60]

        for run in range(N_RUNS):
            try:
                t = generate_time_vector(sig["fs"], sig["duration"])
                s = generate_multitone(t, sig["frequencies"], sig["amplitudes"], fd=dop["fd"])
                cfg_tmp = self.config.copy()
                cfg_tmp["channel"] = self.config["channel"].copy()
                cfg_tmp["channel"]["snr_db"] = snr_values[run % len(snr_values)]
                
                rec, _ = apply_channel(s, cfg_tmp)
                if np.any(np.isnan(rec)): issues["nan"] += 1
                if np.any(np.isinf(rec)): issues["inf"] += 1
            except Exception:
                issues["crash"] += 1

        passed = sum(issues.values()) == 0
        details = (
            f"Runs: {N_RUNS} | Crashes: {issues['crash']} | NaN: {issues['nan']} | Inf: {issues['inf']}\n"
            f"SNR testés: {snr_values}"
        )
        self.report.add_result("Stabilité Numérique", passed, details)
        return passed

    def test_dynamic_doppler(self) -> bool:
        """Test 6: Précision Doppler Dynamique (Δfd_rate < 10%)"""
        THRESHOLD_PCT = 10.0
        sig = self.config["signal"]
        dop = self.config["doppler"]
        
        if not dop.get("dynamic", False):
            self.report.add_result("Doppler Dynamique", True, "Test sauté (dynamic=false)")
            return True

        fd_rate_target = dop["fd_rate"]
        # On va estimer la dérive en comparant la fréquence au début et à la fin
        # Pour une meilleure résolution spectrale, on utilise une durée plus longue pour ce test spécifique
        duration_test = 1.0
        fs = sig["fs"]
        t = generate_time_vector(fs, duration_test)
        
        # Segment 1: t=0 à t=duration/4
        # Segment 2: t=3*duration/4 à t=duration
        N = len(t)
        N_seg = N // 4
        t1 = t[:N_seg]
        t2 = t[-N_seg:]
        
        s = generate_multitone(t, sig["frequencies"], sig["amplitudes"], phases=[0]*len(sig["frequencies"]), fd=dop["fd"], fd_rate=fd_rate_target)
        s1 = s[:N_seg]
        s2 = s[-N_seg:]
        
        f1_peaks = detect_spectral_peaks(s1, fs, 1) # On regarde le premier ton
        f2_peaks = detect_spectral_peaks(s2, fs, 1)
        
        df = f2_peaks[0] - f1_peaks[0]
        dt = t2[N_seg//2] - t1[N_seg//2]
        fd_rate_est = df / dt
        
        error_pct = compute_doppler_rate_error(fd_rate_target, fd_rate_est)
        passed = error_pct < THRESHOLD_PCT
        
        details = (
            f"Dérive cible  : {fd_rate_target:.2f} Hz/s\n"
            f"Dérive estimée: {fd_rate_est:.2f} Hz/s\n"
            f"Erreur        : {error_pct:.4f}% (seuil: {THRESHOLD_PCT}%)"
        )
        self.report.add_result("Doppler Dynamique", passed, details)
        return passed

    def run_all(self):
        """Exécute tous les tests et affiche le bilan."""
        print("\n" + "#" * 70)
        print("  SGD SYSTEM — VALIDATION DU MODÈLE DE SIGNAL")
        print("  SYNAPTIC Lab — Ir divinkas")
        print("#" * 70)

        self.test_spectral_fidelity()
        self.test_snr_accuracy()
        self.test_doppler_accuracy()
        self.test_noise_gaussianity()
        self.test_numerical_stability()
        self.test_dynamic_doppler()

        self.report.print_summary()
        return all(r["passed"] for r in self.report.results)

if __name__ == "__main__":
    config = load_config()
    validator = SignalValidator(config)
    success = validator.run_all()
    sys.exit(0 if success else 1)
