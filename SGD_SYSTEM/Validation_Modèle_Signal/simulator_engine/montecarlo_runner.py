"""
Ir divinkas — SYNAPTIC Lab
SGD System — Monte Carlo Runner

Orchestrateur de simulation massive :
  1. Charge config.yaml
  2. Boucle sur snr_range × n_runs
  3. Génère signal → applique canal → exécute estimateurs → calcule métriques
  4. Vérifie RMSE ≥ CRLB (critère GO)
  5. Exporte résultats consolidés

Usage :
    python montecarlo_runner.py                    # config par défaut
    python montecarlo_runner.py --config custom.yaml
    python montecarlo_runner.py --quick            # 10 runs (test rapide)
"""

import numpy as np
import os
import sys
import time
import argparse
from pathlib import Path

# Adjust sys.path to allow absolute imports from the root if run directly
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator_engine.signal_model import load_config, generate_signal_from_config, generate_time_vector, generate_multitone
from simulator_engine.channel_model import apply_channel
from core_algorithms.estimator_music import estimate_aoa_music
from core_algorithms.crlb import crlb_frequency, crlb_aoa
from core_algorithms.metrics import rmse


# ══════════════════════════════════════════════
# Monte Carlo Engine
# ══════════════════════════════════════════════

def run_montecarlo(config: dict, n_runs_override: int = None, verbose: bool = True) -> dict:
    """
    Exécute la boucle Monte Carlo.

    Returns
    -------
    results : dict
        {
            'snr_values': [...],
            'aoa_rmse': [...],         # RMSE AOA par SNR
            'aoa_crlb': [...],         # CRLB AOA par SNR
            'aoa_estimates': {...},    # estimations brutes
            'go_status': bool,
        }
    """
    sig = config["signal"]
    dop = config["doppler"]
    arr = config["array"]
    mc = config["montecarlo"]

    snr_values = mc["snr_range"]
    n_runs = n_runs_override if n_runs_override else mc["n_runs"]
    aoa_true = arr["aoa_deg"]

    results = {
        "snr_values": snr_values,
        "aoa_rmse": [],
        "aoa_crlb": [],
        "freq_rmse": [],
        "freq_crlb": [],
        "aoa_estimates": {},
        "go_status": True,
    }

    # Calcule la puissance totale théorique (Sigma A_k^2)
    total_power = np.sum(np.array(sig["amplitudes"])**2)

    N = int(sig["fs"] * sig["duration"])
    total = len(snr_values) * n_runs
    count = 0

    if verbose:
        print("\n" + "█" * 60)
        print("  MONTE CARLO SIMULATION")
        print(f"  SNR range : {snr_values}")
        print(f"  Runs/SNR  : {n_runs}")
        print(f"  Total     : {total} runs")
        print("█" * 60)

    start_time = time.time()

    for snr_db in snr_values:
        aoa_estimates = []
        freq_ests = []

        # Config temporaire avec SNR modifié
        cfg_run = config.copy()
        cfg_run["channel"] = config["channel"].copy()
        cfg_run["channel"]["snr_db"] = snr_db

        for run in range(n_runs):
            count += 1

            # ── Génération ──
            t, s, X, fd_eff, fdr_eff = generate_signal_from_config(cfg_run)

            # ── Canal ──
            X_noisy, _ = apply_channel(X, cfg_run)

            # ── Estimation MUSIC ──
            aoa_est = estimate_aoa_music(
                X_noisy, n_sources=1, d_lambda=arr["d_lambda"]
            )
            aoa_estimates.append(aoa_est[0])

            # ── Estimation Frequency (First Tone) ──
            from core_algorithms.metrics import detect_spectral_peaks
            peaks = detect_spectral_peaks(X_noisy[0], sig["fs"], n_peaks=len(sig["frequencies"]))
            freq_raw = peaks[0] if len(peaks) > 0 else np.nan
            freq_ests.append(freq_raw)

            if verbose and count % max(1, total // 20) == 0:
                pct = count / total * 100
                elapsed = time.time() - start_time
                eta = elapsed / count * (total - count)
                print(f"  [{pct:5.1f}%] Run {count}/{total} — "
                      f"SNR={snr_db:+3d} dB — "
                      f"ETA: {eta:.0f}s")

        # ── Métriques pour ce SNR ──
        aoa_arr = np.array(aoa_estimates)
        valid = ~np.isnan(aoa_arr)

        if np.sum(valid) > 0:
            rmse_aoa = rmse(aoa_arr[valid], np.full(np.sum(valid), aoa_true))
        else:
            rmse_aoa = np.inf

        crlb_val = crlb_aoa(
            arr["n_elements"], arr["d_lambda"], snr_db, aoa_true, n_snapshots=N
        )

        # Freq Metrics
        freq_arr = np.array(freq_ests)
        valid_f = ~np.isnan(freq_arr)
        # Use NOMINAL central frequency as truth. 
        # The LEO chirp creates a physical bias which our reformulated CRLB now accounts for.
        f_true = sig["frequencies"][0] + fd_eff 
        if np.sum(valid_f) > 0:
            rmse_f = rmse(freq_arr[valid_f], np.full(np.sum(valid_f), f_true))
        else:
            rmse_f = np.inf
        
        alpha = fdr_eff # Use effective fd_rate from generator
        crlb_f = crlb_frequency(sig["fs"], N, snr_db, sig["amplitudes"][0], total_power, doppler_rate_hz_per_sec=alpha)

        results["aoa_rmse"].append(rmse_aoa)
        results["aoa_crlb"].append(crlb_val)
        results["freq_rmse"].append(rmse_f)
        results["freq_crlb"].append(crlb_f)
        results["aoa_estimates"][snr_db] = aoa_arr

        # Verification GO : RMSE >= CRLB * 0.9 (tolerance stat avec 200+ runs)
        if rmse_aoa < crlb_val * 0.9 or rmse_f < crlb_f * 0.9:
            results["go_status"] = False

        if verbose:
            status = "✅" if (rmse_aoa >= crlb_val * 0.9 and rmse_f >= crlb_f * 0.9) else "⚠️"
            print(f"  {status} SNR={snr_db:+3d} dB — "
                  f"RMSE_A={rmse_aoa:.4f}° (CRLB={crlb_val:.4f}°) — "
                  f"RMSE_F={rmse_f:.4f}Hz (CRLB={crlb_f:.4f}Hz)")

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n  ⏱️  Temps total : {elapsed:.1f}s ({elapsed/total*1000:.1f} ms/run)")
        print(f"\n  {'🚦 GO' if results['go_status'] else '🚦 NO-GO'} — "
              f"{'Tous critères satisfaits' if results['go_status'] else 'Certains critères échoués'}")

    return results


# ══════════════════════════════════════════════
# Export des résultats
# ══════════════════════════════════════════════

def export_results(results: dict, save_dir: str, fmt: str = "npz"):
    """Sauvegarde les résultats consolidés."""
    os.makedirs(save_dir, exist_ok=True)

    if fmt == "npz":
        filepath = os.path.join(save_dir, "montecarlo_results.npz")
        np.savez(
            filepath,
            snr_values=results["snr_values"],
            aoa_rmse=results["aoa_rmse"],
            aoa_crlb=results["aoa_crlb"],
        )
    elif fmt == "csv":
        filepath = os.path.join(save_dir, "montecarlo_results.csv")
        import csv
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SNR_dB", "AOA_RMSE_deg", "AOA_CRLB_deg"])
            for snr, rmse_val, crlb_val in zip(
                results["snr_values"], results["aoa_rmse"], results["aoa_crlb"]
            ):
                writer.writerow([snr, rmse_val, crlb_val])

    print(f"\n  💾 Résultats exportés : {filepath}")
    return filepath


# ══════════════════════════════════════════════
# Visualisation RMSE vs CRLB
# ══════════════════════════════════════════════

def plot_rmse_vs_crlb(results: dict, save_path: str = None):
    """Trace RMSE et CRLB en fonction du SNR."""
    import matplotlib.pyplot as plt

    snr = results["snr_values"]
    rmse_vals = results["aoa_rmse"]
    crlb_vals = results["aoa_crlb"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(snr, rmse_vals, "o-", label="RMSE (MUSIC)", linewidth=2, markersize=8)
    ax.semilogy(snr, crlb_vals, "s--", label="CRLB", linewidth=2, markersize=8, color="red")

    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("RMSE / CRLB — AOA (degrés)", fontsize=13)
    ax.set_title("Performance MUSIC vs Cramér-Rao Lower Bound", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim([min(snr) - 2, max(snr) + 2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Graphique sauvegardé : {save_path}")

    plt.show()


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SGD Monte Carlo Runner")
    parser.add_argument("--config", default=None, help="Chemin vers config.yaml")
    parser.add_argument("--quick", action="store_true", help="Mode rapide (50 runs)")
    parser.add_argument("--runs", type=int, default=200, help="Nombre de runs par SNR (defaut 200)")
    parser.add_argument("--no-plot", action="store_true", help="Désactiver le graphique")
    args = parser.parse_args()

    config = load_config(args.config)
    n_runs = 50 if args.quick else args.runs

    results = run_montecarlo(config, n_runs_override=n_runs)

    # Export
    save_dir = config.get("output", {}).get("save_dir", "./results")
    fmt = config.get("output", {}).get("format", "npz")
    export_results(results, save_dir, fmt)

    # Plot
    if not args.no_plot:
        plot_path = os.path.join(save_dir, "rmse_vs_crlb.png")
        plot_rmse_vs_crlb(results, save_path=plot_path)

    return 0 if results["go_status"] else 1


if __name__ == "__main__":
    sys.exit(main())
