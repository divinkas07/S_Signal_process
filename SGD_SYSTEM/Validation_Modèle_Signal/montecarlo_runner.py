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

from signal_model import load_config, generate_signal_from_config, generate_time_vector, generate_multitone
from channel_model import apply_channel
from estimator_music import estimate_aoa_music
from crlb import crlb_frequency, crlb_aoa
from metrics import rmse


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
        "aoa_estimates": {},
        "go_status": True,
    }

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

        # Config temporaire avec SNR modifié
        cfg_run = config.copy()
        cfg_run["channel"] = config["channel"].copy()
        cfg_run["channel"]["snr_db"] = snr_db

        for run in range(n_runs):
            count += 1

            # ── Génération ──
            t, s, X = generate_signal_from_config(cfg_run)

            # ── Canal ──
            X_noisy, _ = apply_channel(X, cfg_run)

            # ── Estimation MUSIC ──
            aoa_est = estimate_aoa_music(
                X_noisy, n_sources=1, d_lambda=arr["d_lambda"]
            )
            aoa_estimates.append(aoa_est[0])

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

        results["aoa_rmse"].append(rmse_aoa)
        results["aoa_crlb"].append(crlb_val)
        results["aoa_estimates"][snr_db] = aoa_arr

        # Vérification GO : RMSE ≥ CRLB (l'estimateur ne peut pas battre la borne)
        if rmse_aoa < crlb_val * 0.5:  # tolérance facteur 0.5 pour estimation MC
            results["go_status"] = False

        if verbose:
            status = "✅" if rmse_aoa >= crlb_val * 0.5 else "⚠️"
            print(f"  {status} SNR={snr_db:+3d} dB — "
                  f"RMSE={rmse_aoa:.4f}° — "
                  f"CRLB={crlb_val:.4f}°")

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
    parser.add_argument("--quick", action="store_true", help="Mode rapide (10 runs)")
    parser.add_argument("--no-plot", action="store_true", help="Désactiver le graphique")
    args = parser.parse_args()

    config = load_config(args.config)
    n_runs = 10 if args.quick else None

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
