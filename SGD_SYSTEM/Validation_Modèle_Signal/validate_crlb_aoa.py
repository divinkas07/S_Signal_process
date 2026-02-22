"""
Ir divinkas — SYNAPTIC Lab
SGD System — Script de Validation Fondamentale CRLB vs MUSIC

Ce script réalise un balayage de SNR pour valider la cohérence physique de l'estimateur MUSIC
par rapport à la borne théorique de Cramér-Rao (CRLB).
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
from pathlib import Path

# Configuration des chemins pour importer les modules du projet
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

try:
    from signal_model import generate_array_signal, load_config
    from channel_model import generate_awgn
    from crlb import crlb_aoa
    from estimator_music import estimate_aoa_music
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    sys.exit(1)

def run_crlb_validation_sweep():
    # ── Paramètres de Simulation ──
    snr_range_db = np.arange(-5, 41, 2)  # Sweep de -5 à 40 dB
    n_runs = 50                           # Nombre de Monte Carlo par point
    fs = 1e6                              # 1 MHz sampling
    duration = 0.005                      # 5 ms d'observation
    N = int(fs * duration)
    
    # Paramètres Array (ULA)
    M = 2                                 # Nombre d'antennes (doublet λ/2)
    d_lambda = 0.5
    aoa_true = 22.34                      # Degrés (Pas sur le grid 0.1)
    
    rmse_results = []
    crlb_results = []
    
    print(f"Lancement de la validation CRLB (M={M}, AOA={aoa_true}°, N={N})...")
    
    for snr_db in tqdm(snr_range_db, desc="Scanning SNR"):
        errors = []
        
        # 1. Calcul CRLB théorique (sqrt pour être homogène au RMSE en degrés)
        # On utilise la fonction du projet qui implémente la Deterministic CRLB
        val_crlb = crlb_aoa(n_elements=M, d_lambda=d_lambda, snr_db=snr_db, aoa_deg=aoa_true, n_snapshots=N)
        crlb_results.append(val_crlb)
        
        # 2. Monte Carlo
        for _ in range(n_runs):
            # Génération signal (bruit blanc complexe sur source)
            s = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
            
            # Application steering vector
            X = generate_array_signal(s, n_elements=M, d_lambda=d_lambda, aoa_deg=aoa_true)
            
            # Ajout du bruit AWGN
            p_sig = np.mean(np.abs(X)**2)
            noise = generate_awgn(X.shape, snr_db, p_sig)
            X_noisy = X + noise
            
            # Estimation MUSIC
            try:
                aoa_est = estimate_aoa_music(X_noisy, n_sources=1, d_lambda=d_lambda)
                if not np.isnan(aoa_est[0]):
                    errors.append(aoa_est[0] - aoa_true)
            except:
                pass
                
        # Calcul RMSE pour ce SNR
        if len(errors) > 0:
            rmse = np.sqrt(np.mean(np.array(errors)**2))
        else:
            rmse = np.nan
        rmse_results.append(rmse)
        
    # ── Génération des Graphiques ──
    rmse_results = np.array(rmse_results)
    crlb_results = np.array(crlb_results)
    ratio = rmse_results / crlb_results
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Panel 1 : RMSE vs CRLB
    axes[0].semilogy(snr_range_db, rmse_results, 'o-', label='RMSE MUSIC (Simulé)')
    axes[0].semilogy(snr_range_db, crlb_results, 'r--', label='√CRLB (Théorique)')
    axes[0].set_title(f"Validation Cohérence : RMSE vs Borne Théorique (AOA={aoa_true}°)")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Erreur RMS (degrés)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    
    # Panel 2 : Ratio (Efficacité)
    axes[1].plot(snr_range_db, ratio, 'g-s', label='Ratio RMSE / √CRLB')
    axes[1].axhline(1.0, color='black', linestyle='-')
    axes[1].axhline(1.5, color='orange', linestyle='--', label='Limite Performance (1.5)')
    axes[1].set_title("Efficacité de l'estimateur (Doit être >= 1)")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(0, 5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    output_png = ROOT_DIR / "crlb_validation_results.png"
    plt.savefig(output_png)
    print(f"\nValidation terminee. Graphique enregistre dans : {output_png}")
    
    # Verification KPI final
    high_snr_mask = snr_range_db > 15
    avg_ratio = np.nanmean(ratio[high_snr_mask])
    print(f"Ratio moyen a haut SNR (>15dB) : {avg_ratio:.2f}")
    
    if np.any(ratio < 0.95): # Marge pour stats
        print("ATTENTION : L'estimateur semble passer sous la CRLB (Verifier modele !)")
    elif avg_ratio < 1.5:
        print("KPI VALIDE : L'estimateur est optimal (Ratio < 1.5)")
    else:
        print("INFO : L'estimateur est coherent mais pourrait etre optimise.")

if __name__ == "__main__":
    run_crlb_validation_sweep()
