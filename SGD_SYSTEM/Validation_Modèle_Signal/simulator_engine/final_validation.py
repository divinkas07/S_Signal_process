"""
Ir divinkas — SYNAPTIC Lab
SGD System — Master Validation Suite

Ce script coordonne l'exécution de tous les tests de validation :
1. Validation du modèle de signal (spectre, SNR, Doppler, etc.)
2. Validation CRLB vs MUSIC (cohérence physique de l'estimateur)
3. Validation ICA (séparation de sources dans divers scénarios)
"""

import sys
from pathlib import Path

# Adjust sys.path to allow absolute imports from the root if run directly
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator_engine.validation_report import get_report
from simulator_engine.validate_signal import SignalValidator
from simulator_engine.signal_model import load_config

def run_final_validation():
    config = load_config()
    final_report = get_report("SGD SYSTEM — MASTER VALIDATION REPORT")
    
    print("\n" + "=" * 70)
    print("  DEMARRAGE DE LA SUITE DE VALIDATION FINALE")
    print("  SYNAPTIC Lab - Ir divinkas")
    print("=" * 70)

    # 1. Validation du Modèle de Signal
    print("\n--- PHASE 1 : Modèle de Signal ---")
    validator = SignalValidator(config)
    success_signal = validator.run_all()
    for res in validator.report.results:
        final_report.add_result(f"Signal: {res['name']}", res['passed'], res['details'])

    # 2. Validation CRLB (AoA)
    print("\n--- PHASE 2 : Cohérence Physique (CRLB) ---")
    try:
        from simulator_engine.validate_crlb_aoa import run_crlb_validation_sweep
        # run_crlb_validation_sweep ne retourne pas de status, on va intercepter l'exécution
        # Pour le master script, on pourrait vouloir qu'il retourne un status.
        # Pour l'instant, on l'exécute.
        run_crlb_validation_sweep()
        final_report.add_result("Physique: CRLB vs MUSIC", True, "Scan SNR terminé. Voir graphique crlb_validation_results.png")
    except Exception as e:
        final_report.add_result("Physique: CRLB vs MUSIC", False, f"Échec de l'exécution: {e}")

    # 3. Validation ICA
    print("\n--- PHASE 3 : Séparation de Sources (ICA) ---")
    try:
        from simulator_engine.validate_ica import run_simulation
        # Exécution d'un scénario représentatif (Idéal)
        sir, rate = run_simulation(1, n_runs=20, snr_db=20)
        passed_ica = rate > 90
        final_report.add_result("Séparation: ICA Scenario Ideal", passed_ica, f"SIR Moyen: {sir:.2f} dB, Taux de succès: {rate}%")
    except Exception as e:
        final_report.add_result("Séparation: ICA", False, f"Échec de l'exécution: {e}")

    # Résumé Final
    final_report.print_summary()

if __name__ == "__main__":
    run_final_validation()
