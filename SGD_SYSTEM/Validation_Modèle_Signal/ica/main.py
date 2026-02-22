"""
Ir divinkas — SYNAPTIC Lab
SGD System — Point d'Entrée Validation ICA

Exécute le plan de validation complet pour la séparation de sources.
"""

import sys
import os
from pathlib import Path

# Ajout du répertoire parent au path pour importer les modules de base
sys.path.append(str(Path(__file__).parent.parent))

from signal_model import load_config
from ica.scenarios import get_all_scenarios
from ica.validator import ICAValidator

def main():
    print("\n" + "#" * 60)
    print("  SGD SYSTEM - VALIDATION SEPARATION DE SOURCES (ICA)")
    print("  SYNAPTIC Lab - Ir divinkas")
    print("#" * 60)

    # 1. Chargement Configuration
    config = load_config()
    validator = ICAValidator(config)
    scenarios = get_all_scenarios()
    
    # 2. Paramètres de simulation
    N_RUNS = 50  # Nombre de runs par scénario
    
    global_results = {}
    
    # 3. Boucle sur les scénarios
    for scenario in scenarios:
        # Adaptation du SNR selon le scénario
        snr_db = 20
        if scenario.name == "Brouillage Fort":
            snr_db = 15
            
        res = validator.run_scenario(scenario, n_runs=N_RUNS, snr_db=snr_db)
        global_results[scenario.name] = res

    # 4. Rapport Final
    print("\n" + "=" * 60)
    print("  RESUME DU PLAN DE VALIDATION ICA")
    print("-" * 60)
    
    all_ok = True
    for name, res in global_results.items():
        status = "[OK]" if res['success_rate'] > 90 else "[FAIL]"
        if res['success_rate'] <= 90: all_ok = False
        print(f"  {status} {name:20s} | SIR: {res['avg_sir']:5.1f} dB | Succes: {res['success_rate']:5.1f}%")
    
    print("=" * 60)
    
    if all_ok:
        print("\n  GO - La brique ICA est validee pour l'integration.")
    else:
        print("\n  NO-GO - Des optimisations sont necessaires sur l'ICA.")
    
    print()

if __name__ == "__main__":
    main()
