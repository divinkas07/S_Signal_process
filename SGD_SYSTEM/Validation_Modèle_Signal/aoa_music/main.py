"""
Ir divinkas — SYNAPTIC Lab
SGD System — Validation AOA Dashboard

Lancement des tests de validation pour l'algorithme MUSIC.
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

import sys
import os
from pathlib import Path

# Ajout du répertoire racine au path (1 niveau au-dessus)
sys.path.append(str(Path(__file__).parent.parent))

from signal_model import load_config
from aoa_music.scenarios import get_aoa_scenarios
from aoa_music.validator import AOAValidator
from aoa_music.estimator import music_spectrum

console = Console()

def run_aoa_validation():
    # 1. Chargement config
    cfg = load_config()
    validator = AOAValidator(cfg)
    scenarios = get_aoa_scenarios()
    
    console.print(Panel.fit("[bold blue]SGD System — Validation AOA (MUSIC)[/bold blue]", border_style="blue"))
    
    results = []
    
    # 2. Exécution des scénarios
    with Progress() as progress:
        task = progress.add_task("[cyan]Simulation des scénarios...", total=len(scenarios))
        
        for scenario in scenarios:
            res = validator.run_scenario(scenario, n_runs=100)
            results.append(res)
            progress.update(task, advance=1)

    # 3. Affichage du tableau de synthèse
    table = Table(title="Résultats de Validation AOA")
    table.add_column("Scénario", style="cyan")
    table.add_column("RMSE (°)", justify="right")
    table.add_column("Bias (°)", justify="right")
    table.add_column("CRLB (°)", justify="right", style="dim")
    table.add_column("Ratio RMSE/CRLB", justify="right")
    table.add_column("Succès (%)", justify="right")
    table.add_column("Status", justify="center")

    for res in results:
        status = "[green]PASS[/green]" if res['success_rate'] >= 95 and res['rmse'] < 2.0 else "[red]FAIL[/red]"
        table.add_row(
            res['name'],
            f"{res['rmse']:.3f}",
            f"{res['bias']:.3f}",
            f"{res['crlb']:.3f}",
            f"{res['ratio_crlb']:.2f}",
            f"{res['success_rate']:.1f}%",
            status
        )

    console.print(table)
    
    # 4. Visualisation (Exemple sur le scénario idéal)
    plot_aoa_visuals(validator, scenarios[0], cfg)

def plot_aoa_visuals(validator, scenario, config):
    """Génère un graphique du spectre MUSIC pour un snapshot."""
    N = int(validator.fs * 0.01)
    X = scenario.generate_received_signal(N, validator.fs, validator.arr['d_lambda'])
    
    from channel_model import generate_awgn
    p_sig = np.mean(np.abs(X)**2)
    noise = generate_awgn(X.shape, scenario.snr_db, p_sig)
    X_noisy = X + noise
    
    angles, spectrum = music_spectrum(X_noisy, scenario.n_sources, validator.arr['d_lambda'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(angles, spectrum, label='Pseudo-spectre MUSIC')
    
    for a in scenario.aoa_deg:
        plt.axvline(a, color='red', linestyle='--', alpha=0.6, label=f'Vrai AOA: {a}°')
        
    plt.title(f"Spectre MUSIC - {scenario.name} (SNR={scenario.snr_db}dB)")
    plt.xlabel("Angle (°)")
    plt.ylabel("Pseudo-spectre (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = "e:/SSN_lab/SGD_SYSTEM/Validation_Modèle_Signal/Validation_AOA/aoa_music_preview.png"
    plt.savefig(output_path)
    console.print(f"\n[yellow]Graphique de prévisualisation enregistré dans : {output_path}[/yellow]")

if __name__ == "__main__":
    run_aoa_validation()
