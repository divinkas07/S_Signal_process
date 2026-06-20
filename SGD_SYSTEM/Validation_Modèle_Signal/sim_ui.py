"""
Ir divinkas — SYNAPTIC Lab
SGD System — Interface Terminal (Rich TUI)

Point d'entrée interactif pour simuler et interagir avec le pipeline :
    Signal → Canal → Estimateur → Métriques → CRLB

Usage :
    python sim_ui.py
    python sim_ui.py --config custom.yaml
"""

import sys
import time
import copy
import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

# ──────────────────────────────────────────────
# Rich
# ──────────────────────────────────────────────
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.live import Live
from rich.layout import Layout
from rich import box

# ──────────────────────────────────────────────
# Modules SGD (Simulation Engine)
# ──────────────────────────────────────────────
from simulator_engine.signal_model import load_config, generate_signal_from_config, generate_time_vector, generate_multitone
from simulator_engine.channel_model import apply_channel, signal_power

# ──────────────────────────────────────────────
# Modules SGD (Core Algorithms)
# ──────────────────────────────────────────────
from core_algorithms.estimator_music import estimate_aoa_music, music_spectrum
from core_algorithms.estimator_ica import estimate_ica
from core_algorithms.estimator_imm import create_doppler_imm
from core_algorithms.crlb import crlb_aoa, crlb_frequency, compute_crlb_from_config
from core_algorithms.metrics import rmse, mae, detect_spectral_peaks, compute_doppler_error, compute_sir

# ── Dynamic IMM Industrial ──
from core_algorithms.IMM_industrial.leo_doppler_sim import generate_leo_trajectory, add_maneuver
from core_algorithms.IMM_industrial.imm_tracker import IndustrialIMMTracker

console = Console()

# ══════════════════════════════════════════════
# Utilitaires UI
# ══════════════════════════════════════════════

BANNER = """
[bold cyan]╔══════════════════════════════════════════════════════╗[/]
[bold cyan]║[/]  [bold white]SGD SYSTEM[/] [dim]—[/] [bold yellow]Terminal Interface[/]                       [bold cyan]║[/]
[bold cyan]║[/]  [dim]SYNAPTIC Lab · Ir divinkas[/]                          [bold cyan]║[/]
[bold cyan]╚══════════════════════════════════════════════════════╝[/]
"""


def print_banner():
    console.print(BANNER)


def section(title: str):
    console.print()
    console.rule(f"[bold yellow]{title}[/]", style="yellow")
    console.print()


def ok(msg: str):
    console.print(f"  [bold green]✔[/]  {msg}")


def warn(msg: str):
    console.print(f"  [bold yellow]⚠[/]  {msg}")


def err(msg: str):
    console.print(f"  [bold red]✖[/]  {msg}")


def fmt_float(v, decimals=4) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


# ══════════════════════════════════════════════
# Affichage de la configuration courante
# ══════════════════════════════════════════════

def display_config(cfg: dict):
    """Affiche un résumé de la configuration actuelle sous forme de tableaux."""
    sig = cfg["signal"]
    dop = cfg["doppler"]
    ch  = cfg["channel"]
    arr = cfg["array"]
    mc  = cfg["montecarlo"]

    # ── Signal ──
    t_sig = Table(title="Signal", box=box.SIMPLE_HEAD, show_header=True)
    t_sig.add_column("Paramètre", style="cyan")
    t_sig.add_column("Valeur", style="white")
    t_sig.add_row("fs (Hz)",       str(sig["fs"]))
    t_sig.add_row("duration (s)",  str(sig["duration"]))
    t_sig.add_row("frequencies",   str(sig["frequencies"]))
    t_sig.add_row("amplitudes",    str(sig["amplitudes"]))
    t_sig.add_row("phases",        str(sig.get("phases")))
    t_sig.add_row("fd Doppler (Hz)", str(dop["fd"]))

    # ── Canal ──
    t_ch = Table(title="Canal", box=box.SIMPLE_HEAD)
    t_ch.add_column("Paramètre", style="cyan")
    t_ch.add_column("Valeur", style="white")
    t_ch.add_row("SNR (dB)",      str(ch["snr_db"]))
    t_ch.add_row("Multipath",     "[green]ON[/]" if ch["multipath"]["enabled"] else "[dim]OFF[/]")

    # ── Array ──
    t_arr = Table(title="Réseau ULA", box=box.SIMPLE_HEAD)
    t_arr.add_column("Paramètre", style="cyan")
    t_arr.add_column("Valeur", style="white")
    t_arr.add_row("N éléments",    str(arr["n_elements"]))
    t_arr.add_row("d/λ",           str(arr["d_lambda"]))
    t_arr.add_row("AOA vrai (°)",  str(arr["aoa_deg"]))

    # ── Monte Carlo ──
    t_mc = Table(title="Monte Carlo", box=box.SIMPLE_HEAD)
    t_mc.add_column("Paramètre", style="cyan")
    t_mc.add_column("Valeur", style="white")
    t_mc.add_row("n_runs",    str(mc["n_runs"]))
    t_mc.add_row("snr_range", str(mc["snr_range"]))

    # ── LEO Scenario ──
    t_leo = Table(title="Satellite LEO", box=box.SIMPLE_HEAD)
    t_leo.add_column("Paramètre", style="cyan")
    t_leo.add_column("Valeur", style="white")
    if cfg.get("leo_scenario"):
        leo = cfg["leo_scenario"]
        t_leo.add_row("Activé", "[green]OUI[/]" if leo["enabled"] else "[dim]NON[/]")
        t_leo.add_row("Altitude", f"{leo['altitude_km']} km")
        t_leo.add_row("Vitesse", f"{leo['v_km_s']} km/s")
        t_leo.add_row("Porteuse", f"{leo['f_carrier_hz']/1e9:.1f} GHz")
        t_leo.add_row("Passage", f"{leo['duration_s']} s")
    else:
        t_leo.add_row("Status", "[dim]Non configuré[/]")

    console.print(Columns([t_sig, t_ch, t_arr, t_mc, t_leo], equal=False, expand=False))


# ══════════════════════════════════════════════
# Éditeurs de paramètres
# ══════════════════════════════════════════════

def _ask_list_float(prompt_label: str, current: list) -> list:
    """Demande une liste de flottants séparés par des virgules."""
    raw = Prompt.ask(
        f"  {prompt_label} [dim](actuel: {current}, séparés par virgules)[/]",
        default=",".join(str(v) for v in current),
    )
    try:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        warn("Format invalide — valeur conservée.")
        return current


def _ask_list_int(prompt_label: str, current: list) -> list:
    raw = Prompt.ask(
        f"  {prompt_label} [dim](actuel: {current}, séparés par virgules)[/]",
        default=",".join(str(v) for v in current),
    )
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        warn("Format invalide — valeur conservée.")
        return current


def edit_signal_params(cfg: dict):
    section("Édition — Paramètres Signal & Doppler")
    sig = cfg["signal"]
    dop = cfg["doppler"]

    console.print("  Appuyez [bold]Entrée[/] pour conserver la valeur actuelle.\n")

    # fs
    raw = Prompt.ask(f"  fs — Fréquence d'échantillonnage (Hz) [dim](actuel: {sig['fs']})[/]",
                     default=str(sig["fs"]))
    try:
        sig["fs"] = float(raw)
    except ValueError:
        warn("Valeur invalide — conservée.")

    # duration
    raw = Prompt.ask(f"  duration — Durée du signal (s) [dim](actuel: {sig['duration']})[/]",
                     default=str(sig["duration"]))
    try:
        sig["duration"] = float(raw)
    except ValueError:
        warn("Valeur invalide — conservée.")

    # frequencies
    sig["frequencies"] = _ask_list_float("frequencies — Fréquences des tons (Hz)", sig["frequencies"])

    # amplitudes
    n_tones = len(sig["frequencies"])
    amps = sig.get("amplitudes", [1.0] * n_tones)
    # Ajuster la longueur si nombre de tons a changé
    if len(amps) < n_tones:
        amps += [1.0] * (n_tones - len(amps))
    sig["amplitudes"] = _ask_list_float("amplitudes — Amplitudes linéaires", amps[:n_tones])

    # phases
    current_phases = sig.get("phases")
    use_random = Confirm.ask("  phases — Utiliser des phases aléatoires ?",
                              default=(current_phases is None))
    if use_random:
        sig["phases"] = None
    else:
        default_phases = current_phases if current_phases else [0.0] * n_tones
        sig["phases"] = _ask_list_float("phases (rad)", default_phases)

    # Doppler
    raw = Prompt.ask(f"  fd — Décalage Doppler (Hz) [dim](actuel: {dop['fd']})[/]",
                     default=str(dop["fd"]))
    try:
        dop["fd"] = float(raw)
    except ValueError:
        warn("Valeur invalide — conservée.")

    ok("Paramètres signal & Doppler mis à jour.")


def edit_channel_params(cfg: dict):
    section("Édition — Paramètres Canal")
    ch = cfg["channel"]

    raw = Prompt.ask(f"  SNR cible (dB) [dim](actuel: {ch['snr_db']})[/]",
                     default=str(ch["snr_db"]))
    try:
        ch["snr_db"] = float(raw)
    except ValueError:
        warn("Valeur invalide — conservée.")

    # Multipath
    mp_on = Confirm.ask("  Activer le multipath ?",
                         default=ch["multipath"]["enabled"])
    ch["multipath"]["enabled"] = mp_on

    if mp_on:
        console.print(f"  Trajets actuels : {ch['multipath']['paths']}")
        if Confirm.ask("  Redéfinir les trajets ?", default=False):
            paths = []
            n_paths = IntPrompt.ask("  Nombre de trajets", default=2)
            for i in range(n_paths):
                g_raw = Prompt.ask(f"  Trajet {i+1} — gain", default="0.5")
                d_raw = Prompt.ask(f"  Trajet {i+1} — delay_s (s)", default="0.000001")
                try:
                    paths.append({"gain": float(g_raw), "delay_s": float(d_raw)})
                except ValueError:
                    warn("  Format invalide — trajet ignoré.")
            ch["multipath"]["paths"] = paths

    ok("Paramètres canal mis à jour.")


def edit_array_params(cfg: dict):
    section("Édition — Paramètres Réseau ULA")
    arr = cfg["array"]

    raw = Prompt.ask(f"  n_elements — Nombre d'antennes [dim](actuel: {arr['n_elements']})[/]",
                     default=str(arr["n_elements"]))
    try:
        arr["n_elements"] = int(raw)
    except ValueError:
        warn("Valeur invalide — conservée.")

    raw = Prompt.ask(f"  d_lambda — Espacement inter-antennes (fraction de λ) [dim](actuel: {arr['d_lambda']})[/]",
                     default=str(arr["d_lambda"]))
    try:
        arr["d_lambda"] = float(raw)
    except ValueError:
        warn("Valeur invalide — conservée.")

    raw = Prompt.ask(f"  aoa_deg — Angle d'arrivée vrai (°) [dim](actuel: {arr['aoa_deg']})[/]",
                     default=str(arr["aoa_deg"]))
    try:
        arr["aoa_deg"] = float(raw)
    except ValueError:
        warn("Valeur invalide — conservée.")

    ok("Paramètres réseau mis à jour.")


def edit_montecarlo_params(cfg: dict):
    section("Édition — Paramètres Monte Carlo")
    mc = cfg["montecarlo"]

    raw = Prompt.ask(f"  n_runs — Nombre de tirages par SNR [dim](actuel: {mc['n_runs']})[/]",
                     default=str(mc["n_runs"]))
    try:
        mc["n_runs"] = int(raw)
    except ValueError:
        warn("Valeur invalide — conservée.")

    mc["snr_range"] = _ask_list_int("snr_range — Plage SNR (dB)", mc["snr_range"])

    ok("Paramètres Monte Carlo mis à jour.")


def edit_leo_params(cfg: dict):
    section("Édition — Scenario Satellite LEO")
    if "leo_scenario" not in cfg:
        cfg["leo_scenario"] = {
            "enabled": True, "altitude_km": 700, "v_km_s": 7.5,
            "f_carrier_hz": 2e9, "noise_std_hz": 5.0, "duration_s": 600, "dt_s": 0.1
        }
    leo = cfg["leo_scenario"]

    leo["enabled"] = Confirm.ask("  Activer le scenario LEO ?", default=leo["enabled"])
    
    raw = Prompt.ask(f"  Altitude (km) [dim]({leo['altitude_km']})[/]", default=str(leo["altitude_km"]))
    leo["altitude_km"] = float(raw)
    
    raw = Prompt.ask(f"  Vitesse (km/s) [dim]({leo['v_km_s']})[/]", default=str(leo["v_km_s"]))
    leo["v_km_s"] = float(raw)
    
    raw = Prompt.ask(f"  Fréquence Porteuse (Hz) [dim]({leo['f_carrier_hz']})[/]", default=str(int(leo["f_carrier_hz"])))
    leo["f_carrier_hz"] = float(raw)
    
    raw = Prompt.ask(f"  Bruit Mesure σ (Hz) [dim]({leo['noise_std_hz']})[/]", default=str(leo["noise_std_hz"]))
    leo["noise_std_hz"] = float(raw)

    ok("Scenario LEO mis à jour.")


# ══════════════════════════════════════════════
# Simulation rapide (single shot)
# ══════════════════════════════════════════════

def run_quick_sim(cfg: dict):
    section("Simulation Rapide — Single Shot")

    sig  = cfg["signal"]
    arr  = cfg["array"]
    ch   = cfg["channel"]

    console.print(f"  SNR        : [cyan]{ch['snr_db']} dB[/]")
    console.print(f"  AOA vrai   : [cyan]{arr['aoa_deg']}°[/]")
    console.print(f"  Fréquences : [cyan]{sig['frequencies']} Hz[/]")
    if cfg.get("leo_scenario", {}).get("enabled"):
        ok("Scenario LEO actif (Doppler calculé dynamiquement)")
    console.print()

    try:
        with console.status("[bold cyan]Génération du signal…[/]", spinner="dots"):
            t, s, X, fd_eff, fdr_eff = generate_signal_from_config(cfg)

        with console.status("[bold cyan]Application du canal…[/]", spinner="dots"):
            X_noisy, noise = apply_channel(X, cfg)
            p_sig   = signal_power(X)
            p_noise = signal_power(noise)
            snr_meas = 10 * np.log10(p_sig / p_noise) if p_noise > 0 else np.inf

        with console.status("[bold cyan]Estimation MUSIC…[/]", spinner="dots"):
            aoa_est = estimate_aoa_music(X_noisy, n_sources=1, d_lambda=arr["d_lambda"])
            aoa_true = arr["aoa_deg"]

        # CRLB AOA (Standard)
        N = len(t)
        crlb_aoa_val = crlb_aoa(arr["n_elements"], arr["d_lambda"],
                               ch["snr_db"], aoa_true, n_snapshots=N)
        
        # Doppler Estimation (First Tone)
        # s is the mono-channel reference, but we estimate from X_noisy[0] (first antenna)
        # or from s directly if we want to test the freq estimator alone?
        # Let's estimate from X_noisy[0] to be realistic.
        fs = sig["fs"]
        peaks = detect_spectral_peaks(X_noisy[0], fs, n_peaks=len(sig["frequencies"]))
        f_est = peaks[0] if len(peaks) > 0 else np.nan
        f_true = sig["frequencies"][0] + cfg["doppler"]["fd"]
        
        # CRLB Frequency (LEO Rigorous)
        # Doppler rate alpha for bias calculation
        fdr = cfg["doppler"]["fd_rate"] if cfg["doppler"].get("dynamic") else 0.0
        crlb_freq_val = crlb_frequency(fs, N, ch["snr_db"], sig["amplitudes"][0], 
                                      total_power=np.sum(np.array(sig["amplitudes"])**2),
                                      doppler_rate_hz_per_sec=fdr)

    except Exception as e:
        err(f"Erreur durant la simulation : {e}")
        return

    # ── Tableau résultats ──
    t_res = Table(title="Résultats — Simulation Rapide", box=box.ROUNDED,
                  show_header=True, header_style="bold magenta")
    t_res.add_column("Grandeur",          style="cyan",  width=28)
    t_res.add_column("Valeur",            style="white", width=20)
    t_res.add_column("Unité",             style="dim",   width=12)

    t_res.add_section()
    t_res.add_row("CRLB AOA (√Var)",      fmt_float(crlb_aoa_val, 4), "°")

    t_res.add_section()
    t_res.add_row("Fréq. vraie (f1+fd)",  fmt_float(f_true, 1),      "Hz")
    if not np.isnan(f_est):
        f_err = abs(f_est - f_true)
        f_color = "green" if f_err < crlb_freq_val * 3 else "yellow"
        t_res.add_row("Fréq. estimée",      f"[{f_color}]{fmt_float(f_est, 1)}[/]", "Hz")
        t_res.add_row("Erreur |Δf|",        f"[{f_color}]{fmt_float(f_err, 4)}[/]", "Hz")
    else:
        t_res.add_row("Fréq. estimée",      "[red]NaN[/]",             "Hz")
    
    t_res.add_row("CRLB Fréq (√Var)",     fmt_float(crlb_freq_val, 4), "Hz")

    # Verdict
    if not np.all(np.isnan(aoa_est)):
        # Critères physiques : RMSE >= CRLB
        # Critères performance : RMSE < 1.5 * CRLB
        aoa_val = aoa_est[0]
        err_aoa = abs(aoa_val - aoa_true)
        
        is_physical = err_aoa >= crlb_aoa_val * 0.9  # 0.9 pour tolérance MC/single
        is_optimal  = err_aoa < crlb_aoa_val * 2.0   # 2.0 pour single shot (bruité)
        
        if err_aoa < crlb_aoa_val * 0.2:
             verdict = "[bold red]✖ NON-PHYSIQUE (Trop beau pour être vrai)[/]"
             v_style = "red"
        elif is_optimal:
             verdict = "[bold green]✔ Estimateur optimal[/]"
             v_style = "green"
        else:
             verdict = "[bold yellow]⚠ Performance dégradée (> 2×CRLB)[/]"
             v_style = "yellow"
    else:
        verdict = "[bold red]✖ Estimation échouée[/]"
        v_style = "red"

    console.print(t_res)
    console.print(Panel(verdict, title="Verdict", border_style="cyan"))


# ══════════════════════════════════════════════
# Monte Carlo complet
# ══════════════════════════════════════════════

def run_montecarlo(cfg: dict):
    section("Monte Carlo — Simulation Massive")

    mc  = cfg["montecarlo"]
    arr = cfg["array"]
    sig = cfg["signal"]
    dop = cfg["doppler"]

    snr_values = mc["snr_range"]

    # Quick mode ?
    quick = Confirm.ask("  Mode rapide (20 runs/SNR pour test) ?", default=False)
    n_runs = 20 if quick else 200

    total = len(snr_values) * n_runs

    console.print(f"\n  SNR range : [cyan]{snr_values}[/]")
    console.print(f"  Runs/SNR  : [cyan]{n_runs}[/]")
    console.print(f"  Total     : [cyan]{total} runs[/]")
    if cfg.get("leo_scenario", {}).get("enabled"):
        ok("Scenario LEO actif (Doppler calculé dynamiquement)")
    console.print()

    results = {
        "snr_values": snr_values,
        "aoa_rmse":   [],
        "aoa_crlb":   [],
        "freq_rmse":  [],
        "freq_crlb":  [],
        "go_flags":   [],
    }

    start_time = time.time()
    N = int(sig["fs"] * sig["duration"])

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:

        overall = progress.add_task("Monte Carlo…", total=total)
        snr_task = progress.add_task("SNR courant…", total=n_runs)

        for snr_db in snr_values:
            progress.reset(snr_task)
            progress.update(snr_task, description=f"SNR = {snr_db:+3d} dB")

            cfg_run = copy.deepcopy(cfg)
            cfg_run["channel"]["snr_db"] = snr_db

            # Puissance totale théorique (somme des A_k^2 pour complexes)
            total_power = np.sum(np.array(sig["amplitudes"])**2)

            aoa_true = arr["aoa_deg"]
            aoa_ests = []
            freq_ests = []
            crlb_a_list = []
            crlb_f_list = []

            for _ in range(n_runs):
                t, s, X, fd_eff, fdr_eff = generate_signal_from_config(cfg_run)
                X_noisy, noise = apply_channel(X, cfg_run)
                
                # Mesure du SNR réel pour cette itération
                p_sig_real = signal_power(X)
                p_noise_real = signal_power(noise)
                snr_lin_real = p_sig_real / p_noise_real if p_noise_real > 0 else 1e12
                
                # AOA
                est = estimate_aoa_music(X_noisy, n_sources=1,
                                         d_lambda=arr["d_lambda"])
                aoa_ests.append(est[0])
                
                # Doppler (f1)
                peaks = detect_spectral_peaks(X_noisy[0], sig["fs"], n_peaks=len(sig["frequencies"]))
                freq_ests.append(peaks[0] if len(peaks) > 0 else np.nan)
                
                # CRLB individuels (Rigoureux)
                alpha = dop.get("fd_rate", 0.0) if dop.get("dynamic") else 0.0
                crlb_a_list.append(crlb_aoa(arr["n_elements"], arr["d_lambda"],
                                           10 * np.log10(snr_lin_real), aoa_true, n_snapshots=N))
                crlb_f_list.append(crlb_frequency(sig["fs"], N, 10 * np.log10(snr_lin_real), 
                                                 sig["amplitudes"][0], total_power, 
                                                 doppler_rate_hz_per_sec=alpha))
                
                progress.advance(overall)
                progress.advance(snr_task)

            # Statistiques
            aoa_arr = np.array(aoa_ests, dtype=float)
            valid_aoa = ~np.isnan(aoa_arr)
            rmse_aoa = rmse(aoa_arr[valid_aoa], np.full(np.sum(valid_aoa), aoa_true)) \
                       if np.sum(valid_aoa) > 0 else np.inf
            
            # On utilise la MOYENNE des CRLB (en variance c'est plus propre, mais ici les SNR sont proches)
            crlb_a = np.mean(crlb_a_list)
            crlb_f = np.mean(crlb_f_list)

            # Freq Stats
            freq_arr = np.array(freq_ests, dtype=float)
            valid_f = ~np.isnan(freq_arr)
            # On utilise le Doppler EFFECTIF retourné par le générateur
            # On ajoute le décalage spectral moyen dû à la dérive Doppler (0.5 * fd_rate * Duration)
            f_true = sig["frequencies"][0] + fd_eff + 0.5 * fdr_eff * sig["duration"]
            rmse_f = rmse(freq_arr[valid_f], np.full(np.sum(valid_f), f_true)) \
                     if np.sum(valid_f) > 0 else np.inf

            # Critères de Validation (D'après crlb_descript)
            # 1. Physical Validity: RMSE >= sqrt(CRLB) * 0.95 (marge stat)
            # 2. Performance: RMSE < sqrt(CRLB) * 1.5
            ratio_a = rmse_aoa / crlb_a if crlb_a > 0 else np.nan
            
            # GO si physique ET performant
            is_phys = (rmse_aoa >= crlb_a * 0.95)
            is_perf = (ratio_a < 1.5)
            go = is_phys and is_perf

            results["aoa_rmse"].append(rmse_aoa)
            results["aoa_crlb"].append(crlb_a)
            results["freq_rmse"].append(rmse_f)
            results["freq_crlb"].append(crlb_f)
            results["go_flags"].append(go)

    elapsed = time.time() - start_time

    # ── Tableau de résultats ──
    t_mc = Table(title="Résultats Monte Carlo — RMSE vs CRLB (AOA)",
                 box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t_mc.add_column("SNR (dB)",   style="cyan",  justify="right", width=10)
    t_mc.add_column("RMSE (°)",   style="white", justify="right", width=14)
    t_mc.add_column("CRLB (°)",   style="white", justify="right", width=14)
    t_mc.add_column("Ratio",      style="white", justify="right", width=10)
    t_mc.add_column("GO ?",       justify="center", width=8)

    global_go = all(results["go_flags"])

    for idx in range(len(results["snr_values"])):
        snr_db = results["snr_values"][idx]
        rmse_v = results["aoa_rmse"][idx]
        crlb_v = results["aoa_crlb"][idx]
        go     = results["go_flags"][idx]
        
        ratio = rmse_v / crlb_v if crlb_v > 0 and rmse_v != np.inf else float("nan")
        
        if np.isnan(ratio):
            go_str = "[dim]—[/]"
            ratio_str = "—"
        elif rmse_v < crlb_v * 0.95:
            go_str = "[bold red]NON-PHYS[/]"
            ratio_str = f"[red]{ratio:.2f}[/]"
        elif ratio < 1.5:
            go_str = "[bold green]PASS[/]"
            ratio_str = f"[green]{ratio:.2f}[/]"
        else:
            go_str = "[bold yellow]FAIL[/]"
            ratio_str = f"[yellow]{ratio:.2f}[/]"

        t_mc.add_row(
            f"{snr_db:+3d}",
            fmt_float(rmse_v, 6),
            fmt_float(crlb_v, 6),
            ratio_str,
            go_str,
        )

    # Tableau Doppler (Optionnel si RMSE dispo)
    t_f = Table(title="Résultats Monte Carlo — RMSE Frequence",
                 box=box.ROUNDED, show_header=True, header_style="bold cyan")
    t_f.add_column("SNR (dB)",   style="cyan",  justify="right", width=10)
    t_f.add_column("RMSE f (Hz)", style="white", justify="right", width=14)
    t_f.add_column("CRLB f (Hz)", style="white", justify="right", width=14)
    
    for snr_db, rf, cf in zip(results["snr_values"], results["freq_rmse"], results["freq_crlb"]):
        t_f.add_row(f"{snr_db:+3d}", fmt_float(rf, 6), fmt_float(cf, 6))

    console.print(t_mc)
    console.print(t_f)

    verdict_color = "green" if global_go else "red"
    verdict_text  = "🚦 GO — Tous les critères satisfaits" if global_go \
                    else "🚦 NO-GO — Certains critères échoués"
    console.print(Panel(
        f"[bold {verdict_color}]{verdict_text}[/]\n"
        f"[dim]Temps total : {elapsed:.1f}s  |  {elapsed/total*1000:.1f} ms/run[/]",
        title="Verdict Final", border_style=verdict_color,
    ))

    return results


# ══════════════════════════════════════════════
# Inspecteur CRLB
# ══════════════════════════════════════════════

def inspect_crlb(cfg: dict):
    section("Inspecteur CRLB — Cramér-Rao Lower Bound")

    sig = cfg["signal"]
    arr = cfg["array"]
    mc  = cfg["montecarlo"]
    dop = cfg["doppler"]
    N   = int(sig["fs"] * sig["duration"])

    snr_range = mc["snr_range"]
    total_power = np.sum(np.array(sig["amplitudes"])**2)

    # ── CRLB Fréquentielle ──
    t_freq = Table(title="CRLB — Estimation Fréquentielle", box=box.SIMPLE_HEAD,
                   header_style="bold magenta")
    t_freq.add_column("Ton", style="cyan", justify="right")
    t_freq.add_column("Fréquence (Hz)", style="white", justify="right")
    t_freq.add_column("Amplitude", style="white", justify="right")

    for snr_db in snr_range:
        col_name = f"CRLB@{snr_db:+d}dB (Hz)"
        t_freq.add_column(col_name, style="dim", justify="right")

    for k, (f, a) in enumerate(zip(sig["frequencies"], sig["amplitudes"])):
        row = [str(k+1), f"{f:.1f}", f"{a:.3f}"]
        for snr_db in snr_range:
            fdr = dop.get("fd_rate", 0.0) if dop.get("dynamic") else 0.0
            c = crlb_frequency(sig["fs"], N, snr_db, a, total_power, doppler_rate_hz_per_sec=fdr)
            row.append(f"{c:.4e}")
        t_freq.add_row(*row)

    console.print(t_freq)
    console.print()

    # ── CRLB AOA ──
    t_aoa = Table(title="CRLB — Estimation AOA sur ULA", box=box.SIMPLE_HEAD,
                  header_style="bold magenta")
    t_aoa.add_column("SNR (dB)", style="cyan", justify="right")
    t_aoa.add_column("CRLB AOA (°)", style="white", justify="right")
    t_aoa.add_column("CRLB AOA (rad)", style="dim", justify="right")
    t_aoa.add_column("Résolution (N snapshots)", style="dim", justify="right")

    for snr_db in snr_range:
        c_deg = crlb_aoa(arr["n_elements"], arr["d_lambda"],
                          snr_db, arr["aoa_deg"], n_snapshots=N)
        c_rad = np.deg2rad(c_deg)
        c_1snap = crlb_aoa(arr["n_elements"], arr["d_lambda"],
                            snr_db, arr["aoa_deg"], n_snapshots=1)
        t_aoa.add_row(
            f"{snr_db:+3d}",
            fmt_float(c_deg, 6),
            fmt_float(c_rad, 6),
            fmt_float(c_1snap, 4),
        )

    console.print(t_aoa)
    console.print(Panel(
        f"[dim]M = {arr['n_elements']} antennes  |  d/λ = {arr['d_lambda']}  "
        f"|  θ = {arr['aoa_deg']}°  |  N = {N} pts[/]",
        border_style="dim",
    ))


# ══════════════════════════════════════════════
# Pipeline Complet (Signal -> Canal -> ICA -> MUSIC -> IMM)
# ══════════════════════════════════════════════

def run_full_pipeline(cfg: dict):
    section("Pipeline Complet — Algorithme Intégral")
    
    sig = cfg["signal"]
    arr = cfg["array"]
    dop = cfg["doppler"]
    ch  = cfg["channel"]
    
    console.print(f"  [bold]1. Génération & Canal[/]")
    with console.status("Génération..."):
        t, s_ref, X, fd_eff, fdr_eff = generate_signal_from_config(cfg)
        X_noisy, _ = apply_channel(X, cfg)
        ok("Signal reçu bruité généré.")

    console.print(f"\n  [bold]2. Séparation de Sources (ICA)[/]")
    with console.status("Exécution ICA..."):
        # On essaie d'extraire autant de composantes qu'il y a de fréquences ou simplement 1 si 1 source
        n_comp = len(sig["frequencies"]) if len(sig["frequencies"]) > 0 else 1
        S_est, W_glob, _ = estimate_ica(X_noisy, n_components=n_comp)
        ok(f"ICA terminée. Sources extraites : {S_est.shape[0]}")

    console.print(f"\n  [bold]3. Estimation Directionnelle (MUSIC) & Doppler[/]")
    with console.status("Estimation..."):
        # MUSIC sur le signal original bruité (traditionnel)
        aoa_est = estimate_aoa_music(X_noisy, n_sources=1, d_lambda=arr["d_lambda"])
        
        # Doppler sur la composante ICA la plus forte (ou première)
        # On utilise S_est[0] qui devrait être notre source principale séparée
        peaks = detect_spectral_peaks(S_est[0], sig["fs"], n_peaks=1)
        freq_raw = peaks[0] if len(peaks) > 0 else np.nan
        ok(f"AOA estimé : {aoa_est[0]:.2f}°")
        ok(f"Fréq estimée (post-ICA) : {freq_raw:.1f} Hz")

    if dop.get("dynamic", False) or cfg.get("leo_scenario", {}).get("enabled"):
        console.print(f"\n  [bold]4. Tracking IMM (Doppler Dynamique)[/]")
        with console.status("Tracking..."):
            # Pour le pipeline sur bloc unique, on utilise un dt arbitraire ou celui du LEO
            dt = cfg.get("leo_scenario", {}).get("dt_s", 0.1)
            tracker = IndustrialIMMTracker(dt=dt)
            
            # Prediction/Update sur le point estimé
            tracker.predict()
            tracker.update(freq_raw)
            
            freq_final = tracker.x[0]
            ok(f"IMM filtré : {freq_final:.1f} Hz (Model Probs: {tracker.mu})")
    else:
        freq_final = freq_raw

    # Résultats Finaux
    section("Synthèse Pipeline")
    t_res = Table(box=box.DOUBLE)
    t_res.add_column("Bloc", style="cyan")
    t_res.add_column("Grandeur", style="white")
    t_res.add_column("Valeur", justify="right")
    
    t_res.add_row("Canal", "SNR Mesuré", f"{ch['snr_db']} dB")
    t_res.add_row("ICA", "Entrées/Sorties", f"{X.shape[0]} / {S_est.shape[0]}")
    t_res.add_row("MUSIC", "AOA", f"{aoa_est[0]:.4f} °")
    t_res.add_row("IMM", "Fréq Finale", f"{freq_final:.2f} Hz")
    
    console.print(t_res)
    ok("Simulation du Pipeline Complet terminée.")


def run_imm_industrial_sim(cfg: dict):
    """Exécute la simulation IMM industrielle (LEO Satellite)."""
    section("Tracking Doppler IMM — Satellite LEO")

    leo = cfg.get("leo_scenario", {
        "enabled": True, "altitude_km": 700, "v_km_s": 7.5,
        "f_carrier_hz": 2e9, "noise_std_hz": 5.0, "duration_s": 600, "dt_s": 0.1
    })

    # Paramètres depuis config
    duration = leo["duration_s"]
    dt = leo["dt_s"]
    maneuver_start = 400
    
    # Affichage des paramètres
    t_param = Table(title="Paramètres Tracking LEO", box=box.SIMPLE)
    t_param.add_column("Paramètre", style="cyan")
    t_param.add_column("Valeur", style="white")
    t_param.add_row("Altitude", f"{leo['altitude_km']} km")
    t_param.add_row("Vitesse", f"{leo['v_km_s']} km/s")
    t_param.add_row("Porteuse (fc)", f"{leo['f_carrier_hz']/1e9:.1f} GHz")
    t_param.add_row("Bruit (σ)", f"{leo['noise_std_hz']} Hz")
    console.print(t_param)

    with console.status("[bold cyan]Simulation de la trajectoire LEO…[/]", spinner="earth"):
        t, fd_true, fd_meas = generate_leo_trajectory(
            duration=duration, dt=dt, 
            altitude_km=leo["altitude_km"], v_km_s=leo["v_km_s"], 
            f_carrier_hz=leo["f_carrier_hz"], noise_std_hz=leo["noise_std_hz"]
        )
        # Ajout d'une manœuvre pour tester la robustesse
        fd_truth_maneuver = add_maneuver(t, fd_true, maneuver_start=maneuver_start, maneuver_duration=50, accel_hz_s2=2.0)
        fd_meas_maneuver = fd_truth_maneuver + np.random.normal(0, 5.0, size=len(t))
    
    tracker = IndustrialIMMTracker(dt=dt)
    n = len(t)
    results_imm = np.zeros(n)
    probs_imm = np.zeros((n, 3))

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Tracking IMM…[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Filtrage", total=n)
        for i in range(n):
            z = fd_meas_maneuver[i]
            tracker.predict()
            tracker.update(z)
            results_imm[i] = tracker.x[0]
            probs_imm[i] = tracker.mu
            progress.advance(task)

    # Calcul RMSE
    rmse_val = np.sqrt(np.mean((results_imm - fd_truth_maneuver)**2))
    
    # Affichage des résultats
    t_res = Table(title="Résultats IMM Tracking", box=box.ROUNDED)
    t_res.add_column("Métrique", style="cyan")
    t_res.add_column("Valeur", justify="right")
    t_res.add_row("RMSE Finale", f"{rmse_val:.4f} Hz")
    t_res.add_row("Bruit Mesure (sim)", "5.0 Hz RMS")
    
    console.print(t_res)
    
    # Verdict sur la robustesse
    prob_maneuver_max = np.max(probs_imm[maneuver_start*10:(maneuver_start+50)*10, 2])
    if prob_maneuver_max > 0.5:
        ok(f"Détection de manœuvre réussie (Prob Max: {prob_maneuver_max:.2f})")
    else:
        warn(f"Sensibilité manœuvre faible (Prob Max: {prob_maneuver_max:.2f})")

    if Confirm.ask("\n  Générer le rapport graphique ?", default=True):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, fd_truth_maneuver, 'k', label='Ground Truth')
        axes[0].plot(t, results_imm, 'r', label='IMM Track', alpha=0.8)
        axes[0].set_ylabel("Doppler (Hz)")
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(t, probs_imm[:, 0], label='Mode 1 (Const)')
        axes[1].plot(t, probs_imm[:, 1], label='Mode 2 (Drift)')
        axes[1].plot(t, probs_imm[:, 2], label='Mode 3 (Maneuver)')
        axes[1].set_ylabel("Probabilités")
        axes[1].set_xlabel("Temps (s)")
        axes[1].legend()
        axes[1].grid(True)
        
        save_path = "dynamic_imm_report.png"
        plt.tight_layout()
        plt.savefig(save_path)
        ok(f"Rapport sauvegardé : [cyan]{save_path}[/]")
        # plt.show() # On évite de bloquer la TUI si possible


def save_config(cfg: dict, config_path: Path):
    section("Sauvegarde — config.yaml")
    if Confirm.ask(f"  Écrire la configuration dans [cyan]{config_path}[/] ?", default=True):
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        ok(f"Configuration sauvegardée → {config_path}")
    else:
        warn("Sauvegarde annulée.")


# ══════════════════════════════════════════════
# Menu principal — MENU_REGISTRY
# ══════════════════════════════════════════════
# Pour ajouter un nouveau brick : ajouter une entrée ici.
# Format : "clé": ("Label affiché", fonction(cfg))

def build_menu(cfg: dict, config_path: Path) -> dict:
    """Construit le registre du menu principal. Extensible."""

    MENU_REGISTRY = {
        "1": ("Paramètres Signal & Doppler",   lambda: edit_signal_params(cfg)),
        "2": ("Paramètres Canal",              lambda: edit_channel_params(cfg)),
        "3": ("Paramètres Réseau ULA",         lambda: edit_array_params(cfg)),
        "4": ("Paramètres Monte Carlo",        lambda: edit_montecarlo_params(cfg)),
        "11": ("Paramètres Scenario LEO",      lambda: edit_leo_params(cfg)),
        # ── Runners ──
        "5": ("Run — Simulation Rapide",       lambda: run_quick_sim(cfg)),
        "6": ("Run — Monte Carlo",             lambda: run_montecarlo(cfg)),
        "10": ("Run — IMM Industrial (LEO)",   lambda: run_imm_industrial_sim(cfg)),
        "9": ("Run — Pipeline Complet",        lambda: run_full_pipeline(cfg)),
        # ── Analyse ──
        "7": ("Inspecteur CRLB",               lambda: inspect_crlb(cfg)),
        # ── Système ──
        "8": ("Afficher Configuration",        lambda: display_config(cfg)),
        "s": ("Sauvegarder config.yaml",       lambda: save_config(cfg, config_path)),
        "q": ("Quitter",                       None),
    }
    return MENU_REGISTRY


def print_menu(registry: dict):
    """Affiche le menu principal stylisé."""
    console.print()
    t = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    t.add_column("Touche", style="bold cyan",   width=6)
    t.add_column("Action",  style="white",       width=42)

    separators = {
        "5": ("── Runners ──", "dim"),
        "7": ("── Analyse ──", "dim"),
        "8": ("── Système ──", "dim"),
    }

    for key, (label, _) in registry.items():
        if key in separators:
            sep_label, sep_style = separators[key]
            t.add_row("", f"[{sep_style}]{sep_label}[/]")
        style = "bold red" if key == "q" else ("bold yellow" if key == "s" else "")
        t.add_row(f"[{style}]{key}[/]" if style else key, label)

    console.print(Panel(t, title="[bold yellow]✦ Menu Principal[/]",
                        border_style="cyan", padding=(0, 1)))


# ══════════════════════════════════════════════
# Boucle principale
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SGD System — Interface Terminal (Rich)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None,
        help="Chemin vers un fichier config.yaml alternatif"
    )
    args = parser.parse_args()

    # ── Chargement de la configuration ──
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent / "config.yaml"

    try:
        cfg = load_config(str(config_path))
    except FileNotFoundError:
        console.print(f"[bold red]Erreur :[/] fichier config introuvable : {config_path}")
        sys.exit(1)

    # ── Interface ──
    console.clear()
    print_banner()
    console.print(f"  [dim]Configuration chargée : {config_path}[/]\n")
    display_config(cfg)

    registry = build_menu(cfg, config_path)

    while True:
        print_menu(registry)
        choice = Prompt.ask("\n  [bold cyan]Votre choix[/]",
                             choices=list(registry.keys()),
                             show_choices=False).strip().lower()

        if choice == "q":
            console.print("\n  [bold cyan]Au revoir — SYNAPTIC Lab.[/]\n")
            break

        label, action = registry[choice]
        if action is None:
            break

        try:
            action()
        except KeyboardInterrupt:
            console.print("\n  [dim]Interruption — retour au menu.[/]")
        except Exception as exc:
            err(f"Erreur inattendue : {exc}")
            import traceback
            console.print_exception(show_locals=False)


if __name__ == "__main__":
    main()
