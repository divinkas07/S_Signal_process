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
# Modules SGD
# ──────────────────────────────────────────────
from signal_model import load_config, generate_signal_from_config, generate_time_vector, generate_multitone
from channel_model import apply_channel, signal_power
from estimator_music import estimate_aoa_music, music_spectrum
from crlb import crlb_aoa, crlb_frequency, compute_crlb_from_config
from metrics import rmse, mae

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

    console.print(Columns([t_sig, t_ch, t_arr, t_mc], equal=False, expand=False))


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
    console.print(f"  Fréquences : [cyan]{sig['frequencies']} Hz[/]\n")

    try:
        with console.status("[bold cyan]Génération du signal…[/]", spinner="dots"):
            t, s, X = generate_signal_from_config(cfg)

        with console.status("[bold cyan]Application du canal…[/]", spinner="dots"):
            X_noisy, noise = apply_channel(X, cfg)
            p_sig   = signal_power(X)
            p_noise = signal_power(noise)
            snr_meas = 10 * np.log10(p_sig / p_noise) if p_noise > 0 else np.inf

        with console.status("[bold cyan]Estimation MUSIC…[/]", spinner="dots"):
            aoa_est = estimate_aoa_music(X_noisy, n_sources=1, d_lambda=arr["d_lambda"])
            aoa_true = arr["aoa_deg"]

        # CRLB
        N = len(t)
        crlb_val = crlb_aoa(arr["n_elements"], arr["d_lambda"],
                             ch["snr_db"], aoa_true, n_snapshots=N)

    except Exception as e:
        err(f"Erreur durant la simulation : {e}")
        return

    # ── Tableau résultats ──
    t_res = Table(title="Résultats — Simulation Rapide", box=box.ROUNDED,
                  show_header=True, header_style="bold magenta")
    t_res.add_column("Grandeur",          style="cyan",  width=28)
    t_res.add_column("Valeur",            style="white", width=20)
    t_res.add_column("Unité",             style="dim",   width=12)

    t_res.add_row("Échantillons N",       str(N),              "pts")
    t_res.add_row("P_signal (array)",     fmt_float(p_sig),    "lin")
    t_res.add_row("P_bruit",              fmt_float(p_noise),  "lin")
    t_res.add_row("SNR mesuré",           fmt_float(snr_meas), "dB")
    t_res.add_row("AOA vrai",             fmt_float(aoa_true, 2), "°")

    if not np.all(np.isnan(aoa_est)):
        aoa_val = aoa_est[0]
        err_val = abs(aoa_val - aoa_true)
        color = "green" if err_val < crlb_val * 2 else "red"
        t_res.add_row("AOA estimé (MUSIC)",  f"[{color}]{fmt_float(aoa_val, 4)}[/]", "°")
        t_res.add_row("Erreur |Δ AOA|",      f"[{color}]{fmt_float(err_val, 4)}[/]", "°")
    else:
        t_res.add_row("AOA estimé (MUSIC)",  "[red]NaN — pas de pic détecté[/]", "°")
        t_res.add_row("Erreur |Δ AOA|",      "—", "°")

    t_res.add_section()
    t_res.add_row("CRLB AOA (√Var)",      fmt_float(crlb_val, 4), "°")

    # Verdict
    if not np.all(np.isnan(aoa_est)):
        go = aoa_est[0] is not None and abs(aoa_est[0] - aoa_true) < crlb_val * 3
        verdict = "[bold green]✔ Estimateur cohérent[/]" if go else "[bold yellow]⚠ Erreur > 3×CRLB[/]"
    else:
        verdict = "[bold red]✖ Estimation échouée[/]"

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

    snr_values = mc["snr_range"]

    # Quick mode ?
    quick = Confirm.ask("  Mode rapide (10 runs/SNR pour test) ?", default=False)
    n_runs = 10 if quick else mc["n_runs"]

    total = len(snr_values) * n_runs

    console.print(f"\n  SNR range : [cyan]{snr_values}[/]")
    console.print(f"  Runs/SNR  : [cyan]{n_runs}[/]")
    console.print(f"  Total     : [cyan]{total} runs[/]\n")

    results = {
        "snr_values": snr_values,
        "aoa_rmse":   [],
        "aoa_crlb":   [],
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

            aoa_ests = []

            for _ in range(n_runs):
                t, s, X = generate_signal_from_config(cfg_run)
                X_noisy, _ = apply_channel(X, cfg_run)
                est = estimate_aoa_music(X_noisy, n_sources=1,
                                         d_lambda=arr["d_lambda"])
                aoa_ests.append(est[0])
                progress.advance(overall)
                progress.advance(snr_task)

            aoa_arr = np.array(aoa_ests, dtype=float)
            valid = ~np.isnan(aoa_arr)
            aoa_true = arr["aoa_deg"]

            rmse_val = rmse(aoa_arr[valid], np.full(np.sum(valid), aoa_true)) \
                       if np.sum(valid) > 0 else np.inf

            crlb_val = crlb_aoa(arr["n_elements"], arr["d_lambda"],
                                  snr_db, aoa_true, n_snapshots=N)

            go = rmse_val >= crlb_val * 0.5
            results["aoa_rmse"].append(rmse_val)
            results["aoa_crlb"].append(crlb_val)
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

    for snr_db, rmse_v, crlb_v, go in zip(
        results["snr_values"], results["aoa_rmse"],
        results["aoa_crlb"], results["go_flags"]
    ):
        ratio = rmse_v / crlb_v if crlb_v > 0 and rmse_v != np.inf else float("nan")
        go_str = "[bold green]✔[/]" if go else "[bold red]✖[/]"
        ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "—"
        t_mc.add_row(
            f"{snr_db:+3d}",
            fmt_float(rmse_v, 6),
            fmt_float(crlb_v, 6),
            ratio_str,
            go_str,
        )

    console.print(t_mc)

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
    N   = int(sig["fs"] * sig["duration"])

    snr_range = mc["snr_range"]

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
            c = crlb_frequency(sig["fs"], N, snr_db, a)
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
# Sauvegarde de la config
# ══════════════════════════════════════════════

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
        # ── Runners ──
        "5": ("Run — Simulation Rapide",       lambda: run_quick_sim(cfg)),
        "6": ("Run — Monte Carlo",             lambda: run_montecarlo(cfg)),
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
    t = Table(box=box.SIMPLE_ROUNDED, show_header=False, padding=(0, 2))
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
