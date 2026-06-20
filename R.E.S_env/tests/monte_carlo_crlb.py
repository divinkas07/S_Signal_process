"""
Monte Carlo Simulation for CRLB Validation and Pipeline Performance.
Evaluates the signal processing pipeline across various SNR levels.
"""

import sys
import os
import argparse

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from application.pipeline_manager import PipelineManager
from simulation_engine.signal_generator import SignalGenerator
from simulation_engine.rf_channel import RFChannel


def run_monte_carlo(snr_range_db, n_runs=500, n_samples=256, use_fading=False):
    """
    Run Monte Carlo simulation across a range of SNR values.
    
    Args:
        snr_range_db: Array of SNR values to test (in dB)
        n_runs: Number of independent simulation runs per SNR point
        n_samples: Number of samples per array signal snapshot
        use_fading: Whether to enable Rician fading and multipath
        
    Returns:
        Dictionary containing aggregated metrics per SNR point
    """
    print(f"Starting Monte Carlo Simulation: SNR={snr_range_db[0]} to {snr_range_db[-1]} dB")
    print(f"Runs per point: {n_runs} | Samples/snapshot: {n_samples} | Fading: {use_fading}")
    
    sg = SignalGenerator()
    ch = RFChannel()
    
    # We use the full pipeline.
    pm = PipelineManager({'dt': 0.01, 'n_sources': 1})
    pipe = pm.build_default_pipeline()
    
    results = {
        'snr_db': snr_range_db.tolist(),
        'rmse_music': [],
        'rmse_imm': [],
        'crlb_music': [],
        'imm_cv_prob': [],
        'spoofing_alert_rate': []
    }
    
    # Random true AOA for each run between -45 and 45 degrees
    true_aoas = np.random.uniform(-45, 45, n_runs)
    
    for snr in tqdm(snr_range_db, desc="SNR Sweep"):
        errors_music = []
        errors_imm = []
        crlb_list = []
        imm_probs = []
        spoofing_alerts = 0
        
        for run in range(n_runs):
            true_aoa = true_aoas[run]
            
            # Reset timeline state for IMM to avoid carrying over between independent snaps
            for block in pipe.blocks:
                if hasattr(block._processor, 'reset'):
                    block._processor.reset()
                    
            # 1. Generate Signal
            bb = sg.generate_baseband(n_samples)
            arr = sg.generate_array_signal(bb, true_aoa, snr_db=snr)
            
            # (Optional) Apply RF Channel Impairments
            if use_fading:
                arr_dict = ch.apply(arr, fc=14e9, snr_db=snr)
                arr = arr_dict.get('signal_out', arr)
            
            # 2. Run Pipeline
            data = {
                'array_signal': arr,
                'snr_db': snr,
                'true_aoa_deg': true_aoa,
                'time': run * 0.01
            }
            
            result = pipe.process(data)
            
            # 3. Collect Metrics
            # MUSIC AOA
            aoa_est = result.get('aoa_estimate_deg', [])
            if aoa_est:
                est = aoa_est[0] if isinstance(aoa_est, list) else aoa_est
                errors_music.append((est - true_aoa)**2)
            
            # IMM Filtered
            imm_est = result.get('imm_filtered_aoa')
            if imm_est is not None:
                errors_imm.append((imm_est - true_aoa)**2)
                
            # CRLB
            crlb = result.get('crlb_deg')
            if crlb is not None and crlb != float("inf"):
                crlb_list.append(crlb)
            
            # IMM & Spoofing internal metrics
            pm_metrics = result.get('pipeline_metrics', {})
            imm_metrics = pm_metrics.get('IMM_Tracker', {})
            imm_probs.append(imm_metrics.get('cv_prob', 0.5))
            
            if result.get('spoofing_alert', False):
                spoofing_alerts += 1

        # Aggregate per SNR point
        rmse_m = np.sqrt(np.mean(errors_music)) if errors_music else float('nan')
        rmse_i = np.sqrt(np.mean(errors_imm)) if errors_imm else float('nan')
        avg_crlb = np.mean(crlb_list) if crlb_list else float('nan')
        avg_imm_prob = np.mean(imm_probs) if imm_probs else 0.5
        alert_rate = spoofing_alerts / max(n_runs, 1)
        
        results['rmse_music'].append(float(rmse_m))
        results['rmse_imm'].append(float(rmse_i))
        results['crlb_music'].append(float(avg_crlb))
        results['imm_cv_prob'].append(float(avg_imm_prob))
        results['spoofing_alert_rate'].append(float(alert_rate))
        
    return results

def plot_results(results, output_dir="tests/reports"):
    """Generate graphical reports from Monte Carlo results."""
    os.makedirs(output_dir, exist_ok=True)
    
    snr = np.array(results['snr_db'])
    rmse_m = np.array(results['rmse_music'])
    rmse_i = np.array(results['rmse_imm'])
    crlb = np.array(results['crlb_music'])
    
    # Plot 1: RMSE vs CRLB
    plt.figure(figsize=(10, 6), facecolor='white')
    
    # Handle NaN values elegantly
    valid_m = ~np.isnan(rmse_m)
    if np.any(valid_m):
        plt.semilogy(snr[valid_m], rmse_m[valid_m], 'o-', label='MUSIC RMSE', linewidth=2, color='#ff6644')
    
    valid_i = ~np.isnan(rmse_i)
    if np.any(valid_i):
        plt.semilogy(snr[valid_i], rmse_i[valid_i], 's-', label='IMM Filtered RMSE', linewidth=2, color='#00d4ff')
        
    valid_c = ~np.isnan(crlb)
    if np.any(valid_c):
        plt.semilogy(snr[valid_c], crlb[valid_c], 'k--', label='Theoretical CRLB', linewidth=2)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    plt.ylabel('RMSE (Degrees)', fontsize=12)
    plt.title('AOA Estimation Performance: RMSE vs CRLB', fontsize=14, pad=15)
    plt.legend(fontsize=11)
    
    filepath_rmse = os.path.join(output_dir, "rmse_vs_crlb.png")
    plt.tight_layout()
    plt.savefig(filepath_rmse, dpi=300)
    plt.close()
    print(f"Plot saved: {filepath_rmse}")
    
    # Plot 2: IMM Confidence & Spoofing Alerts
    plt.figure(figsize=(10, 6), facecolor='white')
    plt.plot(snr, results['imm_cv_prob'], 'g^-', label='IMM Constant Velocity Prob', linewidth=2)
    plt.plot(snr, results['spoofing_alert_rate'], 'r.-', label='False Spoofing Alert Rate', linewidth=2)
    
    plt.grid(True, ls="--", alpha=0.5)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Probability / Rate', fontsize=12)
    plt.title('Algorithm Stability vs SNR', fontsize=14, pad=15)
    plt.legend(fontsize=11)
    
    filepath_conf = os.path.join(output_dir, "stability_metrics.png")
    plt.tight_layout()
    plt.savefig(filepath_conf, dpi=300)
    plt.close()
    print(f"Plot saved: {filepath_conf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo CRLB Validation")
    parser.add_argument("--runs", type=int, default=500, help="Number of runs per SNR point")
    parser.add_argument("--snr-min", type=float, default=-10.0, help="Minimum SNR (dB)")
    parser.add_argument("--snr-max", type=float, default=30.0, help="Maximum SNR (dB)")
    parser.add_argument("--snr-step", type=float, default=5.0, help="SNR step size (dB)")
    parser.add_argument("--fading", action="store_true", help="Enable channel fading")
    
    args = parser.parse_args()
    
    snr_sweep = np.arange(args.snr_min, args.snr_max + args.snr_step/2, args.snr_step)
    
    res = run_monte_carlo(snr_sweep, n_runs=args.runs, use_fading=args.fading)
    plot_results(res)
    
    print("\nMonte Carlo Validation Complete.")
