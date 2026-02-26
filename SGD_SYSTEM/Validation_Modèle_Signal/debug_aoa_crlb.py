import numpy as np
import sys
from pathlib import Path

# Add project modules
ROOT_DIR = Path("e:/SSN_lab/SGD_SYSTEM/Validation_Modèle_Signal")
sys.path.append(str(ROOT_DIR))

from crlb import crlb_aoa
from estimator_music import estimate_aoa_music
from channel_model import apply_channel, signal_power
from signal_model import generate_signal_from_config, load_config

def debug_aoa_sweep():
    cfg = load_config(str(ROOT_DIR / "config.yaml"))
    
    # Use an OFF-GRID angle to avoid SNR artifacts at high SNR
    aoa_true = 32.3456 
    cfg["array"]["aoa_deg"] = aoa_true
    
    sig = cfg["signal"]
    arr = cfg["array"]
    N = int(sig["fs"] * sig["duration"])
    
    print(f"--- AOA SNR Sweep (N = {N}, M = {arr['n_elements']}) ---")
    print(f"{'SNR(dB)':>7} | {'RMSE':>10} | {'CRLB':>10} | {'Ratio':>8} | {'Bias':>10} | {'StdDev':>10}")
    print("-" * 75)
    
    for snr_db in [-10, 0, 10, 20, 30, 40]:
        cfg["channel"]["snr_db"] = snr_db
        val_crlb = crlb_aoa(arr["n_elements"], arr["d_lambda"], snr_db, arr["aoa_deg"], n_snapshots=N)
        
        errors = []
        n_runs = 50
        for _ in range(n_runs):
            t, s, X = generate_signal_from_config(cfg)
            X_noisy, _ = apply_channel(X, cfg)
            est = estimate_aoa_music(X_noisy, n_sources=1, d_lambda=arr["d_lambda"])
            if not np.isnan(est[0]):
                errors.append(est[0] - arr["aoa_deg"])
        
        err_arr = np.array(errors)
        rmse = np.sqrt(np.mean(err_arr**2))
        bias = np.mean(err_arr)
        std_dev = np.std(err_arr)
        ratio = rmse / val_crlb
        print(f"{snr_db:7d} | {rmse:10.6f} | {val_crlb:10.6f} | {ratio:8.4f} | {bias:10.6f} | {std_dev:10.6f}")

if __name__ == "__main__":
    debug_aoa_sweep()
