import numpy as np
import sys
from pathlib import Path

# Add project modules
ROOT_DIR = Path("e:/SSN_lab/SGD_SYSTEM/Validation_Modèle_Signal")
sys.path.append(str(ROOT_DIR))

from signal_model import generate_signal_from_config, load_config
from metrics import detect_spectral_peaks

def debug_freq():
    cfg = load_config(str(ROOT_DIR / "config.yaml"))
    # Ensure LEO is on for high drift
    cfg["leo_scenario"]["enabled"] = True
    
    t, s, X, fd_eff, fdr_eff = generate_signal_from_config(cfg)
    
    # Peak detection (noiseless)
    peaks = detect_spectral_peaks(X[0], cfg["signal"]["fs"], n_peaks=1)
    
    f0 = cfg["signal"]["frequencies"][0]
    duration = cfg["signal"]["duration"]
    
    f_expected_start = f0 + fd_eff
    f_expected_mid = f0 + fd_eff + 0.5 * fdr_eff * duration
    
    print(f"--- Freq Debug (Noiseless) ---")
    print(f"f0: {f0} Hz")
    print(f"fd_eff: {fd_eff:.4f} Hz")
    print(f"fdr_eff: {fdr_eff:.4f} Hz/s")
    print(f"Peak detected: {peaks[0]:.4f} Hz")
    print(f"Mid-freq expected: {f_expected_mid:.4f} Hz")
    print(f"Error vs Mid: {peaks[0] - f_expected_mid:.4f} Hz")
    print(f"Error vs Start: {peaks[0] - f_expected_start:.4f} Hz")

if __name__ == "__main__":
    debug_freq()
