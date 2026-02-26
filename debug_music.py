import numpy as np
import sys
from pathlib import Path

# Add project dir to path
sys.path.append(str(Path("e:/SSN_lab/SGD_SYSTEM/Validation_Modèle_Signal")))

from signal_model import load_config, generate_signal_from_config
from channel_model import apply_channel
from estimator_music import music_spectrum, estimate_aoa_music
import matplotlib.pyplot as plt

def debug_music():
    cfg = load_config("e:/SSN_lab/SGD_SYSTEM/Validation_Modèle_Signal/config.yaml")
    cfg["channel"]["snr_db"] = 20
    
    t, s, X, fd, fdr = generate_signal_from_config(cfg)
    X_noisy, _ = apply_channel(X, cfg)
    
    print(f"Signal Multi-ton: fd={fd:.2f}, fdr={fdr:.2f}")
    print(f"True AOA: {cfg['array']['aoa_deg']}")
    
    angles, spectrum = music_spectrum(X_noisy, n_sources=1, d_lambda=cfg['array']['d_lambda'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles, spectrum)
    plt.axvline(cfg['array']['aoa_deg'], color='r', linestyle='--', label='True AOA')
    plt.title("MUSIC Pseudospectrum Debug")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Pseudo-spectrum (dB)")
    plt.grid(True)
    plt.legend()
    plt.savefig("debug_music_spectrum.png")
    
    aoa_est = estimate_aoa_music(X_noisy, n_sources=1, d_lambda=cfg['array']['d_lambda'])
    print(f"Estimated AOA: {aoa_est}")
    
    from scipy.signal import find_peaks
    peaks, props = find_peaks(spectrum, height=-30)
    print(f"Peaks detected (find_peaks): {angles[peaks] if len(peaks)>0 else 'None'}")
    if len(peaks) > 0:
        print(f"Peak heights: {spectrum[peaks]}")

if __name__ == "__main__":
    debug_music()
