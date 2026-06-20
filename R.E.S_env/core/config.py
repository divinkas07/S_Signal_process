"""
Configuration settings for the LEO Ku-Band Dynamic Reality Simulator.
All physical parameters are based on realistic Ku-band LEO systems.
"""

import numpy as np

# ─── Simulation Parameters ───────────────────────────────────────────────
SIM_CONFIG = {
    "dt": 0.01,                # Simulation time step (s)
    "duration": 600,           # Total simulation duration (s)
    "speed_multiplier": 1.0,   # Time acceleration factor
}

# ─── Satellite Orbital Parameters ────────────────────────────────────────
SATELLITE_PARAMS = {
    "altitude_km": 550,        # LEO altitude (km)
    "inclination_deg": 53.0,   # Orbital inclination (degrees)
    "raan_deg": 0.0,           # Right Ascension of Ascending Node (degrees)
    "eccentricity": 0.0001,    # Near-circular orbit
    "arg_perigee_deg": 0.0,    # Argument of perigee (degrees)
    "mean_anomaly_deg": 0.0,   # Initial mean anomaly (degrees)
    "tx_power_dbw": 30.0,      # Transmit power (dBW)
    "tx_antenna_gain_dbi": 38.0,  # Satellite antenna gain (dBi)
}

# ─── RF / Ku-Band Parameters ────────────────────────────────────────────
RF_PARAMS = {
    "fc": 14e9,                # Carrier frequency (Hz) - Ku-band
    "bandwidth": 250e6,        # Bandwidth (Hz)
    "sampling_rate": 500e6,    # ADC sampling rate (Hz)
    "n_fft": 1024,             # OFDM FFT size
    "cp_len": 64,              # Cyclic prefix length
    "modulation": "QPSK",      # Modulation scheme: QPSK, 16QAM, 64QAM
    "rolloff": 0.25,           # Root Raised Cosine roll-off factor
    "noise_figure_db": 2.0,    # Receiver noise figure (dB)
    "system_temp_k": 290.0,    # System noise temperature (K)
}

# ─── Antenna Array Parameters ───────────────────────────────────────────
ANTENNA_PARAMS = {
    "n_antennas": 8,           # Number of antenna elements (ULA)
    "antenna_spacing_lambda": 0.5,  # Element spacing in wavelengths (d/λ)
    "rx_antenna_gain_dbi": 12.0,    # Receiver antenna gain (dBi)
}

# ─── Channel Parameters ─────────────────────────────────────────────────
CHANNEL_PARAMS = {
    "rician_k_db": 10.0,       # Rician K-factor (dB) - LOS dominant
    "num_multipath_taps": 3,   # Number of multipath components
    "shadowing_sigma_db": 3.0, # Log-normal shadowing std dev (dB)
    "rain_rate_mmh": 0.0,      # Rain rate (mm/h), 0 = clear sky
    "phase_noise_var": 1e-4,   # Phase noise variance (Wiener process)
}

# ─── Ground Station Default ─────────────────────────────────────────────
STATION_PARAMS = {
    "latitude_deg": 36.75,     # Station latitude (degrees)
    "longitude_deg": 3.05,     # Station longitude (degrees)
    "altitude_m": 100.0,       # Station altitude (m above sea level)
    "mode": "fixed",           # Mobility: "fixed", "pedestrian", "vehicle"
    "speed_mps": 0.0,          # Speed (m/s) for mobile modes
}

# ─── Physical Constants ─────────────────────────────────────────────────
CONSTANTS = {
    "c": 299792458.0,          # Speed of light (m/s)
    "k_boltzmann": 1.380649e-23,  # Boltzmann constant (J/K)
    "R_earth_km": 6371.0,     # Earth radius (km)
    "R_earth_m": 6371000.0,   # Earth radius (m)
    "mu_earth": 3.986004418e14,   # Earth gravitational parameter (m³/s²)
    "omega_earth": 7.2921159e-5,  # Earth rotation rate (rad/s)
}

# ─── Derived Parameters (computed at import time) ───────────────────────
def compute_derived():
    """Compute derived physical parameters."""
    c = CONSTANTS["c"]
    fc = RF_PARAMS["fc"]
    wavelength = c / fc
    d_antenna = ANTENNA_PARAMS["antenna_spacing_lambda"] * wavelength
    alt_m = SATELLITE_PARAMS["altitude_km"] * 1e3
    a = CONSTANTS["R_earth_m"] + alt_m
    T_orbit = 2 * np.pi * np.sqrt(a**3 / CONSTANTS["mu_earth"])
    max_doppler = (7500.0 / c) * fc  # ~350 kHz at 14 GHz

    return {
        "wavelength": wavelength,
        "antenna_spacing_m": d_antenna,
        "orbit_semi_major_m": a,
        "orbit_period_s": T_orbit,
        "max_doppler_hz": max_doppler,
    }

DERIVED = compute_derived()
