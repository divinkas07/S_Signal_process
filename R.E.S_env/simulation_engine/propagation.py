"""
Link Budget & Propagation Model – Computes received power, SNR,
and total link losses for a satellite-to-ground link.

    P_r = P_t + G_t + G_r − L_fs − L_atm − L_rain − L_shadow
    SNR = P_r − N_0 − 10·log10(B)
"""

import numpy as np
from core.config import CONSTANTS, RF_PARAMS, ANTENNA_PARAMS


class LinkBudget:
    """
    Complete link budget calculator for LEO Ku-band links.
    """

    def __init__(self, fc: float = None, bandwidth: float = None):
        self.fc = fc or RF_PARAMS["fc"]
        self.bandwidth = bandwidth or RF_PARAMS["bandwidth"]
        self.c = CONSTANTS["c"]
        self.k = CONSTANTS["k_boltzmann"]

    def compute(self, tx_power_dbw: float, tx_gain_dbi: float,
                rx_gain_dbi: float, distance_m: float,
                additional_losses_db: float = 0.0,
                noise_figure_db: float = 2.0,
                system_temp_k: float = 290.0) -> dict:
        """
        Compute full link budget.
        
        Args:
            tx_power_dbw: Transmit power (dBW)
            tx_gain_dbi: Transmitter antenna gain (dBi)
            rx_gain_dbi: Receiver antenna gain (dBi)
            distance_m: Slant range (m)
            additional_losses_db: Sum of rain + shadow + atmospheric losses
            noise_figure_db: Receiver noise figure (dB)
            system_temp_k: System noise temperature (K)
            
        Returns:
            dict with eirp_dbw, fspl_db, rx_power_dbw, noise_floor_dbw, snr_db, etc.
        """
        # EIRP
        eirp_dbw = tx_power_dbw + tx_gain_dbi

        # Free-space path loss
        fspl_db = self._compute_fspl(distance_m)

        # Received power
        rx_power_dbw = eirp_dbw + rx_gain_dbi - fspl_db - additional_losses_db

        # Noise floor: N = k·T·B (in dBW)
        noise_power_w = self.k * system_temp_k * self.bandwidth
        noise_floor_dbw = 10 * np.log10(noise_power_w) + noise_figure_db

        # C/N
        snr_db = rx_power_dbw - noise_floor_dbw

        # G/T (receiver figure of merit)
        g_over_t = rx_gain_dbi - 10 * np.log10(system_temp_k)

        return {
            "eirp_dbw": eirp_dbw,
            "fspl_db": fspl_db,
            "additional_losses_db": additional_losses_db,
            "rx_power_dbw": rx_power_dbw,
            "noise_floor_dbw": noise_floor_dbw,
            "snr_db": snr_db,
            "g_over_t": g_over_t,
            "distance_km": distance_m / 1e3,
        }

    def compute_atmospheric_loss(self, elevation_deg: float, frequency_ghz: float = None) -> float:
        """
        Atmospheric gaseous attenuation (simplified model).
        Based on ITU-R P.676 approximation for Ku-band.
        
        Higher attenuation at low elevations due to longer path through atmosphere.
        """
        f = frequency_ghz or (self.fc / 1e9)
        elev = max(elevation_deg, 5.0)
        
        # Zenith attenuation at sea level (Ku-band ~0.04-0.15 dB)
        if f < 15:
            a_zenith = 0.04 + 0.003 * (f - 10)
        else:
            a_zenith = 0.06 + 0.01 * (f - 15)

        # Scale by path elevation (1/sin approximation)
        return a_zenith / np.sin(np.radians(elev))

    def _compute_fspl(self, distance_m: float) -> float:
        """Free-space path loss in dB."""
        if distance_m <= 0:
            return 0.0
        d_km = distance_m / 1e3
        f_mhz = self.fc / 1e6
        return 20 * np.log10(d_km) + 20 * np.log10(f_mhz) + 32.44

    @staticmethod
    def db_to_linear(db: float) -> float:
        return 10 ** (db / 10)

    @staticmethod
    def linear_to_db(linear: float) -> float:
        if linear <= 0:
            return -np.inf
        return 10 * np.log10(linear)
