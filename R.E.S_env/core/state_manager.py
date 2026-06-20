"""
State Manager – Centralized world state for all simulation objects.
Holds satellites, ground stations, links, and provides query methods.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─── Data Classes ────────────────────────────────────────────────────────

@dataclass
class SatelliteState:
    """State of a single satellite at current simulation time."""
    id: str
    name: str
    # Orbital elements
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    raan_deg: float = 0.0
    eccentricity: float = 0.0001
    arg_perigee_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    # Dynamic state (updated each tick)
    position_eci: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_eci: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_ecef: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # RF parameters
    tx_power_dbw: float = 30.0
    antenna_gain_dbi: float = 38.0
    frequency_hz: float = 14e9
    # Status
    is_visible: bool = False


@dataclass
class StationState:
    """State of a ground station or user terminal."""
    id: str
    name: str
    # Position (geodetic)
    latitude_deg: float = 36.75
    longitude_deg: float = 3.05
    altitude_m: float = 100.0
    # ECEF position (computed)
    position_ecef: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_ecef: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Mobility
    mode: str = "fixed"  # "fixed", "pedestrian", "vehicle"
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    # RF parameters
    rx_gain_dbi: float = 12.0
    noise_figure_db: float = 2.0
    n_antennas: int = 8


@dataclass
class Link:
    """A communication link between a satellite and a station."""
    id: str
    satellite_id: str
    station_id: str
    # Geometry
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    range_m: float = 0.0
    radial_velocity_mps: float = 0.0
    # RF state
    fspl_db: float = 0.0
    total_loss_db: float = 0.0
    snr_db: float = 0.0
    doppler_hz: float = 0.0
    # Status
    is_active: bool = False


# ─── State Manager ───────────────────────────────────────────────────────

class StateManager:
    """Central repository for all simulation objects."""

    def __init__(self):
        self.satellites: dict[str, SatelliteState] = {}
        self.stations: dict[str, StationState] = {}
        self.links: dict[str, Link] = {}
        self._time: float = 0.0

    # ── Satellite Management ───────────────────────────────────────────
    def add_satellite(self, sat: SatelliteState):
        self.satellites[sat.id] = sat

    def remove_satellite(self, sat_id: str):
        self.satellites.pop(sat_id, None)
        # Remove associated links
        self.links = {k: v for k, v in self.links.items() if v.satellite_id != sat_id}

    def get_satellite(self, sat_id: str) -> Optional[SatelliteState]:
        return self.satellites.get(sat_id)

    # ── Station Management ─────────────────────────────────────────────
    def add_station(self, station: StationState):
        self.stations[station.id] = station

    def remove_station(self, station_id: str):
        self.stations.pop(station_id, None)
        self.links = {k: v for k, v in self.links.items() if v.station_id != station_id}

    def get_station(self, station_id: str) -> Optional[StationState]:
        return self.stations.get(station_id)

    # ── Link Management ────────────────────────────────────────────────
    def update_link(self, link: Link):
        self.links[link.id] = link

    def get_link(self, link_id: str) -> Optional[Link]:
        return self.links.get(link_id)

    def get_active_links(self) -> list[Link]:
        return [lnk for lnk in self.links.values() if lnk.is_active]

    # ── Time ───────────────────────────────────────────────────────────
    def set_time(self, t: float):
        self._time = t

    @property
    def current_time(self) -> float:
        return self._time

    # ── Reset ──────────────────────────────────────────────────────────
    def clear(self):
        """Remove all objects and reset state."""
        self.satellites.clear()
        self.stations.clear()
        self.links.clear()
        self._time = 0.0

    def __repr__(self):
        return (f"StateManager(sats={len(self.satellites)}, "
                f"stations={len(self.stations)}, links={len(self.links)}, "
                f"t={self._time:.2f}s)")
