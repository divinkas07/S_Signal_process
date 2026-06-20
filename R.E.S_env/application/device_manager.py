"""
Device Manager – Manages satellites, ground stations, and terminals.
Provides lifecycle management (add, remove, configure) for all devices.
"""

import uuid
from core.state_manager import StateManager, SatelliteState, StationState
from core.event_bus import EventBus, Events
from core.config import SATELLITE_PARAMS, STATION_PARAMS


class DeviceManager:
    """
    Creates and manages simulation devices.
    Connects to StateManager for storage and EventBus for notifications.
    """

    def __init__(self, state_manager: StateManager, event_bus: EventBus = None):
        self.state = state_manager
        self.bus = event_bus or EventBus.get_instance()

    # ── Satellite Management ───────────────────────────────────────────

    def add_satellite(self, name: str = None, **kwargs) -> str:
        """
        Add a satellite to the simulation.
        
        Args:
            name: Satellite name (auto-generated if None)
            **kwargs: Override default SATELLITE_PARAMS
            
        Returns:
            Satellite ID
        """
        sat_id = f"sat_{uuid.uuid4().hex[:8]}"
        params = SATELLITE_PARAMS.copy()
        params.update(kwargs)

        sat = SatelliteState(
            id=sat_id,
            name=name or f"LEO-{len(self.state.satellites) + 1}",
            altitude_km=params.get("altitude_km", 550),
            inclination_deg=params.get("inclination_deg", 53.0),
            raan_deg=params.get("raan_deg", 0.0),
            eccentricity=params.get("eccentricity", 0.0001),
            arg_perigee_deg=params.get("arg_perigee_deg", 0.0),
            mean_anomaly_deg=params.get("mean_anomaly_deg", 0.0),
            tx_power_dbw=params.get("tx_power_dbw", 30.0),
            antenna_gain_dbi=params.get("tx_antenna_gain_dbi", 38.0),
            frequency_hz=params.get("frequency_hz", 14e9),
        )

        self.state.add_satellite(sat)
        self.bus.publish(Events.DEVICE_ADDED, {"type": "satellite", "id": sat_id, "name": sat.name})
        return sat_id

    def remove_satellite(self, sat_id: str):
        """Remove a satellite from the simulation."""
        self.state.remove_satellite(sat_id)
        self.bus.publish(Events.DEVICE_REMOVED, {"type": "satellite", "id": sat_id})

    # ── Station Management ─────────────────────────────────────────────

    def add_station(self, name: str = None, **kwargs) -> str:
        """
        Add a ground station or user terminal.
        
        Args:
            name: Station name
            **kwargs: Override default STATION_PARAMS
            
        Returns:
            Station ID
        """
        sta_id = f"sta_{uuid.uuid4().hex[:8]}"
        params = STATION_PARAMS.copy()
        params.update(kwargs)

        station = StationState(
            id=sta_id,
            name=name or f"GS-{len(self.state.stations) + 1}",
            latitude_deg=params.get("latitude_deg", 36.75),
            longitude_deg=params.get("longitude_deg", 3.05),
            altitude_m=params.get("altitude_m", 100.0),
            mode=params.get("mode", "fixed"),
            speed_mps=params.get("speed_mps", 0.0),
            n_antennas=params.get("n_antennas", 8),
        )

        self.state.add_station(station)
        self.bus.publish(Events.DEVICE_ADDED, {"type": "station", "id": sta_id, "name": station.name})
        return sta_id

    def remove_station(self, sta_id: str):
        """Remove a ground station from the simulation."""
        self.state.remove_station(sta_id)
        self.bus.publish(Events.DEVICE_REMOVED, {"type": "station", "id": sta_id})

    # ── Utilities ──────────────────────────────────────────────────────

    def list_devices(self) -> dict:
        """List all devices with their IDs and names."""
        return {
            "satellites": {sid: s.name for sid, s in self.state.satellites.items()},
            "stations": {sid: s.name for sid, s in self.state.stations.items()},
        }

    def get_device_count(self) -> dict:
        return {
            "satellites": len(self.state.satellites),
            "stations": len(self.state.stations),
        }

    def clear_all(self):
        """Remove all devices."""
        self.state.clear()

    def __repr__(self):
        counts = self.get_device_count()
        return f"DeviceManager(sats={counts['satellites']}, stations={counts['stations']})"
