"""
Orbital Dynamics – Keplerian 2-body orbital propagator.
Computes satellite position/velocity from orbital elements,
performs ECI→ECEF conversion, and calculates look angles.
"""

import numpy as np
from core.config import CONSTANTS


class OrbitalDynamics:
    """
    Keplerian 2-body orbital propagator.
    
    Given orbital elements (a, e, i, Ω, ω, M₀), computes:
    - ECI position/velocity at time t
    - ECEF position via GMST rotation
    - Look angles (azimuth, elevation, range) from a ground station
    """

    def __init__(self, altitude_km: float = 550, inclination_deg: float = 53.0,
                 raan_deg: float = 0.0, eccentricity: float = 0.0001,
                 arg_perigee_deg: float = 0.0, mean_anomaly_deg: float = 0.0):
        
        self.mu = CONSTANTS["mu_earth"]
        self.R_earth = CONSTANTS["R_earth_m"]
        self.omega_earth = CONSTANTS["omega_earth"]

        # Orbital elements
        self.a = self.R_earth + altitude_km * 1e3  # Semi-major axis (m)
        self.e = eccentricity
        self.i = np.radians(inclination_deg)
        self.raan = np.radians(raan_deg)
        self.omega = np.radians(arg_perigee_deg)
        self.M0 = np.radians(mean_anomaly_deg)

        # Derived
        self.n = np.sqrt(self.mu / self.a**3)  # Mean motion (rad/s)
        self.T = 2 * np.pi / self.n             # Orbital period (s)

    # ── Propagation ────────────────────────────────────────────────────

    def propagate(self, t: float) -> dict:
        """
        Compute satellite state at time t.
        
        Returns:
            dict with keys: position_eci, velocity_eci, position_ecef
        """
        # Mean anomaly at time t
        M = self.M0 + self.n * t

        # Solve Kepler's equation: M = E - e*sin(E)
        E = self._solve_kepler(M, self.e)

        # True anomaly
        nu = 2 * np.arctan2(
            np.sqrt(1 + self.e) * np.sin(E / 2),
            np.sqrt(1 - self.e) * np.cos(E / 2)
        )

        # Distance from focus
        r = self.a * (1 - self.e * np.cos(E))

        # Position in orbital plane (perifocal coordinates)
        x_pf = r * np.cos(nu)
        y_pf = r * np.sin(nu)

        # Velocity in orbital plane
        h = np.sqrt(self.mu * self.a * (1 - self.e**2))
        vx_pf = -(self.mu / h) * np.sin(nu)
        vy_pf = (self.mu / h) * (self.e + np.cos(nu))

        # Rotation matrix: perifocal → ECI
        R_pf_to_eci = self._perifocal_to_eci_matrix()

        pos_eci = R_pf_to_eci @ np.array([x_pf, y_pf, 0.0])
        vel_eci = R_pf_to_eci @ np.array([vx_pf, vy_pf, 0.0])

        # ECI → ECEF rotation
        pos_ecef = self._eci_to_ecef(pos_eci, t)

        return {
            "position_eci": pos_eci,
            "velocity_eci": vel_eci,
            "position_ecef": pos_ecef,
        }

    # ── Look Angles ────────────────────────────────────────────────────

    def compute_look_angles(self, sat_ecef: np.ndarray, station_ecef: np.ndarray,
                            station_lat: float, station_lon: float) -> dict:
        """
        Compute azimuth, elevation, range, and radial velocity
        from a ground station to a satellite.

        Args:
            sat_ecef: Satellite ECEF position (m)
            station_ecef: Station ECEF position (m)
            station_lat: Station latitude (radians)
            station_lon: Station longitude (radians)

        Returns:
            dict: azimuth_deg, elevation_deg, range_m, radial_velocity_mps
        """
        # Range vector in ECEF
        delta = sat_ecef - station_ecef
        range_m = np.linalg.norm(delta)

        if range_m < 1.0:
            return {"azimuth_deg": 0, "elevation_deg": 90, "range_m": 0, "radial_velocity_mps": 0}

        # ECEF → ENU rotation
        R_ecef_to_enu = self._ecef_to_enu_matrix(station_lat, station_lon)
        delta_enu = R_ecef_to_enu @ delta

        e, n, u = delta_enu

        # Azimuth (from North, clockwise)
        azimuth = np.arctan2(e, n)
        if azimuth < 0:
            azimuth += 2 * np.pi

        # Elevation
        elevation = np.arctan2(u, np.sqrt(e**2 + n**2))

        return {
            "azimuth_deg": np.degrees(azimuth),
            "elevation_deg": np.degrees(elevation),
            "range_m": range_m,
        }

    # ── Visibility ─────────────────────────────────────────────────────

    @staticmethod
    def is_visible(elevation_deg: float, min_elevation: float = 5.0) -> bool:
        """Check if satellite is above minimum elevation angle."""
        return elevation_deg >= min_elevation

    # ── Geodetic Utilities ─────────────────────────────────────────────

    @staticmethod
    def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        """Convert geodetic coordinates (lat, lon, alt) to ECEF."""
        R = CONSTANTS["R_earth_m"]
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        
        # Simplified spherical Earth model
        r = R + alt_m
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        return np.array([x, y, z])

    # ── Private Methods ────────────────────────────────────────────────

    @staticmethod
    def _solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
        """Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson."""
        E = M  # Initial guess
        for _ in range(max_iter):
            dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            E -= dE
            if abs(dE) < tol:
                break
        return E

    def _perifocal_to_eci_matrix(self) -> np.ndarray:
        """Rotation matrix from perifocal to ECI frame."""
        cos_o = np.cos(self.omega)
        sin_o = np.sin(self.omega)
        cos_O = np.cos(self.raan)
        sin_O = np.sin(self.raan)
        cos_i = np.cos(self.i)
        sin_i = np.sin(self.i)

        R = np.array([
            [cos_O*cos_o - sin_O*sin_o*cos_i, -cos_O*sin_o - sin_O*cos_o*cos_i,  sin_O*sin_i],
            [sin_O*cos_o + cos_O*sin_o*cos_i, -sin_O*sin_o + cos_O*cos_o*cos_i, -cos_O*sin_i],
            [sin_o*sin_i,                       cos_o*sin_i,                        cos_i      ],
        ])
        return R

    def _eci_to_ecef(self, pos_eci: np.ndarray, t: float) -> np.ndarray:
        """Rotate ECI position to ECEF using GMST."""
        theta = self.omega_earth * t  # Simplified GMST
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([
            [ cos_t, sin_t, 0],
            [-sin_t, cos_t, 0],
            [     0,     0, 1],
        ])
        return R @ pos_eci

    @staticmethod
    def _ecef_to_enu_matrix(lat: float, lon: float) -> np.ndarray:
        """Rotation matrix from ECEF to local ENU frame."""
        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        cos_lon = np.cos(lon)
        sin_lon = np.sin(lon)

        return np.array([
            [-sin_lon,           cos_lon,          0       ],
            [-sin_lat*cos_lon,  -sin_lat*sin_lon,  cos_lat ],
            [ cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat ],
        ])
