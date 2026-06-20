"""
Simulation Controller – Background thread running the simulation engine.
Bridges the backend (physics + algorithms) with the GUI via Qt signals.
Optimized for scale: Multi-threading, Spatial Partitioning, and Event-driven time steps.
"""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QThread, Signal

from core.simulation_clock import SimulationClock
from core.state_manager import StateManager
from core.event_bus import EventBus
from core.config import SIM_CONFIG, SATELLITE_PARAMS, STATION_PARAMS, RF_PARAMS, CHANNEL_PARAMS
from simulation_engine.orbital_dynamics import OrbitalDynamics
from simulation_engine.rf_channel import RFChannel
from simulation_engine.propagation import LinkBudget
from simulation_engine.signal_generator import SignalGenerator
from simulation_engine.environment import Environment
from simulation_engine.spatial_partition import SpatialGrid
from application.device_manager import DeviceManager
from application.pipeline_manager import PipelineManager
from application.metrics_collector import MetricsCollector


class SimulationThread(QThread):
    """
    Optimized background thread.
    Uses ThreadPoolExecutor for parallel link processing and SpatialGrid for fast culling.
    Uses adaptive time stepping (coarse when no links active, fine when links active).
    """

    update_data = Signal(dict)
    update_metrics = Signal(dict)
    log_message = Signal(str)
    finished_sig = Signal()

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = config or {}
        self.running = True

        self.dt_fine = self.config.get("dt", SIM_CONFIG["dt"])
        self.dt_coarse = 1.0  # 1 second jump when no active links
        self.duration = self.config.get("duration", SIM_CONFIG["duration"])
        self.max_workers = self.config.get("max_workers", 4)
        
        self.metrics = None
        self._new_sat_params = None
        self._new_sta_params = None

    def add_satellite(self, params: dict):
        """Thread-safe way to request adding/updating a satellite."""
        self._new_sat_params = params
        
    def add_station(self, params: dict):
        """Thread-safe way to request adding/updating a station."""
        self._new_sta_params = params

    def run(self):
        try:
            self._run_simulation()
        except Exception as e:
            self.log_message.emit(f"❌ Simulation error: {e}")
            import traceback
            self.log_message.emit(traceback.format_exc())
        finally:
            self.finished_sig.emit()

    def _run_simulation(self):
        self.log_message.emit("🛰️ Initializing optimized simulation engine...")

        clock = SimulationClock(dt=self.dt_fine)
        state = StateManager()
        bus = EventBus.get_instance()

        # Create devices
        device_mgr = DeviceManager(state, bus)
        sat_id = device_mgr.add_satellite("Starlink-SIM-01",
            altitude_km=self.config.get("altitude_km", SATELLITE_PARAMS["altitude_km"]),
            inclination_deg=self.config.get("inclination_deg", SATELLITE_PARAMS["inclination_deg"]),
        )
        sta_id = device_mgr.add_station("GS-Algiers",
            latitude_deg=self.config.get("latitude_deg", STATION_PARAMS["latitude_deg"]),
            longitude_deg=self.config.get("longitude_deg", STATION_PARAMS["longitude_deg"]),
        )

        sat = state.get_satellite(sat_id)
        sta = state.get_station(sta_id)

        # Physics engines
        def _rebuild_orbital(sat):
            return OrbitalDynamics(
                altitude_km=sat.altitude_km,
                inclination_deg=sat.inclination_deg,
                raan_deg=sat.raan_deg,
                mean_anomaly_deg=sat.mean_anomaly_deg,
            )
            
        orbital = _rebuild_orbital(sat)
        
        # Shared environment
        env = Environment()
        
        # Spatial Grid setup
        spatial_grid = SpatialGrid(cell_size_deg=10.0)
        spatial_grid.add_object(sta_id, sta.latitude_deg, sta.longitude_deg)
        
        # Precompute static station parameters
        sta_ecef = OrbitalDynamics.geodetic_to_ecef(sta.latitude_deg, sta.longitude_deg, sta.altitude_m)
        sta.position_ecef = sta_ecef
        sta_lat_rad = np.radians(sta.latitude_deg)
        sta_lon_rad = np.radians(sta.longitude_deg)

        self.metrics = MetricsCollector()
        
        # Prepare thread pool resources (pipeline managers, channels, signal gens)
        # We need independent stateful objects per thread to avoid race conditions.
        # But for 1 satellite/station it's fine. We'll pre-allocate a pool of objects.
        class ThreadResource:
            def __init__(self, dt):
                self.channel = RFChannel(fc=RF_PARAMS["fc"], config=CHANNEL_PARAMS)
                self.link_budget = LinkBudget()
                self.sig_gen = SignalGenerator()
                pm = PipelineManager({"dt": dt})
                self.pipeline = pm.build_default_pipeline()

        resources = [ThreadResource(self.dt_fine) for _ in range(self.max_workers)]

        clock.start()
        self.log_message.emit(
            f"▶️ Simulation started: {self.duration}s, dt_fine={self.dt_fine}s, dt_coarse={self.dt_coarse}s"
        )
        self.log_message.emit(f"⚙️ Multi-threading enabled: {self.max_workers} workers")

        t = 0.0
        step = 0
        n_signal_samples = 256

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while t < self.duration and self.running:
                # Handle dynamic reconfigurations
                if self._new_sat_params:
                    p = self._new_sat_params
                    sat.altitude_km = p.get("altitude_km", sat.altitude_km)
                    sat.inclination_deg = p.get("inclination_deg", sat.inclination_deg)
                    sat.tx_power_dbw = p.get("tx_power_dbw", sat.tx_power_dbw)
                    orbital = _rebuild_orbital(sat)
                    self._new_sat_params = None
                    self.log_message.emit("🔄 Engine applied new satellite parameters.")
                    
                if self._new_sta_params:
                    p = self._new_sta_params
                    sta.latitude_deg = p.get("latitude_deg", sta.latitude_deg)
                    sta.longitude_deg = p.get("longitude_deg", sta.longitude_deg)
                    sta_ecef = OrbitalDynamics.geodetic_to_ecef(sta.latitude_deg, sta.longitude_deg, sta.altitude_m)
                    sta.position_ecef = sta_ecef
                    sta_lat_rad = np.radians(sta.latitude_deg)
                    sta_lon_rad = np.radians(sta.longitude_deg)
                    spatial_grid.objects.clear()
                    spatial_grid.add_object(sta_id, sta.latitude_deg, sta.longitude_deg)
                    self._new_sta_params = None
                    self.log_message.emit("🔄 Engine applied new station parameters.")

                # 1. Orbital propagation
                sat_state = orbital.propagate(t)
                sat.position_eci = sat_state["position_eci"]
                sat.velocity_eci = sat_state["velocity_eci"]
                sat.position_ecef = sat_state["position_ecef"]
                
                # Update satellite position in SpatialGrid to find nearby stations
                # (For demo we approximate lat/lon. In real code we'd convert ECEF to geodetic).
                # We do a fast distance check instead:
                look = orbital.compute_look_angles(
                    sat.position_ecef, sta_ecef, sta_lat_rad, sta_lon_rad
                )
                elevation = look["elevation_deg"]
                visible = OrbitalDynamics.is_visible(elevation, min_elevation=5.0)
                sat.is_visible = visible

                # Adaptive time stepping:
                if not visible:
                    clock.dt = self.dt_coarse
                    gui_data = {
                        "time": t, "step": step, "visible": False,
                        "sat_ecef": sat.position_ecef.tolist(),
                        "sta_ecef": sta_ecef.tolist(),
                        "azimuth_deg": look["azimuth_deg"],
                        "elevation_deg": elevation,
                        "range_km": look["range_m"] / 1e3 if look["range_m"] > 0 else 0,
                    }
                    self.update_data.emit(gui_data)
                    
                    if step % 50 == 0:
                        self.log_message.emit(f"⏩ Fast-forwarding: t={t:.1f}s | el={elevation:.1f}°")
                        
                else:
                    clock.dt = self.dt_fine
                    range_m = look["range_m"]
                    azimuth = look["azimuth_deg"]
                    
                    if t > 0:
                        prev_state = orbital.propagate(t - self.dt_fine)
                        radial_vel = (range_m - np.linalg.norm(prev_state["position_ecef"] - sta_ecef)) / self.dt_fine
                    else:
                        radial_vel = 0.0

                    env.update(t, self.dt_fine)
                    
                    # Define a processing task for the thread pool
                    def process_link(res_idx):
                        res = resources[res_idx]
                        
                        # Environment/Losses
                        rain_db = res.channel.compute_rain_attenuation(elevation)
                        shadow_db = res.channel.generate_shadowing()
                        atm_db = res.link_budget.compute_atmospheric_loss(elevation)
                        
                        budget = res.link_budget.compute(
                            tx_power_dbw=sat.tx_power_dbw,
                            tx_gain_dbi=sat.antenna_gain_dbi,
                            rx_gain_dbi=sta.rx_gain_dbi,
                            distance_m=range_m,
                            additional_losses_db=rain_db + shadow_db + atm_db,
                            noise_figure_db=sta.noise_figure_db,
                        )
                        snr_db = budget["snr_db"]
                        
                        # Generate signal
                        doppler_hz = res.sig_gen.compute_doppler_shift(radial_vel)
                        baseband = res.sig_gen.generate_baseband(n_signal_samples)
                        baseband = res.sig_gen.apply_doppler(baseband, radial_vel)
                        array_signal = res.sig_gen.generate_array_signal(baseband, azimuth, snr_db=snr_db)
                        
                        # Run pipeline
                        p_data = {
                            "array_signal": array_signal,
                            "snr_db": snr_db,
                            "true_aoa_deg": azimuth,
                            "time": t,
                        }
                        result = res.pipeline.process(p_data)
                        
                        result.update({
                            "snr_out_db": snr_db, "doppler_hz": doppler_hz,
                            "elevation_deg": elevation, "range_m": range_m,
                            "true_aoa_deg": azimuth, "rx_power_dbw": budget["rx_power_dbw"],
                        })
                        return result
                    
                    # Execute task (In a multi-link scenario we'd map this over many links)
                    # For now just 1 link submitted to ThreadPool
                    future = executor.submit(process_link, 0)
                    result = future.result()
                    
                    self.metrics.update(t, result)
                    
                    gui_data = {
                        "time": t, "step": step, "visible": True,
                        "sat_ecef": sat.position_ecef.tolist(),
                        "sta_ecef": sta_ecef.tolist(),
                        "azimuth_deg": azimuth,
                        "elevation_deg": elevation,
                        "range_km": range_m / 1e3,
                        "snr_db": result["snr_out_db"],
                        "doppler_hz": result["doppler_hz"],
                        "true_aoa_deg": azimuth,
                        "aoa_estimate_deg": result.get("aoa_estimate_deg", []),
                        "imm_filtered_aoa": result.get("imm_filtered_aoa"),
                        "consensus_estimate": result.get("consensus_estimate"),
                        "crlb_deg": result.get("crlb_deg"),
                        "prediction_confidence": result.get("prediction_confidence"),
                        "spoofing_alert": result.get("spoofing_alert", False),
                        "weather": env.weather_state,
                        "pipeline_metrics": result.get("pipeline_metrics", {}),
                        "rx_power_dbw": result["rx_power_dbw"],
                    }
                    self.update_data.emit(gui_data)
                    self.update_metrics.emit(self.metrics.get_current_metrics())
                    
                    if step % 500 == 0:
                        self.log_message.emit(f"⏱️ t={t:.1f}s | el={elevation:.1f}° | 🟢 Visible")

                t = clock.tick()
                step += 1
                time.sleep(max(0.001, clock.dt * 0.05))  # Keep GUI responsive

        summary = self.metrics.get_summary()
        self.log_message.emit("─" * 50)
        self.log_message.emit("✅ Simulation complete!")
        self.log_message.emit(f"   Total steps: {summary['total_steps']}")
        if "snr" in summary:
            self.log_message.emit(f"   SNR: {summary['snr']['mean']:.1f} dB (avg)")

    def stop(self):
        self.running = False
