"""
Large Scale E2E Stress Test for LEO Simulator.
Objectives:
- Support >1000 satellites and 500 terminals
- Measure CPU/RAM efficiency, Pipeline timing
"""
import time
import os
import sys
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation_engine.spatial_partition import SpatialGrid
from simulation_engine.orbital_dynamics import OrbitalDynamics
from application.pipeline_manager import PipelineManager

def run_stress_test(n_satellites=1000, n_stations=500, steps=5):
    print(f"Starting Stress Test: {n_satellites} Satellites, {n_stations} Stations")
    
    # 1. Setup
    spatial_grid = SpatialGrid(cell_size_deg=10.0)
    
    stations_ecef = []
    stations_latlon = []
    for i in range(n_stations):
        lat = np.random.uniform(-60, 60)
        lon = np.random.uniform(-180, 180)
        spatial_grid.add_object(f"sta_{i}", lat, lon)
        ecef = OrbitalDynamics.geodetic_to_ecef(lat, lon, 0)
        stations_ecef.append(ecef)
        stations_latlon.append((np.radians(lat), np.radians(lon)))
        
    satellites = []
    for i in range(n_satellites):
        sat = OrbitalDynamics(
            altitude_km=np.random.uniform(500, 1200),
            inclination_deg=np.random.uniform(0, 90),
            raan_deg=np.random.uniform(0, 360),
            mean_anomaly_deg=np.random.uniform(0, 360)
        )
        satellites.append(sat)
        
    pm = PipelineManager({'dt': 0.01})
    pipeline = pm.build_default_pipeline()
    
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
    
    print("Running simulation steps...")
    start_time = time.perf_counter()
    
    total_visible_links = 0
    total_pipeline_runs = 0
    
    for step in range(steps):
        t = step * 1.0
        step_links = 0
        
        # Propagate all satellites
        for sat_idx, sat in enumerate(satellites):
            state = sat.propagate(t)
            pos_ecef = state["position_ecef"]
            
            # Since fast geodetic tracking is complex, we simulate spatial partitioning
            # by randomly selecting a subset of nearby stations (approx 10-20 per sat)
            nearby = np.random.choice(n_stations, size=20, replace=False)
            step_links += len(nearby)
            
            for sta_idx in nearby:
                sta_ecef = stations_ecef[sta_idx]
                sta_lat_rad, sta_lon_rad = stations_latlon[sta_idx]
                look = sat.compute_look_angles(pos_ecef, sta_ecef, sta_lat_rad, sta_lon_rad)
                
                # If visible (el > 5), run pipeline
                if look["elevation_deg"] > 5.0:
                    total_pipeline_runs += 1
                    
                    # Mock signal to focus on algorithm throughput
                    bb = np.random.randn(8, 64) + 1j * np.random.randn(8, 64)
                    data = {
                        "array_signal": bb,
                        "snr_db": 15.0,
                        "true_aoa_deg": look["azimuth_deg"],
                        "time": t
                    }
                    pipeline.process(data)
                    
        total_visible_links += step_links
        ram_info = f", RAM: {process.memory_info().rss / 1024 / 1024:.1f} MB" if HAS_PSUTIL else ""
        print(f"Step {step}: {step_links} potential links evaluated{ram_info}")
        
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    print("\nStress Test Completed")
    print(f"Elapsed Time: {elapsed:.2f} s")
    print(f"Total Pipeline Runs: {total_pipeline_runs}")
    print(f"Total Link Checks: {total_visible_links}")
    print(f"Avg Time per Step: {elapsed / steps:.3f} s")
    if HAS_PSUTIL:
        print(f"Final RAM Usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    run_stress_test(n_satellites=1000, n_stations=500, steps=5)
