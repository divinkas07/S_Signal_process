"""End-to-end smoke test for the LEO simulator backend."""
import sys
sys.path.insert(0, '.')
import numpy as np

from application.pipeline_manager import PipelineManager
from simulation_engine.signal_generator import SignalGenerator
from simulation_engine.orbital_dynamics import OrbitalDynamics
from simulation_engine.rf_channel import RFChannel
from simulation_engine.propagation import LinkBudget
from application.metrics_collector import MetricsCollector

sg = SignalGenerator()
orb = OrbitalDynamics(550, 53)
ch = RFChannel()
lb = LinkBudget()
pm = PipelineManager({'dt': 0.01})
pipe = pm.build_default_pipeline()
mc = MetricsCollector()

print("Pipeline:", pipe)

t = 300
state = orb.propagate(t)
sta_ecef = orb.geodetic_to_ecef(36.75, 3.05, 100)
look = orb.compute_look_angles(
    state['position_ecef'], sta_ecef,
    np.radians(36.75), np.radians(3.05)
)
print(f"Look: el={look['elevation_deg']:.1f} az={look['azimuth_deg']:.1f} range={look['range_m']/1e3:.0f}km")

bb = sg.generate_baseband(256)
arr = sg.generate_array_signal(bb, look['azimuth_deg'], snr_db=20)

data = {
    'array_signal': arr,
    'snr_db': 20,
    'true_aoa_deg': look['azimuth_deg'],
}
result = pipe.process(data)
result['snr_out_db'] = 20
result['elevation_deg'] = look['elevation_deg']
mc.update(t, result)

print("Metrics:", mc.get_current_metrics())
print("=== E2E PIPELINE: PASS ===")
