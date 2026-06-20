# 🛰️ LEO Ku-Band Dynamic Reality Simulator (RES_env)

A comprehensive **satellite signal processing and simulation framework** for Low Earth Orbit (LEO) Ku-band communication systems. This project models realistic satellite-to-ground RF channels, implements advanced signal processing algorithms (MUSIC, ICA, IMM tracking), and provides an interactive GUI for real-time scenario simulation and analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Module Documentation](#module-documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **RES_env** (Reality Engine Simulator Environment) framework simulates:

- **Orbital Dynamics**: Accurate LEO satellite trajectories (550 km altitude, 53° inclination)
- **RF Propagation**: Ku-band (14 GHz) signal generation with realistic multipath, fading, and Doppler effects
- **Signal Processing**: OFDM modulation, antenna array beamforming, and angle-of-arrival (AOA) estimation
- **Advanced Algorithms**:
  - MUSIC (Multiple Signal Classification) for AOA estimation
  - ICA (Independent Component Analysis) for source separation
  - IMM (Interacting Multiple Model) for target tracking
  - PINN (Physics-Informed Neural Networks) for prediction
  - GPS spoofing detection and mitigation
- **Interactive Visualization**: Real-time 3D rendering, signal spectrum analysis, and metrics dashboard
- **Hardware Integration**: Support for Arduino-based measurement hardware (via `pyduino`)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Realistic Physical Modeling** | LEO orbital mechanics, Rician fading, rain attenuation, phase noise |
| **Multi-Antenna Arrays** | 8-element ULA with configurable spacing and beamforming |
| **Modulation Schemes** | QPSK, 16-QAM, 64-QAM with root-raised-cosine shaping |
| **Doppler Simulation** | Up to 350 kHz Doppler shift at 14 GHz |
| **Real-Time GUI** | PySide6-based interactive dashboard with dark theme |
| **Scenario Management** | Load/save simulation configurations and run multiple scenarios |
| **Metrics Collection** | SNR, BER, Doppler offset, AOA error tracking |
| **Hardware Support** | Arduino integration for experimental validation |
| **Extensible Pipeline** | Plugin-based architecture for custom algorithms |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GUI Layer (PySide6)                          │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────────────┐  │
│  │Device Panel │ │3D Renderer   │ │Metrics Dashboard         │  │
│  └─────────────┘ └──────────────┘ └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────────┐
│                  Application Layer                              │
│  ┌──────────────────┐ ┌──────────────┐ ┌────────────────────┐  │
│  │Scenario Manager  │ │Device Manager│ │Pipeline Manager    │  │
│  └──────────────────┘ └──────────────┘ └────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────────┐
│                   Core Infrastructure                           │
│  ┌─────────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │Config Manager   │ │Event Bus     │ │Simulation Clock      │ │
│  │State Manager    │ │Plugin Loader │ │Scheduler             │ │
│  └─────────────────┘ └──────────────┘ └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────────┐
│              Signal Processing & Simulation Engine              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Orbital Dynamics  │  RF Channel  │  Signal Generator   │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  MUSIC AOA Estimator  │  ICA  │  IMM Tracker  │  PINN   │   │
│  │  Spoofing Detector    │  Consensus  │  Propagation      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
d:\SSN_lab\
├── R.E.S_env/                    # Main simulator package
│   ├── main.py                   # Application entry point
│   ├── requirements.txt           # Python dependencies
│   │
│   ├── core/                      # Framework infrastructure
│   │   ├── config.py             # Centralized configuration (orbital, RF, antenna params)
│   │   ├── event_bus.py          # Event-driven architecture
│   │   ├── state_manager.py      # Global state management
│   │   ├── simulation_clock.py   # Time management
│   │   ├── scheduler.py          # Task scheduling
│   │   └── plugin_loader.py      # Dynamic plugin loading
│   │
│   ├── simulation_engine/        # Physics & signal modeling
│   │   ├── orbital_dynamics.py   # Satellite trajectory computation
│   │   ├── environment.py        # Environmental conditions
│   │   ├── rf_channel.py         # Fading, attenuation, multipath
│   │   ├── signal_generator.py   # OFDM modulation, steering vectors
│   │   ├── propagation.py        # Path loss, Doppler calculations
│   │   └── spatial_partition.py  # Spatial indexing for efficiency
│   │
│   ├── algorithms/               # DSP & machine learning
│   │   ├── music_aoa.py          # MUSIC angle-of-arrival estimator
│   │   ├── ica.py                # Independent Component Analysis
│   │   ├── imm_tracker.py        # Interacting Multiple Model filter
│   │   ├── pinn_predictor.py     # Physics-Informed Neural Network
│   │   ├── spoofing_detector.py  # GPS spoofing detection
│   │   ├── mesh_consensus.py     # Distributed consensus algorithm
│   │   └── signal_pipeline.py    # Signal processing pipeline orchestration
│   │
│   ├── application/              # Application-level logic
│   │   ├── scenario_manager.py   # Scenario loading/saving
│   │   ├── device_manager.py     # Hardware device management
│   │   ├── metrics_collector.py  # Real-time metrics aggregation
│   │   └── pipeline_manager.py   # Pipeline execution management
│   │
│   ├── gui/                      # Interactive user interface
│   │   ├── main_window.py        # Main application window
│   │   ├── controller.py         # Simulation thread management
│   │   ├── device_panel.py       # Device configuration UI
│   │   ├── renderer.py           # 3D visualization
│   │   ├── pipeline_view.py      # Signal pipeline visualization
│   │   ├── metrics_dashboard.py  # Real-time metrics display
│   │   └── [theme assets]        # Dark theme stylesheet
│   │
│   ├── tests/                    # Test suites
│   │   ├── test_e2e.py          # End-to-end integration tests
│   │   ├── monte_carlo_crlb.py  # Monte Carlo validation
│   │   └── stress_test.py       # Performance stress tests
│   │
│   └── build/                    # PyInstaller build artifacts
│
├── SGD_SYSTEM/                   # Legacy validation framework
│   ├── Validation_Modèle_Signal/
│   │   ├── core_algorithms/     # Reference algorithm implementations
│   │   ├── simulator_engine/    # Legacy simulation engine
│   │   ├── IMM_dynamic/         # IMM filter validation
│   │   └── Validation_AOA/      # AOA estimation validation
│   │
├── data/                         # Simulation data & results
├── material_test/                # Hardware test sketches
│   └── On_arduino/              # Arduino integration
│
├── [Debug & Utility Scripts]
│   ├── debug_freq_bias.py
│   ├── debug_music.py
│   ├── verify_crlb_math.py
│   └── starfield.html           # 3D visualization reference
│
└── README.md                     # This file
```

---

## Installation

### Prerequisites

- **Python 3.9+** (tested on 3.10, 3.11)
- **pip** or **conda** package manager
- **Windows/Linux/macOS** (DLL handling included for Windows)

### Step 1: Clone Repository

```bash
git clone https://github.com/divinkas07/S_Signal_process.git
cd d:\SSN_lab
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
venv\Scripts\activate

# Or using conda
conda create -n res_env python=3.11
conda activate res_env
```

### Step 3: Install Dependencies

```bash
cd R.E.S_env
pip install -r requirements.txt
```

**Required packages:**
- `numpy>=1.24` – Numerical computations
- `scipy>=1.10` – Scientific algorithms
- `PySide6>=6.5` – Qt-based GUI framework
- `pyqtgraph>=0.13` – Real-time plotting

### Step 4: Verify Installation

```bash
python main.py
```

You should see the GUI window launch with the interactive dashboard.

---

## Quick Start

### Run the Simulator

```bash
cd R.E.S_env
python main.py
```

### Basic Workflow

1. **Configure Scenario**: Use the Device Panel to set satellite/ground station parameters
2. **Generate Signals**: Signal Generator creates OFDM waveforms with Doppler and fading
3. **Run Algorithm**: MUSIC/ICA/IMM processes the array signals in real-time
4. **Visualize Results**: Monitor metrics and 3D orbital rendering on the dashboard

### Configuration (core/config.py)

Key parameters can be tuned in `core/config.py`:

```python
# Satellite orbital parameters
SATELLITE_PARAMS = {
    "altitude_km": 550,           # LEO altitude
    "inclination_deg": 53.0,      # Orbital inclination
    "tx_power_dbw": 30.0,         # Transmit power
    "tx_antenna_gain_dbi": 38.0,  # Antenna gain
}

# RF/Ku-Band parameters
RF_PARAMS = {
    "fc": 14e9,                   # 14 GHz carrier
    "bandwidth": 250e6,           # 250 MHz bandwidth
    "modulation": "QPSK",         # Modulation: QPSK/16QAM/64QAM
}

# Antenna array configuration
ANTENNA_PARAMS = {
    "n_antennas": 8,              # ULA with 8 elements
    "antenna_spacing_lambda": 0.5,# Half-wavelength spacing
}
```

---

## Usage Guide

### Interactive GUI

The main window provides three panels:

#### 1. **Device Panel** (Left)
- Configure satellite TLE or orbital elements
- Set ground station location (lat/lon/alt)
- Adjust RF parameters (frequency, modulation, power)
- Select signal processing algorithm
- Monitor device status

#### 2. **3D Renderer** (Top-Right)
- Real-time satellite trajectory visualization
- Ground track display
- Antenna beam pattern overlay
- Orbital mechanics animation
- Zoom/pan/rotate controls

#### 3. **Metrics Dashboard** (Bottom-Right)
- SNR (Signal-to-Noise Ratio)
- BER (Bit Error Rate)
- Doppler offset
- AOA estimation error
- Signal spectrum (FFT)
- Channel impulse response

### Running Scenarios

```python
from application.scenario_manager import ScenarioManager

# Load a scenario
mgr = ScenarioManager()
scenario = mgr.load_scenario("scenarios/leo_ku_nominal.yaml")

# Run simulation
mgr.run(duration_seconds=600, speed_multiplier=1.0)

# Save results
mgr.save_results("output/results.npz")
```

### Custom Signal Processing

Add your own algorithm to the pipeline:

```python
from algorithms.signal_pipeline import SignalPipeline

# Create custom processor
class CustomAlgorithm:
    def process(self, data: dict) -> dict:
        # Your processing logic
        data['custom_output'] = compute_something(data['array_signal'])
        return data

# Register with pipeline
pipeline = SignalPipeline()
pipeline.register_stage("custom", CustomAlgorithm())
pipeline.execute(signal_data)
```

---

## Module Documentation

### Core Modules

#### `orbital_dynamics.py`
Computes satellite position/velocity using SGP4/Kepler equations:
- TLE parsing and propagation
- Ground track calculations
- Visibility cone computation
- Doppler shift estimation

#### `rf_channel.py`
Models realistic RF propagation:
- Path loss (Friis equation)
- Rician fading (K-factor configurable)
- Rain attenuation (ITU-R model)
- Multipath impulse response
- Phase noise (Wiener process)

#### `signal_generator.py`
OFDM signal synthesis:
- Subcarrier mapping (QAM constellation)
- IFFT with cyclic prefix
- Root-raised-cosine filtering
- ULA steering vector generation
- Doppler pre-compensation

### Algorithm Modules

#### `music_aoa.py`
MUSIC (MUltiple SIgnal Classification):
- Spatial covariance matrix estimation
- Eigendecomposition
- Noise subspace projection
- AOA peak detection via scanning

#### `ica.py`
Independent Component Analysis:
- Blind source separation
- Signal separation from multiuser channel
- FastICA algorithm

#### `imm_tracker.py`
Interacting Multiple Model:
- Multi-hypothesis tracking
- Kalman filter bank
- Model-conditional probability updates
- Track maintenance

#### `pinn_predictor.py`
Physics-Informed Neural Networks:
- Neural network with physical constraints
- Trajectory/signal prediction
- Domain knowledge integration

#### `spoofing_detector.py`
GPS Spoofing Detection:
- Signal authentication via multipath fingerprinting
- Anomaly detection
- Confidence scoring

### GUI Components

#### `renderer.py`
3D visualization with pyqtgraph:
- Orbital mechanics animation
- Antenna pattern overlay
- Spatial partitioning for efficient rendering
- Interactive camera controls

#### `metrics_dashboard.py`
Real-time metric aggregation and display:
- Time-series plots
- Spectrum analyzer
- Performance KPI dashboards

---

## Contributing

We welcome contributions! Follow these guidelines:

### 1. Fork & Branch

```bash
git clone https://github.com/YOUR_USERNAME/S_Signal_process.git
cd S_Signal_process
git checkout -b feature/my-feature
```

### 2. Development Setup

```bash
pip install -r R.E.S_env/requirements.txt
# Install dev tools if needed
pip install pytest black flake8
```

### 3. Code Style

- Follow **PEP 8** guidelines
- Use **black** for formatting: `black R.E.S_env/`
- Add docstrings (NumPy style) to all functions
- Type hints for public APIs

Example:

```python
def estimate_aoa(self, array_signal: np.ndarray) -> np.ndarray:
    """
    Estimate angle of arrival using MUSIC.
    
    Parameters
    ----------
    array_signal : np.ndarray, shape (n_antennas, n_samples)
        Array input signal
    
    Returns
    -------
    np.ndarray
        Estimated AOA in degrees
    
    Notes
    -----
    Assumes narrowband signal and ULA geometry.
    """
    # Implementation
```

### 4. Testing

```bash
# Run tests
pytest R.E.S_env/tests/

# Run specific test
pytest R.E.S_env/tests/test_e2e.py::test_scenario_execution
```

### 5. Create Pull Request

```bash
git add .
git commit -m "feat: add FMCW chirp modulation support"
git push origin feature/my-feature
```

Then create a PR on GitHub with:
- Clear description of changes
- Reference to related issues (if any)
- Test results summary
- Before/after comparison (if applicable)

### Contribution Areas

- 🔬 Algorithm improvements (AOA, ICA, tracking)
- 🎨 GUI enhancements and usability
- 📊 Performance optimization
- 📖 Documentation and examples
- 🐛 Bug fixes and issue resolution
- ✅ Test coverage improvements

---

## Testing

### Run Full Test Suite

```bash
cd R.E.S_env
pytest tests/ -v
```

### Test Categories

1. **End-to-End (E2E)**: `test_e2e.py`
   - Full simulation workflow
   - Algorithm correctness
   - GUI integration

2. **Monte Carlo**: `monte_carlo_crlb.py`
   - Validates Cramér-Rao Lower Bound
   - Statistical performance verification
   - 1000+ trials per scenario

3. **Stress Test**: `stress_test.py`
   - Long-duration simulations
   - Memory profiling
   - Real-time performance under load

### Validation Against Reference

The `SGD_SYSTEM/Validation_Modèle_Signal/` folder contains legacy reference implementations:

```bash
python SGD_SYSTEM/Validation_Modèle_Signal/simulator_engine/validate_crlb_aoa.py
python SGD_SYSTEM/Validation_Modèle_Signal/simulator_engine/validate_ica.py
```

---

## Performance Notes

- **Optimization**: Spatial partitioning reduces channel model complexity from O(n²) to O(n log n)
- **Real-time**: GUI updates at ~30 Hz; algorithm processing at 10× real-time
- **Memory**: ~500 MB for typical 10-minute simulation with 8 antennas
- **Parallelization**: Multi-scenario runs supported via ProcessPoolExecutor

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GUI doesn't start on Windows | Ensure `PySide6` DLL directory is added (check `main.py` setup) |
| Import errors | Verify virtual environment activation and dependency installation |
| Slow rendering | Reduce FFT size (`n_fft`) or disable real-time 3D visualization |
| Memory issues | Lower sampling rate or reduce simulation duration |
| Algorithm divergence | Tune state transition probabilities in IMM or ICA hyperparameters |

---

## References

- **Ku-Band Specification**: 3GPP TS 38.104 (FR2 bands)
- **Satellite Propagation**: ITU-R Recommendations (P.618, P.676, P.840)
- **Signal Processing**: Stoica & Moses, "Spectral Analysis of Signals" (MUSIC, array theory)
- **Fading Channels**: Proakis, "Digital Communications" (Rician model)
- **Doppler Effect**: Classic orbital mechanics with relativistic corrections

---

## License

This project is distributed under the **MIT License** – see [LICENSE](LICENSE) file for details.

---

## Contact & Support

- **Issues**: Report bugs via [GitHub Issues](https://github.com/divinkas07/S_Signal_process/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/divinkas07/S_Signal_process/discussions)
- **Email**: Contact the maintainers (see GitHub profile)

---

## Acknowledgments

- SGP4 orbital propagation
- IEEE 802.11 OFDM reference implementations
- Academic research on MUSIC, ICA, and IMM algorithms

**Last Updated**: June 2026

---

*Built with ❤️ for satellite communications research and development.*
