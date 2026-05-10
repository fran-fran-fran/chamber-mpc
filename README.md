# chamber-mpc

Model Predictive Control for heated chambers in [Kalico](https://github.com/KalicoCrew/kalico) and [Klipper](https://github.com/Klipper3d/klipper) (Klipper compatibility is expected but untested).

Replaces standard PID control with a model-based controller that provides feedforward compensation, temperature-dependent heat loss modeling, and optional bed heater disturbance rejection. Designed for both standalone annealing ovens and 3D printer heated chambers.

## Features

- **Model-based feedforward** - predicts required heater power from a thermal model, reducing overshoot and improving ramp tracking
- **Multi-point h(T) calibration** - characterizes temperature-dependent heat loss across the operating range in a single ascending pass
- **Single-point calibration** - for narrow operating ranges (e.g. printer chambers at 50-80 deg C), one calibration point is sufficient
- **Heating element temperature limiting** - optional secondary sensor constrains output to protect heating elements from overtemperature, with model-aware anti-windup
- **Bed heater feedforward** - optional measured disturbance rejection for printer chambers where the bed contributes heat
- **Auto-estimated smoothing** - calibration routine estimates the optimal model correction aggressiveness from prediction error statistics
- **Progressive calibration** - multi-point calibration runs in one ascending pass without cool-down between points

## Requirements

- Kalico (tested) or Klipper (untested, expected to work)
- Python 3.9+
- A `[heater_generic]` configured for your chamber heater
- A working PID calibration on the heater (used as scaffolding during MPC calibration)

## Installation

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/chamber-mpc.git
cd chamber-mpc
chmod +x scripts/install.sh
./scripts/install.sh
```

To uninstall:

```bash
./scripts/install.sh --uninstall
```

## Configuration

### Standalone annealing oven (e.g. airfryer)

```ini
[chamber_mpc]
heater: airfryer
heater_power: 1800
heating_element_sensor: element
heating_element_max_temp: 270
heating_element_margin: 20
```

### 3D printer heated chamber

```ini
[chamber_mpc]
heater: chamber
heater_power: 600
bed_heater: heater_bed
bed_transfer: 0.35
```

### Configuration reference

| Parameter | Requirement | Default | Description |
|-----------|-------------|---------|-------------|
| `heater` | required | none | Name of the `[heater_generic]` to control |
| `heater_power` | required | none | Heater nameplate power in watts |
| `model_type` | optional | `basic` | Thermal model type: `basic` (2-state) or `advanced` (4-state) |
| `estimator_type` | optional | `fixed` | State estimator type: `fixed` (constant smoothing) or `kalman` (adaptive gains) |
| `chamber_heat_capacity` | required | calibrated | Chamber thermal mass in J/K |
| `sensor_responsiveness` | required | calibrated | S2 sensor lag coefficient (1/s) |
| `smoothing` | optional | 0.5 | Model correction aggressiveness for basic+fixed mode (0.0-1.0) |
| `smoothing_heater` | optional | min(0.95, smoothing*1.4) | Override correction aggressiveness for heater/S1 pair in advanced+fixed mode (0.0-1.0). Derived from smoothing if not set |
| `smoothing_chamber` | optional | smoothing | Override correction aggressiveness for chamber/S2 pair in advanced+fixed mode (0.0-1.0). Derived from smoothing if not set |
| `target_reach_time` | optional | 2.0 | Prediction horizon in seconds |
| `max_temp_margin` | optional | 5.0 | Control setpoint is clamped to the heater's max_temp minus this margin (deg C) |
| `ambient_temp` | optional | 25.0 (calibrated) | Ambient temperature used by the model (deg C) |
| `ambient_temp_sensor` | optional | none | Name of external `[temperature_sensor]` for ambient temperature |
| `h_calibration_points` | required | calibrated | Temperature-dependent h(T) values (multi-line T, h pairs) |
| `heating_element_sensor` | conditional | none | Name of `[temperature_sensor]` on heating element. Required for advanced model, optional for basic |
| `heating_element_max_temp` | optional | 250.0 | Heating element temperature hard limit. Must be greater than heater max_temp (deg C) |
| `heating_element_margin` | optional | 20.0 | Zone below the hard limit where output starts tapering down proportionally (deg C) |
| `bed_heater` | optional | none | Name of bed heater for disturbance feedforward |
| `bed_transfer` | optional | 0.0 (calibrated) | Bed-to-chamber heat transfer coefficient in W/K |
| `heater_heat_capacity` | advanced only | calibrated | Heating element thermal mass in J/K |
| `heater_chamber_coupling` | advanced only | calibrated | Heater-to-chamber coupling coefficient in W/K |
| `s1_responsiveness` | advanced only | calibrated | S1 sensor lag coefficient (1/s) |
| `process_noise_chamber` | kalman only | calibrated | Process noise variance for chamber state |
| `process_noise_sensor` | kalman only | calibrated | Process noise variance for sensor state (basic+kalman) |
| `process_noise_heater` | kalman only | calibrated | Process noise variance for heater state (advanced+kalman) |
| `process_noise_s1` | kalman only | calibrated | Process noise variance for S1 state (advanced+kalman) |
| `process_noise_s2` | kalman only | calibrated | Process noise variance for S2 state (advanced+kalman) |
| `measurement_noise` | kalman only | calibrated | Measurement noise variance (basic+kalman) |
| `measurement_noise_s1` | kalman only | calibrated | S1 measurement noise variance (advanced+kalman) |
| `measurement_noise_s2` | kalman only | calibrated | S2 measurement noise variance (advanced+kalman) |

## Calibration

### Single-point (printer chamber)

```gcode
MPC_CHAMBER_CALIBRATE HEATER=chamber POINTS=70
```

### Multi-point (annealing oven)

```gcode
MPC_CHAMBER_CALIBRATE HEATER=airfryer POINTS=60,100,150,200
```

### With known ambient temperature (chamber already warm)

```gcode
MPC_CHAMBER_CALIBRATE HEATER=chamber POINTS=80,130 T_AMBIENT=28
```

### With bed transfer measurement

```gcode
MPC_CHAMBER_CALIBRATE HEATER=chamber POINTS=70 BED_TEMP=110 T_AMBIENT=25
```

Calibration results are saved automatically. Run `SAVE_CONFIG` to persist to printer.cfg.

## GCode command reference

| Command | Parameters | Description |
|---------|-----------|-------------|
| `MPC_CHAMBER_CALIBRATE` | `HEATER=` `POINTS=` `T_AMBIENT=` `BED_TEMP=` | Run calibration |
| `MPC_CHAMBER_STATUS` | `HEATER=` | Report model state |

## How it works

The controller maintains a two-state thermal model:

1. **Chamber temperature** - estimated true chamber temperature (thermal mass)
2. **Sensor temperature** - estimated sensor reading (lagged by sensor_responsiveness)

Each control tick:
1. Propagate model forward using last actual applied power
2. Correct model state by blending prediction with measurement
3. Compute desired output (feedforward + loss compensation)
4. Constrain for heating element limit (if configured)
5. Apply constrained output and record actual power for next tick

The model propagation uses the actual applied power (after constraints), not the desired power. This prevents integral windup when the heating element limiter is active.

## Thermal Model and Control Theory

For the full mathematical formulation of the thermal models, state estimation
algorithms, and control law, see [docs/thermal_model_theory.md](docs/thermal_model_theory.md).

## Development

```bash
python setup_dev_env.py --run-tests
```

## Project structure

```
chamber-mpc/
+-- src/chamber_mpc/
|   +-- __init__.py            # Klippy entry point, heater control registration
|   +-- __version__.py         # Version
|   +-- thermal_model.py       # Two-state basic model (T_chamber + T_sensor)
|   +-- thermal_model_advanced.py # Four-state advanced model (T_heater + T_chamber + T_s1 + T_s2)
|   +-- kalman.py              # Kalman filters for 2-state and 4-state models
|   +-- h_interpolator.py      # h(T) interpolation from calibration points
|   +-- control.py             # Klipper heater control interface
|   +-- calibrate.py           # Calibration analysis (step response, smoothing)
|   +-- calibrate_runner.py    # Calibration sequence orchestration
+-- scripts/
|   +-- install.sh             # Install / --uninstall
+-- tests/unit/                # pytest unit tests (cross-platform)
+-- docs/examples/             # Example configs
+-- setup_dev_env.py           # Cross-platform dev environment setup
+-- moonraker.conf             # Moonraker update_manager snippet
+-- README.md
```

## License

GPL-3.0 - see LICENSE file.
