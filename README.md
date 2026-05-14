# chamber-mpc

Model Predictive Control for heated chambers in [Kalico](https://github.com/KalicoCrew/kalico) and [Klipper](https://github.com/Klipper3d/klipper) (Klipper compatibility is expected but untested).

Replaces standard PID control with a model-based controller that provides feedforward compensation, temperature-dependent heat loss modeling, and optional bed heater disturbance rejection. Designed for both standalone annealing ovens and 3D printer heated chambers.

## Features

- **Model-based feedforward** - predicts required heater power from a thermal model, reducing overshoot and improving ramp tracking
- **Multi-point h(T) calibration** - characterizes temperature-dependent heat loss across the operating range in a single ascending pass
- **Single-point calibration** - for narrow operating ranges (e.g. printer chambers at 50-80 deg C), one calibration point is sufficient
- **Heating element temperature limiting** - optional secondary sensor constrains output to protect heating elements from overtemperature, with model-aware anti-windup
- **Bed heater feedforward** - optional measured disturbance rejection for printer chambers where the bed contributes heat
- **Kalman filter with disturbance estimation** - optimal per-state correction gains adapt automatically to operating conditions, with an integrating disturbance state that provides offset-free control regardless of model accuracy
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
secondary_sensor: element
s2_safe_temp: 270
s2_safe_temp_zone: 20
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
| `model_type` | optional | basic | Thermal model type: `basic` (2-state) or `advanced` (4-state) |
| `chamber_heat_capacity` | required | calibrated | Chamber thermal mass in J/K |
| `s1_responsiveness` | required | calibrated | S1 lag coefficient [1/s]. S1 is the main sensor (used for setpoint tracking), as configured in the `[heater_generic]` defined in `heater` |
| `target_reach_time` | optional | 2.0 | Prediction horizon of the MPC [s] |
| `max_temp_margin` | optional | 5.0 | Control setpoint is clamped to the `heater`'s `max_temp` minus this margin [°C], to avoid shutdown due to temperature overshoot |
| `ambient_temp` | optional | 25.0 (calibrated) | Ambient temperature [°C] used by the model |
| `ambient_temp_sensor` | optional | none | Name of external `[temperature_sensor]` for ambient temperature |
| `ambient_transfer_points` | required | calibrated | Temperature-dependent chamber-to-ambient heat transfer coefficient points. Defined as multi-line points (T, h(T)), with minimum of one point |
| `secondary_sensor` | conditional | none | Name of the `[temperature_sensor]` placed on (or closer to) the heating element. This sensor is useful when the main sensor (defined in `heater`) is used to measure the chamber temperature. This is then used for limiting the temperature of the heating element for safety/longevity. Required for advanced model, optional for basic |
| `s2_safe_temp` | optional | 250.0 | Temperature limit [°C] for the heating element. The control output is turned off when the `secondary_sensor` reaches this temperature. Must be greater than `heater`'s `max_temp`. Must be smaller than `heating_element_sensor`'s `max_temp` |
| `s2_safe_temp_zone` | optional | 20.0 | Temperature zone [°C] below the `s2_safe_temp` where control output starts tapering down proportionally to 0 |
| `bed_heater` | optional | none | Name of `[heater_bed]` for bed disturbance feedforward |
| `bed_transfer` | optional | 0.0 (calibrated) | Bed-to-chamber heat transfer coefficient [W/K] |
| `heater_heat_capacity` | required for advanced model | calibrated | Heating element thermal mass [J/K] |
| `heating_element_transfer` | required for advanced model | calibrated | Heater-to-chamber heat transfer coefficient [W/K] |
| `s2_responsiveness` | required for advanced model | calibrated | S2 lag coefficient [1/s]. S2 is the optional secondary sensor, as defined in `secondary_sensor` |
| `process_noise_chamber` | optional | 1.0 | Kalman Q matrix: process noise for chamber state $Q_c$ |
| `process_noise_s1` | optional | 0.1 | Kalman Q matrix: process noise for S1 state $Q_{s1}$ |
| `process_noise_disturbance` | optional | 10.0 | Kalman Q matrix: process noise for disturbance state $Q_d$ (controls how fast the disturbance adapts) |
| `measurement_noise_s1` | optional | 0.5 | Kalman R: S1 measurement noise variance $R$ (basic model) or $R_1$ in R matrix (advanced model) |
| `process_noise_heater` | optional (advanced model only) | 1.0 | Kalman Q matrix: process noise for heater state $Q_h$ |
| `process_noise_s2` | optional (advanced model only) | 0.5 | Kalman Q matrix: process noise for S2 state $Q_{s2}$ |
| `measurement_noise_s2` | optional (advanced model only) | 0.5 | Kalman R matrix: S2 measurement noise variance $R_2$ |

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
2. **Sensor temperature** - estimated sensor reading (lagged by s1_responsiveness)

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
|   +-- calibrate.py           # Calibration analysis (step response, h identification)
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
