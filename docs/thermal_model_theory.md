# Thermal Model and Control Theory

This document describes the mathematical models and control algorithms
implemented in `chamber-mpc`. The controller supports four operating modes
from two configuration parameters:

| `model_type` | `estimator_type` | Description |
|---|---|---|
| `basic` | `fixed` | 2-state model, constant smoothing correction |
| `basic` | `kalman` | 2-state model, Kalman filter correction |
| `advanced` | `fixed` | 4-state model, dual constant smoothing correction |
| `advanced` | `kalman` | 4-state model, Kalman filter correction |

---

## Basic Model (2-state)

Two states: chamber temperature $T_c$ and sensor temperature $T_s$.
One measurement: $y = T_s$ (the chamber sensor reading).

### Dynamics

$$C_c \frac{dT_c}{dt} = P - h(T_c) \cdot (T_c - T_{amb})$$

$$\frac{dT_s}{dt} = r_s \cdot (T_c - T_s)$$

where:
- $C_c$ is the chamber thermal mass (J/K), identified as `chamber_heat_capacity`
- $P$ is the applied heater power (W)
- $h(T_c)$ is the temperature-dependent ambient heat transfer coefficient (W/K), interpolated from `h_calibration_points`
- $T_{amb}$ is the ambient temperature (deg C), from `ambient_temp` config or external sensor
- $r_{s}$ is the sensor responsiveness (1/s), identified as `sensor_responsiveness`

### Optional bed disturbance feedforward

When `bed_heater` is configured, the chamber equation becomes:

$$C_c \frac{dT_c}{dt} = P + h_{bed} \cdot (T_{bed} - T_c) - h(T_c) \cdot (T_c - T_{amb})$$

where $h_{bed}$ is `bed_transfer` (W/K) and $T_{bed}$ is the measured bed temperature.

### Heating element limiting (external clamp)

In the basic model, the heating element limit is applied as an output
constraint after the MPC output computation:

$$u_{actual} = \begin{cases}
0 & \text{if } T_{element} \geq T_{max} \\
u_{desired} \cdot \frac{T_{max} - T_{element}}{T_{margin}} & \text{if } T_{element} > T_{max} - T_{margin} \\
u_{desired} & \text{otherwise}
\end{cases}$$

The model propagation uses $u_{actual}$ (the constrained output), not
$u_{desired}$. This prevents integral windup when the element limiter
is active.

---

## Advanced Model (4-state)

Four states: heater temperature $T_h$, chamber temperature $T_c$,
S1 sensor temperature $T_{s1}$, S2 sensor temperature $T_{s2}$.
Two measurements: $y_1 = T_{s1}$, $y_2 = T_{s2}$.

### Dynamics

$$C_h \frac{dT_h}{dt} = P - k_{hc} \cdot (T_h - T_c)$$

$$C_c \frac{dT_c}{dt} = k_{hc} \cdot (T_h - T_c) - h(T_c) \cdot (T_c - T_{amb})$$

$$\frac{dT_{s1}}{dt} = r_{s1} \cdot (T_h - T_{s1})$$

$$\frac{dT_{s2}}{dt} = r_{s2} \cdot (T_c - T_{s2})$$

where additionally:
- $C_h$ is the heating element thermal mass (J/K), identified as `heater_heat_capacity`
- $k_{hc}$ is the heater-to-chamber coupling coefficient (W/K), identified as `heater_chamber_coupling`
- $r_{s1}$ is the S1 sensor responsiveness (1/s), identified as `s1_responsiveness`

Energy flow is strictly serial: $P \rightarrow T_h \rightarrow T_c \rightarrow T_{amb}$.
The two sensors observe their respective physical states with first-order lag.

### Heating element limiting (model-integrated)

In the advanced model, the element limit is integrated into the output
computation using the modeled $T_h$ state. From the heater equation at
the thermal limit where $dT_h/dt \leq 0$:

$$P_{limit} = k_{hc} \cdot (T_{max} - T_c)$$

With proportional pullback as $T_h$ approaches $T_{max}$:

$$P_{actual} = \min\left(P_{desired},\ P_{limit} \cdot \frac{T_{max} - T_h}{T_{margin}}\right)$$

This constraint is computed within the model, eliminating the external
clamp and its associated windup risk.

---

## MPC Output Computation

The output law is the same for both model types. The controller computes
power to drive $T_c$ toward the target in a specified time horizon:

$$P_{heat} = \frac{(T_{target} - T_c) \cdot C_c}{\tau_{reach}}$$

$$P_{loss} = h(T_c) \cdot (T_c - T_{amb})$$

$$P_{bed} = -h_{bed} \cdot (T_{bed} - T_c) \quad \text{(if bed configured)}$$

$$u = \frac{\min\left(\max(0,\ P_{heat} + P_{loss} + P_{bed}),\ P_{max}\right)}{P_{heater}}$$

where $\tau_{reach}$ is `target_reach_time` (seconds).

---

## Fixed Smoothing Estimator

### Basic model

The correction from the single measurement $y = T_{s2}$:

$$\alpha = 1 - (1 - s)^{\Delta t}$$

$$\epsilon = y - \hat{T}_s$$

$$\hat{T}_c \leftarrow \hat{T}_c + \alpha \cdot \epsilon$$

$$\hat{T}_s \leftarrow \hat{T}_s + \alpha \cdot \epsilon$$

where $s$ is the `smoothing` parameter (0.0 to 1.0). The same correction
is applied to both states (Kalico pattern). This is an approximation:
it over-corrects $T_s$ and under-corrects $T_c$, but the error is
self-correcting through the model dynamics on subsequent ticks.

### Advanced model

Two independent corrections from two measurements:

**S1 correction** (applied to $T_h$, $T_{s1}$ pair):

$$\alpha_h = 1 - (1 - s_h)^{\Delta t}$$

$$\epsilon_1 = y_1 - \hat{T}_{s1}$$

$$\hat{T}_h \leftarrow \hat{T}_h + \alpha_h \cdot \epsilon_1, \quad \hat{T}_{s1} \leftarrow \hat{T}_{s1} + \alpha_h \cdot \epsilon_1$$

**S2 correction** (applied to $T_c$, $T_{s2}$ pair):

$$\alpha_c = 1 - (1 - s_c)^{\Delta t}$$

$$\epsilon_2 = y_2 - \hat{T}_{s2}$$

$$\hat{T}_c \leftarrow \hat{T}_c + \alpha_c \cdot \epsilon_2, \quad \hat{T}_{s2} \leftarrow \hat{T}_{s2} + \alpha_c \cdot \epsilon_2$$

where $s_h$ is `smoothing_heater` and $s_c$ is `smoothing_chamber`.

---

## Kalman Filter Estimator

The Kalman filter replaces fixed smoothing with optimal per-state
correction gains computed from the prediction error covariance.
The algorithm is identical for both model sizes; only the matrix
dimensions change (2x2 for basic, 4x4 for advanced).

### Predict step

After model propagation, update the state covariance:

$$\mathbf{P}^- = \mathbf{A} \, \mathbf{P} \, \mathbf{A}^T + \mathbf{Q}$$

where $\mathbf{A}$ is the Jacobian of the discretized state dynamics and
$\mathbf{Q}$ is the diagonal process noise covariance.

### Update step

Given innovation $\boldsymbol{\epsilon} = \mathbf{y} - \mathbf{H} \, \hat{\mathbf{x}}$:

$$\mathbf{S} = \mathbf{H} \, \mathbf{P}^- \, \mathbf{H}^T + \mathbf{R}$$

$$\mathbf{K} = \mathbf{P}^- \, \mathbf{H}^T \, \mathbf{S}^{-1}$$

$$\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \, \boldsymbol{\epsilon}$$

$$\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K} \, \mathbf{H}) \, \mathbf{P}^-$$

### Basic model matrices

$$\mathbf{x} = \begin{bmatrix} T_c \\ T_s \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 0 & 1 \end{bmatrix}, \quad \mathbf{Q} = \begin{bmatrix} q_c & 0 \\ 0 & q_s \end{bmatrix}, \quad R = r$$

The gain vector $\mathbf{K} = [K_c, K_s]^T$ provides different corrections
for each state. At steady state, $K_c > K_s$ when sensor lag is
significant, correctly attributing more of the measurement surprise
to the chamber state than to the sensor state.

### Advanced model matrices

$$\mathbf{x} = \begin{bmatrix} T_h \\ T_c \\ T_{s1} \\ T_{s2} \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

$$\mathbf{Q} = \text{diag}(q_h, q_c, q_{s1}, q_{s2}), \quad \mathbf{R} = \begin{bmatrix} r_1 & 0 \\ 0 & r_2 \end{bmatrix}$$

The gain matrix $\mathbf{K}$ is $4 \times 2$: each of the four states
receives optimally weighted corrections from both measurements.

### Advantages over fixed smoothing

- **Adaptive gains**: higher during transients (model uncertain), lower at steady state (model reliable)
- **Physically correct distribution**: each state receives a correction proportional to its contribution to the prediction error, rather than equal correction to all states
- **Automatic tuning**: Q and R are estimated from calibration data, no manual smoothing adjustment needed

---

## Ambient Temperature

Ambient temperature $T_{amb}$ is not adaptively estimated.
It is set from one of two sources:

1. **External sensor** (`ambient_temp_sensor`): updated every control tick
2. **Saved calibration value** (`ambient_temp`): fixed for the entire run

Adaptive estimation was found to cause drift for chamber applications
due to persistent model-sensor bias being misattributed as ambient change.
The Kalico hotend MPC uses adaptive ambient estimation successfully because
hotend dynamics are fast and model errors are small; chamber dynamics are
slow and small biases in $h$ or sensor responsiveness accumulate.

---

## Calibration

### Parameters identified per mode

| Parameter | basic | advanced | Method |
|---|---|---|---|
| `chamber_heat_capacity` $C_c$ | required | required | Step response (max rate of rise) |
| `sensor_responsiveness` $r_{s2}$ | required | required | Step response (lag analysis) |
| `h_calibration_points` $h(T)$ | required | required | Steady-state power at each temperature |
| `smoothing` $s$ | fixed only | - | Prediction error statistics |
| `smoothing_heater` $s_h$ | - | fixed only | Prediction error statistics (S1) |
| `smoothing_chamber` $s_c$ | - | fixed only | Prediction error statistics (S2) |
| `heater_heat_capacity` $C_h$ | - | required | Dual-sensor step response |
| `heater_chamber_coupling` $k_{hc}$ | - | required | Dual-sensor step response |
| `s1_responsiveness` $r_{s1}$ | - | required | S1 lag during step response |
| `process_noise_*` | kalman only | kalman only | Prediction error variance decomposition |
| `measurement_noise_*` | kalman only | kalman only | Sensor noise variance during hold |
| `ambient_temp` $T_{amb}$ | required | required | Measured or provided via `T_AMBIENT` |

### h(T) identification

The ambient heat transfer coefficient is identified from the steady-state
power balance at each calibration temperature:

$$h(T_i) = \frac{P_{ss,i}}{T_i - T_{amb}}$$

Single calibration point: $h$ is constant across all temperatures.
Multiple points: $h(T)$ is linearly interpolated between calibrated values.
