# Thermal Model and Control Theory

This document describes the mathematical models and control algorithms
implemented in `chamber-mpc`. Two model types are available, both using
Kalman filtering with disturbance estimation for state correction:

| `model_type` | Description |
|---|---|
| `basic` | 2-state model (T_chamber + T_sensor) with 3-state Kalman filter (includes disturbance state) |
| `advanced` | 4-state model (T_heater + T_chamber + T_s1 + T_s2) with Kalman filter |

---

## Basic Model (2-state)

Two states: chamber temperature $T_c$ and sensor temperature $T_s$.
One measurement: $y = T_{s1}$ (the primary sensor reading).

### Dynamics

```math
C_c \frac{dT_c}{dt} = P - h(T_c) \cdot (T_c - T_{amb})
```

```math
\frac{dT_{s1}}{dt} = r_{s1} \cdot (T_c - T_{s1})
```

where:
- $C_c$ is the chamber thermal mass (J/K), identified as `chamber_heat_capacity`
- $P$ is the applied heater power (W)
- $h(T_c)$ is the temperature-dependent ambient heat transfer coefficient (W/K), interpolated from `ambient_transfer_points`
- $T_{amb}$ is the ambient temperature (deg C), from `ambient_temp` config or external sensor
- $r_{s1}$ is the S1 sensor responsiveness (1/s), identified as `s1_responsiveness`

### Optional bed disturbance feedforward

When `bed_heater` is configured, the chamber equation becomes:

```math
C_c \frac{dT_c}{dt} = P + h_{bed} \cdot (T_{bed} - T_c) - h(T_c) \cdot (T_c - T_{amb})
```

where $h_{bed}$ is `bed_transfer` (W/K) and $T_{bed}$ is the measured bed temperature.

### Heating element limiting (external clamp)

In the basic model, the heating element limit is applied as an output
constraint after the MPC output computation:

```math
u_{actual} = \begin{cases}
0 & \text{if } T_{element} \geq T_{max} \\
u_{desired} \cdot \frac{T_{max} - T_{element}}{T_{margin}} & \text{if } T_{element} > T_{max} - T_{margin} \\
u_{desired} & \text{otherwise}
\end{cases}
```

The model propagation uses $u_{actual}$ (the constrained output), not
$u_{desired}$. This prevents integral windup when the element limiter
is active.

---

## Advanced Model (4-state)

Four states: heater temperature $T_h$, chamber temperature $T_c$,
S1 sensor temperature $T_{s1}$, S2 sensor temperature $T_{s2}$.
Two measurements: $y_1 = T_{s1}$ (primary, chamber), $y_2 = T_{s2}$ (secondary, element).

### Dynamics

```math
C_h \frac{dT_h}{dt} = P - k_{hc} \cdot (T_h - T_c)
```

```math
C_c \frac{dT_c}{dt} = k_{hc} \cdot (T_h - T_c) - h(T_c) \cdot (T_c - T_{amb})
```

```math
\frac{dT_{s2}}{dt} = r_{s2} \cdot (T_h - T_{s2})
```

```math
\frac{dT_{s1}}{dt} = r_{s1} \cdot (T_c - T_{s1})
```

where additionally:
- $C_h$ is the heating element thermal mass (J/K), identified as `heater_heat_capacity`
- $k_{hc}$ is the heater-to-chamber coupling coefficient (W/K), identified as `heating_element_transfer`
- $r_{s2}$ is the S2 sensor responsiveness (1/s), identified as `s2_responsiveness`

Energy flow is strictly serial: $P \rightarrow T_h \rightarrow T_c \rightarrow T_{amb}$.
The two sensors observe their respective physical states with first-order lag.

### Heating element limiting (model-integrated)

In the advanced model, the element limit is integrated into the output
computation using the modeled $T_h$ state. From the heater equation at
the thermal limit where $dT_h/dt \leq 0$:

```math
P_{limit} = k_{hc} \cdot (T_{max} - T_c)
```

With proportional pullback as $T_h$ approaches $T_{max}$:

```math
P_{actual} = \min\left(P_{desired},\ P_{limit} \cdot \frac{T_{max} - T_h}{T_{margin}}\right)
```

This constraint is computed within the model, eliminating the external
clamp and its associated windup risk.

---

## MPC Output Computation

The output law is the same for both model types. The controller computes
power to drive $T_c$ toward the target in a specified time horizon:

```math
P_{heat} = \frac{(T_{target} - T_c) \cdot C_c}{\tau_{reach}}
```

```math
P_{loss} = h(T_c) \cdot (T_c - T_{amb})
```

```math
P_{bed} = -h_{bed} \cdot (T_{bed} - T_c) \quad \text{(if bed configured)}
```

```math
u = \frac{\min\left(\max(0,\ P_{heat} + P_{loss} + P_{bed}),\ P_{max}\right)}{P_{heater}}
```

where $\tau_{reach}$ is `target_reach_time` (seconds).

---

## Kalman Filter Estimator

The Kalman filter computes optimal per-state
correction gains computed from the prediction error covariance.
The algorithm is identical for both model sizes; only the matrix
dimensions change (2x2 for basic, 4x4 for advanced).

### Predict step

After model propagation, update the state covariance:

```math
\mathbf{P}^- = \mathbf{A} \, \mathbf{P} \, \mathbf{A}^T + \mathbf{Q}
```

where $\mathbf{A}$ is the Jacobian of the discretized state dynamics and
$\mathbf{Q}$ is the diagonal process noise covariance.

### Update step

Given innovation $\boldsymbol{\epsilon} = \mathbf{y} - \mathbf{H} \, \hat{\mathbf{x}}$:

```math
\mathbf{S} = \mathbf{H} \, \mathbf{P}^- \, \mathbf{H}^T + \mathbf{R}
```

```math
\mathbf{K} = \mathbf{P}^- \, \mathbf{H}^T \, \mathbf{S}^{-1}
```

```math
\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \, \boldsymbol{\epsilon}
```

```math
\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K} \, \mathbf{H}) \, \mathbf{P}^-
```

### Basic model: 3-state Kalman with disturbance

The basic model augments the two physical states with a disturbance
state $d$ (in watts) that captures persistent model errors such as
incorrect $h$ values. The disturbance is an integrating state with
no dynamics of its own:

```math
\mathbf{x} = \begin{bmatrix} T_c \\ T_s \\ d \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}
```

```math
\mathbf{Q} = \begin{bmatrix} Q_c & 0 & 0 \\ 0 & Q_s & 0 \\ 0 & 0 & Q_d \end{bmatrix}, \quad R = r
```

The state transition includes the disturbance coupling into the
chamber temperature via $\Delta t / C_c$:

```math
\mathbf{A} = \begin{bmatrix} 1 & 0 & \Delta t / C_c \\ r_{s1} \Delta t & 1 - r_{s1} \Delta t & 0 \\ 0 & 0 & 1 \end{bmatrix}
```

The disturbance $d$ enters the model propagation as additional power:

```math
C_c \frac{dT_c}{dt} = P + d - h(T_c) \cdot (T_c - T_{amb})
```

But $d$ does **not** appear in the output computation. The MPC
computes output from the corrected $T_c$ state, which already
reflects $d$'s effect through the model propagation.

The gain vector $\mathbf{K} = [K_c, K_s, K_d]^T$ provides different
corrections for each state. At steady state:
- $K_c > K_s$ when sensor lag is significant
- $K_d$ is small (disturbance changes slowly)
- With correct model parameters, $d \to 0$
- With incorrect $h$, $d$ converges to the missing power:
  $d \to (h_{true} - h_{model}) \cdot (T_c - T_{amb})$

This provides **offset-free MPC**: the controller tracks the setpoint
correctly regardless of $h$ calibration accuracy, without the integral
windup problems of PID. The Kalman covariance update naturally prevents
windup because the gain $K_d$ shrinks as the innovation approaches zero.

### Advanced model matrices

```math
\mathbf{x} = \begin{bmatrix} T_h \\ T_c \\ T_{s1} \\ T_{s2} \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
```

```math
\mathbf{Q} = \text{diag}(q_h, q_c, q_{s1}, q_{s2}), \quad \mathbf{R} = \begin{bmatrix} r_1 & 0 \\ 0 & r_2 \end{bmatrix}
```

The gain matrix $\mathbf{K}$ is $4 \times 2$: each of the four states
receives optimally weighted corrections from both measurements.

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
| `s1_responsiveness` $r_{s1}$ | required | required | Step response (lag analysis) |
| `ambient_transfer_points` $h(T)$ | required | required | Steady-state power at each temperature |
| `heater_heat_capacity` $C_h$ | - | required | Dual-sensor step response |
| `heating_element_transfer` $k_{hc}$ | - | required | Dual-sensor step response |
| `s2_responsiveness` $r_{s2}$ | - | required | S2 lag during step response |
| `process_noise_*` | defaults provided | defaults provided | Kalman Q matrix (advanced tuning) |
| `measurement_noise_*` | defaults provided | defaults provided | Kalman R matrix (advanced tuning) |
| `ambient_temp` $T_{amb}$ | required | required | Measured or provided via `T_AMBIENT` |

### h(T) identification

The ambient heat transfer coefficient is identified from the steady-state
power balance at each calibration temperature:

```math
h(T_i) = \frac{P_{ss,i}}{T_i - T_{amb}}
```

Single calibration point: $h$ is constant across all temperatures.
Multiple points: $h(T)$ is linearly interpolated between calibrated values.
