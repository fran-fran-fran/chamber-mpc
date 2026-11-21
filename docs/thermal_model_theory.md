# Thermal Model and Control Theory

This document describes the mathematical model and control algorithms
implemented in `chamber-mpc`. The model uses a 2-state thermal model
with a 3-state Kalman filter that includes a disturbance state.

---

## Thermal Model (2-state)

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

## MPC Output Computation

The controller computes power to drive $T_c$ toward the target in a
specified time horizon:

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

The Kalman filter computes optimal per-state correction gains from the
prediction error covariance. It maintains a 3x3 covariance matrix for
the basic model.

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

### 3-state Kalman with disturbance

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

This reduces the steady-state error without the integral windup problems
of PID. Because $d$ is excluded from the output law, a small residual offset
remains when $h$ is inaccurate:

```math
e \approx (h_{true} - h_{model}) \cdot (T_c - T_{amb}) \cdot
\frac{\tau_{reach}}{C_c}
```

The high proportional gain $C_c / \tau_{reach}$ makes this offset small in
practice. The Kalman covariance update naturally prevents windup because the
gain $K_d$ shrinks as the innovation approaches zero.

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

### Parameters identified by calibration

| Parameter | Method |
|---|---|
| `chamber_heat_capacity` $C_c$ | Step response (max rate of rise) |
| `s1_responsiveness` $r_{s1}$ | Step response (lag analysis) |
| `ambient_transfer_points` $h(T)$ | Steady-state power at each temperature |
| `ambient_temp` $T_{amb}$ | Measured or provided via `T_AMBIENT` |

### h(T) identification

The ambient heat transfer coefficient is identified from the steady-state
power balance at each calibration temperature:

```math
h(T_i) = \frac{P_{ss,i}}{T_i - T_{amb}}
```

Single calibration point: $h$ is constant across all temperatures.
Multiple points: $h(T)$ is linearly interpolated between calibrated values.
