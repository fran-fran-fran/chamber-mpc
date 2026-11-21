# Thermal Model and Control Theory

This document describes the mathematical model and control algorithms
implemented in `chamber-mpc`. The model uses a 2-state thermal model
with a 3-state Kalman filter that includes a disturbance state.

---

## Thermal Model (2-state)

Two states: chamber temperature $T_c$ and sensor temperature $T_{s1}$.
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

When `secondary_sensor` is configured, the heating element temperature limit
is applied as an output constraint after the controller output computation:

```math
u_{actual} = \begin{cases}
0 & \text{if } T_{s2} \geq T_{safe} \\
u_{desired} \cdot \frac{T_{safe} - T_{s2}}{T_{zone}} & \text{if } T_{s2} > T_{safe} - T_{zone} \\
u_{desired} & \text{otherwise}
\end{cases}
```

where $T_{safe}$ is `s2_safe_temp` and $T_{zone}$ is `s2_safe_temp_zone`.

The model propagation uses $u_{actual}$ (the constrained output), not
$u_{desired}$. This prevents windup of the Kalman disturbance state $d$ and
state-estimate divergence when the element limiter is active.

---

## Controller Output Computation

Following Kalico's naming convention, this controller is called MPC, though
mathematically it is a single-step feedback-linearizing proportional +
feedforward law (rather than a multi-step receding-horizon optimizer).
When $T_{target} = 0$, the output is forced to $u = 0$. Otherwise:

```math
P_{heat} = \frac{(T_{target} - T_c) \cdot C_c}{\tau_{reach}}
```

```math
P_{loss} = h(T_c) \cdot (T_c - T_{amb})
```

```math
P_{bed} = h_{bed} \cdot (T_{bed} - T_c) \quad \text{(if bed configured, else } 0 \text{)}
```

```math
u_{desired} = \frac{\min\left(\max(0,\ P_{heat} + P_{loss} - P_{bed} - d),\ P_{max}\right)}{P_{heater}}
```

where $\tau_{reach}$ is `target_reach_time` (seconds), $d$ is the Kalman
disturbance estimate (watts), and $P_{max} = u_{max} \cdot P_{heater}$ is
the maximum allowed heater power.

With an exact model ($d = 0$) and an unsaturated heater, substituting
$P = P_{heat} + P_{loss} - P_{bed}$ into the chamber dynamics yields:

```math
\frac{dT_c}{dt} = \frac{T_{target} - T_c}{\tau_{reach}}
```

Thus $\tau_{reach}$ acts as the closed-loop exponential time constant
(closing ~63% of the remaining error in $\tau_{reach}$ and ~95% in
$3\tau_{reach}$), rather than a finite reach time.

---

## Kalman Filter Estimator

The Kalman filter computes per-state correction gains from the prediction
error covariance, maintaining a 3x3 covariance matrix for the augmented
state vector.

### Predict step

After model propagation over time step $\Delta t$, update the state covariance:

```math
\mathbf{P}^- = \mathbf{A} \, \mathbf{P} \, \mathbf{A}^T + \mathbf{Q} \, \Delta t
```

where $\mathbf{A}$ is the linearized state transition matrix and $\mathbf{Q}$
is the diagonal continuous-time process noise spectral density matrix.

### Update step

Given innovation $\epsilon = y - \mathbf{H} \, \hat{\mathbf{x}}$:

```math
S = \mathbf{H} \, \mathbf{P}^- \, \mathbf{H}^T + R
```

```math
\mathbf{K} = \mathbf{P}^- \, \mathbf{H}^T \, S^{-1}
```

```math
\hat{\mathbf{x}} \leftarrow \hat{\mathbf{x}} + \mathbf{K} \, \epsilon
```

```math
\mathbf{P} \leftarrow (\mathbf{I} - \mathbf{K} \, \mathbf{H}) \, \mathbf{P}^-
```

### 3-state Kalman with disturbance

The two physical states are augmented with an integrating disturbance state
$d$ (in watts) that captures persistent model errors such as inaccurate
$h(T_c)$ or unmodeled heat loads:

```math
\mathbf{x} = \begin{bmatrix} T_c \\ T_{s1} \\ d \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}
```

```math
\mathbf{Q} = \begin{bmatrix} Q_c & 0 & 0 \\ 0 & Q_{s1} & 0 \\ 0 & 0 & Q_d \end{bmatrix}, \quad R = r
```

where $Q_c, Q_{s1}$ are in $\text{K}^2/\text{s}$, $Q_d$ is in $\text{W}^2/\text{s}$,
and $R$ is the per-sample measurement noise variance in $\text{K}^2$.

The discretized state transition matrix includes the disturbance coupling into
the chamber temperature via $\Delta t / C_c$:

```math
\mathbf{A} = \begin{bmatrix} 1 & 0 & \Delta t / C_c \\ r_{s1} \Delta t & 1 - r_{s1} \Delta t & 0 \\ 0 & 0 & 1 \end{bmatrix}
```

Note that $A_{00} = 1$ is an approximation that neglects the thermal-loss
damping term $-\frac{\Delta t}{C_c}\big(h(T_c) + h'(T_c)(T_c - T_{amb}) + h_{bed}\big)$,
which is small ($\sim 10^{-4}$) for typical chambers.

The disturbance $d$ enters the model propagation as additional net power:

```math
C_c \frac{dT_c}{dt} = P + d + h_{bed} \cdot (T_{bed} - T_c) - h(T_c) \cdot (T_c - T_{amb})
```

In the control law, $d$ is subtracted from the required heater power
(offset-free MPC). At steady state (unsaturated heater):

- The covariance $\mathbf{P}$ and gains $\mathbf{K} = [K_c, K_{s1}, K_d]^T$
  converge to constant nonzero steady-state values determined solely by
  $\mathbf{A}$, $\mathbf{Q}$, $R$, and $\Delta t$ (independent of the
  innovation $\epsilon$).
- Under the default tuning ($Q_c \gg Q_{s1}$), $K_c > K_{s1}$ and the ratio
  $K_c / K_{s1}$ grows as sensor lag increases (inverting the sensor lag);
  this ordering depends on the chosen $Q_c / Q_{s1}$ ratio.
- $K_d$ (in W/K) is governed by $Q_d / R$ and determines how quickly $d$
  adapts to persistent prediction errors.
- Because $d$ integrates the innovation ($\Delta d = K_d \cdot \epsilon$) with
  $K_d > 0$, steady state forces $\epsilon = 0$ and therefore eliminates
  state-estimation bias ($\hat{T}_c = \hat{T}_{s1} = T_c$).
- With exact model parameters, $d \to 0$.
- With inaccurate $h$, $d$ converges to the unmodeled net power:
  $d \to (h_{model} - h_{true}) \cdot (T_c - T_{amb}) = -(h_{true} - h_{model}) \cdot (T_c - T_{amb})$
  (negative when true heat loss exceeds modeled heat loss, i.e., $h_{true} > h_{model}$).

Because $d$ cancels the unmodeled power mismatch directly in the output law
($P = P_{heat} + P_{loss} - P_{bed} - d$), the steady-state tracking error is
driven to zero ($T_c \to T_{target}$) regardless of $\tau_{reach}$, as long as
the heater is not saturated. Anti-windup during output limiting is preserved
because model propagation uses $u_{actual}$ rather than $u_{desired}$ (not
because of covariance decay: as $\epsilon \to 0$, the update $K_d \cdot \epsilon$
vanishes while $K_d$ itself remains constant and nonzero). As an additional
fault guard (e.g., against a stuck sensor), $d$ is clamped to
$[-P_{heater},\ +P_{heater}]$.

---

## Ambient Temperature

Ambient temperature $T_{amb}$ is not adaptively estimated.
It is set from one of two sources:

1. **External sensor** (`ambient_temp_sensor`): updated every control tick
2. **Saved calibration value** (`ambient_temp`): fixed for the entire run

Adaptive estimation was found to cause drift for chamber applications
due to persistent model-sensor bias being misattributed as ambient change
(from a single chamber sensor at steady state, errors in $h$, $T_{amb}$, and
$d$ are unobservable/confounded). The Kalico hotend MPC uses adaptive ambient
estimation successfully because hotend dynamics are fast and model errors are
small; chamber dynamics are slow and small biases in $h$ or sensor
responsiveness accumulate.

---

## Calibration

### Parameters identified by calibration

| Parameter | Method |
|---|---|
| `chamber_heat_capacity` $C_c$ | Step response (max rate of rise, $C_c = P / \max(\dot{T}_{s1})$) |
| `s1_responsiveness` $r_{s1}$ | Step response (lag analysis) |
| `target_reach_time` $\tau_{reach}$ | Scaled from $C_c$ and $P_{heater}$ ($\text{clamp}(\Delta T_{band} \cdot C_c / P_{heater},\ 2,\ 60)\text{ s}$, $\Delta T_{band} = 5\text{ °C}$) |
| `ambient_transfer_points` $h(T)$ | Steady-state power at each temperature |
| `ambient_temp` $T_{amb}$ | Measured or provided via `T_AMBIENT` |
| `measurement_noise_s1` $R$ | Detrended idle variance (reported as a suggestion, not auto-saved) |

Note that estimating $C_c$ from $\max(\dot{T}_{s1})$ slightly overestimates
$C_c$ whenever sensor lag or heat loss significantly attenuates the peak
measured slope relative to $P / C_c$.

### `target_reach_time` estimation

During calibration, `target_reach_time` ($\tau_{reach}$) is chosen so that a
temperature error equal to a target proportional band $\Delta T_{band} = 5\text{ °C}$
requests 100% of the heater's nameplate power $P_{heater}$, clamped to
$[2.0,\ 60.0]\text{ s}$:

```math
\tau_{reach} = \text{clamp}\!\left(\frac{\Delta T_{band} \cdot C_c}{P_{heater}},\ 2.0,\ 60.0\right)
```

Substituting this into the proportional heating term cancels $C_c$:

```math
P_{heat} = \frac{(T_{target} - T_c) \cdot C_c}{\tau_{reach}} = \frac{T_{target} - T_c}{\Delta T_{band}} \cdot P_{heater}
```

This makes Phase 2 holds and runtime control robust to $C_c$ estimation error
while giving fast ovens and slow printer chambers the same smooth proportional
band (100% power whenever more than ~5 °C below target, tapering smoothly near
setpoint, with zero steady-state offset due to $-d$ in the output law).

Typical examples:

| System | $P_{heater}$ | $C_c$ | Unclamped $5 \cdot C_c / P_{heater}$ | Saved `target_reach_time` |
|---|---|---|---|---|
| Airfryer annealing oven | 1800 W | 500 J/K | 1.39 s | **2.0 s** (floor) |
| 3D printer chamber (e.g. 420×420×460 mm) | 500 W | 2000 J/K | 20.0 s | **20.0 s** |
| Large / low-power chamber | 300 W | 10000 J/K | 166.7 s | **60.0 s** (ceiling) |

To override the calibrated value manually, remove `#*# target_reach_time` from
the `SAVE_CONFIG` block and set `target_reach_time:` in `[chamber_mpc]`.

### h(T) identification

The ambient heat transfer coefficient is identified from the steady-state
power balance at each calibration temperature (with the bed heater off and
near chamber temperature):

```math
h(T_i) = \frac{P_{ss,i}}{T_i - T_{amb}}
```

Single calibration point: $h$ is constant across all temperatures.
Multiple points: $h(T)$ is linearly interpolated between calibrated values and
clamped to the nearest endpoint outside the calibrated range.
