# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: thermal_model.py
# Description: Two-state thermal model for MPC. Tracks chamber (thermal mass)
#              temperature and sensor temperature with Euler integration.
#              Supports h(T) interpolation, bed disturbance feedforward,
#              and heating element temperature limiting.

from .h_interpolator import HInterpolator

AMBIENT_TEMP_DEFAULT = 25.0


class ThermalModel:
    """Two-state thermal model for chamber MPC.

    State variables:
        state_chamber_temp:   estimated true chamber temperature (deg C)
        state_sensor_temp:  estimated sensor reading (deg C, lagged)
        state_ambient_temp: estimated ambient temperature (deg C)

    Model equation:
        C * dT_block/dt = P_heater + P_bed - h(T) * (T_block - T_ambient)
        dT_sensor/dt = sensor_responsiveness * (T_block - T_sensor)
    """

    def __init__(self, chamber_heat_capacity, sensor_responsiveness,
                 h_interpolator, heater_power, smoothing=0.5,
                 target_reach_time=2.0, min_ambient_change=1.0,
                 steady_state_rate=0.5):
        # Identified parameters
        self.chamber_heat_capacity = float(chamber_heat_capacity)
        self.sensor_responsiveness = float(sensor_responsiveness)
        self.h_interpolator = h_interpolator
        self.heater_power = float(heater_power)

        # Tuning parameters
        self.smoothing = float(smoothing)
        self.target_reach_time = float(target_reach_time)
        self.min_ambient_change = float(min_ambient_change)
        self.steady_state_rate = float(steady_state_rate)

        # State
        self.state_chamber_temp = AMBIENT_TEMP_DEFAULT
        self.state_sensor_temp = AMBIENT_TEMP_DEFAULT
        self.state_ambient_temp = AMBIENT_TEMP_DEFAULT
        self.last_power = 0.0
        self.last_time = 0.0

        # Rolling power average
        self._power_history = []
        self._power_avg_window = 30.0  # seconds

        # Bed disturbance (optional)
        self.bed_transfer = 0.0
        self._bed_temp_fn = None  # callable returning (temp, target) tuple

        # Heating element limit (optional)
        self._heating_element_temp_fn = None
        self.heating_element_max_temp = 300.0
        self.heating_element_margin = 20.0

        # Ambient sensor (optional)
        self._ambient_temp_fn = None
        self.want_ambient_refresh = False

    def set_initial_state(self, temp):
        """Set initial state from a known temperature."""
        self.state_chamber_temp = temp
        self.state_sensor_temp = temp

    def set_ambient(self, temp):
        """Set known ambient temperature."""
        self.state_ambient_temp = temp

    def set_bed_disturbance(self, bed_temp_fn, bed_transfer):
        """Configure bed disturbance feedforward.

        Args:
            bed_temp_fn: callable(read_time) returning (temp, target) tuple
            bed_transfer: h_bed in W/K
        """
        self._bed_temp_fn = bed_temp_fn
        self.bed_transfer = float(bed_transfer)

    def set_heating_element_limit(self, temp_fn, max_temp, margin):
        """Configure heating element temperature limiting.

        Args:
            temp_fn: callable(read_time) returning (temp, target) tuple
            max_temp: maximum allowed heating element temperature (deg C)
            margin: proportional pullback zone width (deg C)
        """
        self._heating_element_temp_fn = temp_fn
        self.heating_element_max_temp = float(max_temp)
        self.heating_element_margin = float(margin)

    def set_ambient_sensor(self, temp_fn):  # pragma: no cover
        """Configure an external ambient temperature sensor.

        Args:
            temp_fn: callable(read_time) returning (temp, target) tuple
        """
        self._ambient_temp_fn = temp_fn
        self.want_ambient_refresh = True

    # -- Main update cycle --

    def update(self, read_time, temp_measured, target_temp, max_power):
        """Run one MPC cycle.

        Args:
            read_time: timestamp from temperature_callback
            temp_measured: raw sensor reading (deg C)
            target_temp: desired temperature (deg C)
            max_power: maximum heater output as fraction (0.0-1.0)

        Returns:
            duty: heater duty cycle to apply (0.0-1.0)
        """
        dt = read_time - self.last_time
        if self.last_time == 0.0 or dt < 0.0 or dt > 1.0:
            dt = 0.1
        self.last_time = read_time

        # -- Simulate: propagate model with last actual power --
        self._propagate(dt, read_time)

        # -- Correct: blend model with measurement --
        self._correct(dt, temp_measured)

        # -- Update ambient estimate --
        self._update_ambient(dt, read_time)

        # -- Compute desired output --
        u_desired = self._compute_output(target_temp, read_time, max_power)

        # -- Apply heating element constraint --
        u_actual = self._constrain_for_heating_element(read_time, u_desired)

        # -- Clamp to heater limits --
        u_actual = max(0.0, min(max_power, u_actual))

        # -- Store actual power for next propagation --
        self.last_power = u_actual * self.heater_power

        # -- Update rolling power average --
        self._power_history.append((read_time, u_actual))
        cutoff = read_time - self._power_avg_window
        while self._power_history and self._power_history[0][0] < cutoff:
            self._power_history.pop(0)

        return u_actual

    # -- Internal steps --

    def _propagate(self, dt, read_time):
        """Propagate model state using last actual applied power."""
        # Heat from our heater
        p_heating = self.last_power

        # Heat loss to ambient (temperature-dependent h)
        h = self.h_interpolator.h(self.state_chamber_temp)
        p_loss_ambient = h * (self.state_chamber_temp - self.state_ambient_temp)

        # Bed disturbance (optional)
        p_bed = 0.0
        if self._bed_temp_fn is not None and self.bed_transfer > 0:  # pragma: no cover
            try:
                bed_temp, _ = self._bed_temp_fn(read_time)
                p_bed = self.bed_transfer * (bed_temp - self.state_chamber_temp)
            except Exception:
                pass  # pragma: no cover -- sensor read failure, skip feedforward

        # Chamber temperature update
        dT_block = (
            (p_heating + p_bed - p_loss_ambient) * dt
            / self.chamber_heat_capacity
        )
        self.state_chamber_temp += dT_block

        # Sensor temperature update (lagged response)
        dT_sensor = (
            (self.state_chamber_temp - self.state_sensor_temp)
            * self.sensor_responsiveness * dt
        )
        self.state_sensor_temp += dT_sensor

    def _correct(self, dt, temp_measured):
        """Correct model state by blending prediction with measurement."""
        effective_smoothing = 1.0 - (1.0 - self.smoothing) ** dt
        adjustment = (temp_measured - self.state_sensor_temp) * effective_smoothing
        # Apply correction to both states (Kalico pattern)
        self.state_chamber_temp += adjustment
        self.state_sensor_temp += adjustment

    def _update_ambient(self, dt, read_time):
        """Update ambient temperature estimate.

        Conservative approach: only adjust when the system is genuinely
        near steady state (small model-sensor gap AND small temperature
        rate of change). This prevents runaway ambient drift during
        transients like startup, large setpoint changes, or overshoot.
        """
        # Use external sensor if available (overrides adaptive estimation)
        if self.want_ambient_refresh and self._ambient_temp_fn is not None:  # pragma: no cover
            try:
                temp, _ = self._ambient_temp_fn(read_time)
                if temp != 0.0:
                    self.state_ambient_temp = temp
                    self.want_ambient_refresh = False
                    return
            except Exception:
                pass  # sensor read failure

        # Adaptive ambient estimation - very conservative for chambers
        # Only adjust when ALL conditions are met:
        # 1. Heater is partially on (not off or fully saturated)
        # 2. Model and sensor are close (system is settled)
        # 3. Temperature is not changing rapidly
        model_sensor_gap = abs(
            self.state_chamber_temp - self.state_sensor_temp)
        if model_sensor_gap > 2.0:
            return  # model hasn't converged yet, don't adjust ambient

        if not (0 < self.last_power < self.heater_power * 0.95):
            return  # heater is off or saturated, measurement unreliable

        # Very small adjustment rate for chambers (0.01 deg C/s max)
        max_rate = 0.01
        error = self.state_sensor_temp - self.state_chamber_temp
        delta = max(-max_rate * dt, min(max_rate * dt, error * 0.1 * dt))
        self.state_ambient_temp += delta

        # Clamp ambient to reasonable range
        self.state_ambient_temp = max(
            -10.0, min(50.0, self.state_ambient_temp))

    def _compute_output(self, target_temp, read_time, max_power):
        """Compute desired heater output."""
        if target_temp == 0.0:
            return 0.0

        # Power needed to reach target in the desired time
        heating_power = (
            (target_temp - self.state_chamber_temp)
            * self.chamber_heat_capacity
            / self.target_reach_time
        )

        # Loss to ambient at current chamber temperature
        h = self.h_interpolator.h(self.state_chamber_temp)
        loss_ambient = (
            (self.state_chamber_temp - self.state_ambient_temp) * h
        )

        # Bed contribution (reduces our required output)
        loss_bed = 0.0
        if self._bed_temp_fn is not None and self.bed_transfer > 0:  # pragma: no cover
            try:
                bed_temp, _ = self._bed_temp_fn(read_time)
                # Negative loss = heat gain from bed
                loss_bed = -self.bed_transfer * (
                    bed_temp - self.state_chamber_temp)
            except Exception:
                pass  # sensor read failure

        # Total power needed
        power = max(0.0, min(
            max_power * self.heater_power,
            heating_power + loss_ambient + loss_bed
        ))

        return power / self.heater_power

    def _constrain_for_heating_element(self, read_time, u_desired):
        """Apply heating element temperature limit.

        Returns constrained output. The MPC model propagation uses the
        constrained value (via last_power), preventing windup.
        """
        if self._heating_element_temp_fn is None:
            return u_desired

        try:  # pragma: no cover
            temp_element, _ = self._heating_element_temp_fn(read_time)
        except Exception:
            return u_desired  # pragma: no cover -- sensor read failure

        if temp_element >= self.heating_element_max_temp:
            return 0.0

        pullback_start = (
            self.heating_element_max_temp - self.heating_element_margin
        )
        if temp_element > pullback_start:
            headroom = self.heating_element_max_temp - temp_element
            scale = headroom / self.heating_element_margin
            return u_desired * scale

        return u_desired

    # -- Status --

    def get_avg_power(self):
        if not self._power_history:
            return 0.0
        total = sum(duty for _, duty in self._power_history)
        return total / len(self._power_history) * self.heater_power

    def get_avg_duty(self):
        if not self._power_history:
            return 0.0
        total = sum(duty for _, duty in self._power_history)
        return total / len(self._power_history)

    def get_status(self):
        """Return model state for Moonraker/UI."""
        return {
            'temp_chamber': round(self.state_chamber_temp, 2),
            'temp_sensor': round(self.state_sensor_temp, 2),
            'temp_ambient': round(self.state_ambient_temp, 2),
            'power': round(self.last_power, 2),
            'avg_power': round(self.get_avg_power(), 2),
            'avg_duty': round(self.get_avg_duty(), 4),
        }
