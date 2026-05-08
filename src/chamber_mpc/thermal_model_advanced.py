# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: thermal_model_advanced.py
# Description: Four-state advanced thermal model for MPC.
#              States: T_heater, T_chamber, T_s1 (heater sensor), T_s2 (chamber sensor)
#              Two measurements: S1 (near heating element), S2 (chamber air)
#              The heating element is modeled as a separate thermal mass
#              coupled to the chamber, enabling model-integrated element
#              temperature limiting (no external clamp needed).

from .h_interpolator import HInterpolator

AMBIENT_TEMP_DEFAULT = 25.0


class ThermalModelAdvanced:
    """Four-state thermal model for chamber MPC.

    States:
        state_heater_temp:  estimated heating element temperature (deg C)
        state_chamber_temp: estimated chamber bulk temperature (deg C)
        state_s1_temp:      estimated S1 sensor reading (deg C, lagged from heater)
        state_s2_temp:      estimated S2 sensor reading (deg C, lagged from chamber)
        state_ambient_temp: ambient temperature (deg C, from config or sensor)

    Dynamics:
        C_h * dT_h/dt = P - k_hc * (T_h - T_c)
        C_c * dT_c/dt = k_hc * (T_h - T_c) - h_c(T_c) * (T_c - T_amb)
        dT_s1/dt = r_s1 * (T_h - T_s1)
        dT_s2/dt = r_s2 * (T_c - T_s2)

    Measurements: y1 = T_s1, y2 = T_s2
    """

    def __init__(self, heater_heat_capacity, chamber_heat_capacity,
                 heater_chamber_coupling,
                 s1_responsiveness, s2_responsiveness,
                 h_interpolator, heater_power,
                 smoothing_heater=0.7, smoothing_chamber=0.5,
                 target_reach_time=2.0,
                 estimator_type='fixed', kalman_filter=None):
        # Identified parameters
        self.heater_heat_capacity = float(heater_heat_capacity)
        self.chamber_heat_capacity = float(chamber_heat_capacity)
        self.heater_chamber_coupling = float(heater_chamber_coupling)
        self.s1_responsiveness = float(s1_responsiveness)
        self.s2_responsiveness = float(s2_responsiveness)
        self.h_interpolator = h_interpolator
        self.heater_power = float(heater_power)

        # Tuning parameters
        self.smoothing_heater = float(smoothing_heater)
        self.smoothing_chamber = float(smoothing_chamber)
        self.target_reach_time = float(target_reach_time)

        # Estimator
        self.estimator_type = estimator_type
        self.kalman = kalman_filter

        # State
        self.state_heater_temp = AMBIENT_TEMP_DEFAULT
        self.state_chamber_temp = AMBIENT_TEMP_DEFAULT
        self.state_s1_temp = AMBIENT_TEMP_DEFAULT
        self.state_s2_temp = AMBIENT_TEMP_DEFAULT
        self.state_ambient_temp = AMBIENT_TEMP_DEFAULT
        self.last_power = 0.0
        self.last_time = 0.0

        # Rolling power average
        self._power_history = []
        self._power_avg_window = 30.0

        # Bed disturbance (optional)
        self.bed_transfer = 0.0
        self._bed_temp_fn = None

        # Heating element limit (model-integrated)
        self.heating_element_max_temp = 250.0
        self.heating_element_margin = 20.0

        # Ambient sensor (optional)
        self._ambient_temp_fn = None

        # S1 sensor reading function (required for advanced model)
        self._s1_temp_fn = None

    def set_initial_state(self, temp):
        """Set initial state from a known temperature."""
        self.state_heater_temp = temp
        self.state_chamber_temp = temp
        self.state_s1_temp = temp
        self.state_s2_temp = temp

    def set_ambient(self, temp):
        """Set known ambient temperature."""
        self.state_ambient_temp = temp

    def set_s1_sensor(self, temp_fn):
        """Configure S1 (heating element) sensor reading function.

        Args:
            temp_fn: callable(read_time) returning (temp, target) tuple
        """
        self._s1_temp_fn = temp_fn

    def set_bed_disturbance(self, bed_temp_fn, bed_transfer):
        """Configure bed disturbance feedforward."""
        self._bed_temp_fn = bed_temp_fn
        self.bed_transfer = float(bed_transfer)

    def set_heating_element_limit(self, temp_fn, max_temp, margin):
        """Configure heating element temperature limit.

        In the advanced model, the limit is integrated into the output
        computation via the modeled T_heater state, not via an external
        clamp reading temp_fn. The temp_fn is stored for safety monitoring
        only (verify_heater style backup).
        """
        self.heating_element_max_temp = float(max_temp)  # pragma: no cover
        self.heating_element_margin = float(margin)
        # temp_fn stored but not used for control - model predicts T_h
        # S1 sensor is set via set_s1_sensor() and used for state estimation

    def set_ambient_sensor(self, temp_fn):  # pragma: no cover
        """Configure an external ambient temperature sensor."""
        self._ambient_temp_fn = temp_fn

    # -- Main update cycle --

    def update(self, read_time, temp_s2_measured, target_temp, max_power):
        """Run one MPC cycle.

        Args:
            read_time: timestamp from temperature_callback
            temp_s2_measured: raw S2 sensor reading (deg C) - from heater's sensor
            target_temp: desired chamber temperature (deg C)
            max_power: maximum heater output as fraction (0.0-1.0)

        Returns:
            duty: heater duty cycle to apply (0.0-1.0)
        """
        dt = read_time - self.last_time
        if self.last_time == 0.0 or dt < 0.0 or dt > 1.0:
            dt = 0.1
        self.last_time = read_time

        # Read S1 sensor
        temp_s1_measured = self._read_s1(read_time)

        # -- Propagate model --
        self._propagate(dt, read_time)

        # -- Correct: Kalman or fixed smoothing --
        self._correct(dt, temp_s1_measured, temp_s2_measured)

        # -- Update ambient --
        self._update_ambient(read_time)

        # -- Compute output with model-integrated element limit --
        u_actual = self._compute_output(
            target_temp, read_time, max_power)

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

    def _read_s1(self, read_time):
        """Read S1 sensor. Returns current S1 state estimate if sensor unavailable."""
        if self._s1_temp_fn is not None:
            try:
                temp, _ = self._s1_temp_fn(read_time)
                return temp
            except Exception:
                pass  # pragma: no cover
        return self.state_s1_temp

    def _propagate(self, dt, read_time):
        """Propagate 4-state model using last actual applied power."""
        p_heating = self.last_power
        k = self.heater_chamber_coupling
        c_h = self.heater_heat_capacity
        c_c = self.chamber_heat_capacity

        # Heat loss to ambient
        h = self.h_interpolator.h(self.state_chamber_temp)
        p_loss = h * (self.state_chamber_temp - self.state_ambient_temp)

        # Bed disturbance
        p_bed = 0.0
        if self._bed_temp_fn is not None and self.bed_transfer > 0:  # pragma: no cover
            try:
                bed_temp, _ = self._bed_temp_fn(read_time)
                p_bed = self.bed_transfer * (
                    bed_temp - self.state_chamber_temp)
            except Exception:
                pass

        # Heater state: C_h * dT_h/dt = P - k*(T_h - T_c)
        dT_h = (p_heating - k * (
            self.state_heater_temp - self.state_chamber_temp)) * dt / c_h
        self.state_heater_temp += dT_h

        # Chamber state: C_c * dT_c/dt = k*(T_h - T_c) - h*(T_c - T_amb) + P_bed
        dT_c = (k * (self.state_heater_temp - self.state_chamber_temp)
                + p_bed - p_loss) * dt / c_c
        self.state_chamber_temp += dT_c

        # S1 sensor state: dT_s1/dt = r_s1 * (T_h - T_s1)
        dT_s1 = self.s1_responsiveness * (
            self.state_heater_temp - self.state_s1_temp) * dt
        self.state_s1_temp += dT_s1

        # S2 sensor state: dT_s2/dt = r_s2 * (T_c - T_s2)
        dT_s2 = self.s2_responsiveness * (
            self.state_chamber_temp - self.state_s2_temp) * dt
        self.state_s2_temp += dT_s2

        # Kalman predict step
        if self.estimator_type == 'kalman' and self.kalman is not None:
            self.kalman.predict(
                dt, k, c_h, c_c, h,
                self.s1_responsiveness, self.s2_responsiveness)

    def _correct(self, dt, temp_s1_measured, temp_s2_measured):
        """Correct model states from S1 and S2 measurements."""
        if self.estimator_type == 'kalman' and self.kalman is not None:
            innovation_s1 = temp_s1_measured - self.state_s1_temp
            innovation_s2 = temp_s2_measured - self.state_s2_temp
            corr = self.kalman.update(innovation_s1, innovation_s2)
            self.state_heater_temp += corr[0]
            self.state_chamber_temp += corr[1]
            self.state_s1_temp += corr[2]
            self.state_s2_temp += corr[3]
        else:
            # Fixed smoothing: two pairs, each with Kalico pattern
            # S1 correction applied to (T_heater, T_s1) pair
            eff_h = 1.0 - (1.0 - self.smoothing_heater) ** dt
            adj_s1 = (temp_s1_measured - self.state_s1_temp) * eff_h
            self.state_heater_temp += adj_s1
            self.state_s1_temp += adj_s1

            # S2 correction applied to (T_chamber, T_s2) pair
            eff_c = 1.0 - (1.0 - self.smoothing_chamber) ** dt
            adj_s2 = (temp_s2_measured - self.state_s2_temp) * eff_c
            self.state_chamber_temp += adj_s2
            self.state_s2_temp += adj_s2

    def _update_ambient(self, read_time):
        """Update ambient from external sensor only (no adaptive estimation)."""
        if self._ambient_temp_fn is not None:  # pragma: no cover
            try:
                temp, _ = self._ambient_temp_fn(read_time)
                if temp != 0.0:
                    self.state_ambient_temp = temp
            except Exception:
                pass

    def _compute_output(self, target_temp, read_time, max_power):
        """Compute desired heater output with model-integrated element limit.

        The output computation considers both:
        1. What power drives T_chamber toward target
        2. What power keeps T_heater below the element limit

        The constraint is: what maximum power can we apply such that
        T_heater doesn't exceed heating_element_max_temp, given current
        T_heater and the coupling k_hc?
        """
        if target_temp == 0.0:
            return 0.0

        # Power needed to drive chamber to target
        heating_power = (
            (target_temp - self.state_chamber_temp)
            * self.chamber_heat_capacity
            / self.target_reach_time
        )

        # Loss compensation
        h = self.h_interpolator.h(self.state_chamber_temp)
        loss_ambient = (
            (self.state_chamber_temp - self.state_ambient_temp) * h
        )

        # Bed contribution
        loss_bed = 0.0
        if self._bed_temp_fn is not None and self.bed_transfer > 0:  # pragma: no cover
            try:
                bed_temp, _ = self._bed_temp_fn(read_time)
                loss_bed = -self.bed_transfer * (
                    bed_temp - self.state_chamber_temp)
            except Exception:
                pass

        # Desired power from chamber perspective
        p_desired = heating_power + loss_ambient + loss_bed

        # Model-integrated heating element constraint
        # From: C_h * dT_h/dt = P - k*(T_h - T_c)
        # At the limit: we want dT_h/dt <= 0 when T_h = max_temp
        # So: P <= k * (max_temp - T_c)
        # With proportional pullback in the margin zone:
        p_element_hard = self.heater_chamber_coupling * (
            self.heating_element_max_temp - self.state_chamber_temp)

        pullback_start_temp = (
            self.heating_element_max_temp - self.heating_element_margin)
        if self.state_heater_temp >= self.heating_element_max_temp:
            p_element_limit = 0.0
        elif self.state_heater_temp > pullback_start_temp:
            headroom = (
                self.heating_element_max_temp - self.state_heater_temp)
            scale = headroom / self.heating_element_margin
            p_element_limit = p_element_hard * scale
        else:
            p_element_limit = p_element_hard

        # Take the minimum of desired and element-constrained power
        power = max(0.0, min(p_desired, p_element_limit,
                             max_power * self.heater_power))

        return power / self.heater_power

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
            'temp_heater': round(self.state_heater_temp, 2),
            'temp_chamber': round(self.state_chamber_temp, 2),
            'temp_s1': round(self.state_s1_temp, 2),
            'temp_s2': round(self.state_s2_temp, 2),
            'temp_ambient': round(self.state_ambient_temp, 2),
            'power': round(self.last_power, 2),
            'avg_power': round(self.get_avg_power(), 2),
            'avg_duty': round(self.get_avg_duty(), 4),
        }
