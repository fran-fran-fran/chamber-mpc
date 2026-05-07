# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: control.py
# Description: Klipper heater control class that integrates the thermal
#              model with Kalico's heater system. Registered as
#              control type 'mpc_chamber' in heater config.

import logging

from .thermal_model import ThermalModel
from .h_interpolator import HInterpolator


class ControlMPCChamber:
    """Klipper heater control implementation for chamber MPC.

    Registered as control type 'mpc_chamber'. Implements the same
    interface as Kalico's ControlPID and ControlMPC: temperature_update(),
    check_busy(), get_profile(), get_type().
    """

    def __init__(self, profile, heater, load_clean=False):
        self.profile = profile
        self.heater = heater
        self.printer = heater.printer
        self.log = logging.getLogger('chamber_mpc')

        # Parse model parameters from profile
        self._load_profile(profile, load_clean)

        # Max temp clamping with margin
        self.max_temp = heater.max_temp
        self.max_temp_margin = profile.get('max_temp_margin', 5.0)
        self._cap_warned = False

        # Deferred setup (needs other objects to be loaded)
        self.printer.register_event_handler(
            'klippy:ready', self._handle_ready)

    def _load_profile(self, profile, load_clean):
        """Initialize thermal model from profile parameters."""
        # Required parameters
        heater_power = profile.get('heater_power', 0)
        chamber_heat_capacity = profile.get('chamber_heat_capacity', None)
        sensor_responsiveness = profile.get('sensor_responsiveness', None)

        # h(T) calibration points
        h_points_raw = profile.get('h_calibration_points', None)
        ambient_transfer = profile.get('ambient_transfer', None)

        # Build interpolator
        if h_points_raw is not None:
            if isinstance(h_points_raw, str):
                h_points = HInterpolator.parse_config_string(h_points_raw)
            else:
                h_points = h_points_raw
            h_interp = HInterpolator(h_points)
        elif ambient_transfer is not None:
            # Single h value (backward compat or single-point calibration)
            h_interp = HInterpolator([(100.0, ambient_transfer)])
        else:
            h_interp = None

        # Tuning parameters
        smoothing = profile.get('smoothing', 0.5)
        target_reach_time = profile.get('target_reach_time', 2.0)
        min_ambient_change = profile.get('min_ambient_change', 1.0)
        steady_state_rate = profile.get('steady_state_rate', 0.5)

        # Build model (may be incomplete if not yet calibrated)
        if chamber_heat_capacity and sensor_responsiveness and h_interp:
            self.model = ThermalModel(
                chamber_heat_capacity=chamber_heat_capacity,
                sensor_responsiveness=sensor_responsiveness,
                h_interpolator=h_interp,
                heater_power=heater_power,
                smoothing=smoothing,
                target_reach_time=target_reach_time,
                min_ambient_change=min_ambient_change,
                steady_state_rate=steady_state_rate,
            )
            if not load_clean:
                temp = self.heater.get_temp(
                    self.heater.reactor.monotonic())[0]
                self.model.set_initial_state(temp)
            self._valid = True
        else:
            self.model = None
            self._valid = False

        # Store optional config names for deferred lookup
        self._bed_heater_name = profile.get('bed_heater', None)
        self._bed_transfer = profile.get('bed_transfer', 0.0)
        self._heating_element_sensor_name = profile.get(
            'heating_element_sensor', None)
        self._heating_element_max_temp = profile.get(
            'heating_element_max_temp', 300.0)
        self._heating_element_margin = profile.get(
            'heating_element_margin', 20.0)
        self._ambient_sensor_name = profile.get(
            'ambient_temp_sensor', None)

    def _handle_ready(self):
        """Resolve named references to other Klipper objects."""
        if self.model is None:
            return

        printer = self.printer

        # Bed disturbance feedforward
        if self._bed_heater_name:
            try:
                pheaters = printer.lookup_object('heaters')
                bed_heater = pheaters.lookup_heater(self._bed_heater_name)
                self.model.set_bed_disturbance(
                    bed_heater.get_temp, self._bed_transfer)
                self.log.info(
                    "Bed feedforward enabled: %s (h_bed=%.4f W/K)",
                    self._bed_heater_name, self._bed_transfer)
            except Exception as e:
                self.log.warning(
                    "Could not set up bed feedforward '%s': %s",
                    self._bed_heater_name, e)

        # Heating element limiter
        if self._heating_element_sensor_name:
            try:
                sensor = printer.lookup_object(
                    'temperature_sensor %s' %
                    self._heating_element_sensor_name)
                self.model.set_heating_element_limit(
                    sensor.get_temp,
                    self._heating_element_max_temp,
                    self._heating_element_margin)
                self.log.info(
                    "Heating element limit enabled: %s (max=%.0f, margin=%.0f)",
                    self._heating_element_sensor_name,
                    self._heating_element_max_temp,
                    self._heating_element_margin)
            except Exception as e:
                self.log.warning(
                    "Could not set up heating element limit '%s': %s",
                    self._heating_element_sensor_name, e)

        # Ambient temperature sensor
        if self._ambient_sensor_name:
            try:
                sensor = printer.lookup_object(
                    'temperature_sensor %s' % self._ambient_sensor_name)
                self.model.set_ambient_sensor(sensor.get_temp)
                self.log.info(
                    "Ambient sensor enabled: %s",
                    self._ambient_sensor_name)
            except Exception as e:
                self.log.warning(
                    "Could not set up ambient sensor '%s': %s",
                    self._ambient_sensor_name, e)

    # -- Klipper control interface --

    def temperature_update(self, read_time, temp, target_temp):
        """Called by Klipper's heater on every sensor reading."""
        if not self._valid or self.model is None:
            self.heater.set_pwm(read_time, 0.0)
            return

        # Setpoint cap with margin
        cap = self.max_temp - self.max_temp_margin
        if target_temp > cap:
            if not self._cap_warned:
                self.log.warning(
                    "mpc_chamber: target %.1f deg C clamped to %.1f deg C "
                    "(max_temp %.1f - margin %.1f)",
                    target_temp, cap, self.max_temp, self.max_temp_margin)
                self._cap_warned = True
            target_temp = cap
        else:
            self._cap_warned = False

        max_power = self.heater.max_power
        duty = self.model.update(read_time, temp, target_temp, max_power)
        self.heater.set_pwm(read_time, duty)

    def check_busy(self, eventtime, smoothed_temp, target_temp):
        """Check if heater is still working toward target."""
        return abs(target_temp - smoothed_temp) > 1.0

    def update_smooth_time(self):
        pass

    def get_profile(self):
        return self.profile

    def get_type(self):
        return 'mpc_chamber'

    def is_valid(self):
        return self._valid

    # -- GCode commands --

    cmd_MPC_CHAMBER_CALIBRATE_help = "Run MPC chamber calibration"

    def cmd_MPC_CHAMBER_CALIBRATE(self, gcmd):
        from .calibrate_runner import MpcChamberCalibrateRunner
        runner = MpcChamberCalibrateRunner(
            self.printer, self.heater, self)
        runner.run(gcmd)

    cmd_MPC_CHAMBER_STATUS_help = "Report MPC chamber model state"

    def cmd_MPC_CHAMBER_STATUS(self, gcmd):
        if self.model is None:
            gcmd.respond_info("MPC chamber: not calibrated")
            return
        status = self.model.get_status()
        gcmd.respond_info(
            "MPC chamber state: "
            "chamber=%.1f deg C, sensor=%.1f deg C, "
            "ambient=%.1f deg C, power=%.1f W"
            % (status['temp_chamber'], status['temp_sensor'],
               status['temp_ambient'], status['power']))

    # -- Status for Moonraker --

    def get_status(self, eventtime):
        if self.model is None:
            return {}
        return self.model.get_status()
