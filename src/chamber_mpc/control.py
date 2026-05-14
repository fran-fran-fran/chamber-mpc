# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: control.py
# Description: Klipper heater control class that builds the appropriate
#              thermal model and estimator from the profile configuration.

import logging

from .thermal_model import ThermalModel
from .thermal_model_advanced import ThermalModelAdvanced
from .kalman import KalmanFilter3, KalmanFilter4
from .h_interpolator import HInterpolator


class ControlMPCChamber:
    """Klipper heater control implementation for chamber MPC.

    Supports four control modes built from two config parameters:
        basic+kalman:   2-state model, Kalman filter correction
        advanced+kalman: 4-state model, Kalman filter correction
    """

    def __init__(self, profile, heater, load_clean=False):
        self.profile = profile
        self.heater = heater
        self.printer = heater.printer
        self.log = logging.getLogger('chamber_mpc')

        # Build model and estimator
        self._build_model(profile, heater, load_clean)

        # Max temp clamping with margin
        self.max_temp = heater.max_temp
        self.max_temp_margin = profile.get('max_temp_margin', 5.0)
        self._cap_warned = False

        # Deferred setup (resolve sensor references)
        self.printer.register_event_handler(
            'klippy:ready', self._handle_ready)

    def _build_model(self, profile, heater, load_clean):
        """Build the appropriate model+estimator from profile."""
        model_type = profile.get('model_type', 'basic')
        heater_power = profile.get('heater_power', 0)

        # Build h interpolator
        h_interp = self._build_h_interpolator(profile)
        if h_interp is None:
            self.model = None
            self._valid = False
            return

        # Common parameters
        chamber_heat_capacity = profile.get('chamber_heat_capacity')
        s1_responsiveness = profile.get('s1_responsiveness')
        target_reach_time = profile.get('target_reach_time', 2.0)

        if not chamber_heat_capacity or not s1_responsiveness:
            self.model = None
            self._valid = False
            return

        if model_type == 'advanced':
            self._build_advanced_model(
                profile, h_interp, heater, load_clean)
        else:
            self._build_basic_model(
                profile, h_interp, heater, load_clean)

        # Store config names for deferred lookup
        self._bed_heater_name = profile.get('bed_heater')
        self._bed_transfer = profile.get('bed_transfer', 0.0)
        self._secondary_sensor_name = profile.get(
            'secondary_sensor')
        self._s2_safe_temp = profile.get(
            's2_safe_temp', 250.0)
        self._s2_safe_temp_zone = profile.get(
            's2_safe_temp_zone', 20.0)
        self._ambient_sensor_name = profile.get('ambient_temp_sensor')

    def _build_h_interpolator(self, profile):
        """Build HInterpolator from profile parameters."""
        h_raw = profile.get('ambient_transfer_points')
        ambient_transfer = profile.get('ambient_transfer')

        if h_raw is not None:
            if isinstance(h_raw, str):
                h_points = HInterpolator.parse_config_string(h_raw)
            else:
                h_points = h_raw
            return HInterpolator(h_points)
        elif ambient_transfer is not None:
            return HInterpolator([(100.0, ambient_transfer)])
        return None

    def _build_basic_model(self, profile, h_interp, heater, load_clean):
        """Build 2-state basic model with Kalman estimator."""
        kalman = KalmanFilter3(
            process_noise_chamber=profile.get(
                'process_noise_chamber', 1.0),
            process_noise_s1=profile.get(
                'process_noise_s1', 0.1),
            process_noise_disturbance=profile.get(
                'process_noise_disturbance', 10.0),
            measurement_noise_s1=profile.get(
                'measurement_noise_s1', 0.5),
        )

        self.model = ThermalModel(
            chamber_heat_capacity=profile.get('chamber_heat_capacity'),
            s1_responsiveness=profile.get('s1_responsiveness'),
            h_interpolator=h_interp,
            heater_power=profile.get('heater_power'),
            target_reach_time=profile.get('target_reach_time', 2.0),
            kalman_filter=kalman,
        )

        if not load_clean:
            temp = heater.get_temp(heater.reactor.monotonic())[0]
            self.model.set_initial_state(temp)
        ambient = profile.get('ambient_temp', 25.0)
        self.model.set_ambient(ambient)
        self._valid = True

    def _build_advanced_model(self, profile, h_interp, heater, load_clean):
        """Build 4-state advanced model with Kalman estimator."""

        heater_heat_capacity = profile.get('heater_heat_capacity')
        heating_element_transfer = profile.get('heating_element_transfer')
        s1_responsiveness = profile.get('s1_responsiveness')

        if (not heater_heat_capacity or not heating_element_transfer
                or not s1_responsiveness):
            self.model = None
            self._valid = False
            return

        kalman = KalmanFilter4(
                process_noise_heater=profile.get(
                    'process_noise_heater', 1.0),
                process_noise_chamber=profile.get(
                    'process_noise_chamber', 1.0),
                process_noise_s1=profile.get(
                    'process_noise_s1', 0.5),
                process_noise_s2=profile.get(
                    'process_noise_s2', 0.5),
                measurement_noise_s1=profile.get(
                    'measurement_noise_s1', 0.5),
                measurement_noise_s2=profile.get(
                    'measurement_noise_s2', 0.5),
        )

        self.model = ThermalModelAdvanced(
            heater_heat_capacity=heater_heat_capacity,
            chamber_heat_capacity=profile.get('chamber_heat_capacity'),
            heating_element_transfer=heating_element_transfer,
            s1_responsiveness=s1_responsiveness,
            s2_responsiveness=profile.get('s2_responsiveness'),
            h_interpolator=h_interp,
            heater_power=profile.get('heater_power'),
            target_reach_time=profile.get('target_reach_time', 2.0),
            kalman_filter=kalman,
        )

        if not load_clean:
            temp = heater.get_temp(heater.reactor.monotonic())[0]
            self.model.set_initial_state(temp)
        ambient = profile.get('ambient_temp', 25.0)
        self.model.set_ambient(ambient)
        self._valid = True

    def _handle_ready(self):
        """Resolve named references to other Klipper objects."""
        if self.model is None:
            return

        printer = self.printer
        is_advanced = isinstance(self.model, ThermalModelAdvanced)

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

        # Heating element sensor
        if self._secondary_sensor_name:
            try:
                sensor = printer.lookup_object(
                    'temperature_sensor %s' %
                    self._secondary_sensor_name)

                if is_advanced:
                    # Advanced model: S1 sensor for state observation
                    self.model.set_s1_sensor(sensor.get_temp)
                    self.model.set_secondary_sensor_limit(
                        sensor.get_temp,
                        self._s2_safe_temp,
                        self._s2_safe_temp_zone)
                    self.log.info(
                        "Heating element sensor (S1) enabled: %s "
                        "(model-integrated limit, max=%.0f, margin=%.0f)",
                        self._secondary_sensor_name,
                        self._s2_safe_temp,
                        self._s2_safe_temp_zone)
                else:
                    # Basic model: external clamp
                    self.model.set_secondary_sensor_limit(
                        sensor.get_temp,
                        self._s2_safe_temp,
                        self._s2_safe_temp_zone)
                    self.log.info(
                        "Secondary sensor limit enabled: %s "
                        "(external clamp, max=%.0f, margin=%.0f)",
                        self._secondary_sensor_name,
                        self._s2_safe_temp,
                        self._s2_safe_temp_zone)
            except Exception as e:
                self.log.warning(
                    "Could not set up secondary sensor sensor '%s': %s",
                    self._secondary_sensor_name, e)

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
        is_advanced = isinstance(self.model, ThermalModelAdvanced)

        if is_advanced:
            gcmd.respond_info(
                "MPC chamber [advanced]: "
                "heater=%.1f deg C, chamber=%.1f deg C, "
                "s1=%.1f deg C, s2=%.1f deg C, "
                "ambient=%.1f deg C, power=%.1f W, "
                "avg_power=%.1f W (%.0f%%)"
                % (status['temp_heater'], status['temp_chamber'],
                   status['temp_s1'], status['temp_s2'],
                   status['temp_ambient'], status['power'],
                   status['avg_power'], status['avg_duty'] * 100))
        else:
            k = status.get('kalman_gain')
            k_str = ""
            if k is not None:
                k_str = ", kalman_gain=[%s]" % ", ".join("%.4f" % v for v in k)
            d = status.get('disturbance', 0.0)
            d_str = ""
            if d != 0.0:
                d_str = ", d=%.1f W" % d
            gcmd.respond_info(
                "MPC chamber [basic]: "
                "chamber=%.1f deg C, sensor=%.1f deg C, "
                "ambient=%.1f deg C, power=%.1f W, "
                "avg_power=%.1f W (%.0f%%)%s%s"
                % (status['temp_chamber'], status['temp_s1'],
                   status['temp_ambient'], status['power'],
                   status['avg_power'], status['avg_duty'] * 100,
                   d_str, k_str))

    # -- Status for Moonraker --

    def get_status(self, eventtime):
        if self.model is None:
            return {}
        return self.model.get_status()
