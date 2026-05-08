# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: __init__.py
# Description: Klippy plugin entry point for chamber_mpc.
#              Configures MPC control on a heater_generic at startup
#              by replacing its default control algorithm.
#              Supports four control modes:
#                basic+fixed, basic+kalman, advanced+fixed, advanced+kalman

try:
    from chamber_mpc.__version__ import version as __version__
except ImportError:
    __version__ = "unknown"

import logging

from .control import ControlMPCChamber
from .h_interpolator import HInterpolator

VALID_MODEL_TYPES = ('basic', 'advanced')
VALID_ESTIMATOR_TYPES = ('fixed', 'kalman')


class ChamberMpcModule:
    """Klipper module for [chamber_mpc] config section.

    Reads MPC chamber configuration, validates parameter consistency,
    builds the appropriate model+estimator combination, and replaces
    the heater's control at klippy:ready.

    Four control modes from two config parameters:
        model_type: basic (2-state) or advanced (4-state)
        estimator_type: fixed (constant smoothing) or kalman (adaptive gains)
    """
    def __init__(self, config):
        self.printer = config.get_printer()
        self.config = config
        self.log = logging.getLogger('chamber_mpc')
        self.heater_name = config.get('heater')
        self.control = None

        # Parse all MPC parameters into a profile dict
        self.profile = self._parse_profile(config)
        self.profile['name'] = 'chamber_mpc'

        # Register GCode commands during config loading phase
        gcode = self.printer.lookup_object('gcode')
        gcode.register_mux_command(
            'MPC_CHAMBER_CALIBRATE', 'HEATER', self.heater_name,
            self.cmd_MPC_CHAMBER_CALIBRATE,
            desc="Run MPC chamber calibration")
        gcode.register_mux_command(
            'MPC_CHAMBER_STATUS', 'HEATER', self.heater_name,
            self.cmd_MPC_CHAMBER_STATUS,
            desc="Report MPC chamber model state")

        self.printer.register_event_handler(
            'klippy:ready', self._handle_ready)

    def _parse_profile(self, config):
        """Parse MPC parameters from config into a profile dict."""
        profile = {}

        # Required
        profile['heater_power'] = config.getfloat('heater_power')

        # Model and estimator type
        profile['model_type'] = config.get('model_type', 'basic').lower()
        profile['estimator_type'] = config.get(
            'estimator_type', 'fixed').lower()

        if profile['model_type'] not in VALID_MODEL_TYPES:
            raise config.error(
                "model_type must be 'basic' or 'advanced', got '%s'"
                % profile['model_type'])
        if profile['estimator_type'] not in VALID_ESTIMATOR_TYPES:
            raise config.error(
                "estimator_type must be 'fixed' or 'kalman', got '%s'"
                % profile['estimator_type'])

        # Basic model parameters (always parsed)
        profile['chamber_heat_capacity'] = config.getfloat(
            'chamber_heat_capacity', default=None)
        profile['sensor_responsiveness'] = config.getfloat(
            'sensor_responsiveness', default=None)

        # Advanced model parameters (optional, only for advanced mode)
        profile['heater_heat_capacity'] = config.getfloat(
            'heater_heat_capacity', default=None)
        profile['heater_chamber_coupling'] = config.getfloat(
            'heater_chamber_coupling', default=None)
        profile['s1_responsiveness'] = config.getfloat(
            's1_responsiveness', default=None)
        # s2_responsiveness = sensor_responsiveness (same parameter)

        # h(T) calibration points
        h_raw = config.get('h_calibration_points', default=None)
        if h_raw is not None:
            profile['h_calibration_points'] = h_raw
        else:
            profile['ambient_transfer'] = config.getfloat(
                'ambient_transfer', default=None)

        # Fixed estimator parameters
        profile['smoothing'] = config.getfloat('smoothing', 0.5)
        profile['smoothing_heater'] = config.getfloat(
            'smoothing_heater', default=None)
        profile['smoothing_chamber'] = config.getfloat(
            'smoothing_chamber', default=None)

        # Kalman estimator parameters
        profile['process_noise_chamber'] = config.getfloat(
            'process_noise_chamber', default=None)
        profile['process_noise_sensor'] = config.getfloat(
            'process_noise_sensor', default=None)
        profile['measurement_noise'] = config.getfloat(
            'measurement_noise', default=None)
        # Advanced Kalman parameters
        profile['process_noise_heater'] = config.getfloat(
            'process_noise_heater', default=None)
        profile['process_noise_s1'] = config.getfloat(
            'process_noise_s1', default=None)
        profile['process_noise_s2'] = config.getfloat(
            'process_noise_s2', default=None)
        profile['measurement_noise_s1'] = config.getfloat(
            'measurement_noise_s1', default=None)
        profile['measurement_noise_s2'] = config.getfloat(
            'measurement_noise_s2', default=None)

        # General tuning
        profile['target_reach_time'] = config.getfloat(
            'target_reach_time', 2.0)
        profile['ambient_temp'] = config.getfloat(
            'ambient_temp', 25.0)
        profile['max_temp_margin'] = config.getfloat(
            'max_temp_margin', 5.0)

        # Optional: bed disturbance feedforward
        profile['bed_heater'] = config.get('bed_heater', default=None)
        profile['bed_transfer'] = config.getfloat('bed_transfer', 0.0)

        # Optional: heating element temperature limiting
        profile['heating_element_sensor'] = config.get(
            'heating_element_sensor', default=None)
        profile['heating_element_max_temp'] = config.getfloat(
            'heating_element_max_temp', 250.0)
        profile['heating_element_margin'] = config.getfloat(
            'heating_element_margin', 20.0)

        # Optional: ambient temperature sensor
        profile['ambient_temp_sensor'] = config.get(
            'ambient_temp_sensor', default=None)

        return profile

    def _validate_config(self, heater):
        """Validate that calibrated parameters match the selected mode.

        Returns list of error/warning messages. Empty list = valid.
        """
        p = self.profile
        model = p['model_type']
        estimator = p['estimator_type']
        issues = []

        # Basic model requires
        if p['chamber_heat_capacity'] is None:
            issues.append("chamber_heat_capacity not calibrated")
        if p['sensor_responsiveness'] is None:
            issues.append("sensor_responsiveness not calibrated")

        h_raw = p.get('h_calibration_points')
        ambient_transfer = p.get('ambient_transfer')
        if h_raw is None and ambient_transfer is None:
            issues.append(
                "h_calibration_points or ambient_transfer not calibrated")

        # Advanced model additionally requires
        if model == 'advanced':
            if p['heater_heat_capacity'] is None:
                issues.append(
                    "model_type=advanced requires heater_heat_capacity "
                    "(not calibrated)")
            if p['heater_chamber_coupling'] is None:
                issues.append(
                    "model_type=advanced requires heater_chamber_coupling "
                    "(not calibrated)")
            if p['s1_responsiveness'] is None:
                issues.append(
                    "model_type=advanced requires s1_responsiveness "
                    "(not calibrated)")
            if p['heating_element_sensor'] is None:
                issues.append(
                    "model_type=advanced requires heating_element_sensor "
                    "(S1 sensor for heater state observation)")

        # Kalman estimator requires noise parameters
        if estimator == 'kalman':
            if model == 'basic':
                if p['process_noise_chamber'] is None:
                    issues.append(
                        "estimator_type=kalman requires "
                        "process_noise_chamber (not calibrated)")
                if p['process_noise_sensor'] is None:
                    issues.append(
                        "estimator_type=kalman requires "
                        "process_noise_sensor (not calibrated)")
                if p['measurement_noise'] is None:
                    issues.append(
                        "estimator_type=kalman requires "
                        "measurement_noise (not calibrated)")
            elif model == 'advanced':
                for param in ('process_noise_heater',
                              'process_noise_chamber',
                              'process_noise_s1', 'process_noise_s2',
                              'measurement_noise_s1',
                              'measurement_noise_s2'):
                    if p[param] is None:
                        issues.append(
                            "estimator_type=kalman with model_type=advanced "
                            "requires %s (not calibrated)" % param)

        # Validate heating_element_max_temp > heater max_temp
        he_max = p['heating_element_max_temp']
        heater_max = heater.max_temp
        if he_max <= heater_max:
            issues.append(
                "heating_element_max_temp (%.0f) must be greater than "
                "heater max_temp (%.0f)" % (he_max, heater_max))

        return issues

    def _handle_ready(self):
        """Validate config, build control, replace heater's controller."""
        pheaters = self.printer.lookup_object('heaters')
        heater = pheaters.lookup_heater(self.heater_name)

        # Validate config matches selected mode
        issues = self._validate_config(heater)
        if issues:
            mode_str = "%s+%s" % (
                self.profile['model_type'],
                self.profile['estimator_type'])
            self.log.warning(
                "MPC chamber [%s] config mismatch for mode '%s':",
                self.heater_name, mode_str)
            for issue in issues:
                self.log.warning("  - %s", issue)
            self.log.warning(
                "Falling back to default control (PID). "
                "Run MPC_CHAMBER_CALIBRATE HEATER=%s to calibrate.",
                self.heater_name)
            return

        self.control = ControlMPCChamber(
            self.profile, heater, load_clean=False)

        if self.control.is_valid():
            heater.set_control(self.control)
            self.log.info(
                "MPC chamber control active on '%s' "
                "(model=%s, estimator=%s)",
                self.heater_name,
                self.profile['model_type'],
                self.profile['estimator_type'])
        else:
            self.log.warning(
                "MPC chamber failed to initialize for '%s'. "
                "Using default control.",
                self.heater_name)

    def cmd_MPC_CHAMBER_CALIBRATE(self, gcmd):
        if self.control is None:
            # Allow calibration even without valid control
            # (first-time calibration scenario)
            pheaters = self.printer.lookup_object('heaters')
            heater = pheaters.lookup_heater(self.heater_name)
            temp_control = ControlMPCChamber(
                self.profile, heater, load_clean=True)
            temp_control.cmd_MPC_CHAMBER_CALIBRATE(gcmd)
        else:
            self.control.cmd_MPC_CHAMBER_CALIBRATE(gcmd)

    def cmd_MPC_CHAMBER_STATUS(self, gcmd):
        if self.control is None:
            gcmd.respond_info(
                "MPC chamber: not active (mode=%s+%s, check log for issues)"
                % (self.profile['model_type'],
                   self.profile['estimator_type']))
            return
        self.control.cmd_MPC_CHAMBER_STATUS(gcmd)

    def get_status(self, eventtime):
        """Expose status for Moonraker."""
        if self.control and self.control.is_valid():
            return self.control.get_status(eventtime)
        return {
            'state': 'uncalibrated',
            'model_type': self.profile['model_type'],
            'estimator_type': self.profile['estimator_type'],
        }


def load_config(config):
    return ChamberMpcModule(config)
