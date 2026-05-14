# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: __init__.py
# Description: Klippy plugin entry point for chamber_mpc.
#              Configures MPC control on a heater_generic at startup
#              by replacing its default control algorithm.
#              Supports two model types: basic (2-state) and advanced (4-state).
#              Kalman filtering is used for state estimation in both.

try:
    from chamber_mpc.__version__ import version as __version__
except ImportError:
    __version__ = "unknown"

import logging

from .control import ControlMPCChamber
from .h_interpolator import HInterpolator

VALID_MODEL_TYPES = ('basic', 'advanced')


class ChamberMpcModule:
    """Klipper module for [chamber_mpc] config section.

    Reads MPC chamber configuration, validates parameter consistency,
    builds the appropriate model+estimator combination, and replaces
    the heater's control at klippy:ready.

    Model type: basic (2-state) or advanced (4-state).
    Kalman filtering with disturbance estimation for all modes.
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

        if profile['model_type'] not in VALID_MODEL_TYPES:
            raise config.error(
                "model_type must be 'basic' or 'advanced', got '%s'"
                % profile['model_type'])

        # Basic model parameters (always parsed)
        profile['chamber_heat_capacity'] = config.getfloat(
            'chamber_heat_capacity', default=None)
        profile['s1_responsiveness'] = config.getfloat(
            's1_responsiveness', default=None)

        # Advanced model parameters (optional, only for advanced mode)
        profile['heater_heat_capacity'] = config.getfloat(
            'heater_heat_capacity', default=None)
        profile['heating_element_transfer'] = config.getfloat(
            'heating_element_transfer', default=None)
        profile['s1_responsiveness'] = config.getfloat(
            's1_responsiveness', default=None)
        

        # h(T) calibration points
        h_raw = config.get('ambient_transfer_points', default=None)
        if h_raw is not None:
            profile['ambient_transfer_points'] = h_raw
        else:
            profile['ambient_transfer'] = config.getfloat(
                'ambient_transfer', default=None)


        # Kalman estimator parameters
        profile['process_noise_chamber'] = config.getfloat(
            'process_noise_chamber', 1.0)
        profile['process_noise_s1'] = config.getfloat(
            'process_noise_s1', 0.1)
        profile['measurement_noise_s1'] = config.getfloat(
            'measurement_noise_s1', 0.5)
        profile['process_noise_disturbance'] = config.getfloat(
            'process_noise_disturbance', 10.0)
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
        profile['secondary_sensor'] = config.get(
            'secondary_sensor', default=None)
        profile['s2_safe_temp'] = config.getfloat(
            's2_safe_temp', 250.0)
        profile['s2_safe_temp_zone'] = config.getfloat(
            's2_safe_temp_zone', 20.0)

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
        issues = []

        # Basic model requires
        if p['chamber_heat_capacity'] is None:
            issues.append("chamber_heat_capacity not calibrated")
        if p['s1_responsiveness'] is None:
            issues.append("s1_responsiveness not calibrated")

        h_raw = p.get('ambient_transfer_points')
        ambient_transfer = p.get('ambient_transfer')
        if h_raw is None and ambient_transfer is None:
            issues.append(
                "ambient_transfer_points or ambient_transfer not calibrated")

        # Advanced model additionally requires
        if model == 'advanced':
            if p['heater_heat_capacity'] is None:
                issues.append(
                    "model_type=advanced requires heater_heat_capacity "
                    "(not calibrated)")
            if p['heating_element_transfer'] is None:
                issues.append(
                    "model_type=advanced requires heating_element_transfer "
                    "(not calibrated)")
            if p['s1_responsiveness'] is None:
                issues.append(
                    "model_type=advanced requires s1_responsiveness "
                    "(not calibrated)")
            if p['secondary_sensor'] is None:
                issues.append(
                    "model_type=advanced requires secondary_sensor "
                    "(S1 sensor for heater state observation)")

        # Advanced Kalman requires additional noise parameters
        if model == 'advanced':
            for param in ('process_noise_heater',
                          'process_noise_chamber',
                          'process_noise_s1', 'process_noise_s2',
                          'measurement_noise_s1',
                          'measurement_noise_s2'):
                if p[param] is None:
                    issues.append(
                        "model_type=advanced requires %s "
                        "(not calibrated)" % param)

        # Validate s2_safe_temp > heater max_temp
        s2_max = p['s2_safe_temp']
        heater_max = heater.max_temp
        if s2_max <= heater_max:
            issues.append(
                "s2_safe_temp (%.0f) must be greater than "
                "heater max_temp (%.0f)" % (s2_max, heater_max))

        return issues

    def _handle_ready(self):
        """Validate config, build control, replace heater's controller."""
        pheaters = self.printer.lookup_object('heaters')
        heater = pheaters.lookup_heater(self.heater_name)

        # Validate config matches selected mode
        issues = self._validate_config(heater)
        if issues:
            mode_str = self.profile['model_type']
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
                "MPC chamber control active on '%s' (model=%s)",
                self.heater_name,
                self.profile['model_type'])
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
                "MPC chamber: not active (model=%s, check log for issues)"
                % self.profile['model_type'])
            return
        self.control.cmd_MPC_CHAMBER_STATUS(gcmd)

    def get_status(self, eventtime):
        """Expose status for Moonraker."""
        if self.control and self.control.is_valid():
            return self.control.get_status(eventtime)
        return {
            'state': 'uncalibrated',
            'model_type': self.profile['model_type'],
        }


def load_config(config):
    return ChamberMpcModule(config)
