# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: __init__.py
# Description: Klippy plugin entry point for chamber_mpc.
#              Configures MPC control on a heater_generic at startup
#              by replacing its default control algorithm.
#              Uses the 2-state model with Kalman filtering for state
#              estimation.

try:
    from chamber_mpc.__version__ import version as __version__
except ImportError:
    __version__ = "unknown"

import logging

from .control import ControlMPCChamber
from .temp_sensor import ChamberMpcTempSensor


class ChamberMpcModule:
    """Klipper module for [chamber_mpc] config section.

    Reads MPC chamber configuration, validates parameter consistency,
    builds the model and estimator, and replaces the heater's control
    at klippy:ready.

    Uses the 2-state model with Kalman filtering and disturbance
    estimation.
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

        # Register virtual temperature_sensor type (mpc_chamber_temperature)
        pheaters = self.printer.load_object(config, 'heaters')
        if hasattr(pheaters, 'add_sensor_factory'):
            pheaters.add_sensor_factory(
                'mpc_chamber_temperature', ChamberMpcTempSensor)

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

        # Model parameters
        profile['chamber_heat_capacity'] = config.getfloat(
            'chamber_heat_capacity', default=None)
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

        # Optional: [fan_generic] tied to chamber heating & calibration
        profile['chamber_fan'] = config.get('chamber_fan', default=None)
        profile['chamber_fan_speed'] = config.getfloat(
            'chamber_fan_speed', 1.0, minval=0.0, maxval=1.0)

        return profile

    def _validate_config(self, heater):
        """Validate that the calibrated model parameters are present.

        Returns list of error/warning messages. Empty list = valid.
        """
        p = self.profile
        issues = []

        if p['chamber_heat_capacity'] is None:
            issues.append("chamber_heat_capacity not calibrated")
        if p['s1_responsiveness'] is None:
            issues.append("s1_responsiveness not calibrated")

        h_raw = p.get('ambient_transfer_points')
        ambient_transfer = p.get('ambient_transfer')
        if h_raw is None and ambient_transfer is None:
            issues.append(
                "ambient_transfer_points or ambient_transfer not calibrated")

        # Validate s2_safe_temp > heater max_temp when secondary sensor is used
        if p['secondary_sensor']:
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
            self.log.warning(
                "MPC chamber [%s] config mismatch:",
                self.heater_name)
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
                "MPC chamber control active on '%s'",
                self.heater_name)
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
                "MPC chamber: not active (check log for issues)")
            return
        self.control.cmd_MPC_CHAMBER_STATUS(gcmd)

    def get_status(self, eventtime):
        """Expose status for Moonraker."""
        if self.control and self.control.is_valid():
            return self.control.get_status(eventtime)
        return {'state': 'uncalibrated'}


def load_config(config):
    return ChamberMpcModule(config)
