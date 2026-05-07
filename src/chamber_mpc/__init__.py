# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: __init__.py
# Description: Klippy plugin entry point for chamber_mpc.
#              Configures MPC control on a heater_generic at startup
#              by replacing its default control algorithm.

try:
    from chamber_mpc.__version__ import version as __version__
except ImportError:
    __version__ = "unknown"

import logging

from .control import ControlMPCChamber
from .h_interpolator import HInterpolator


class ChamberMpcModule:
    """Klipper module for [chamber_mpc] config section.

    Reads MPC chamber configuration, builds the control object,
    and replaces the heater's control at klippy:ready.

    Config example:
        [chamber_mpc]
        heater: airfryer
        heater_power: 1800
        # ... other MPC parameters
    """
    def __init__(self, config):
        self.printer = config.get_printer()
        self.config = config
        self.log = logging.getLogger('chamber_mpc')
        self.heater_name = config.get('heater')
        self.control = None

        # Parse all MPC parameters into a profile dict
        self.profile = self._parse_profile(config)

        self.printer.register_event_handler(
            'klippy:ready', self._handle_ready)

    def _parse_profile(self, config):
        """Parse MPC parameters from config into a profile dict."""
        profile = {}

        # Required
        profile['heater_power'] = config.getfloat('heater_power')

        # Calibrated parameters (may not exist before first calibration)
        profile['chamber_heat_capacity'] = config.getfloat(
            'chamber_heat_capacity', default=None)
        profile['sensor_responsiveness'] = config.getfloat(
            'sensor_responsiveness', default=None)

        # h(T) calibration points
        h_raw = config.get('h_calibration_points', default=None)
        if h_raw is not None:
            profile['h_calibration_points'] = h_raw
        else:
            profile['ambient_transfer'] = config.getfloat(
                'ambient_transfer', default=None)

        # Tuning parameters
        profile['smoothing'] = config.getfloat('smoothing', 0.5)
        profile['target_reach_time'] = config.getfloat(
            'target_reach_time', 2.0)
        profile['min_ambient_change'] = config.getfloat(
            'min_ambient_change', 1.0)
        profile['steady_state_rate'] = config.getfloat(
            'steady_state_rate', 0.5)
        profile['max_temp_margin'] = config.getfloat(
            'max_temp_margin', 5.0)

        # Optional: bed disturbance feedforward
        profile['bed_heater'] = config.get('bed_heater', default=None)
        profile['bed_transfer'] = config.getfloat(
            'bed_transfer', 0.0)

        # Optional: heating element temperature limiting
        profile['heating_element_sensor'] = config.get(
            'heating_element_sensor', default=None)
        profile['heating_element_max_temp'] = config.getfloat(
            'heating_element_max_temp', 300.0)
        profile['heating_element_margin'] = config.getfloat(
            'heating_element_margin', 20.0)

        # Optional: ambient temperature sensor
        profile['ambient_temp_sensor'] = config.get(
            'ambient_temp_sensor', default=None)

        return profile

    def _handle_ready(self):
        """Replace the heater's control with MPC chamber control."""
        pheaters = self.printer.lookup_object('heaters')
        heater = pheaters.lookup_heater(self.heater_name)

        self.control = ControlMPCChamber(
            self.profile, heater, load_clean=False)

        if self.control.is_valid():
            heater.set_control(self.control)
            self.log.info(
                "MPC chamber control active on '%s'", self.heater_name)
        else:
            self.log.warning(
                "MPC chamber not calibrated for '%s'. "
                "Run MPC_CHAMBER_CALIBRATE HEATER=%s to calibrate. "
                "Using default control until then.",
                self.heater_name, self.heater_name)

    def get_status(self, eventtime):
        """Expose status for Moonraker."""
        if self.control and self.control.is_valid():
            return self.control.get_status(eventtime)
        return {'state': 'uncalibrated'}


def load_config(config):
    return ChamberMpcModule(config)
