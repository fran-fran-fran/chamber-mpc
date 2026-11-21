# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: temp_sensor.py
# Description: Virtual Klipper temperature_sensor exposing the Kalman-estimated
#              chamber temperature (state_chamber_temp), analogous to Kalico's
#              mpc_block_temperature sensor.

REPORT_TIME = 1.0


class ChamberMpcTempSensor:
    """Klipper temperature sensor reporting the MPC-estimated chamber temp.

    Configured via:
        [temperature_sensor Chamber_MPC]
        sensor_type: mpc_chamber_temperature
        min_temp: -10
        max_temp: 100
        # heater_name: chamber  # optional; defaults to [chamber_mpc]'s heater

    Note: [chamber_mpc] must appear above [temperature_sensor ...] in
    printer.cfg so the sensor factory is registered before Klipper parses
    the sensor section.
    """

    def __init__(self, config):
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()
        self.heater_name = config.get('heater_name', default=None)
        self.temp = 0.0
        self.min_temp = 0.0
        self.max_temp = 0.0
        self._callback = None
        self._heater = None
        self._sample_timer = None
        self.printer.register_event_handler(
            'klippy:ready', self._handle_ready)

    def setup_minmax(self, min_temp, max_temp):
        self.min_temp = min_temp
        self.max_temp = max_temp

    def setup_callback(self, cb):
        self._callback = cb

    def get_report_time_delta(self):
        return REPORT_TIME

    def _handle_ready(self):
        heater_name = self.heater_name
        if heater_name is None:
            mpc_mod = self.printer.lookup_object('chamber_mpc', None)
            if mpc_mod is not None:
                heater_name = mpc_mod.heater_name
        if heater_name is not None:
            pheaters = self.printer.lookup_object('heaters')
            self._heater = pheaters.lookup_heater(heater_name)
        self._sample_timer = self.reactor.register_timer(
            self._sample_temp, self.reactor.NOW)

    def _read_temp(self, eventtime):
        if self._heater is None:
            return None
        # Use the active controller's model if running MPC (runtime or
        # calibration Phase 2/3); otherwise fall back to the raw heater temp.
        ctrl = getattr(self._heater, 'control', None)
        model = getattr(ctrl, 'model', None)
        if model is not None:
            return model.state_chamber_temp
        temp, _ = self._heater.get_temp(eventtime)
        return temp

    def _sample_temp(self, eventtime):
        temp = self._read_temp(eventtime)
        if temp is not None:
            self.temp = temp
            if self._callback is not None:
                self._callback(self.reactor.monotonic(), temp)
        return eventtime + REPORT_TIME

    def get_temp(self, eventtime):
        return (self.temp, 0.0)

    def get_status(self, eventtime):
        return {'temperature': round(self.temp, 2)}
