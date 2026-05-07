# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: calibrate_runner.py
# Description: Orchestrates the MPC chamber calibration sequence within
#              Klipper's reactor. Drives the heater, collects measurements,
#              and delegates analysis to calibrate.py.

import logging

from .calibrate import (
    StepResponseAnalyzer, SmoothingEstimator, CalibrationResult,
    compute_cooling_rates, format_calibration_report,
    format_cooling_rate_comments,
    SETTLE_TOLERANCE_C, SETTLE_DURATION_S, POWER_MEASURE_WINDOW_S,
    STEP_RESPONSE_POWER,
)
from .h_interpolator import HInterpolator
from .thermal_model import ThermalModel


class TuningControl:
    """Temporary heater control used during calibration.

    Applies a fixed PWM output and optionally logs temperature readings.
    Replaces the normal control during the calibration procedure.
    """

    def __init__(self, heater):
        self.heater = heater
        self.value = 0.0
        self.target = 0.0
        self.log = []
        self.logging = False

    def temperature_update(self, read_time, temp, target_temp):
        if self.logging:
            self.log.append((read_time, temp))
        self.heater.set_pwm(read_time, self.value)

    def check_busy(self, eventtime, smoothed_temp, target_temp):
        return self.value != 0.0 or self.target != 0.0

    def set_output(self, value, target):
        self.value = value
        self.target = target
        self.heater.set_temp(target)

    def get_profile(self):
        return {'name': 'tuning'}

    def get_type(self):
        return 'tuning'

    def update_smooth_time(self):
        pass


class MpcChamberCalibrateRunner:
    """Runs the MPC chamber calibration sequence.

    Sequence:
        Phase 0: Measure T_ambient (chamber at room temperature)
        Phase 1: Step response to first point (identify C, sensor_resp)
        Phase 2: Progressive steady-state holds (identify h at each point)
        Phase 3: Optional bed transfer measurement
        Phase 4: Estimate smoothing, compute results, save
    """

    def __init__(self, printer, heater, orig_control):
        self.printer = printer
        self.heater = heater
        self.orig_control = orig_control
        self.log = logging.getLogger('chamber_mpc.calibrate')

    def run(self, gcmd):
        """Main calibration entry point, called from GCode command."""
        # Parse parameters
        points_str = gcmd.get('POINTS', '85')
        points = [float(p.strip()) for p in points_str.split(',')]
        points.sort()
        bed_temp = gcmd.get_float('BED_TEMP', default=None)

        if not points:
            raise gcmd.error("POINTS must specify at least one temperature")

        for p in points:
            if p < 30 or p > self.heater.max_temp - 10:
                raise gcmd.error(
                    "Calibration point %.0f deg C is outside safe range "
                    "[30, %.0f]" % (p, self.heater.max_temp - 10))

        # Set up tuning control
        tuning = TuningControl(self.heater)
        old_control = self.heater.set_control(tuning)

        try:
            result = self._run_calibration(gcmd, tuning, points, bed_temp)
            self._save_results(gcmd, result)
        except self.printer.command_error as e:
            raise gcmd.error("Calibration failed: %s" % e)
        finally:
            self.heater.set_control(old_control)
            self.heater.alter_target(0.0)

    def _run_calibration(self, gcmd, tuning, points, bed_temp):
        """Execute the calibration sequence."""
        result = CalibrationResult()

        # Phase 0: measure ambient
        gcmd.respond_info("Phase 0: Measuring ambient temperature...")
        result.t_ambient = self._measure_ambient(gcmd, tuning)
        gcmd.respond_info(
            "  T_ambient = %.1f deg C" % result.t_ambient)

        # Phase 1: step response from ambient to first point
        gcmd.respond_info(
            "Phase 1: Step response to %.0f deg C..." % points[0])
        step_data = self._run_step_response(gcmd, tuning, points[0])
        heater_power = (self.orig_control.profile.get('heater_power', 0)
                        if self.orig_control.profile
                        else self.heater.get_max_power())
        if heater_power <= 0:
            heater_power = self.heater.get_max_power()
        analyzer = StepResponseAnalyzer(
            step_data, heater_power, result.t_ambient)
        step_result = analyzer.analyze()
        result.chamber_heat_capacity = step_result['chamber_heat_capacity']
        result.sensor_responsiveness = step_result['sensor_responsiveness']
        gcmd.respond_info(
            "  C = %.1f J/K, sensor_resp = %.4f"
            % (result.chamber_heat_capacity, result.sensor_responsiveness))

        # Phase 2: progressive steady-state holds
        gcmd.respond_info("Phase 2: Steady-state measurements...")
        all_prediction_errors = []
        for i, target in enumerate(points):
            gcmd.respond_info(
                "  Point %d/%d: %.0f deg C" % (i + 1, len(points), target))
            h, errors = self._measure_steady_state(
                gcmd, target, result, tuning)
            result.h_points.append((target, h))
            all_prediction_errors.extend(errors)
            gcmd.respond_info(
                "    h = %.4f W/K" % h)

        # Phase 3: optional bed transfer
        if bed_temp is not None:
            gcmd.respond_info(
                "Phase 3: Bed transfer measurement at bed=%.0f deg C..."
                % bed_temp)
            result.bed_transfer = self._measure_bed_transfer(
                gcmd, bed_temp, points[-1], result, tuning)
            gcmd.respond_info(
                "  bed_transfer = %.4f W/K" % result.bed_transfer)

        # Phase 4: estimate smoothing
        result.smoothing = SmoothingEstimator.estimate(all_prediction_errors)
        gcmd.respond_info(
            "Phase 4: Estimated smoothing = %.2f" % result.smoothing)

        return result

    def _measure_ambient(self, gcmd, tuning):
        """Wait for temperature to stabilize and record ambient."""
        tuning.set_output(0.0, 0.0)

        samples = []

        def process(eventtime):
            temp, _ = self.heater.get_temp(eventtime)
            samples.append((eventtime, temp))
            # Keep last 30s of samples
            while (samples and
                   samples[0][0] < eventtime - 30.0):
                samples.pop(0)
            # Need at least 10s of data
            if len(samples) < 30:
                return True
            # Check stability
            temps = [s[1] for s in samples]
            dt = samples[-1][1] - samples[0][1]
            duration = samples[-1][0] - samples[0][0]
            if duration < 10.0:
                return True
            rate = abs(dt / duration)
            return rate > 0.05  # wait until < 0.05 deg C/s drift

        self.printer.wait_while(process)
        return samples[-1][1]

    def _run_step_response(self, gcmd, tuning, target):
        """Apply full power and record temperature trajectory."""
        tuning.logging = True
        tuning.log = []
        tuning.set_output(STEP_RESPONSE_POWER, target)

        def process(eventtime):
            temp, _ = self.heater.get_temp(eventtime)
            return temp < target

        self.printer.wait_while(process)
        tuning.logging = False
        tuning.set_output(0.0, 0.0)

        log = list(tuning.log)
        tuning.log = []
        return log

    def _measure_steady_state(self, gcmd, target, result, tuning):
        """Hold at target temperature and measure steady-state power.

        Returns (h_value, prediction_errors) tuple.
        """
        # Use a temporary MPC to hold temperature (better than raw PID
        # since we already have C and sensor_resp from step response)
        # For the first point we use the step response h estimate,
        # for subsequent points we use the last identified h
        if result.h_points:
            h_est = result.h_points[-1][1]
        else:
            h_est = 0.15  # reasonable initial guess

        temp_model = ThermalModel(
            chamber_heat_capacity=result.chamber_heat_capacity,
            sensor_responsiveness=result.sensor_responsiveness,
            h_interpolator=HInterpolator([(target, h_est)]),
            heater_power=(self.orig_control.model.heater_power
                          if self.orig_control.model
                          else self.heater.get_max_power()),
            smoothing=0.5,
        )
        current_temp = self.heater.get_temp(
            self.heater.reactor.monotonic())[0]
        temp_model.set_initial_state(current_temp)
        temp_model.set_ambient(result.t_ambient)

        # Hold control using the temp model
        hold_control = _TempModelControl(self.heater, temp_model, target)
        self.heater.set_control(hold_control)
        self.heater.set_temp(target)

        # Wait for temperature to settle
        gcmd.respond_info("    Waiting for steady state...")
        self._wait_settle(target)

        # Measure power over window
        gcmd.respond_info("    Measuring steady-state power...")
        power, errors = self._measure_power_and_errors(
            hold_control, POWER_MEASURE_WINDOW_S)

        # Restore tuning control
        self.heater.set_control(tuning)

        # Compute h
        delta_T = target - result.t_ambient
        if delta_T <= 0:
            raise self.printer.command_error(
                "Target %.1f <= ambient %.1f, cannot compute h"
                % (target, result.t_ambient))
        h = power / delta_T

        return h, errors

    def _measure_bed_transfer(self, gcmd, bed_temp, chamber_target,
                              result, tuning):
        """Measure bed heat transfer coefficient.

        Holds chamber at the last calibration point, measures P_ss
        without bed, then with bed on, computes h_bed from the difference.
        """
        # Measure power without bed (should already be at chamber_target)
        gcmd.respond_info("    Measuring baseline power (bed off)...")
        h_est = result.h_points[-1][1] if result.h_points else 0.15

        temp_model = ThermalModel(
            chamber_heat_capacity=result.chamber_heat_capacity,
            sensor_responsiveness=result.sensor_responsiveness,
            h_interpolator=HInterpolator([(chamber_target, h_est)]),
            heater_power=(self.orig_control.model.heater_power
                          if self.orig_control.model
                          else self.heater.get_max_power()),
            smoothing=0.5,
        )
        current_temp = self.heater.get_temp(
            self.heater.reactor.monotonic())[0]
        temp_model.set_initial_state(current_temp)
        temp_model.set_ambient(result.t_ambient)

        hold_control = _TempModelControl(
            self.heater, temp_model, chamber_target)
        self.heater.set_control(hold_control)
        self.heater.set_temp(chamber_target)

        self._wait_settle(chamber_target)
        power_no_bed, _ = self._measure_power_and_errors(
            hold_control, POWER_MEASURE_WINDOW_S)

        # Turn bed on
        gcmd.respond_info(
            "    Turning bed on to %.0f deg C..." % bed_temp)
        try:
            pheaters = self.printer.lookup_object('heaters')
            bed_name = self.orig_control._bed_heater_name
            if not bed_name:
                raise self.printer.command_error(
                    "bed_heater not configured")
            bed_heater = pheaters.lookup_heater(bed_name)
            bed_heater.set_temp(bed_temp)
        except Exception as e:
            raise self.printer.command_error(
                "Cannot control bed heater: %s" % e)

        # Wait for bed to reach target and chamber to re-settle
        gcmd.respond_info("    Waiting for bed and chamber to stabilize...")

        def wait_bed(eventtime):
            temp, _ = bed_heater.get_temp(eventtime)
            return temp < bed_temp - 2.0

        self.printer.wait_while(wait_bed)
        self._wait_settle(chamber_target)

        # Measure power with bed on
        gcmd.respond_info("    Measuring power with bed on...")
        power_with_bed, _ = self._measure_power_and_errors(
            hold_control, POWER_MEASURE_WINDOW_S)

        # Turn bed off
        bed_heater.set_temp(0.0)
        self.heater.set_control(tuning)

        # h_bed = (P_no_bed - P_with_bed) / (T_bed - T_chamber)
        # Bed adds heat to chamber, so chamber heater needs less power
        actual_bed_temp = bed_heater.get_temp(
            self.heater.reactor.monotonic())[0]
        delta_power = power_no_bed - power_with_bed
        delta_T = actual_bed_temp - chamber_target
        if abs(delta_T) < 5.0:
            raise self.printer.command_error(
                "Bed temperature too close to chamber temperature "
                "for accurate h_bed measurement")

        return delta_power / delta_T

    def _wait_settle(self, target):
        """Wait for temperature to settle within tolerance."""
        stable_since = [None]

        def process(eventtime):
            temp, _ = self.heater.get_temp(eventtime)
            if abs(temp - target) < SETTLE_TOLERANCE_C:
                if stable_since[0] is None:
                    stable_since[0] = eventtime
                elif eventtime - stable_since[0] > SETTLE_DURATION_S:
                    return False
            else:
                stable_since[0] = None
            return True

        self.printer.wait_while(process)

    def _measure_power_and_errors(self, hold_control, window_s):
        """Measure average power and collect prediction errors.

        Returns (average_power_watts, prediction_errors_list).
        """
        samples = []
        errors = []
        start_time = [None]

        def process(eventtime):
            if start_time[0] is None:
                start_time[0] = eventtime
            status = self.heater.get_status(eventtime)
            power = hold_control.last_power
            temp, _ = self.heater.get_temp(eventtime)
            predicted = hold_control.model.state_sensor_temp
            samples.append(power)
            errors.append(temp - predicted)
            return eventtime - start_time[0] < window_s

        self.printer.wait_while(process, True, 0.3)

        if not samples:
            return 0.0, []

        avg_power = sum(samples) / len(samples)
        return avg_power, errors

    def _save_results(self, gcmd, result):
        """Save calibration results to config and report to user."""
        # Compute cooling rates
        cooling_rates = compute_cooling_rates(
            result.h_points, result.chamber_heat_capacity, result.t_ambient)

        # Terminal report
        report = format_calibration_report(result, cooling_rates)
        gcmd.respond_info("\n".join(report))

        # Save to config
        cfgname = self.heater.get_name()
        configfile = self.printer.lookup_object('configfile')
        configfile.set(cfgname, 'control', 'mpc_chamber')
        configfile.set(
            cfgname, 'chamber_heat_capacity',
            "%.1f" % result.chamber_heat_capacity)
        configfile.set(
            cfgname, 'sensor_responsiveness',
            "%.4f" % result.sensor_responsiveness)
        configfile.set(
            cfgname, 'smoothing',
            "%.2f" % result.smoothing)

        # h calibration points
        h_interp = HInterpolator(result.h_points)
        configfile.set(
            cfgname, 'h_calibration_points',
            h_interp.format_config_string())

        if result.bed_transfer is not None:
            configfile.set(
                cfgname, 'bed_transfer',
                "%.4f" % result.bed_transfer)

        # Add cooling rate comments
        comment_lines = format_cooling_rate_comments(result, cooling_rates)
        for line in comment_lines:
            configfile.set(cfgname, line, None)

        gcmd.respond_info(
            "Results saved. Run SAVE_CONFIG to persist to printer.cfg.")


class _TempModelControl:
    """Temporary control wrapper that uses a ThermalModel to hold temperature.

    Used during calibration steady-state holds. Not for general use.
    """

    def __init__(self, heater, model, target):
        self.heater = heater
        self.model = model
        self.target = target
        self.last_power = 0.0

    def temperature_update(self, read_time, temp, target_temp):
        max_power = self.heater.get_max_power()
        duty = self.model.update(read_time, temp, self.target, max_power)
        self.last_power = duty * self.model.heater_power
        self.heater.set_pwm(read_time, duty)

    def check_busy(self, eventtime, smoothed_temp, target_temp):
        return abs(target_temp - smoothed_temp) > 1.0

    def get_profile(self):
        return {'name': 'calibrating'}

    def get_type(self):
        return 'calibrating'

    def update_smooth_time(self):
        pass
