# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: calibrate_runner.py
# Description: Orchestrates the MPC chamber calibration sequence within
#              Klipper's reactor. Drives the heater, collects measurements,
#              and delegates analysis to calibrate.py.
#
#              Phase 0-2 use the heater's existing controller (typically PID)
#              for temperature holding. Phase 3 (optional bed transfer) uses
#              a fully calibrated MPC model since all chamber parameters are
#              known by that point.

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
    """Temporary heater control for open-loop calibration phases.

    Applies a fixed PWM output and optionally logs temperature readings.
    Used during ambient measurement (output=0) and step response (output=1).
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


class _MpcHoldControl:
    """Temporary control using a fully calibrated MPC model.

    Used only during Phase 3 (bed transfer measurement) where all chamber
    parameters are already identified and MPC provides better holding than PID.
    """

    def __init__(self, heater, model, target):
        self.heater = heater
        self.model = model
        self.target = target
        self.last_power_fraction = 0.0

    def temperature_update(self, read_time, temp, target_temp):
        max_power = self.heater.max_power
        duty = self.model.update(read_time, temp, self.target, max_power)
        self.last_power_fraction = duty
        self.heater.set_pwm(read_time, duty)

    def check_busy(self, eventtime, smoothed_temp, target_temp):
        return abs(target_temp - smoothed_temp) > 1.0

    def get_profile(self):
        return {'name': 'calibrating'}

    def get_type(self):
        return 'calibrating'

    def update_smooth_time(self):
        pass


class MpcChamberCalibrateRunner:
    """Runs the MPC chamber calibration sequence.

    Sequence:
        Phase 0: Measure T_ambient (chamber at room temperature)
        Phase 1: Step response to first point (identify C, sensor_resp)
        Phase 2: Steady-state holds using existing PID (identify h at each point)
        Phase 3: Optional bed transfer using calibrated MPC (identify h_bed)
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

        # Capture the original control (typically PID) by swapping
        # in a temporary control - set_control() returns the old one
        tuning = TuningControl(self.heater)
        pid_control = self.heater.set_control(tuning)

        try:
            result = self._run_calibration(
                gcmd, tuning, pid_control, points, bed_temp)
            self._save_results(gcmd, result)
        except self.printer.command_error as e:
            raise gcmd.error("Calibration failed: %s" % e)
        finally:
            # Always restore original control and turn off heater
            self.heater.set_control(pid_control)
            self.heater.set_temp(0.0)

    def _run_calibration(self, gcmd, tuning, pid_control, points, bed_temp):
        """Execute the calibration sequence."""
        result = CalibrationResult()

        # Get heater power from config
        heater_power = 0
        if self.orig_control.profile:
            heater_power = self.orig_control.profile.get('heater_power', 0)
        if heater_power <= 0:
            heater_power = self.heater.max_power
            gcmd.respond_info(
                "WARNING: heater_power not set in [chamber_mpc], "
                "using max_power=%.0f (this may be incorrect)" % heater_power)

        # Phase 0: measure ambient (tuning control already active with output=0)
        gcmd.respond_info("Phase 0: Measuring ambient temperature...")
        tuning.set_output(0.0, 0.0)
        result.t_ambient = self._measure_ambient()
        gcmd.respond_info(
            "  T_ambient = %.1f deg C" % result.t_ambient)

        # Phase 1: step response from ambient to first point
        gcmd.respond_info(
            "Phase 1: Step response to %.0f deg C..." % points[0])
        step_data = self._run_step_response(tuning, points[0])
        analyzer = StepResponseAnalyzer(
            step_data, heater_power, result.t_ambient)
        step_result = analyzer.analyze()
        result.chamber_heat_capacity = step_result['chamber_heat_capacity']
        result.sensor_responsiveness = step_result['sensor_responsiveness']
        gcmd.respond_info(
            "  C = %.1f J/K, sensor_resp = %.4f"
            % (result.chamber_heat_capacity, result.sensor_responsiveness))

        # Phase 2: steady-state holds using existing PID
        gcmd.respond_info(
            "Phase 2: Steady-state measurements (using PID to hold)...")
        self.heater.set_control(pid_control)

        all_prediction_errors = []
        for i, target in enumerate(points):
            gcmd.respond_info(
                "  Point %d/%d: %.0f deg C" % (i + 1, len(points), target))
            h, errors = self._measure_steady_state_with_pid(
                gcmd, target, result, heater_power)
            result.h_points.append((target, h))
            all_prediction_errors.extend(errors)
            gcmd.respond_info("    h = %.4f W/K" % h)

        # Phase 3: optional bed transfer using calibrated MPC
        if bed_temp is not None:
            gcmd.respond_info(
                "Phase 3: Bed transfer measurement at bed=%.0f deg C..."
                % bed_temp)
            result.bed_transfer = self._measure_bed_transfer(
                gcmd, bed_temp, points[-1], result, heater_power)
            gcmd.respond_info(
                "  bed_transfer = %.4f W/K" % result.bed_transfer)

        # Phase 4: estimate smoothing
        result.smoothing = SmoothingEstimator.estimate(all_prediction_errors)
        gcmd.respond_info(
            "Phase 4: Estimated smoothing = %.2f" % result.smoothing)

        return result

    # -- Phase 0: Ambient measurement --

    def _measure_ambient(self):
        """Wait for temperature to stabilize and record ambient."""
        samples = []

        def process(eventtime):
            temp, _ = self.heater.get_temp(eventtime)
            samples.append((eventtime, temp))
            # Keep last 30s of samples
            while samples and samples[0][0] < eventtime - 30.0:
                samples.pop(0)
            if len(samples) < 30:
                return True
            duration = samples[-1][0] - samples[0][0]
            if duration < 10.0:
                return True
            dt = samples[-1][1] - samples[0][1]
            rate = abs(dt / duration)
            return rate > 0.05  # wait until < 0.05 deg C/s drift

        self.printer.wait_while(process)
        return samples[-1][1]

    # -- Phase 1: Step response --

    def _run_step_response(self, tuning, target):
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

    # -- Phase 2: Steady-state holds with PID --

    def _measure_steady_state_with_pid(self, gcmd, target, result,
                                        heater_power):
        """Hold at target using existing PID and measure steady-state power.

        The heater's original controller (PID) is already active.
        We simply set the target and wait for it to settle.

        Returns (h_value, prediction_errors) tuple.
        """
        # Set target - PID will drive to it
        self.heater.set_temp(target)

        # Wait for temperature to settle
        gcmd.respond_info("    Waiting for steady state...")
        self._wait_settle(target)

        # Measure average power over window
        gcmd.respond_info("    Measuring steady-state power...")
        avg_power_frac = self._measure_avg_power(POWER_MEASURE_WINDOW_S)

        # Convert duty cycle fraction to watts
        avg_power_watts = avg_power_frac * heater_power

        # Compute h = P_ss / (T_target - T_ambient)
        delta_T = target - result.t_ambient
        if delta_T <= 0:
            raise self.printer.command_error(
                "Target %.1f <= ambient %.1f, cannot compute h"
                % (target, result.t_ambient))
        h = avg_power_watts / delta_T

        # Collect prediction errors for smoothing estimation
        # Build a temporary model to compute prediction errors
        errors = self._collect_prediction_errors(
            target, result, heater_power)

        return h, errors

    def _measure_avg_power(self, window_s):
        """Measure average heater duty cycle fraction over a time window.

        Reads the heater's actual PWM output (what the PID is commanding).
        Returns average duty cycle as a fraction 0.0-1.0.
        """
        samples = []
        start_time = [None]

        def process(eventtime):
            if start_time[0] is None:
                start_time[0] = eventtime
            # Read the heater's last commanded power
            status = self.heater.get_status(eventtime)
            power = status.get('power', 0.0)
            samples.append(power)
            return eventtime - start_time[0] < window_s

        self.printer.wait_while(process)

        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def _collect_prediction_errors(self, target, result, heater_power):
        """Run a temporary MPC model alongside PID to collect prediction errors.

        The PID is still controlling the heater. We just run the MPC model
        in observer mode (no output) to see how well it predicts the
        measured temperature. The prediction errors are used to estimate
        the smoothing parameter.
        """
        # Build a model with current best estimates
        h_est = result.h_points[-1][1] if result.h_points else 0.15
        model = ThermalModel(
            chamber_heat_capacity=result.chamber_heat_capacity,
            sensor_responsiveness=result.sensor_responsiveness,
            h_interpolator=HInterpolator([(target, h_est)]),
            heater_power=heater_power,
            smoothing=0.5,
        )
        current_temp, _ = self.heater.get_temp(
            self.heater.reactor.monotonic())
        model.set_initial_state(current_temp)
        model.set_ambient(result.t_ambient)

        errors = []
        start_time = [None]

        def process(eventtime):
            if start_time[0] is None:
                start_time[0] = eventtime
            temp, _ = self.heater.get_temp(eventtime)
            # Run model in observer mode - predict but don't control
            # Feed it the actual power the PID is applying
            status = self.heater.get_status(eventtime)
            actual_power_frac = status.get('power', 0.0)
            model.last_power = actual_power_frac * heater_power
            dt = eventtime - model.last_time
            if model.last_time > 0 and 0 < dt < 1.0:
                model._propagate(dt, eventtime)
            model.last_time = eventtime
            # Record prediction error before correction
            errors.append(temp - model.state_sensor_temp)
            # Apply correction so model stays in sync
            model._correct(dt if dt > 0 else 0.1, temp)
            return eventtime - start_time[0] < 60.0  # 60s collection

        self.printer.wait_while(process)
        return errors

    # -- Phase 3: Bed transfer with calibrated MPC --

    def _measure_bed_transfer(self, gcmd, bed_temp, chamber_target,
                              result, heater_power):
        """Measure bed heat transfer coefficient using calibrated MPC.

        By this point all chamber parameters (C, sensor_resp, h(T)) are
        identified, so we can build a proper MPC model to hold the chamber
        temperature while measuring the bed's thermal contribution.
        """
        # Build fully calibrated MPC model
        model = ThermalModel(
            chamber_heat_capacity=result.chamber_heat_capacity,
            sensor_responsiveness=result.sensor_responsiveness,
            h_interpolator=HInterpolator(result.h_points),
            heater_power=heater_power,
            smoothing=result.smoothing,
        )
        current_temp, _ = self.heater.get_temp(
            self.heater.reactor.monotonic())
        model.set_initial_state(current_temp)
        model.set_ambient(result.t_ambient)

        # Use MPC to hold chamber temperature
        hold = _MpcHoldControl(self.heater, model, chamber_target)
        self.heater.set_control(hold)
        self.heater.set_temp(chamber_target)

        # Measure baseline power (bed off)
        gcmd.respond_info("    Measuring baseline power (bed off)...")
        self._wait_settle(chamber_target)
        power_no_bed = self._measure_avg_power(POWER_MEASURE_WINDOW_S)
        power_no_bed_watts = power_no_bed * heater_power

        # Turn bed on
        gcmd.respond_info(
            "    Turning bed on to %.0f deg C..." % bed_temp)
        bed_heater = self._get_bed_heater()
        bed_heater.set_temp(bed_temp)

        # Wait for bed to reach target
        gcmd.respond_info("    Waiting for bed to reach target...")

        def wait_bed(eventtime):
            temp, _ = bed_heater.get_temp(eventtime)
            return temp < bed_temp - 2.0

        self.printer.wait_while(wait_bed)

        # Wait for chamber to re-settle with bed heat contribution
        gcmd.respond_info("    Waiting for chamber to re-settle...")
        self._wait_settle(chamber_target)

        # Measure power with bed on
        gcmd.respond_info("    Measuring power with bed on...")
        power_with_bed = self._measure_avg_power(POWER_MEASURE_WINDOW_S)
        power_with_bed_watts = power_with_bed * heater_power

        # Turn bed off
        bed_heater.set_temp(0.0)

        # h_bed = (P_no_bed - P_with_bed) / (T_bed - T_chamber)
        actual_bed_temp, _ = bed_heater.get_temp(
            self.heater.reactor.monotonic())
        delta_power = power_no_bed_watts - power_with_bed_watts
        delta_T = actual_bed_temp - chamber_target
        if abs(delta_T) < 5.0:
            raise self.printer.command_error(
                "Bed temperature too close to chamber temperature "
                "for accurate h_bed measurement")

        return delta_power / delta_T

    def _get_bed_heater(self):
        """Look up the bed heater object."""
        bed_name = None
        if self.orig_control.profile:
            bed_name = self.orig_control.profile.get('bed_heater')
        if not bed_name:
            raise self.printer.command_error(
                "bed_heater not configured in [chamber_mpc]")
        pheaters = self.printer.lookup_object('heaters')
        return pheaters.lookup_heater(bed_name)

    # -- Shared utilities --

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

    def _save_results(self, gcmd, result):
        """Save calibration results to config and report to user."""
        cooling_rates = compute_cooling_rates(
            result.h_points, result.chamber_heat_capacity, result.t_ambient)

        # Terminal report
        report = format_calibration_report(result, cooling_rates)
        gcmd.respond_info("\n".join(report))

        # Save to [chamber_mpc] config section
        cfgname = 'chamber_mpc'
        configfile = self.printer.lookup_object('configfile')
        configfile.set(
            cfgname, 'chamber_heat_capacity',
            "%.1f" % result.chamber_heat_capacity)
        configfile.set(
            cfgname, 'sensor_responsiveness',
            "%.4f" % result.sensor_responsiveness)
        configfile.set(
            cfgname, 'smoothing',
            "%.2f" % result.smoothing)
        configfile.set(
            cfgname, 'ambient_temp',
            "%.1f" % result.t_ambient)

        # h calibration points
        h_interp = HInterpolator(result.h_points)
        configfile.set(
            cfgname, 'h_calibration_points',
            h_interp.format_config_string())

        if result.bed_transfer is not None:
            configfile.set(
                cfgname, 'bed_transfer',
                "%.4f" % result.bed_transfer)

        gcmd.respond_info(
            "Results saved to [chamber_mpc]. "
            "Run SAVE_CONFIG to persist to printer.cfg.")
