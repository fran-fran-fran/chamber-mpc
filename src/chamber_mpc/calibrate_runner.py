# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: calibrate_runner.py
# Description: Orchestrates the MPC chamber calibration sequence within
#              Klipper's reactor.
#
#              The calibration is fully self-bootstrapping:
#              - Precondition: Bed is off and cool
#              - Phase 0: Measure or accept ambient temperature
#              - Phase 1: Open-loop step response to first target
#                         (identifies C and s1_responsiveness)
#              - Phase 1b: Passive cooling estimate of rough h
#              - Phase 2: MPC holds at each target using progressively
#                         refined parameters (identifies h at each point)
#              - Phase 3: Optional bed transfer measurement using
#                         fully calibrated MPC
#
#              No PID or other external controller is needed.

import logging

from .calibrate import (
    StepResponseAnalyzer, CalibrationResult,
    estimate_h_from_cooling,
    compute_cooling_rates, format_calibration_report,
    SETTLE_TOLERANCE_C, SETTLE_DURATION_S, SETTLE_TIMEOUT_S,
    POWER_MEASURE_WINDOW_S, STEP_RESPONSE_POWER,
    BED_COOLDOWN_TOLERANCE_C, BED_HEAT_TIMEOUT_S,
)
from .h_interpolator import HInterpolator
from .kalman import KalmanFilter3
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
    """Temporary control using an MPC model for temperature holding.

    Used during calibration steady-state holds (Phase 2 onward).
    The model is progressively refined as more h points are identified.

    Only calls set_pwm from temperature_update (never set_temp),
    matching Kalico's control pattern and avoiding deadlock with
    the heater's threading lock.
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
        return abs(self.target - smoothed_temp) > 1.0

    def get_profile(self):
        return {'name': 'calibrating'}

    def get_type(self):
        return 'calibrating'

    def update_smooth_time(self):
        pass


class MpcChamberCalibrateRunner:
    """Runs the MPC chamber calibration sequence.

    Fully self-bootstrapping: the step response identifies enough
    parameters for MPC to hold temperature, then MPC holds at each
    calibration point while steady-state power is measured.
    No PID or external controller is needed.

    Sequence:
        Precondition: Bed heater off and within a few deg C of the chamber
        Phase 0: Measure T_ambient (or accept from T_AMBIENT parameter)
        Phase 1: Step response to first point (identify C and s1_resp)
        Phase 1b: Passive cooling estimate of rough h
        Phase 2: MPC holds at each point (identify h at each temperature)
        Phase 3: Optional bed transfer using calibrated MPC (identify h_bed)
    """

    def __init__(self, printer, heater, orig_control):
        self.printer = printer
        self.heater = heater
        self.orig_control = orig_control
        self.profile = orig_control.profile or {}
        self.log = logging.getLogger('chamber_mpc.calibrate')
        self._bed_heater = None

    def run(self, gcmd):
        """Main calibration entry point, called from GCode command."""
        # Parse parameters
        points_str = gcmd.get('POINTS', '85')
        points = [float(p.strip()) for p in points_str.split(',')]
        points.sort()
        bed_temp = gcmd.get_float('BED_TEMP', default=None)
        t_ambient_override = gcmd.get_float('T_AMBIENT', default=None)
        initial_h = gcmd.get_float('INITIAL_H', default=None)
        settle_timeout = gcmd.get_float(
            'SETTLE_TIMEOUT', SETTLE_TIMEOUT_S, above=0.)
        bed_timeout = gcmd.get_float(
            'BED_TIMEOUT', BED_HEAT_TIMEOUT_S, above=0.)

        if not points:
            raise gcmd.error("POINTS must specify at least one temperature")

        for p in points:
            if p < 30 or p > self.heater.max_temp - 10:
                raise gcmd.error(
                    "Calibration point %.0f deg C is outside safe range "
                    "[30, %.0f]" % (p, self.heater.max_temp - 10))

        # Swap in tuning control for open-loop phases
        tuning = TuningControl(self.heater)
        old_control = self.heater.set_control(tuning)

        try:
            result = self._run_calibration(
                gcmd, tuning, points, bed_temp, t_ambient_override,
                initial_h, settle_timeout, bed_timeout)
            self._save_results(gcmd, result)
        except self.printer.command_error as e:
            raise gcmd.error("Calibration failed: %s" % e)
        finally:
            # Always restore original control and turn off heater
            self.heater.set_control(old_control)
            self.heater.set_temp(0.0)
            if self._bed_heater is not None:
                self._bed_heater.set_temp(0.0)

    def _run_calibration(self, gcmd, tuning, points, bed_temp,
                         t_ambient_override=None, initial_h=None,
                         settle_timeout=SETTLE_TIMEOUT_S,
                         bed_timeout=BED_HEAT_TIMEOUT_S):
        """Execute the calibration sequence."""
        result = CalibrationResult()

        # The bed heater must be off before measuring ambient, step response,
        # or h(T): any residual bed heat biases all of those measurements.
        self._ensure_bed_is_cold()

        # Get heater power from config
        heater_power = self.profile.get('heater_power', 0)
        if heater_power <= 0:
            raise self.printer.command_error(
                "heater_power must be configured in [chamber_mpc] "
                "for calibration")

        # Phase 0: ambient temperature
        tuning.set_output(0.0, 0.0)
        if t_ambient_override is not None:
            result.t_ambient = t_ambient_override
            gcmd.respond_info(
                "Phase 0: Using provided T_ambient = %.1f\u00b0C"
                % result.t_ambient)
            gcmd.respond_info(
                "  Waiting for temperature to stabilize...")
            self._wait_for_stable_temperature()
        else:
            gcmd.respond_info("Phase 0: Measuring ambient temperature...")
            result.t_ambient = self._wait_for_stable_temperature()
            gcmd.respond_info(
                "  T_ambient = %.1f\u00b0C" % result.t_ambient)

        # Phase 1: step response from ambient to first point
        gcmd.respond_info(
            "Phase 1: Step response to %.0f\u00b0C..." % points[0])
        step_data = self._run_step_response(tuning, points[0])
        analyzer = StepResponseAnalyzer(
            step_data, heater_power, result.t_ambient)
        step_result = analyzer.analyze()
        result.chamber_heat_capacity = step_result['chamber_heat_capacity']
        result.s1_responsiveness = step_result['s1_responsiveness']

        gcmd.respond_info(
            "  C = %.1f J/K, s1_resp = %.4f"
            % (result.chamber_heat_capacity,
               result.s1_responsiveness))

        # Phase 1b: estimate rough h from passive cooling
        if initial_h is not None:
            rough_h = initial_h
            gcmd.respond_info(
                "  Using provided INITIAL_H = %.3f W/K" % rough_h)
        else:
            gcmd.respond_info(
                "Phase 1b: Passive cooling to estimate rough h...")
            cooling_data = self._record_cooling(
                tuning, points[0], gcmd)
            rough_h = estimate_h_from_cooling(
                cooling_data, result.chamber_heat_capacity,
                result.t_ambient, center_temp=points[0])
            if rough_h is None or rough_h <= 0:
                rough_h = 1.0
                gcmd.respond_info(
                    "  Could not estimate h from cooling, "
                    "using default h = 1.0 W/K")
            else:
                gcmd.respond_info(
                    "  Rough h from cooling = %.3f W/K" % rough_h)

        # Phase 2: progressive steady-state holds using self-bootstrapping MPC
        gcmd.respond_info(
            "Phase 2: Steady-state measurements (MPC self-bootstrapping)...")

        for i, target in enumerate(points):
            gcmd.respond_info(
                "  Point %d/%d: %.0f\u00b0C" % (i + 1, len(points), target))

            # Build MPC model with best available parameters
            h_points_so_far = list(result.h_points)
            if not h_points_so_far:
                # First point: use rough h from step response
                h_points_so_far = [(target, rough_h)]
            model = self._build_model(
                result, h_points_so_far, heater_power)

            # MPC holds at target
            hold = _MpcHoldControl(self.heater, model, target)
            self.heater.set_control(hold)
            self.heater.set_temp(target)

            # Wait for temperature to settle (with periodic status)
            gcmd.respond_info("    Waiting for steady state...")
            settled, actual_temp = self._wait_settle(
                target, settle_timeout, model, gcmd)

            if not settled:
                raise self.printer.command_error(
                    "Chamber did not settle at %.1f deg C within %.0f s "
                    "(last temperature %.1f deg C). Increase "
                    "SETTLE_TIMEOUT or check the heater."
                    % (target, settle_timeout, actual_temp))

            # Measure steady-state power and actual hold temperature
            gcmd.respond_info("    Measuring steady-state power...")
            avg_duty, avg_temp = self._measure_steady_state(
                POWER_MEASURE_WINDOW_S)
            avg_power_watts = avg_duty * heater_power

            # Compute h at actual hold temperature
            delta_T = avg_temp - result.t_ambient
            if delta_T <= 0:
                raise self.printer.command_error(
                    "Measured temperature %.1f <= ambient %.1f, "
                    "cannot compute h" % (avg_temp, result.t_ambient))
            h = avg_power_watts / delta_T
            result.h_points.append((avg_temp, h))
            gcmd.respond_info("    h(%.1f) = %.4f W/K" % (avg_temp, h))

            # Swap back to tuning control between points. Klipper's
            # set_control() resets the heater target to 0, so no extra
            # set_temp() is needed here.
            self.heater.set_control(tuning)

        # Phase 3: optional bed transfer using fully calibrated MPC
        if bed_temp is not None:
            gcmd.respond_info(
                "Phase 3: Bed transfer measurement at bed=%.0f\u00b0C..."
                % bed_temp)
            chamber_target = (
                result.h_points[-1][0] if result.h_points else points[-1])
            try:
                result.bed_transfer = self._measure_bed_transfer(
                    gcmd, bed_temp, chamber_target, result, heater_power,
                    settle_timeout, bed_timeout)
                gcmd.respond_info(
                    "  bed_transfer = %.4f W/K" % result.bed_transfer)
            except self.printer.command_error as e:
                result.bed_transfer = None
                gcmd.respond_info(
                    "WARNING: Bed transfer calibration skipped: %s" % e)

        return result

    # -- Model building --

    def _build_model(self, result, h_points, heater_power):
        """Build a ThermalModel with Kalman filter for calibration.

        Uses the profile's estimator tuning and target_reach_time. The
        disturbance state absorbs persistent prediction errors and
        cancels steady-state offset even when the initial h estimate
        is inaccurate.
        """
        kalman = KalmanFilter3(
            process_noise_chamber=self.profile.get(
                'process_noise_chamber', 1.0),
            process_noise_s1=self.profile.get(
                'process_noise_s1', 0.1),
            process_noise_disturbance=self.profile.get(
                'process_noise_disturbance', 10.0),
            measurement_noise_s1=self.profile.get(
                'measurement_noise_s1', 0.5),
        )
        model = ThermalModel(
            chamber_heat_capacity=result.chamber_heat_capacity,
            s1_responsiveness=result.s1_responsiveness,
            h_interpolator=HInterpolator(h_points),
            heater_power=heater_power,
            target_reach_time=self.profile.get('target_reach_time', 2.0),
            kalman_filter=kalman,
        )
        current_temp, _ = self.heater.get_temp(
            self.heater.reactor.monotonic())
        model.set_initial_state(current_temp)
        model.set_ambient(result.t_ambient)
        self._attach_secondary_sensor(model)
        return model

    def _attach_secondary_sensor(self, model):
        """Attach the configured secondary-sensor limit to a model."""
        sensor_name = self.profile.get('secondary_sensor')
        if not sensor_name:
            return
        try:
            sensor = self.printer.lookup_object(
                'temperature_sensor %s' % sensor_name)
            model.set_secondary_sensor_limit(
                sensor.get_temp,
                self.profile.get('s2_safe_temp', 250.0),
                self.profile.get('s2_safe_temp_zone', 20.0))
        except Exception as e:
            self.log.warning(
                "Could not attach secondary sensor '%s' during "
                "calibration: %s", sensor_name, e)

    # -- Phase 0: Ambient measurement --

    def _wait_for_stable_temperature(self):
        """Wait for a slow temperature rate and return the last reading."""
        samples = []

        def process(eventtime):
            temp, _ = self.heater.get_temp(eventtime)
            samples.append((eventtime, temp))
            while samples and samples[0][0] < eventtime - 30.0:
                samples.pop(0)
            if len(samples) < 30:
                return True
            duration = samples[-1][0] - samples[0][0]
            if duration < 10.0:
                return True
            dt = samples[-1][1] - samples[0][1]
            rate = abs(dt / duration)
            return rate > 0.05

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

    def _record_cooling(self, tuning, target, gcmd):
        """Record passive cooling data after step response overshoot.

        After the step response, the heater is off and the system
        overshoots then cools. We wait for the temperature to peak
        and then record samples as it cools through a window around
        the target temperature.

        Args:
            tuning: TuningControl (heater off)
            target: first calibration target temperature
            gcmd: GCode command for status messages

        Returns:
            list of (time, temperature) tuples during cooling
        """
        tuning.set_output(0.0, 0.0)
        cooling_samples = []
        t_high = target + 5.0  # start recording 5 deg C above target
        t_low = target - 5.0   # stop recording 5 deg C below target
        peak_seen = [False]
        last_temp = [None]

        def process(eventtime):
            temp, _ = self.heater.get_temp(eventtime)

            # Detect peak (temperature starts falling)
            if last_temp[0] is not None:
                if temp < last_temp[0] - 0.2:
                    peak_seen[0] = True
            last_temp[0] = temp

            if not peak_seen[0]:
                return True  # still waiting for peak

            # Record samples in the cooling window
            if temp <= t_high and temp >= t_low:
                cooling_samples.append((eventtime, temp))

            # Stop when we've cooled below the window
            if temp < t_low:
                return False

            return True

        self.printer.wait_while(process)

        gcmd.respond_info(
            "  Collected %d cooling samples" % len(cooling_samples))
        return cooling_samples

    # -- Phase 2: Steady-state measurement utilities --

    def _measure_steady_state(self, window_s):
        """Measure average heater duty cycle and chamber temperature over a window.

        Returns (avg_duty, avg_temp) where avg_duty is in 0.0-1.0 and
        avg_temp is in deg C.
        """
        power_samples = []
        temp_samples = []
        start_time = [None]

        def process(eventtime):
            if start_time[0] is None:
                start_time[0] = eventtime
            status = self.heater.get_status(eventtime)
            power_samples.append(status.get('power', 0.0))
            temp, _ = self.heater.get_temp(eventtime)
            temp_samples.append(temp)
            return eventtime - start_time[0] < window_s

        self.printer.wait_while(process)

        if not power_samples:
            temp, _ = self.heater.get_temp(self.heater.reactor.monotonic())
            return (0.0, temp)
        avg_duty = sum(power_samples) / len(power_samples)
        avg_temp = sum(temp_samples) / len(temp_samples)
        return (avg_duty, avg_temp)

    def _measure_avg_power(self, window_s):
        """Measure average heater duty cycle over a time window.

        Returns average duty cycle as a fraction 0.0-1.0.
        """
        avg_duty, _ = self._measure_steady_state(window_s)
        return avg_duty

    # -- Phase 3: Bed transfer --

    def _measure_bed_transfer(self, gcmd, bed_temp, chamber_target,
                              result, heater_power, settle_timeout,
                              bed_timeout):
        """Measure bed heat transfer coefficient using calibrated MPC.

        All chamber parameters are identified by this point. MPC holds
        the chamber while we toggle the bed and measure power difference.
        """
        model = self._build_model(result, result.h_points, heater_power)

        hold = _MpcHoldControl(self.heater, model, chamber_target)
        self.heater.set_control(hold)
        self.heater.set_temp(chamber_target)

        # Measure baseline power (bed off)
        gcmd.respond_info("    Measuring baseline power (bed off)...")
        settled, actual_temp = self._wait_settle(
            chamber_target, settle_timeout, model, gcmd)
        if not settled:
            raise self.printer.command_error(
                "Chamber did not settle at %.1f deg C within %.0f s "
                "before the bed-transfer baseline (last temperature "
                "%.1f deg C)." % (chamber_target, settle_timeout,
                                  actual_temp))
        power_no_bed = self._measure_avg_power(POWER_MEASURE_WINDOW_S)
        power_no_bed_watts = power_no_bed * heater_power

        # Turn bed on
        gcmd.respond_info(
            "    Turning bed on to %.0f\u00b0C..." % bed_temp)
        bed_heater = self._get_bed_heater()
        bed_heater.set_temp(bed_temp)

        # Wait for bed to reach target
        gcmd.respond_info("    Waiting for bed to reach target...")
        bed_start = [None]
        bed_timed_out = [False]

        def wait_bed(eventtime):
            if bed_start[0] is None:
                bed_start[0] = eventtime
            temp, _ = bed_heater.get_temp(eventtime)
            if temp >= bed_temp - 2.0:
                return False
            if eventtime - bed_start[0] > bed_timeout:
                bed_timed_out[0] = True
                return False
            return True

        self.printer.wait_while(wait_bed)
        if bed_timed_out[0]:
            last_bed_temp, _ = bed_heater.get_temp(
                self.heater.reactor.monotonic())
            raise self.printer.command_error(
                "Bed heater did not reach %.0f deg C within %.0f s "
                "(last temperature %.1f deg C). Increase BED_TIMEOUT "
                "or check the bed heater."
                % (bed_temp, bed_timeout, last_bed_temp))

        # Wait for chamber to re-settle
        gcmd.respond_info("    Waiting for chamber to re-settle...")
        settled, actual_temp = self._wait_settle(
            chamber_target, settle_timeout, model, gcmd)
        if not settled:
            raise self.printer.command_error(
                "Chamber did not settle at %.1f deg C within %.0f s "
                "with the bed at %.0f deg C (last temperature %.1f deg C). "
                "The bed may be too hot; lower BED_TEMP or increase "
                "SETTLE_TIMEOUT."
                % (chamber_target, settle_timeout, bed_temp, actual_temp))

        # Measure power with bed on
        gcmd.respond_info("    Measuring power with bed on...")
        power_with_bed = self._measure_avg_power(POWER_MEASURE_WINDOW_S)
        power_with_bed_watts = power_with_bed * heater_power

        # Read actual temperatures before turning the bed off.
        eventtime = self.heater.reactor.monotonic()
        actual_bed_temp, _ = bed_heater.get_temp(eventtime)
        actual_chamber_temp, _ = self.heater.get_temp(eventtime)
        bed_heater.set_temp(0.0)

        # h_bed = (P_no_bed - P_with_bed) / (T_bed - T_chamber)
        delta_power = power_no_bed_watts - power_with_bed_watts
        delta_T = actual_bed_temp - actual_chamber_temp
        if abs(delta_T) < 5.0:
            raise self.printer.command_error(
                "Bed temperature too close to chamber temperature "
                "for accurate h_bed measurement")

        return delta_power / delta_T

    def _ensure_bed_is_cold(self):
        """Turn off the bed and abort if it is too hot to calibrate.

        The bed is turned off before the check so the user can start the
        cool-down immediately. If it is still significantly warmer than the
        chamber, abort instead of waiting: the user should let it cool and
        then rerun the calibration.
        """
        bed_name = self.profile.get('bed_heater')
        if not bed_name:
            return

        bed_heater = self._get_bed_heater()
        bed_heater.set_temp(0.0)

        eventtime = self.heater.reactor.monotonic()
        bed_temp, _ = bed_heater.get_temp(eventtime)
        chamber_temp, _ = self.heater.get_temp(eventtime)
        if bed_temp > chamber_temp + BED_COOLDOWN_TOLERANCE_C:
            raise self.printer.command_error(
                "Bed heater '%s' is still hot (T_bed=%.1f deg C, "
                "T_chamber=%.1f deg C). The bed has been turned off. "
                "Wait for it to cool within %.1f deg C of the chamber, "
                "then run calibration again."
                % (bed_name, bed_temp, chamber_temp,
                   BED_COOLDOWN_TOLERANCE_C))

    def _get_bed_heater(self):
        """Look up the bed heater object."""
        if self._bed_heater is not None:
            return self._bed_heater
        bed_name = self.profile.get('bed_heater')
        if not bed_name:
            raise self.printer.command_error(
                "bed_heater not configured in [chamber_mpc]")
        pheaters = self.printer.lookup_object('heaters')
        self._bed_heater = pheaters.lookup_heater(bed_name)
        return self._bed_heater

    # -- Shared utilities --

    def _wait_settle(self, target, timeout_s, model=None, gcmd=None):
        """Wait for a stable temperature, with optional MPC status output.

        The temperature must stay within SETTLE_TOLERANCE_C for
        SETTLE_DURATION_S. Returns (settled, actual_temp); settled is
        False when timeout_s expires.
        """
        stable_since = [None]
        last_status_time = [None]
        start_time = [None]
        last_temp = [None]
        timed_out = [False]

        def process(eventtime):
            temp, _ = self.heater.get_temp(eventtime)
            last_temp[0] = temp

            # Check timeout
            if start_time[0] is None:
                start_time[0] = eventtime
            elif eventtime - start_time[0] > timeout_s:
                timed_out[0] = True
                return False

            # Print status every 60 seconds when requested
            if model is not None and gcmd is not None:
                if last_status_time[0] is None:
                    last_status_time[0] = eventtime
                elif eventtime - last_status_time[0] >= 60.0:
                    last_status_time[0] = eventtime
                    status = model.get_status()
                    k = status.get('kalman_gain')
                    k_str = ""
                    if k is not None:
                        k_str = ", kalman_gain=[%s]" % ", ".join(
                            "%.4f" % v for v in k)
                    d = status.get('disturbance', 0.0)
                    d_str = ""
                    if d != 0.0:
                        d_str = ", d=%.1f W" % d
                    gcmd.respond_info(
                        "    [status] chamber=%.1f\u00b0C, s1=%.1f\u00b0C, "
                        "ambient=%.1f\u00b0C, power=%.1f W, "
                        "avg=%.1f W (%.0f%%)%s%s"
                        % (status['temp_chamber'], status['temp_s1'],
                           status['temp_ambient'], status['power'],
                           status['avg_power'], status['avg_duty'] * 100,
                           d_str, k_str))

            # Settle detection
            if abs(temp - target) < SETTLE_TOLERANCE_C:
                if stable_since[0] is None:
                    stable_since[0] = eventtime
                elif eventtime - stable_since[0] > SETTLE_DURATION_S:
                    return False
            else:
                stable_since[0] = None
            return True

        self.printer.wait_while(process)
        return (not timed_out[0], last_temp[0]
                if last_temp[0] is not None else target)

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
            cfgname, 's1_responsiveness',
            "%.4f" % result.s1_responsiveness)
        configfile.set(
            cfgname, 'ambient_temp',
            "%.1f" % result.t_ambient)

        # h calibration points
        h_interp = HInterpolator(result.h_points)
        configfile.set(
            cfgname, 'ambient_transfer_points',
            h_interp.format_config_string())

        if result.bed_transfer is not None:
            configfile.set(
                cfgname, 'bed_transfer',
                "%.4f" % result.bed_transfer)

        gcmd.respond_info(
            "Results saved to [chamber_mpc]. "
            "Run SAVE_CONFIG to persist to printer.cfg.")
