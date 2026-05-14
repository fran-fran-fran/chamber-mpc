# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: calibrate.py
# Description: Progressive multi-point MPC calibration. Identifies
#              chamber_heat_capacity, s1_responsiveness, and h(T)
#              at multiple operating points in a single ascending pass.
#              Optionally identifies bed_transfer if BED_TEMP is specified.

import math
import logging
import statistics


# Defaults for calibration control
SETTLE_TOLERANCE_C = 1.0
SETTLE_DURATION_S = 120.0
POWER_MEASURE_WINDOW_S = 180.0
STEP_RESPONSE_POWER = 1.0  # full power for step response


class CalibrationResult:
    """Holds the results of an MPC chamber calibration."""
    __slots__ = (
        'chamber_heat_capacity', 's1_responsiveness',
        'h_points', 't_ambient', 'bed_transfer',
    )

    def __init__(self):
        self.chamber_heat_capacity = 0.0
        self.s1_responsiveness = 0.0
        self.h_points = []  # list of (T, h) tuples
        self.t_ambient = 25.0
        self.bed_transfer = None  # None = not calibrated


class StepResponseAnalyzer:
    """Analyzes a step response trajectory to identify C and s1_responsiveness.

    Uses the same mathematical approach as Kalico's MPC calibration:
    - Asymptotic temperature from three evenly-spaced samples
    - Block heat capacity from maximum rate of temperature rise
    - Sensor responsiveness from the lag analysis
    """

    def __init__(self, samples, heater_power, t_ambient, threshold_temp=50.0):
        """
        Args:
            samples: list of (time, temperature) tuples from full-power step
            heater_power: heater power in watts
            t_ambient: measured ambient temperature
            threshold_temp: minimum temperature for analysis (filters initial transient)
        """
        self.samples = samples
        self.heater_power = heater_power
        self.t_ambient = t_ambient
        self.threshold_temp = threshold_temp
        self.log = logging.getLogger('chamber_mpc.calibrate')

    def analyze(self):
        """Run the step response analysis.

        Uses the differential method (maximum rate of rise) as the primary
        identification approach. This is robust for short step responses
        where the asymptotic method may fail (e.g. heating only to 60 deg C
        from ambient).

        Falls back to the asymptotic method for s1_responsiveness
        refinement when the data supports it.

        Returns:
            dict with keys: chamber_heat_capacity, s1_responsiveness,
                            post_chamber_temp, post_sensor_temp
        """
        if len(self.samples) < 10:
            raise ValueError(
                "Not enough samples for analysis (got %d, need >= 10)"
                % len(self.samples))

        start_temp = self.samples[0][1]
        end_temp = self.samples[-1][1]

        if end_temp - start_temp < 5.0:
            raise ValueError(
                "Temperature rise too small for analysis: "
                "%.1f to %.1f deg C (need >= 5 deg C)"
                % (start_temp, end_temp))

        # Primary method: differential (maximum rate of rise)
        # This works reliably for any step response length
        fastest = self._fastest_rate(self.samples)
        chamber_heat_capacity = self.heater_power / fastest[2]

        # Sensor responsiveness from differential method
        # fastest = [time_from_start, temperature_at_max_rate, max_rate]
        denom = (fastest[2] * fastest[0] + self.t_ambient - fastest[1])
        if abs(denom) > 0.01:
            s1_responsiveness = fastest[2] / denom
        else:
            # Fallback: conservative default for chamber sensors
            s1_responsiveness = 0.1

        # Sanity checks and clamps
        if chamber_heat_capacity <= 0:  # pragma: no cover
            self.log.warning(
                "chamber_heat_capacity negative (%.1f), using absolute value",
                chamber_heat_capacity)
            chamber_heat_capacity = abs(chamber_heat_capacity)

        if s1_responsiveness <= 0 or s1_responsiveness > 1.0:  # pragma: no cover
            self.log.warning(
                "s1_responsiveness out of range (%.4f), clamping",
                s1_responsiveness)
            s1_responsiveness = max(0.01, min(1.0,
                                                   abs(s1_responsiveness)))

        # Try asymptotic method for refined s1_responsiveness
        try:
            asymp = self._try_asymptotic_analysis()  # pragma: no cover
            if asymp is not None:
                self.log.info(
                    "Asymptotic refinement: asymp_T=%.1f",
                    asymp['asymp_temp'])
                if 0.001 < asymp['s1_responsiveness'] < 1.0:
                    s1_responsiveness = asymp['s1_responsiveness']
        except Exception as e:
            self.log.info(
                "Asymptotic analysis skipped (normal for short ramps): %s", e)

        return {
            'chamber_heat_capacity': chamber_heat_capacity,
            's1_responsiveness': s1_responsiveness,
            'post_chamber_temp': end_temp,
            'post_sensor_temp': end_temp,
        }

    def _try_asymptotic_analysis(self):
        """Attempt three-sample asymptotic analysis for refinement.

        Returns dict with asymp_temp and s1_responsiveness,
        or None if data is insufficient.
        """
        above_idx = None
        for i, (t, temp) in enumerate(self.samples):
            if temp > self.threshold_temp and above_idx is None:
                above_idx = i
            elif temp < self.threshold_temp:
                above_idx = None

        if above_idx is None or above_idx >= len(self.samples) - 6:
            return None

        segment = self.samples[above_idx:]
        if len(segment) < 6:
            return None

        t1_time = segment[0][0] - self.samples[0][0]
        pitch = max(1, len(segment) // 3)
        dt = segment[pitch][0] - segment[0][0]
        if dt <= 0:
            return None  # pragma: no cover

        t1 = segment[0][1]
        t2 = segment[pitch][1]
        t3 = segment[2 * pitch][1]

        denom = 2.0 * t2 - t1 - t3
        if abs(denom) < 0.01:
            return None

        asymp_T = (t2 * t2 - t1 * t3) / denom
        if asymp_T <= t1 or asymp_T <= self.t_ambient:
            return None

        ratio = (t2 - asymp_T) / (t1 - asymp_T)
        if ratio <= 0 or ratio >= 1.0:
            return None

        chamber_resp = -math.log(ratio) / dt

        start_temp = self.samples[0][1]
        exp_term = (start_temp - asymp_T) * math.exp(
            -chamber_resp * t1_time)
        denom2 = t1 - asymp_T
        if abs(denom2) < 0.01:
            return None  # pragma: no cover

        s1_resp = chamber_resp / (1.0 - exp_term / denom2)

        return {
            'asymp_temp': asymp_T,
            's1_responsiveness': sensor_resp,
        }

    def _fastest_rate(self, samples):
        """Find the point of maximum temperature rise rate.

        Returns [time_from_start, temperature, rate] of the fastest point.
        """
        best = [-1, 0, 0]
        base_t = samples[0][0]
        for i in range(2, len(samples)):
            dT = samples[i][1] - samples[i - 2][1]
            dt = samples[i][0] - samples[i - 2][0]
            if dt <= 0:
                continue  # pragma: no cover -- shouldn't happen with sorted data
            rate = dT / dt
            if rate > best[2]:
                sample = samples[i - 1]
                best = [sample[0] - base_t, sample[1], rate]
        if best[2] <= 0:
            raise ValueError("No positive temperature rise detected")  # pragma: no cover
        return best


def estimate_h_from_arrival(samples, heater_power, chamber_heat_capacity,
                           t_ambient, window=5):
    """Estimate rough h from the last portion of a step response.

    At the moment of arrival at target, the energy balance gives:
        C * dT/dt = P - h * (T - T_amb)
        h = (P - C * dT/dt) / (T - T_amb)

    Uses the last `window` samples to estimate dT/dt at arrival.

    Args:
        samples: list of (time, temperature) tuples
        heater_power: applied power in watts (P = duty * P_heater)
        chamber_heat_capacity: C in J/K
        t_ambient: ambient temperature in deg C
        window: number of trailing samples for slope estimation

    Returns:
        estimated h in W/K (clamped to positive values)
    """
    if len(samples) < window + 1:
        window = max(2, len(samples) // 2)

    # Estimate dT/dt from trailing samples
    tail = samples[-window:]
    dt = tail[-1][0] - tail[0][0]
    dT = tail[-1][1] - tail[0][1]
    if dt <= 0:
        return 0.1  # fallback

    rate = dT / dt  # deg C/s at arrival
    T_arrival = tail[-1][1]
    delta_T = T_arrival - t_ambient

    if delta_T <= 1.0:
        return 0.1  # too close to ambient for meaningful estimate

    h_est = (heater_power - chamber_heat_capacity * rate) / delta_T

    # Clamp: h must be positive, and a rough estimate might be
    # quite high if the system is still heating fast at arrival
    return max(0.05, h_est)


def estimate_h_from_cooling(samples, chamber_heat_capacity, t_ambient,
                            center_temp, window=10.0):
    """Estimate h from passive cooling slope around a target temperature.

    After a step response with heater off, the system cools passively.
    The energy balance with P=0 gives:
        C * dT/dt = -h * (T - T_amb)
        h = -C * (dT/dt) / (T - T_amb)

    Measures the cooling slope over a temperature window centered on
    center_temp (e.g. 85 to 75 deg C for center_temp=80, window=10).

    Args:
        samples: list of (time, temperature) tuples during cooling
        chamber_heat_capacity: C in J/K
        t_ambient: ambient temperature in deg C
        center_temp: temperature to estimate h at (deg C)
        window: temperature window width (deg C), centered on center_temp

    Returns:
        estimated h in W/K
    """
    t_high = center_temp + window / 2.0
    t_low = center_temp - window / 2.0

    # Find samples within the window
    window_samples = [(t, temp) for t, temp in samples
                      if t_low <= temp <= t_high]

    if len(window_samples) < 5:
        return None  # not enough data in window

    # Linear regression for slope (dT/dt)
    n = len(window_samples)
    sum_t = sum(s[0] for s in window_samples)
    sum_T = sum(s[1] for s in window_samples)
    sum_tT = sum(s[0] * s[1] for s in window_samples)
    sum_tt = sum(s[0] * s[0] for s in window_samples)

    denom = n * sum_tt - sum_t * sum_t
    if abs(denom) < 1e-10:
        return None  # degenerate data

    slope = (n * sum_tT - sum_t * sum_T) / denom  # dT/dt in deg C/s

    if slope >= 0:
        return None  # temperature is rising, not cooling

    # Average temperature in the window
    avg_temp = sum_T / n
    delta_T = avg_temp - t_ambient

    if delta_T <= 1.0:
        return None  # too close to ambient

    h = -chamber_heat_capacity * slope / delta_T

    if h <= 0:
        return None

    return h


def compute_cooling_rates(h_points, chamber_heat_capacity, t_ambient,
                          step_c=10.0):
    """Compute passive cooling rates across the calibrated range.

    Args:
        h_points: list of (T, h) tuples
        chamber_heat_capacity: C in J/K
        t_ambient: ambient temperature in deg C
        step_c: temperature step for the output table

    Returns:
        list of (T, cooling_rate_per_min, is_calibrated) tuples
    """
    from .h_interpolator import HInterpolator
    interp = HInterpolator(h_points)

    t_min = min(t for t, _ in h_points)
    t_max = max(t for t, _ in h_points)
    calibrated_temps = {round(t) for t, _ in h_points}

    # Round bounds to step grid
    t_start = int(round(t_min / step_c)) * int(step_c)
    t_end = int(round(t_max / step_c)) * int(step_c)

    results = []
    t = t_start
    while t <= t_end:
        h = interp.h(float(t))
        rate = -h * (t - t_ambient) / chamber_heat_capacity * 60.0
        is_cal = t in calibrated_temps
        results.append((t, rate, is_cal))
        t += int(step_c)

    return results


def format_calibration_report(result, cooling_rates):
    """Format calibration results for terminal output.

    Args:
        result: CalibrationResult instance
        cooling_rates: output of compute_cooling_rates()

    Returns:
        list of strings for respond_info
    """
    lines = [
        "Calibration complete.",
        "  chamber_heat_capacity   = %.1f J/K" % result.chamber_heat_capacity,
        "  s1_responsiveness = %.4f" % result.s1_responsiveness,
        "",
        "  h(T) calibration:",
    ]

    for T, h in result.h_points:
        rate = None
        for ct, cr, _ in cooling_rates:
            if ct == round(T):
                rate = cr
                break
        rate_str = "%.2f deg C/min" % rate if rate is not None else "N/A"
        lines.append(
            "    T=%5.1f\u00b0C  h=%.4f W/K  -> passive cool rate: %s"
            % (T, h, rate_str))

    if result.bed_transfer is not None:
        lines.append("")
        lines.append(
            "  bed_transfer = %.4f W/K" % result.bed_transfer)

    return lines


def format_cooling_rate_comments(result, cooling_rates):
    """Format cooling rate table for autosave comment block.

    Args:
        result: CalibrationResult instance
        cooling_rates: output of compute_cooling_rates()

    Returns:
        list of comment lines (without #*# prefix)
    """
    lines = [
        "# Calibration T_ambient: %.1f\u00b0C" % result.t_ambient,
        "#",
        "# Passive cooling rate at 10\u00b0C intervals:",
    ]

    for T, rate, is_cal in cooling_rates:
        marker = "  (calibrated)" if is_cal else ""
        lines.append(
            "#   T=%3d\u00b0C: %+.2f\u00b0C/min%s" % (int(T), rate, marker))

    return lines
