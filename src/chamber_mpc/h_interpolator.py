# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: h_interpolator.py
# Description: Interpolates the ambient heat transfer coefficient h(T)
#              from calibration points. Supports single-point (constant h)
#              or multi-point (linear/cubic interpolation).


class HInterpolator:
    """Interpolates h(T) from calibration points.

    Single calibration point: returns constant h for all temperatures.
    Multiple points: linear interpolation between adjacent points,
    clamped to the nearest value outside the calibrated range.
    """

    def __init__(self, calibration_points):
        """
        Args:
            calibration_points: list of (T_celsius, h_value) tuples,
                                where h is in W/K.
        """
        if not calibration_points:
            raise ValueError("At least one calibration point is required")
        self._points = sorted(calibration_points, key=lambda p: p[0])

    def h(self, temp_c):
        """Return interpolated h at the given temperature.

        Args:
            temp_c: temperature in Celsius

        Returns:
            h value in W/K
        """
        pts = self._points

        if len(pts) == 1:
            return pts[0][1]

        # Below lowest calibration point: clamp to lowest h
        if temp_c <= pts[0][0]:
            return pts[0][1]

        # Above highest calibration point: clamp to highest h
        if temp_c >= pts[-1][0]:
            return pts[-1][1]

        # Find the two surrounding points and interpolate linearly
        for i in range(len(pts) - 1):
            t_lo, h_lo = pts[i]
            t_hi, h_hi = pts[i + 1]
            if t_lo <= temp_c <= t_hi:
                frac = (temp_c - t_lo) / (t_hi - t_lo)
                return h_lo + frac * (h_hi - h_lo)

        return pts[-1][1]  # pragma: no cover -- unreachable with valid input

    def passive_cooling_rate(self, temp_c, chamber_heat_capacity, t_ambient):
        """Compute passive cooling rate at given temperature.

        Returns dT/dt in deg C/s (negative value = cooling).
        """
        h = self.h(temp_c)
        return -h * (temp_c - t_ambient) / chamber_heat_capacity

    def passive_cooling_rate_per_min(self, temp_c, chamber_heat_capacity,
                                     t_ambient):
        """Compute passive cooling rate in deg C/min."""
        return self.passive_cooling_rate(
            temp_c, chamber_heat_capacity, t_ambient) * 60.0

    def point_count(self):
        """Number of calibration points."""
        return len(self._points)

    def points(self):
        """Return calibration points as list of (T, h) tuples."""
        return list(self._points)

    def temp_range(self):
        """Return (T_min, T_max) of calibrated range."""
        return (self._points[0][0], self._points[-1][0])

    @staticmethod
    def parse_config_string(value):
        """Parse ambient_transfer_points from config string.

        Format (multi-line, tab-indented):
            60.0, 0.1241
            100.0, 0.1513
            150.0, 0.1872
            200.0, 0.2306

        Returns list of (T, h) tuples.
        """
        points = []
        for line in value.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                raise ValueError(
                    "ambient_transfer_points line must be 'T, h': %s" % line)
            try:
                t = float(parts[0].strip())
                h = float(parts[1].strip())
            except ValueError:
                raise ValueError(
                    "Cannot parse ambient_transfer_points values: %s" % line)
            if h < 0:
                raise ValueError(
                    "h value must be non-negative: %s" % line)
            points.append((t, h))
        return points

    def format_config_string(self):
        """Format calibration points for config file output.

        Returns multi-line string suitable for Klipper's autosave block.
        """
        lines = []
        for t, h in self._points:
            lines.append("    %.1f, %.4f" % (t, h))
        return "\n" + "\n".join(lines)
