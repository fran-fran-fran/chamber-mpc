# Unit tests for chamber_mpc.calibrate
import math
import pytest
from chamber_mpc.calibrate import (
    StepResponseAnalyzer, SmoothingEstimator,
    compute_cooling_rates, CalibrationResult,
    format_calibration_report, format_cooling_rate_comments,
)


class TestStepResponseAnalyzer:
    def _generate_step_response(self, C=360.0, h=0.15, P=1800.0,
                                T_amb=22.0, dt=0.3, duration=300):
        """Generate a synthetic step response for a first-order system."""
        # T(t) = T_amb + (P/h) * (1 - exp(-h*t/C))
        asymp = T_amb + P / h
        samples = []
        t = 0.0
        while t < duration:
            T = T_amb + (P / h) * (1.0 - math.exp(-h * t / C))
            samples.append((t, T))
            t += dt
        return samples, asymp

    def test_identifies_chamber_heat_capacity(self):
        samples, _ = self._generate_step_response(C=360.0)
        analyzer = StepResponseAnalyzer(samples, 1800.0, 22.0)
        result = analyzer.analyze()
        # Should be within 20% of true value
        assert result['chamber_heat_capacity'] == pytest.approx(360.0, rel=0.2)

    def test_identifies_sensor_responsiveness(self):
        samples, _ = self._generate_step_response()
        analyzer = StepResponseAnalyzer(samples, 1800.0, 22.0)
        result = analyzer.analyze()
        assert result['sensor_responsiveness'] > 0

    def test_result_has_required_keys(self):
        samples, _ = self._generate_step_response()
        analyzer = StepResponseAnalyzer(samples, 1800.0, 22.0)
        result = analyzer.analyze()
        assert 'chamber_heat_capacity' in result
        assert 'sensor_responsiveness' in result
        assert 'post_chamber_temp' in result
        assert 'post_sensor_temp' in result

    def test_insufficient_samples_raises(self):
        samples = [(0, 20), (1, 21), (2, 22)]
        analyzer = StepResponseAnalyzer(samples, 1800.0, 22.0)
        with pytest.raises(ValueError, match="Not enough"):
            analyzer.analyze()


class TestSmoothingEstimator:
    def test_low_noise_gives_low_smoothing(self):
        # Model is very accurate - small errors, uncorrelated
        import random
        random.seed(42)
        errors = [random.gauss(0, 0.1) for _ in range(100)]
        smoothing = SmoothingEstimator.estimate(errors)
        assert smoothing < 0.6

    def test_high_noise_gives_higher_smoothing(self):
        import random
        random.seed(42)
        # Large systematic errors (model drift)
        errors = [random.gauss(0, 0.1) + 0.5 * math.sin(i / 10)
                  for i in range(100)]
        smoothing = SmoothingEstimator.estimate(errors)
        assert smoothing > 0.4

    def test_insufficient_data_returns_default(self):
        smoothing = SmoothingEstimator.estimate([0.1, 0.2, 0.3])
        assert smoothing == 0.5

    def test_clamps_to_range(self):
        # Any input should give a result in [0.2, 0.9]
        import random
        random.seed(123)
        for _ in range(10):
            errors = [random.gauss(0, random.uniform(0.01, 10))
                      for _ in range(100)]
            s = SmoothingEstimator.estimate(errors)
            assert 0.2 <= s <= 0.9


class TestComputeCoolingRates:
    def test_single_point(self):
        rates = compute_cooling_rates(
            [(100.0, 0.15)], 360.0, 22.0, step_c=10.0)
        assert len(rates) == 1
        T, rate, is_cal = rates[0]
        assert T == 100
        assert rate < 0  # cooling rate is negative
        assert is_cal is True

    def test_multi_point_has_interpolated(self):
        rates = compute_cooling_rates(
            [(60.0, 0.12), (100.0, 0.15)], 360.0, 22.0, step_c=10.0)
        # Should have entries at 60, 70, 80, 90, 100
        temps = [r[0] for r in rates]
        assert 60 in temps
        assert 80 in temps
        assert 100 in temps

    def test_calibrated_points_marked(self):
        rates = compute_cooling_rates(
            [(60.0, 0.12), (100.0, 0.15)], 360.0, 22.0, step_c=10.0)
        for T, rate, is_cal in rates:
            if T == 60 or T == 100:
                assert is_cal is True
            else:
                assert is_cal is False

    def test_cooling_rate_increases_with_temperature(self):
        rates = compute_cooling_rates(
            [(60.0, 0.12), (200.0, 0.23)], 360.0, 22.0, step_c=10.0)
        # Cooling rate magnitude should increase with temperature
        rate_low = abs([r for r in rates if r[0] == 60][0][1])
        rate_high = abs([r for r in rates if r[0] == 200][0][1])
        assert rate_high > rate_low


class TestFormatting:
    def test_report_includes_parameters(self):
        result = CalibrationResult()
        result.chamber_heat_capacity = 360.0
        result.sensor_responsiveness = 0.08
        result.smoothing = 0.45
        result.h_points = [(100.0, 0.15)]
        result.t_ambient = 22.0
        rates = compute_cooling_rates(
            result.h_points, result.chamber_heat_capacity, result.t_ambient)
        report = format_calibration_report(result, rates)
        text = "\n".join(report)
        assert "360.0" in text
        assert "0.0800" in text
        assert "0.45" in text

    def test_report_includes_bed_transfer(self):
        result = CalibrationResult()
        result.chamber_heat_capacity = 360.0
        result.sensor_responsiveness = 0.08
        result.smoothing = 0.45
        result.h_points = [(100.0, 0.15)]
        result.t_ambient = 22.0
        result.bed_transfer = 0.35
        rates = compute_cooling_rates(
            result.h_points, result.chamber_heat_capacity, result.t_ambient)
        report = format_calibration_report(result, rates)
        text = "\n".join(report)
        assert "bed_transfer" in text
        assert "0.35" in text

    def test_comments_include_calibrated_markers(self):
        result = CalibrationResult()
        result.t_ambient = 22.0
        result.h_points = [(60.0, 0.12), (100.0, 0.15)]
        rates = compute_cooling_rates(
            result.h_points, 360.0, result.t_ambient)
        comments = format_cooling_rate_comments(result, rates)
        text = "\n".join(comments)
        assert "(calibrated)" in text


class TestShortStepResponse:
    def test_short_ramp_does_not_crash(self):
        """A ramp from 22 to 60 deg C should still produce valid results."""
        C = 360.0
        h = 0.15
        P = 1800.0
        T_amb = 22.0
        samples = []
        t = 0.0
        while t < 120:  # only 2 minutes
            T = T_amb + (P / h) * (1.0 - math.exp(-h * t / C))
            if T > 60:
                break
            samples.append((t, T))
            t += 0.3
        analyzer = StepResponseAnalyzer(samples, P, T_amb)
        result = analyzer.analyze()
        assert result['chamber_heat_capacity'] > 0
        assert result['sensor_responsiveness'] > 0

    def test_insufficient_samples_raises(self):
        samples = [(0, 20), (1, 21), (2, 22)]
        analyzer = StepResponseAnalyzer(samples, 1800.0, 22.0)
        with pytest.raises(ValueError, match="Not enough"):
            analyzer.analyze()

    def test_no_temperature_rise_raises(self):
        # Flat temperature - no heating
        samples = [(i * 0.3, 22.0) for i in range(50)]
        analyzer = StepResponseAnalyzer(samples, 1800.0, 22.0)
        with pytest.raises(ValueError):
            analyzer.analyze()


class TestEstimateHFromCooling:
    def test_known_cooling_rate(self):
        """Generate cooling data with known h and verify recovery."""
        from chamber_mpc.calibrate import estimate_h_from_cooling
        C = 500.0
        h_true = 4.0
        T_amb = 25.0
        # Simulate cooling from 90 to 70 deg C
        samples = []
        T = 90.0
        t = 0.0
        dt = 0.3
        while T > 70.0:
            samples.append((t, T))
            dT = -h_true * (T - T_amb) / C * dt
            T += dT
            t += dt
        h_est = estimate_h_from_cooling(samples, C, T_amb, center_temp=80.0)
        assert h_est is not None
        assert h_est == pytest.approx(h_true, rel=0.05)

    def test_returns_none_on_insufficient_data(self):
        from chamber_mpc.calibrate import estimate_h_from_cooling
        samples = [(0, 80.0), (1, 79.5)]
        result = estimate_h_from_cooling(samples, 500.0, 25.0, 80.0)
        assert result is None

    def test_returns_none_if_rising(self):
        from chamber_mpc.calibrate import estimate_h_from_cooling
        # Temperature rising, not cooling
        samples = [(i * 0.3, 75.0 + i * 0.5) for i in range(30)]
        result = estimate_h_from_cooling(samples, 500.0, 25.0, 80.0)
        assert result is None

    def test_returns_none_near_ambient(self):
        from chamber_mpc.calibrate import estimate_h_from_cooling
        # Cooling near ambient
        samples = [(i * 0.3, 26.0 - i * 0.01) for i in range(30)]
        result = estimate_h_from_cooling(samples, 500.0, 25.0, 26.0)
        assert result is None
