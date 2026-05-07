# Unit tests for chamber_mpc.h_interpolator
import pytest
from chamber_mpc.h_interpolator import HInterpolator


class TestSinglePoint:
    def test_returns_constant(self):
        interp = HInterpolator([(100.0, 0.15)])
        assert interp.h(50.0) == 0.15
        assert interp.h(100.0) == 0.15
        assert interp.h(200.0) == 0.15

    def test_point_count(self):
        interp = HInterpolator([(100.0, 0.15)])
        assert interp.point_count() == 1


class TestMultiPoint:
    def setup_method(self):
        self.interp = HInterpolator([
            (60.0, 0.124),
            (100.0, 0.151),
            (150.0, 0.187),
            (200.0, 0.231),
        ])

    def test_at_calibration_points(self):
        assert self.interp.h(60.0) == pytest.approx(0.124)
        assert self.interp.h(100.0) == pytest.approx(0.151)
        assert self.interp.h(200.0) == pytest.approx(0.231)

    def test_interpolation_midpoint(self):
        # Midpoint between 60 and 100: h = (0.124 + 0.151) / 2 = 0.1375
        h = self.interp.h(80.0)
        assert h == pytest.approx(0.1375)

    def test_clamp_below_range(self):
        assert self.interp.h(20.0) == pytest.approx(0.124)
        assert self.interp.h(0.0) == pytest.approx(0.124)

    def test_clamp_above_range(self):
        assert self.interp.h(250.0) == pytest.approx(0.231)
        assert self.interp.h(500.0) == pytest.approx(0.231)

    def test_at_exact_boundary(self):
        assert self.interp.h(60.0) == pytest.approx(0.124)
        assert self.interp.h(200.0) == pytest.approx(0.231)

    def test_point_count(self):
        assert self.interp.point_count() == 4

    def test_temp_range(self):
        assert self.interp.temp_range() == (60.0, 200.0)

    def test_points_returned_sorted(self):
        # Construct with unsorted input
        interp = HInterpolator([(200.0, 0.23), (60.0, 0.12)])
        pts = interp.points()
        assert pts[0][0] < pts[1][0]


class TestPassiveCoolingRate:
    def test_rate_at_calibrated_point(self):
        interp = HInterpolator([(100.0, 0.15)])
        # rate = -h * (T - T_amb) / C = -0.15 * (100 - 22) / 360
        rate = interp.passive_cooling_rate(100.0, 360.0, 22.0)
        expected = -0.15 * 78.0 / 360.0
        assert rate == pytest.approx(expected)

    def test_rate_per_min(self):
        interp = HInterpolator([(100.0, 0.15)])
        rate = interp.passive_cooling_rate_per_min(100.0, 360.0, 22.0)
        expected = -0.15 * 78.0 / 360.0 * 60.0
        assert rate == pytest.approx(expected)

    def test_rate_at_ambient_is_zero(self):
        interp = HInterpolator([(100.0, 0.15)])
        rate = interp.passive_cooling_rate(22.0, 360.0, 22.0)
        assert rate == pytest.approx(0.0)


class TestConfigParsing:
    def test_parse_multi_line(self):
        config = """
            60.0, 0.1241
            100.0, 0.1513
            150.0, 0.1872
            200.0, 0.2306
        """
        pts = HInterpolator.parse_config_string(config)
        assert len(pts) == 4
        assert pts[0] == (60.0, 0.1241)
        assert pts[3] == (200.0, 0.2306)

    def test_parse_single_line(self):
        pts = HInterpolator.parse_config_string("70.0, 0.142")
        assert len(pts) == 1
        assert pts[0] == (70.0, 0.142)

    def test_parse_bad_format(self):
        with pytest.raises(ValueError, match="must be"):
            HInterpolator.parse_config_string("60.0")

    def test_parse_bad_value(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            HInterpolator.parse_config_string("abc, 0.12")

    def test_parse_negative_h(self):
        with pytest.raises(ValueError, match="non-negative"):
            HInterpolator.parse_config_string("60.0, -0.1")

    def test_parse_with_blank_lines(self):
        config = """
            60.0, 0.1241

            100.0, 0.1513

        """
        pts = HInterpolator.parse_config_string(config)
        assert len(pts) == 2

    def test_round_trip(self):
        interp = HInterpolator([
            (60.0, 0.1241),
            (100.0, 0.1513),
        ])
        config_str = interp.format_config_string()
        parsed = HInterpolator.parse_config_string(config_str)
        assert len(parsed) == 2
        assert parsed[0][0] == pytest.approx(60.0)
        assert parsed[0][1] == pytest.approx(0.1241)


class TestConstruction:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="[Aa]t least one"):
            HInterpolator([])
