# Unit tests for chamber_mpc.thermal_model_advanced
import pytest
from chamber_mpc.thermal_model_advanced import ThermalModelAdvanced
from chamber_mpc.h_interpolator import HInterpolator
from chamber_mpc.kalman import KalmanFilter4


def make_advanced_model(kalman=None):
    h_interp = HInterpolator([(100.0, 5.5)])
    model = ThermalModelAdvanced(
        heater_heat_capacity=50.0,
        chamber_heat_capacity=500.0,
        heating_element_transfer=80.0,
        s1_responsiveness=0.5,
        s2_responsiveness=0.08,
        h_interpolator=h_interp,
        heater_power=1800.0,
        target_reach_time=2.0,
        kalman_filter=kalman if kalman else KalmanFilter4(1.0, 1.0, 0.5, 0.5, 0.5, 0.5),
    )
    return model


class TestAdvancedModelInit:
    def test_default_state(self):
        model = make_advanced_model()
        assert model.state_heater_temp == pytest.approx(25.0)
        assert model.state_chamber_temp == pytest.approx(25.0)
        assert model.state_s1_temp == pytest.approx(25.0)
        assert model.state_s2_temp == pytest.approx(25.0)

    def test_set_initial_state(self):
        model = make_advanced_model()
        model.set_initial_state(85.0)
        assert model.state_heater_temp == pytest.approx(85.0)
        assert model.state_chamber_temp == pytest.approx(85.0)
        assert model.state_s1_temp == pytest.approx(85.0)
        assert model.state_s2_temp == pytest.approx(85.0)


class TestAdvancedModelUpdate:
    def test_heating_drives_heater_first(self):
        model = make_advanced_model()
        model.set_initial_state(25.0)
        model.set_ambient(25.0)
        # First tick: propagates with last_power=0
        model.update(0.1, 25.0, 200.0, 1.0)
        # Second tick: propagates with power from first tick
        model.update(0.4, 25.0, 200.0, 1.0)
        # Heater should be ahead of chamber (heat goes through element first)
        assert model.state_heater_temp > model.state_chamber_temp

    def test_zero_target_gives_zero_output(self):
        model = make_advanced_model()
        model.set_initial_state(100.0)
        duty = model.update(0.1, 100.0, 0.0, 1.0)
        assert duty == 0.0

    def test_s1_sensor_provides_measurement(self):
        model = make_advanced_model()
        model.set_initial_state(25.0)
        model.set_ambient(25.0)
        # Set S1 sensor that reads a fixed value
        model.set_s2_sensor(lambda t: (30.0, 0.0))
        model.update(0.1, 25.0, 100.0, 1.0)
        # S2 reading should affect the heater state estimate
        # (correction pushes state_heater_temp toward S1 reading)

    def test_model_without_s1_uses_state(self):
        model = make_advanced_model()
        model.set_initial_state(50.0)
        model.set_ambient(25.0)
        # No S1 sensor configured - _read_s1 returns state estimate
        duty = model.update(0.1, 50.0, 100.0, 1.0)
        assert duty > 0


class TestAdvancedElementLimit:
    def test_element_limit_integrated(self):
        model = make_advanced_model()
        model.set_initial_state(25.0)
        model.set_ambient(25.0)
        model.s2_safe_temp = 100.0
        model.s2_safe_temp_zone = 20.0

        # Drive the model hard - element should be constrained by model
        for i in range(20):
            model.update(0.1 + i * 0.3, 25.0, 200.0, 1.0)

        # Heater state should not wildly exceed the limit
        # (model-integrated constraint prevents runaway)
        assert model.state_heater_temp < 150.0  # generous bound



class TestAdvancedKalmanEstimator:
    def test_kalman_mode_runs_without_crash(self):
        kalman = KalmanFilter4(1.0, 1.0, 0.5, 0.5, 0.3, 0.3)
        model = make_advanced_model(kalman=kalman)
        model.set_initial_state(50.0)
        model.set_ambient(25.0)
        model.set_s2_sensor(lambda t: (55.0, 0.0))
        # Run several ticks - should not crash or produce NaN
        for i in range(10):
            duty = model.update(0.1 + i * 0.3, 52.0, 100.0, 1.0)
            assert duty >= 0.0
            assert duty <= 1.0
        # States should be finite
        import math
        assert math.isfinite(model.state_heater_temp)
        assert math.isfinite(model.state_chamber_temp)


class TestAdvancedBedDisturbance:
    def test_hot_bed_reduces_output(self):
        model_with = make_advanced_model()
        model_with.set_initial_state(60.0)
        model_with.set_ambient(22.0)
        model_with.set_bed_disturbance(lambda t: (110.0, 110.0), 0.35)

        model_without = make_advanced_model()
        model_without.set_initial_state(60.0)
        model_without.set_ambient(22.0)

        d_with = model_with.update(0.1, 60.0, 60.0, 1.0)
        d_without = model_without.update(0.1, 60.0, 60.0, 1.0)
        assert d_with < d_without


class TestAdvancedGetStatus:
    def test_status_keys(self):
        model = make_advanced_model()
        status = model.get_status()
        assert 'temp_heater' in status
        assert 'temp_chamber' in status
        assert 'temp_s1' in status
        assert 'temp_s2' in status
        assert 'temp_ambient' in status
        assert 'power' in status
        assert 'avg_power' in status
        assert 'avg_duty' in status
