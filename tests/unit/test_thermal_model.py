# Unit tests for chamber_mpc.thermal_model
import pytest
from chamber_mpc.thermal_model import ThermalModel
from chamber_mpc.h_interpolator import HInterpolator


def make_model(h_val=0.15, C=360.0, sensor_resp=0.08,
               heater_power=1800.0, smoothing=0.5):
    """Create a ThermalModel with simple single-point h."""
    h_interp = HInterpolator([(100.0, h_val)])
    model = ThermalModel(
        chamber_heat_capacity=C,
        sensor_responsiveness=sensor_resp,
        h_interpolator=h_interp,
        heater_power=heater_power,
        smoothing=smoothing,
    )
    return model


class TestModelInitialization:
    def test_default_state(self):
        model = make_model()
        assert model.state_chamber_temp == pytest.approx(25.0)
        assert model.state_sensor_temp == pytest.approx(25.0)
        assert model.state_ambient_temp == pytest.approx(25.0)
        assert model.last_power == 0.0

    def test_set_initial_state(self):
        model = make_model()
        model.set_initial_state(85.0)
        assert model.state_chamber_temp == pytest.approx(85.0)
        assert model.state_sensor_temp == pytest.approx(85.0)

    def test_set_ambient(self):
        model = make_model()
        model.set_ambient(22.0)
        assert model.state_ambient_temp == pytest.approx(22.0)


class TestModelUpdate:
    def test_heating_increases_chamber_temp(self):
        model = make_model()
        model.set_initial_state(25.0)
        model.set_ambient(25.0)
        # First tick: computes output but propagation uses last_power=0
        duty = model.update(0.1, 25.0, 200.0, 1.0)
        assert duty > 0
        # Second tick: propagates with the power from first tick
        duty = model.update(0.4, 25.0, 200.0, 1.0)
        assert model.state_chamber_temp > 25.0

    def test_zero_target_gives_zero_output(self):
        model = make_model()
        model.set_initial_state(100.0)
        duty = model.update(0.1, 100.0, 0.0, 1.0)
        assert duty == 0.0

    def test_at_target_gives_steady_state_output(self):
        model = make_model()
        model.set_initial_state(100.0)
        model.set_ambient(22.0)
        # At target, output should be just enough to compensate losses
        duty = model.update(0.1, 100.0, 100.0, 1.0)
        # Should be small positive value (steady state compensation)
        assert duty > 0
        assert duty < 0.5  # not full power

    def test_below_target_gives_more_output(self):
        # Small temperature deltas to avoid both clamping to max power
        model = make_model()
        model.set_initial_state(95.0)
        model.set_ambient(22.0)
        duty_far = model.update(0.1, 95.0, 100.0, 1.0)

        model2 = make_model()
        model2.set_initial_state(99.0)
        model2.set_ambient(22.0)
        duty_close = model2.update(0.1, 99.0, 100.0, 1.0)

        assert duty_far > duty_close

    def test_output_clamped_to_max_power(self):
        model = make_model()
        model.set_initial_state(25.0)
        duty = model.update(0.1, 25.0, 300.0, 0.8)
        assert duty <= 0.8

    def test_sensor_lag(self):
        model = make_model(sensor_resp=0.05)
        model.set_initial_state(25.0)
        # Run a few ticks of heating
        t = 0.0
        for i in range(10):
            t += 0.3
            model.update(t, 25.0 + i * 2.0, 200.0, 1.0)
        # Block should be ahead of sensor due to lag
        assert model.state_chamber_temp > model.state_sensor_temp


class TestHeatingElementLimit:
    def test_no_limit_when_not_configured(self):
        model = make_model()
        model.set_initial_state(25.0)
        duty = model.update(0.1, 25.0, 200.0, 1.0)
        assert duty > 0

    def test_limit_cuts_output_at_max(self):
        model = make_model()
        model.set_initial_state(100.0)
        model.set_ambient(22.0)
        # Set up element sensor that reads at max
        model.set_heating_element_limit(
            lambda t: (270.0, 0.0),  # at max temp
            max_temp=270.0, margin=20.0)
        duty = model.update(0.1, 100.0, 200.0, 1.0)
        assert duty == 0.0

    def test_proportional_pullback(self):
        model = make_model()
        model.set_initial_state(100.0)
        model.set_ambient(22.0)
        # Element at pullback_start + half margin
        # max=270, margin=20, pullback_start=250
        # element at 260 -> headroom=10, scale=10/20=0.5
        model.set_heating_element_limit(
            lambda t: (260.0, 0.0),
            max_temp=270.0, margin=20.0)

        # Get unrestricted output for comparison
        model_unrestricted = make_model()
        model_unrestricted.set_initial_state(100.0)
        model_unrestricted.set_ambient(22.0)
        duty_free = model_unrestricted.update(0.1, 100.0, 200.0, 1.0)

        duty_limited = model.update(0.1, 100.0, 200.0, 1.0)
        # Should be approximately half of unrestricted
        assert duty_limited == pytest.approx(duty_free * 0.5, abs=0.05)

    def test_no_pullback_below_margin(self):
        model = make_model()
        model.set_initial_state(100.0)
        model.set_ambient(22.0)
        # Element well below pullback zone
        model.set_heating_element_limit(
            lambda t: (200.0, 0.0),
            max_temp=270.0, margin=20.0)

        model_unrestricted = make_model()
        model_unrestricted.set_initial_state(100.0)
        model_unrestricted.set_ambient(22.0)
        duty_free = model_unrestricted.update(0.1, 100.0, 200.0, 1.0)

        duty_limited = model.update(0.1, 100.0, 200.0, 1.0)
        assert duty_limited == pytest.approx(duty_free, abs=0.01)

    def test_model_propagates_with_actual_power(self):
        """Verify the model uses constrained power, not desired power."""
        model = make_model()
        model.set_initial_state(100.0)
        model.set_ambient(22.0)
        model.set_heating_element_limit(
            lambda t: (270.0, 0.0),  # element at max -> output = 0
            max_temp=270.0, margin=20.0)

        model.update(0.1, 100.0, 200.0, 1.0)
        # last_power should be 0 (constrained), not what MPC wanted
        assert model.last_power == 0.0


class TestBedDisturbance:
    def test_no_bed_by_default(self):
        model = make_model()
        model.set_initial_state(100.0)
        model.set_ambient(22.0)
        duty_no_bed = model.update(0.1, 100.0, 100.0, 1.0)
        # Should just be steady-state compensation
        assert duty_no_bed > 0

    def test_hot_bed_reduces_output(self):
        model_with_bed = make_model()
        model_with_bed.set_initial_state(60.0)
        model_with_bed.set_ambient(22.0)
        model_with_bed.set_bed_disturbance(
            lambda t: (110.0, 110.0),  # bed at 110 deg C
            bed_transfer=0.35)

        model_no_bed = make_model()
        model_no_bed.set_initial_state(60.0)
        model_no_bed.set_ambient(22.0)

        duty_with_bed = model_with_bed.update(0.1, 60.0, 60.0, 1.0)
        duty_no_bed = model_no_bed.update(0.1, 60.0, 60.0, 1.0)

        # Hot bed adds heat to chamber, so chamber heater needs less
        assert duty_with_bed < duty_no_bed


class TestModelCorrection:
    def test_correction_adjusts_both_states(self):
        model = make_model(smoothing=0.8)
        model.set_initial_state(100.0)
        # Feed a measurement that's higher than model predicts
        model.update(0.1, 105.0, 100.0, 1.0)
        # Both chamber and sensor should have been corrected upward
        assert model.state_chamber_temp > 100.0
        assert model.state_sensor_temp > 100.0

    def test_low_smoothing_corrects_less(self):
        model_low = make_model(smoothing=0.1)
        model_low.set_initial_state(100.0)
        model_low.update(0.1, 110.0, 100.0, 1.0)
        correction_low = model_low.state_sensor_temp - 100.0

        model_high = make_model(smoothing=0.9)
        model_high.set_initial_state(100.0)
        model_high.update(0.1, 110.0, 100.0, 1.0)
        correction_high = model_high.state_sensor_temp - 100.0

        assert correction_low < correction_high


class TestGetStatus:
    def test_status_keys(self):
        model = make_model()
        status = model.get_status()
        assert 'temp_chamber' in status
        assert 'temp_sensor' in status
        assert 'temp_ambient' in status
        assert 'power' in status
        assert 'avg_power' in status
        assert 'avg_duty' in status

    def test_status_reflects_state(self):
        model = make_model()
        model.set_initial_state(85.0)
        model.set_ambient(22.0)
        status = model.get_status()
        assert status['temp_chamber'] == pytest.approx(85.0)
        assert status['temp_ambient'] == pytest.approx(22.0)


class TestBasicKalmanEstimator:
    def test_kalman_mode_runs(self):
        from chamber_mpc.kalman import KalmanFilter2
        kalman = KalmanFilter2(1.0, 1.0, 0.5)
        h_interp = HInterpolator([(100.0, 0.15)])
        model = ThermalModel(
            chamber_heat_capacity=360.0,
            sensor_responsiveness=0.08,
            h_interpolator=h_interp,
            heater_power=1800.0,
            estimator_type='kalman',
            kalman_filter=kalman,
        )
        model.set_initial_state(25.0)
        model.set_ambient(25.0)
        for i in range(10):
            model.update(0.1 + i * 0.3, 26.0, 100.0, 1.0)
        assert model.state_chamber_temp > 25.0

    def test_kalman_gives_different_gains_per_state(self):
        from chamber_mpc.kalman import KalmanFilter2
        kalman = KalmanFilter2(1.0, 1.0, 0.5)
        h_interp = HInterpolator([(100.0, 0.15)])
        model = ThermalModel(
            chamber_heat_capacity=360.0,
            sensor_responsiveness=0.08,
            h_interpolator=h_interp,
            heater_power=1800.0,
            estimator_type='kalman',
            kalman_filter=kalman,
        )
        model.set_initial_state(100.0)
        model.set_ambient(22.0)
        # Run a few ticks to let gains converge
        for i in range(20):
            model.update(0.1 + i * 0.3, 100.5, 100.0, 1.0)
        k_chamber, k_sensor = kalman.get_gains()
        # Sensor gain should differ from chamber gain
        assert k_chamber != k_sensor

    def test_kalman_no_crash_with_large_innovation(self):
        from chamber_mpc.kalman import KalmanFilter2
        kalman = KalmanFilter2(1.0, 1.0, 0.5)
        h_interp = HInterpolator([(100.0, 0.15)])
        model = ThermalModel(
            chamber_heat_capacity=360.0,
            sensor_responsiveness=0.08,
            h_interpolator=h_interp,
            heater_power=1800.0,
            estimator_type='kalman',
            kalman_filter=kalman,
        )
        model.set_initial_state(25.0)
        model.set_ambient(25.0)
        # Large measurement jump - should not crash or produce NaN
        duty = model.update(0.1, 200.0, 100.0, 1.0)
        assert duty >= 0.0
        assert duty <= 1.0
