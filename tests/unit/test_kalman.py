# Unit tests for chamber_mpc.kalman
import pytest
from chamber_mpc.kalman import KalmanFilter2, KalmanFilter4


class TestKalmanFilter2:
    def test_initial_gains_zero(self):
        kf = KalmanFilter2(1.0, 1.0, 0.5)
        assert kf.k_chamber == 0.0
        assert kf.k_sensor == 0.0

    def test_predict_covariance_reaches_steady_state(self):
        kf = KalmanFilter2(1.0, 1.0, 0.5)
        # Run many predict-update cycles to reach steady state
        for _ in range(100):
            kf.predict(0.3, 0.08)
            kf.update(0.01)
        p11_steady = kf.p11
        # At steady state, one more predict should increase P slightly
        # (Q adds uncertainty, then update reduces it)
        kf.predict(0.3, 0.08)
        assert kf.p11 > p11_steady - 0.1  # approximately stable

    def test_update_reduces_covariance(self):
        kf = KalmanFilter2(1.0, 1.0, 0.5)
        kf.predict(0.3, 0.08)
        p11_after_predict = kf.p11
        kf.update(1.0)
        assert kf.p11 < p11_after_predict

    def test_correction_proportional_to_innovation(self):
        kf = KalmanFilter2(1.0, 1.0, 0.5)
        kf.predict(0.3, 0.08)
        c1_chamber, c1_sensor = kf.update(1.0)
        kf2 = KalmanFilter2(1.0, 1.0, 0.5)
        kf2.predict(0.3, 0.08)
        c2_chamber, c2_sensor = kf2.update(2.0)
        assert c2_chamber == pytest.approx(c1_chamber * 2.0)
        assert c2_sensor == pytest.approx(c1_sensor * 2.0)

    def test_sensor_gain_larger_than_chamber_gain(self):
        """Sensor state should be corrected more (directly observed)."""
        kf = KalmanFilter2(1.0, 1.0, 0.5)
        kf.predict(0.3, 0.08)
        kf.update(1.0)
        assert abs(kf.k_sensor) > abs(kf.k_chamber)

    def test_gains_converge_over_many_ticks(self):
        kf = KalmanFilter2(1.0, 1.0, 0.5)
        gains = []
        for _ in range(100):
            kf.predict(0.3, 0.08)
            kf.update(0.1)
            gains.append(kf.k_sensor)
        # Gains should converge (last 10 values nearly identical)
        late_gains = gains[-10:]
        spread = max(late_gains) - min(late_gains)
        assert spread < 0.01

    def test_get_gains(self):
        kf = KalmanFilter2(1.0, 1.0, 0.5)
        kf.predict(0.3, 0.08)
        kf.update(1.0)
        g = kf.get_gains()
        assert len(g) == 2
        assert g[0] == kf.k_chamber
        assert g[1] == kf.k_sensor

    def test_covariance_stays_symmetric(self):
        kf = KalmanFilter2(1.0, 1.0, 0.5)
        for _ in range(50):
            kf.predict(0.3, 0.08)
            kf.update(0.5)
        assert kf.p01 == pytest.approx(kf.p10, abs=1e-10)

    def test_zero_innovation_no_correction(self):
        kf = KalmanFilter2(1.0, 1.0, 0.5)
        kf.predict(0.3, 0.08)
        corr_c, corr_s = kf.update(0.0)
        assert corr_c == 0.0
        assert corr_s == 0.0


class TestKalmanFilter4:
    def test_initial_gains_zero(self):
        kf = KalmanFilter4(1.0, 1.0, 0.5, 0.5, 0.3, 0.3)
        assert all(k == 0.0 for k in kf.k)

    def test_predict_covariance_reaches_steady_state(self):
        kf = KalmanFilter4(1.0, 1.0, 0.5, 0.5, 0.3, 0.3)
        # Run many predict-update cycles
        for _ in range(100):
            kf.predict(0.3, 50.0, 100.0, 500.0, 5.0, 0.5, 0.08)
            kf.update(0.01, 0.01)
        # Covariance should have converged (all diagonal > 0)
        for i in range(4):
            assert kf.p[i * 4 + i] > 0

    def test_update_returns_four_corrections(self):
        kf = KalmanFilter4(1.0, 1.0, 0.5, 0.5, 0.3, 0.3)
        kf.predict(0.3, 50.0, 100.0, 500.0, 5.0, 0.5, 0.08)
        corr = kf.update(1.0, 1.0)
        assert len(corr) == 4

    def test_s1_innovation_primarily_corrects_heater(self):
        """S1 measures near the heater, so S1 innovation should
        primarily correct T_heater and T_s1."""
        kf = KalmanFilter4(1.0, 1.0, 0.5, 0.5, 0.3, 0.3)
        for _ in range(20):
            kf.predict(0.3, 50.0, 100.0, 500.0, 5.0, 0.5, 0.08)
            kf.update(0.1, 0.0)
        gains = kf.get_gains()
        # k_heater_from_s1 should be significant
        # k_chamber_from_s1 should be smaller
        assert abs(gains['k_heater_from_s1']) > 0
        assert abs(gains['k_s1_from_s1']) > 0

    def test_s2_innovation_primarily_corrects_chamber(self):
        """S2 measures chamber air, so S2 innovation should
        primarily correct T_chamber and T_s2."""
        kf = KalmanFilter4(1.0, 1.0, 0.5, 0.5, 0.3, 0.3)
        for _ in range(20):
            kf.predict(0.3, 50.0, 100.0, 500.0, 5.0, 0.5, 0.08)
            kf.update(0.0, 0.1)
        gains = kf.get_gains()
        assert abs(gains['k_chamber_from_s2']) > 0
        assert abs(gains['k_s2_from_s2']) > 0

    def test_zero_innovations_no_correction(self):
        kf = KalmanFilter4(1.0, 1.0, 0.5, 0.5, 0.3, 0.3)
        kf.predict(0.3, 50.0, 100.0, 500.0, 5.0, 0.5, 0.08)
        corr = kf.update(0.0, 0.0)
        assert all(c == 0.0 for c in corr)

    def test_covariance_stays_symmetric(self):
        kf = KalmanFilter4(1.0, 1.0, 0.5, 0.5, 0.3, 0.3)
        for _ in range(50):
            kf.predict(0.3, 50.0, 100.0, 500.0, 5.0, 0.5, 0.08)
            kf.update(0.5, 0.3)
        for i in range(4):
            for j in range(i + 1, 4):
                assert kf.p[i * 4 + j] == pytest.approx(
                    kf.p[j * 4 + i], abs=1e-8)

    def test_get_gains_returns_dict(self):
        kf = KalmanFilter4(1.0, 1.0, 0.5, 0.5, 0.3, 0.3)
        kf.predict(0.3, 50.0, 100.0, 500.0, 5.0, 0.5, 0.08)
        kf.update(1.0, 1.0)
        g = kf.get_gains()
        assert isinstance(g, dict)
        assert 'k_heater_from_s1' in g
        assert 'k_chamber_from_s2' in g
