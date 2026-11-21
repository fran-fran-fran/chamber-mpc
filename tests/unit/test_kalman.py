# Unit tests for chamber_mpc.kalman


class TestKalmanFilter3:
    def test_construction(self):
        from chamber_mpc.kalman import KalmanFilter3
        kf = KalmanFilter3(1.0, 0.1, 0.5, 0.5)
        assert kf.q_chamber == 1.0
        assert kf.q_s1 == 0.1
        assert kf.q_disturbance == 0.5
        assert kf.r == 0.5

    def test_predict_increases_covariance(self):
        from chamber_mpc.kalman import KalmanFilter3
        kf = KalmanFilter3(1.0, 0.1, 0.5, 0.5)
        p_before = list(kf.p)
        # Run multiple predict steps
        for _ in range(100):
            kf.predict(0.3, 0.03, 500.0)
        # Chamber and disturbance covariance should grow
        # (predict adds Q without update to reduce it)
        # Sensor covariance may shrink due to coupling
        assert kf.p[0] > p_before[0]
        assert kf.p[8] > p_before[8]
        # No NaN
        assert all(p == p for p in kf.p)

    def test_update_returns_three_corrections(self):
        from chamber_mpc.kalman import KalmanFilter3
        kf = KalmanFilter3(1.0, 0.1, 0.5, 0.5)
        kf.predict(0.3, 0.03, 500.0)
        corr = kf.update(1.0)
        assert len(corr) == 3
        # All corrections should be positive for positive innovation
        assert corr[0] > 0  # chamber
        assert corr[1] > 0  # sensor
        assert corr[2] > 0  # disturbance

    def test_disturbance_accumulates(self):
        """With persistent positive innovation, disturbance grows."""
        from chamber_mpc.kalman import KalmanFilter3
        kf = KalmanFilter3(1.0, 0.1, 0.5, 0.5)
        total_d = 0.0
        for _ in range(500):  # ~150 seconds at 0.3s ticks
            kf.predict(0.3, 0.03, 500.0)
            corr = kf.update(1.0)  # persistent +1 deg C innovation
            total_d += corr[2]
        assert total_d > 0.5  # disturbance accumulates slowly but steadily

    def test_disturbance_stops_when_innovation_zero(self):
        """With zero innovation, disturbance correction is zero."""
        from chamber_mpc.kalman import KalmanFilter3
        kf = KalmanFilter3(1.0, 0.1, 0.5, 0.5)
        # Let covariance converge
        for _ in range(100):
            kf.predict(0.3, 0.03, 500.0)
            kf.update(0.0)
        corr = kf.update(0.0)
        assert abs(corr[2]) < 1e-6

    def test_get_gains_returns_three(self):
        from chamber_mpc.kalman import KalmanFilter3
        kf = KalmanFilter3(1.0, 0.1, 0.5, 0.5)
        kf.predict(0.3, 0.03, 500.0)
        kf.update(1.0)
        gains = kf.get_gains()
        assert len(gains) == 3

    def test_covariance_stays_symmetric(self):
        from chamber_mpc.kalman import KalmanFilter3
        kf = KalmanFilter3(1.0, 0.1, 0.5, 0.5)
        for _ in range(50):
            kf.predict(0.3, 0.03, 500.0)
            kf.update(0.5)
        p = kf.p
        assert abs(p[1] - p[3]) < 1e-10
        assert abs(p[2] - p[6]) < 1e-10
        assert abs(p[5] - p[7]) < 1e-10
