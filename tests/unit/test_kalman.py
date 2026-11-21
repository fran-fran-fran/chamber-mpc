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
        # From diagonal P, d couples into T_c on step 1 and into T_s on step 2
        kf.predict(0.3, 0.03, 500.0)
        corr1 = kf.update(1.0)
        assert len(corr1) == 3
        assert corr1[0] > 0  # chamber
        assert corr1[1] > 0  # sensor
        assert corr1[2] == 0.0  # no d <-> T_s covariance after 1 step

        kf.predict(0.3, 0.03, 500.0)
        corr2 = kf.update(1.0)
        assert len(corr2) == 3
        # All corrections should be positive for positive innovation
        assert corr2[0] > 0  # chamber
        assert corr2[1] > 0  # sensor
        assert corr2[2] > 0  # disturbance

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

    def test_predict_matches_reference_matrix_math(self):
        """Verify predict() computes A * P * A^T + Q * dt exactly."""
        from chamber_mpc.kalman import KalmanFilter3
        dt, r_s1, C = 0.3, 0.08, 360.0
        qc, qs, qd, r_meas = 1.0, 0.1, 10.0, 0.5
        kf = KalmanFilter3(qc, qs, qd, r_meas)
        # Non-diagonal symmetric initial P to exercise all cross terms
        P = [
            [5.0, 1.2, 3.4],
            [1.2, 2.0, 0.7],
            [3.4, 0.7, 50.0],
        ]
        kf.p = [P[i][j] for i in range(3) for j in range(3)]

        kf.predict(dt, r_s1, C)

        r = r_s1 * dt
        a02 = dt / C
        A = [
            [1.0, 0.0, a02],
            [r, 1.0 - r, 0.0],
            [0.0, 0.0, 1.0],
        ]
        # Reference: A @ P @ A^T + Q * dt
        AP = [[sum(A[i][k] * P[k][j] for k in range(3)) for j in range(3)]
              for i in range(3)]
        expected = [[sum(AP[i][k] * A[j][k] for k in range(3)) for j in range(3)]
                    for i in range(3)]
        expected[0][0] += qc * dt
        expected[1][1] += qs * dt
        expected[2][2] += qd * dt

        for i in range(3):
            for j in range(3):
                assert abs(kf.p[i * 3 + j] - expected[i][j]) < 1e-12
