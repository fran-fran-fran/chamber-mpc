# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: kalman.py
# Description: Kalman filter for state estimation in thermal models.
#              Implements the basic 3-state estimator and computes optimal
#              per-state correction gains from the prediction error
#              covariance.


class KalmanFilter3:
    """Kalman filter for 2-state basic model with disturbance estimation.

    States: [T_chamber, T_sensor, d]
    Measurement: T_sensor (single observation)

    The disturbance state d is an integrating state (no dynamics)
    that captures persistent model-parameter errors (e.g. wrong h).
    It is added to the heater power in the propagation step, reducing
    steady-state error without integral windup.
    """

    def __init__(self, process_noise_chamber, process_noise_s1,
                 process_noise_disturbance, measurement_noise_s1):
        # Q diagonal: process noise variances
        self.q_chamber = float(process_noise_chamber)
        self.q_s1 = float(process_noise_s1)
        self.q_disturbance = float(process_noise_disturbance)

        # R: measurement noise variance (scalar, one measurement)
        self.r = float(measurement_noise_s1)

        # P: state covariance matrix (3x3, stored as 9 elements, row-major)
        # Initialize with moderate uncertainty
        self.p = [
            10.0, 0.0, 0.0,
            0.0, 10.0, 0.0,
            0.0, 0.0, 10.0,
        ]

        # Last computed gains (for diagnostics)
        self.k_chamber = 0.0
        self.k_s1 = 0.0
        self.k_disturbance = 0.0

    def predict(self, dt, s1_responsiveness, chamber_heat_capacity):
        """Propagate covariance forward (predict step).

        The state transition Jacobian A:
            T_c(k+1)  = T_c(k) + dt/C * (P + d - h*(T_c - T_amb))
            T_s(k+1)  = T_s(k) + r*dt*(T_c - T_s)
            d(k+1)    = d(k)

        Discretized A:
            A = [[1,     0,  dt/C],
                 [r*dt,  1-r*dt,  0],
                 [0,     0,  1   ]]

        The coupling dt/C converts the disturbance (watts) to
        temperature change per tick.
        """
        r = s1_responsiveness * dt
        p = self.p

        # A matrix (3x3):
        # a00=1,  a01=0,  a02=dt/C  (d couples into T_c)
        # a10=r,  a11=1-r, a12=0
        # a20=0,  a21=0,  a22=1     (d persists)
        a02 = dt / chamber_heat_capacity  # coupling of d into T_chamber

        # A * P (3x3 times 3x3)
        ap = [0.0] * 9
        # Row 0: [1, 0, a02] * P
        ap[0] = p[0] + a02 * p[6]
        ap[1] = p[1] + a02 * p[7]
        ap[2] = p[2] + a02 * p[8]
        # Row 1: [r, 1-r, 0] * P
        ap[3] = r * p[0] + (1.0 - r) * p[3]
        ap[4] = r * p[1] + (1.0 - r) * p[4]
        ap[5] = r * p[2] + (1.0 - r) * p[5]
        # Row 2: [0, 0, 1] * P
        ap[6] = p[6]
        ap[7] = p[7]
        ap[8] = p[8]

        # (A*P) * A^T
        # A^T columns: [1, r, 0], [0, 1-r, 0], [a02, 0, 1]
        new_p = [0.0] * 9
        for i in range(3):
            # Column 0 of A^T: [1, r, 0]
            new_p[i * 3 + 0] = ap[i * 3 + 0] + ap[i * 3 + 1] * r
            # Column 1 of A^T: [0, 1-r, 0]
            new_p[i * 3 + 1] = ap[i * 3 + 1] * (1.0 - r)
            # Column 2 of A^T: [a02, 0, 1]
            new_p[i * 3 + 2] = ap[i * 3 + 0] * a02 + ap[i * 3 + 2]

        # Add Q (diagonal)
        new_p[0] += self.q_chamber * dt
        new_p[4] += self.q_s1 * dt
        new_p[8] += self.q_disturbance * dt

        # Enforce symmetry
        for i in range(3):
            for j in range(i + 1, 3):
                avg = (new_p[i * 3 + j] + new_p[j * 3 + i]) * 0.5
                new_p[i * 3 + j] = avg
                new_p[j * 3 + i] = avg

        self.p = new_p

    def update(self, innovation):
        """Compute Kalman gains and return per-state corrections.

        Measurement matrix H = [0, 1, 0] (we observe T_sensor).

        Args:
            innovation: (temp_measured - state_s1_temp)

        Returns:
            (correction_chamber, correction_sensor, correction_disturbance)
        """
        p = self.p

        # Innovation covariance: S = H * P * H^T + R
        # With H = [0, 1, 0]: S = P[1,1] + R
        s = p[4] + self.r
        if s < 1e-10:
            s = 1e-10  # pragma: no cover

        # Kalman gain: K = P * H^T / S
        # P * H^T = [P[0,1], P[1,1], P[2,1]]^T
        self.k_chamber = p[1] / s
        self.k_s1 = p[4] / s
        self.k_disturbance = p[7] / s

        # Update covariance: P = (I - K*H) * P
        # K*H = [[0, k0, 0], [0, k1, 0], [0, k2, 0]]
        # (I - K*H) = [[1, -k0, 0], [0, 1-k1, 0], [0, -k2, 1]]
        new_p = list(p)
        k0 = self.k_chamber
        k1 = self.k_s1
        k2 = self.k_disturbance
        # Row 0: [1, -k0, 0] * P
        new_p[0] = p[0] - k0 * p[3]
        new_p[1] = p[1] - k0 * p[4]
        new_p[2] = p[2] - k0 * p[5]
        # Row 1: [0, 1-k1, 0] * P
        new_p[3] = (1.0 - k1) * p[3]
        new_p[4] = (1.0 - k1) * p[4]
        new_p[5] = (1.0 - k1) * p[5]
        # Row 2: [0, -k2, 1] * P
        new_p[6] = p[6] - k2 * p[3]
        new_p[7] = p[7] - k2 * p[4]
        new_p[8] = p[8] - k2 * p[5]

        # Enforce symmetry
        for i in range(3):
            for j in range(i + 1, 3):
                avg = (new_p[i * 3 + j] + new_p[j * 3 + i]) * 0.5
                new_p[i * 3 + j] = avg
                new_p[j * 3 + i] = avg

        self.p = new_p

        # Corrections
        return (
            self.k_chamber * innovation,
            self.k_s1 * innovation,
            self.k_disturbance * innovation,
        )

    def get_gains(self):
        """Return current Kalman gains for diagnostics."""
        return (self.k_chamber, self.k_s1, self.k_disturbance)
