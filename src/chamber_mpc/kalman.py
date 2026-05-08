# chamber-mpc: Model Predictive Control for heated chambers
#
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# File: kalman.py
# Description: Kalman filter for state estimation in thermal models.
#              Supports 2-state (basic) and 4-state (advanced) models.
#              Computes optimal per-state correction gains from the
#              prediction error covariance, replacing fixed smoothing.


class KalmanFilter2:
    """Kalman filter for the 2-state basic thermal model.

    States: [T_chamber, T_sensor]
    Measurement: T_sensor (single observation)

    The filter maintains a 2x2 covariance matrix P and computes
    optimal correction gains K on every tick. Gains adapt automatically:
    higher during transients (model uncertain), lower at steady state
    (model reliable).
    """

    def __init__(self, process_noise_chamber, process_noise_sensor,
                 measurement_noise):
        # Q diagonal: process noise variances
        self.q_chamber = float(process_noise_chamber)
        self.q_sensor = float(process_noise_sensor)

        # R: measurement noise variance (scalar, one measurement)
        self.r = float(measurement_noise)

        # P: state covariance matrix (2x2, stored as 4 elements)
        # Initialize with moderate uncertainty
        self.p00 = 10.0  # var(T_chamber)
        self.p01 = 0.0   # cov(T_chamber, T_sensor)
        self.p10 = 0.0   # cov(T_sensor, T_chamber)
        self.p11 = 10.0  # var(T_sensor)

        # Last computed gains (for diagnostics)
        self.k_chamber = 0.0
        self.k_sensor = 0.0

    def predict(self, dt, sensor_responsiveness):
        """Propagate covariance forward (predict step).

        The state transition for covariance is:
            P = A * P * A^T + Q

        where A is the Jacobian of the state dynamics:
            A = [[1, 0], [r*dt, 1-r*dt]]

        with r = sensor_responsiveness.
        """
        r = sensor_responsiveness * dt

        # A * P (matrix multiply A by P)
        # A = [[1, 0], [r, 1-r]]
        ap00 = self.p00
        ap01 = self.p01
        ap10 = r * self.p00 + (1.0 - r) * self.p10
        ap11 = r * self.p01 + (1.0 - r) * self.p11

        # (A * P) * A^T
        # A^T = [[1, r], [0, 1-r]]
        self.p00 = ap00 + self.q_chamber * dt
        self.p01 = ap01 * (1.0 - r) + ap00 * r
        self.p10 = ap10 + ap11 * r  # should equal p01 (symmetric)
        self.p11 = ap10 * r + ap11 * (1.0 - r) + self.q_sensor * dt

        # Enforce symmetry (numerical drift)
        self.p01 = (self.p01 + self.p10) * 0.5
        self.p10 = self.p01

    def update(self, innovation):
        """Compute Kalman gains and return per-state corrections.

        The measurement matrix H = [0, 1] (we observe T_sensor).

        Args:
            innovation: (temp_measured - state_sensor_temp)

        Returns:
            (correction_chamber, correction_sensor) tuple
        """
        # Innovation covariance: S = H * P * H^T + R
        # With H = [0, 1]: S = P[1,1] + R
        s = self.p11 + self.r
        if s < 1e-10:
            s = 1e-10  # pragma: no cover -- prevent division by zero

        # Kalman gain: K = P * H^T / S
        # P * H^T = [P[0,1], P[1,1]]^T
        self.k_chamber = self.p01 / s
        self.k_sensor = self.p11 / s

        # Update covariance: P = (I - K*H) * P
        # K*H = [[0, k0], [0, k1]]
        # (I - K*H) = [[1, -k0], [0, 1-k1]]
        new_p00 = self.p00 - self.k_chamber * self.p10
        new_p01 = self.p01 - self.k_chamber * self.p11
        new_p10 = self.p10 - self.k_sensor * self.p10
        new_p11 = self.p11 - self.k_sensor * self.p11

        self.p00 = new_p00
        self.p01 = (new_p01 + new_p10) * 0.5  # enforce symmetry
        self.p10 = self.p01
        self.p11 = new_p11

        # Corrections
        return (self.k_chamber * innovation, self.k_sensor * innovation)

    def get_gains(self):
        """Return current Kalman gains for diagnostics."""
        return (self.k_chamber, self.k_sensor)


class KalmanFilter4:
    """Kalman filter for the 4-state advanced thermal model.

    States: [T_heater, T_chamber, T_s1, T_s2]
    Measurements: [T_s1, T_s2] (two observations)

    The filter maintains a 4x4 covariance matrix P and computes
    optimal correction gains K (4x2 matrix) on every tick.
    """

    def __init__(self, process_noise_heater, process_noise_chamber,
                 process_noise_s1, process_noise_s2,
                 measurement_noise_s1, measurement_noise_s2):
        # Q diagonal: process noise variances for each state
        self.q = [
            float(process_noise_heater),
            float(process_noise_chamber),
            float(process_noise_s1),
            float(process_noise_s2),
        ]

        # R diagonal: measurement noise variances
        self.r = [float(measurement_noise_s1), float(measurement_noise_s2)]

        # P: 4x4 covariance (stored as flat list, row-major)
        self.p = [
            10.0, 0.0, 0.0, 0.0,
            0.0, 10.0, 0.0, 0.0,
            0.0, 0.0, 10.0, 0.0,
            0.0, 0.0, 0.0, 10.0,
        ]

        # Last computed gains (4x2, for diagnostics)
        self.k = [0.0] * 8

    def predict(self, dt, k_hc, c_h, c_c, h_val, r_s1, r_s2):
        """Propagate covariance forward.

        The Jacobian A of the 4-state dynamics:
            dT_h/dt  = -(k_hc/C_h)*T_h + (k_hc/C_h)*T_c
            dT_c/dt  = (k_hc/C_c)*T_h - (k_hc+h)/C_c*T_c
            dT_s1/dt = r_s1*T_h - r_s1*T_s1
            dT_s2/dt = r_s2*T_c - r_s2*T_s2

        Discretized A = I + Ac*dt (first-order Euler).
        """
        a01 = k_hc / c_h * dt
        a00 = 1.0 - a01
        a10 = k_hc / c_c * dt
        a11 = 1.0 - (k_hc + h_val) / c_c * dt
        a20 = r_s1 * dt
        a22 = 1.0 - a20
        a31 = r_s2 * dt
        a33 = 1.0 - a31

        # Build full 4x4 A matrix (sparse, most entries are 0)
        # A = [[a00, a01,   0,   0],
        #      [a10, a11,   0,   0],
        #      [a20,   0, a22,   0],
        #      [  0, a31,   0, a33]]

        # Compute A*P*A^T + Q using explicit operations
        # This avoids numpy dependency and is fast for 4x4
        p = self.p
        new_p = [0.0] * 16

        # A * P (4x4 times 4x4, but A is sparse)
        ap = [0.0] * 16
        for i in range(4):
            for j in range(4):
                val = 0.0
                # Row i of A times column j of P
                if i == 0:
                    val = a00 * p[0 * 4 + j] + a01 * p[1 * 4 + j]
                elif i == 1:
                    val = a10 * p[0 * 4 + j] + a11 * p[1 * 4 + j]
                elif i == 2:
                    val = a20 * p[0 * 4 + j] + a22 * p[2 * 4 + j]
                elif i == 3:
                    val = a31 * p[1 * 4 + j] + a33 * p[3 * 4 + j]
                ap[i * 4 + j] = val

        # (A*P) * A^T
        for i in range(4):
            for j in range(4):
                val = 0.0
                # Row i of AP times column j of A^T (= row j of A)
                if j == 0:
                    val = ap[i * 4 + 0] * a00 + ap[i * 4 + 1] * a10 + \
                          ap[i * 4 + 2] * a20
                elif j == 1:
                    val = ap[i * 4 + 0] * a01 + ap[i * 4 + 1] * a11 + \
                          ap[i * 4 + 3] * a31
                elif j == 2:
                    val = ap[i * 4 + 2] * a22
                elif j == 3:
                    val = ap[i * 4 + 3] * a33
                new_p[i * 4 + j] = val

        # Add Q (diagonal)
        for i in range(4):
            new_p[i * 4 + i] += self.q[i] * dt

        # Enforce symmetry
        for i in range(4):
            for j in range(i + 1, 4):
                avg = (new_p[i * 4 + j] + new_p[j * 4 + i]) * 0.5
                new_p[i * 4 + j] = avg
                new_p[j * 4 + i] = avg

        self.p = new_p

    def update(self, innovation_s1, innovation_s2):
        """Compute Kalman gains and return per-state corrections.

        Measurement matrix H (2x4):
            H = [[0, 0, 1, 0],   (y1 = T_s1)
                 [0, 0, 0, 1]]   (y2 = T_s2)

        Args:
            innovation_s1: (measured_s1 - state_s1)
            innovation_s2: (measured_s2 - state_s2)

        Returns:
            (corr_heater, corr_chamber, corr_s1, corr_s2) tuple
        """
        p = self.p

        # S = H * P * H^T + R (2x2)
        # H*P selects rows 2,3 of P columns, then H^T selects columns 2,3
        s00 = p[2 * 4 + 2] + self.r[0]
        s01 = p[2 * 4 + 3]
        s10 = p[3 * 4 + 2]
        s11 = p[3 * 4 + 3] + self.r[1]

        # Invert S (2x2)
        det = s00 * s11 - s01 * s10
        if abs(det) < 1e-20:
            det = 1e-20  # pragma: no cover
        inv_det = 1.0 / det
        si00 = s11 * inv_det
        si01 = -s01 * inv_det
        si10 = -s10 * inv_det
        si11 = s00 * inv_det

        # K = P * H^T * S^-1 (4x2)
        # P * H^T: column 0 = P[:,2], column 1 = P[:,3]
        k = [0.0] * 8
        for i in range(4):
            ph0 = p[i * 4 + 2]
            ph1 = p[i * 4 + 3]
            k[i * 2 + 0] = ph0 * si00 + ph1 * si10
            k[i * 2 + 1] = ph0 * si01 + ph1 * si11

        self.k = k

        # Update P: P = (I - K*H) * P
        # K*H is 4x4: row i = [0, 0, K[i,0], K[i,1]]
        new_p = list(p)
        for i in range(4):
            k0 = k[i * 2 + 0]
            k1 = k[i * 2 + 1]
            for j in range(4):
                new_p[i * 4 + j] -= k0 * p[2 * 4 + j] + k1 * p[3 * 4 + j]

        # Enforce symmetry
        for i in range(4):
            for j in range(i + 1, 4):
                avg = (new_p[i * 4 + j] + new_p[j * 4 + i]) * 0.5
                new_p[i * 4 + j] = avg
                new_p[j * 4 + i] = avg

        self.p = new_p

        # State corrections
        corr = [0.0] * 4
        for i in range(4):
            corr[i] = (k[i * 2 + 0] * innovation_s1 +
                       k[i * 2 + 1] * innovation_s2)

        return tuple(corr)

    def get_gains(self):
        """Return current Kalman gains as dict for diagnostics."""
        return {
            'k_heater_from_s1': self.k[0],
            'k_heater_from_s2': self.k[1],
            'k_chamber_from_s1': self.k[2],
            'k_chamber_from_s2': self.k[3],
            'k_s1_from_s1': self.k[4],
            'k_s1_from_s2': self.k[5],
            'k_s2_from_s1': self.k[6],
            'k_s2_from_s2': self.k[7],
        }
