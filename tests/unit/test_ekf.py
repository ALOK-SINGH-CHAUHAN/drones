#!/usr/bin/env python3
"""
ANTIGRAVITY — EKF State Estimator Unit Tests
==============================================
Tests Extended Kalman Filter prediction, measurement update,
outlier rejection, and covariance management.
"""

import unittest
import numpy as np


class SimpleEKF:
    """Minimal 6-state EKF for unit testing (position + velocity)."""

    def __init__(self):
        self.x = np.zeros(6)  # [px, py, pz, vx, vy, vz]
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6) * 0.01  # Process noise
        self.outliers_rejected = 0

    def predict(self, dt):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_position(self, z_pos, R_diag, max_innovation=5.0):
        H = np.zeros((3, 6))
        H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
        R = np.diag(R_diag)
        y = z_pos - H @ self.x
        S = H @ self.P @ H.T + R
        mahal = float(y.T @ np.linalg.inv(S) @ y)
        if mahal > max_innovation ** 2:
            self.outliers_rejected += 1
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        return True


class TestEKFPrediction(unittest.TestCase):
    def test_stationary_prediction(self):
        ekf = SimpleEKF()
        ekf.x[:3] = [1, 2, 3]
        ekf.predict(0.01)
        np.testing.assert_array_almost_equal(ekf.x[:3], [1, 2, 3], decimal=2)

    def test_constant_velocity_prediction(self):
        ekf = SimpleEKF()
        ekf.x = np.array([0, 0, 0, 1.0, 0.5, -0.2])
        ekf.predict(1.0)
        np.testing.assert_array_almost_equal(ekf.x[:3], [1.0, 0.5, -0.2], decimal=2)

    def test_covariance_grows(self):
        ekf = SimpleEKF()
        P_before = np.trace(ekf.P)
        ekf.predict(0.1)
        P_after = np.trace(ekf.P)
        self.assertGreater(P_after, P_before)


class TestEKFUpdate(unittest.TestCase):
    def test_measurement_reduces_uncertainty(self):
        ekf = SimpleEKF()
        ekf.P = np.eye(6) * 1.0
        P_before = np.trace(ekf.P)
        ekf.update_position(np.array([0, 0, 0]), [0.05, 0.05, 0.05])
        P_after = np.trace(ekf.P)
        self.assertLess(P_after, P_before)

    def test_measurement_moves_state(self):
        ekf = SimpleEKF()
        ekf.x[:3] = [0, 0, 0]
        ekf.P = np.eye(6) * 1.0
        ekf.update_position(np.array([1.0, 2.0, 3.0]), [0.01, 0.01, 0.01])
        self.assertAlmostEqual(ekf.x[0], 1.0, places=1)
        self.assertAlmostEqual(ekf.x[1], 2.0, places=1)

    def test_outlier_rejection(self):
        ekf = SimpleEKF()
        ekf.x[:3] = [0, 0, 0]
        ekf.P = np.eye(6) * 0.01
        result = ekf.update_position(np.array([100, 100, 100]), [0.01, 0.01, 0.01], max_innovation=3.0)
        self.assertFalse(result, "Should reject outlier measurement")
        self.assertEqual(ekf.outliers_rejected, 1)
        np.testing.assert_array_almost_equal(ekf.x[:3], [0, 0, 0], decimal=1)

    def test_joseph_form_positive_definite(self):
        ekf = SimpleEKF()
        for _ in range(100):
            ekf.predict(0.01)
            z = ekf.x[:3] + np.random.randn(3) * 0.1
            ekf.update_position(z, [0.1, 0.1, 0.1])
        eigenvalues = np.linalg.eigvalsh(ekf.P)
        self.assertTrue(np.all(eigenvalues > 0), "Covariance must stay positive definite")


class TestTrajectoryOptimizer(unittest.TestCase):
    """Test minimum snap trajectory generation."""

    def _time_allocation(self, waypoints, cruise_speed=1.5):
        times = [0.0]
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(np.array(waypoints[i]) - np.array(waypoints[i - 1]))
            t = dist / cruise_speed
            times.append(times[-1] + max(t, 0.1))
        return times

    def test_time_allocation_monotonic(self):
        wps = [(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 1, 1)]
        times = self._time_allocation(wps)
        for i in range(1, len(times)):
            self.assertGreater(times[i], times[i - 1])

    def test_single_segment(self):
        wps = [(0, 0, 0), (1, 0, 0)]
        times = self._time_allocation(wps)
        self.assertEqual(len(times), 2)
        self.assertAlmostEqual(times[0], 0.0)
        self.assertGreater(times[1], 0.0)


if __name__ == '__main__':
    unittest.main()
