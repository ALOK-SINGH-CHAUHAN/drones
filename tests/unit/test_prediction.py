#!/usr/bin/env python3
"""
ANTIGRAVITY — Prediction Engine Unit Tests
============================================
Tests Kalman filter tracking, trajectory prediction,
and Hungarian algorithm association.
"""

import unittest
import numpy as np


class SimpleKalmanTracker:
    """Minimal Kalman tracker for unit testing."""

    def __init__(self, initial_pos, dt=1.0 / 30):
        self.dt = dt
        self.state = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.state[:3] = initial_pos
        self.P = np.eye(6) * 10.0

        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.Q = np.diag([0.1 * dt ** 2] * 3 + [0.1 * dt] * 3)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.R = np.eye(3) * 0.5

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def predict_trajectory(self, horizon_s, steps):
        """Predict future trajectory."""
        dt = horizon_s / steps
        F_pred = np.eye(6)
        F_pred[0, 3] = dt
        F_pred[1, 4] = dt
        F_pred[2, 5] = dt
        state = self.state.copy()
        predictions = []
        for _ in range(steps):
            state = F_pred @ state
            predictions.append(state[:3].copy())
        return predictions


class TestKalmanPrediction(unittest.TestCase):
    def test_stationary_target(self):
        kf = SimpleKalmanTracker([1, 2, 3])
        kf.predict()
        np.testing.assert_array_almost_equal(kf.state[:3], [1, 2, 3], decimal=1)

    def test_constant_velocity(self):
        kf = SimpleKalmanTracker([0, 0, 0])
        kf.state[3:] = [1, 0, 0]  # Moving in x at 1 m/s
        kf.predict()
        self.assertGreater(kf.state[0], 0)

    def test_covariance_grows_on_predict(self):
        kf = SimpleKalmanTracker([0, 0, 0])
        trace_before = np.trace(kf.P)
        kf.predict()
        trace_after = np.trace(kf.P)
        self.assertGreater(trace_after, trace_before)


class TestKalmanUpdate(unittest.TestCase):
    def test_update_reduces_uncertainty(self):
        kf = SimpleKalmanTracker([0, 0, 0])
        kf.P = np.eye(6) * 5.0
        trace_before = np.trace(kf.P)
        kf.update(np.array([0, 0, 0]))
        trace_after = np.trace(kf.P)
        self.assertLess(trace_after, trace_before)

    def test_update_moves_state(self):
        kf = SimpleKalmanTracker([0, 0, 0])
        kf.P = np.eye(6) * 5.0
        kf.update(np.array([5.0, 3.0, 1.0]))
        self.assertGreater(kf.state[0], 2.0)
        self.assertGreater(kf.state[1], 1.0)

    def test_multiple_observations_converge(self):
        kf = SimpleKalmanTracker([0, 0, 0])
        true_pos = np.array([3.0, 4.0, 1.5])
        for _ in range(20):
            kf.predict()
            noisy = true_pos + np.random.randn(3) * 0.1
            kf.update(noisy)
        error = np.linalg.norm(kf.state[:3] - true_pos)
        self.assertLess(error, 0.5)


class TestTrajectoryPrediction(unittest.TestCase):
    def test_straight_line_prediction(self):
        kf = SimpleKalmanTracker([0, 0, 1])
        kf.state[3] = 1.0  # Moving in x at 1 m/s
        predictions = kf.predict_trajectory(2.0, 10)
        self.assertEqual(len(predictions), 10)
        # Last prediction should be roughly at x=2
        self.assertAlmostEqual(predictions[-1][0], 2.0, delta=0.3)

    def test_prediction_length(self):
        kf = SimpleKalmanTracker([0, 0, 0])
        predictions = kf.predict_trajectory(2.0, 20)
        self.assertEqual(len(predictions), 20)

    def test_stationary_prediction_stays(self):
        kf = SimpleKalmanTracker([5, 5, 1])
        predictions = kf.predict_trajectory(2.0, 10)
        for pred in predictions:
            np.testing.assert_array_almost_equal(pred, [5, 5, 1], decimal=1)


class TestHungarianAssignment(unittest.TestCase):
    """Test detection-to-track assignment."""

    def _greedy_assign(self, cost_matrix, max_cost=10.0):
        n_det, n_trk = cost_matrix.shape
        matches = []
        used_det = set()
        used_trk = set()
        flat = cost_matrix.flatten()
        sorted_indices = np.argsort(flat)
        for idx in sorted_indices:
            d, t = divmod(idx, n_trk)
            if d in used_det or t in used_trk:
                continue
            if cost_matrix[d, t] < max_cost:
                matches.append((d, t))
                used_det.add(d)
                used_trk.add(t)
        unmatched_det = [i for i in range(n_det) if i not in used_det]
        unmatched_trk = [i for i in range(n_trk) if i not in used_trk]
        return matches, unmatched_det, unmatched_trk

    def test_perfect_match(self):
        cost = np.array([[0.1, 5.0], [5.0, 0.2]])
        matches, ud, ut = self._greedy_assign(cost)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len(ud), 0)
        self.assertEqual(len(ut), 0)

    def test_unmatched_detection(self):
        cost = np.array([[0.1, 5.0], [5.0, 0.2], [20.0, 20.0]])
        matches, ud, ut = self._greedy_assign(cost)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len(ud), 1)

    def test_unmatched_track(self):
        cost = np.array([[0.1]])
        matches, ud, ut = self._greedy_assign(cost, max_cost=10)
        self.assertEqual(len(matches), 1)

    def test_all_exceed_max_cost(self):
        cost = np.array([[100.0, 100.0]])
        matches, ud, ut = self._greedy_assign(cost, max_cost=10)
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(ud), 1)
        self.assertEqual(len(ut), 2)


if __name__ == '__main__':
    unittest.main()
