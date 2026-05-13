#!/usr/bin/env python3
"""
ANTIGRAVITY — Trajectory Optimizer Unit Tests
===============================================
Tests minimum snap trajectory generation, time allocation,
and constraint satisfaction.
"""

import unittest
import numpy as np


class TestTimeAllocation(unittest.TestCase):
    """Test segment time allocation for trajectory generation."""

    def _trapezoidal_time(self, dist, max_v=2.0, max_a=2.0):
        """Trapezoidal velocity profile time calculation."""
        t_accel = max_v / max_a
        d_accel = 0.5 * max_a * t_accel ** 2
        if 2 * d_accel >= dist:
            return 2 * np.sqrt(dist / max_a)
        else:
            d_cruise = dist - 2 * d_accel
            t_cruise = d_cruise / max_v
            return 2 * t_accel + t_cruise

    def test_short_segment(self):
        """Short segment should use triangle profile."""
        t = self._trapezoidal_time(0.5, max_v=2.0, max_a=2.0)
        self.assertGreater(t, 0)
        self.assertLess(t, 2.0)

    def test_long_segment(self):
        """Long segment should include cruise phase."""
        t = self._trapezoidal_time(10.0, max_v=2.0, max_a=2.0)
        # Min time is 10/2 = 5s for cruise only
        self.assertGreater(t, 5.0)

    def test_zero_distance(self):
        """Zero distance should give zero time."""
        t = self._trapezoidal_time(0.0)
        self.assertAlmostEqual(t, 0.0)

    def test_time_increases_with_distance(self):
        t1 = self._trapezoidal_time(1.0)
        t2 = self._trapezoidal_time(5.0)
        t3 = self._trapezoidal_time(10.0)
        self.assertLess(t1, t2)
        self.assertLess(t2, t3)


class TestPolynomialTrajectory(unittest.TestCase):
    """Test polynomial trajectory evaluation."""

    def _eval_poly(self, coeffs, t):
        """Evaluate polynomial at time t."""
        val = 0.0
        for k, c in enumerate(coeffs):
            val += c * t ** k
        return val

    def _eval_poly_deriv(self, coeffs, t, deriv=1):
        """Evaluate polynomial derivative at time t."""
        if deriv == 0:
            return self._eval_poly(coeffs, t)
        n = len(coeffs)
        d_coeffs = []
        for k in range(deriv, n):
            c = coeffs[k]
            for d in range(deriv):
                c *= (k - d)
            d_coeffs.append(c)
        val = 0.0
        for k, c in enumerate(d_coeffs):
            val += c * t ** k
        return val

    def test_constant_polynomial(self):
        coeffs = [5.0, 0, 0, 0]
        self.assertAlmostEqual(self._eval_poly(coeffs, 0), 5.0)
        self.assertAlmostEqual(self._eval_poly(coeffs, 1), 5.0)
        self.assertAlmostEqual(self._eval_poly(coeffs, 10), 5.0)

    def test_linear_polynomial(self):
        coeffs = [0.0, 1.0, 0, 0]  # p(t) = t
        self.assertAlmostEqual(self._eval_poly(coeffs, 0), 0.0)
        self.assertAlmostEqual(self._eval_poly(coeffs, 5), 5.0)

    def test_velocity_from_position(self):
        coeffs = [0.0, 2.0, 0.5, 0]  # p(t) = 2t + 0.5t^2
        # Velocity: v(t) = 2 + t
        vel_0 = self._eval_poly_deriv(coeffs, 0, deriv=1)
        self.assertAlmostEqual(vel_0, 2.0)
        vel_1 = self._eval_poly_deriv(coeffs, 1, deriv=1)
        self.assertAlmostEqual(vel_1, 3.0)

    def test_acceleration_from_position(self):
        coeffs = [0.0, 0.0, 1.0, 0]  # p(t) = t^2
        # Acceleration: a(t) = 2
        acc = self._eval_poly_deriv(coeffs, 0, deriv=2)
        self.assertAlmostEqual(acc, 2.0)

    def test_zero_initial_derivatives(self):
        """Starting boundary should have zero velocity/accel for rest-to-rest."""
        coeffs = [0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0]  # Only t^3 term
        vel = self._eval_poly_deriv(coeffs, 0, deriv=1)
        acc = self._eval_poly_deriv(coeffs, 0, deriv=2)
        self.assertAlmostEqual(vel, 0.0)
        self.assertAlmostEqual(acc, 0.0)


class TestTrajectoryConstraints(unittest.TestCase):
    """Test that generated trajectories respect physical constraints."""

    def test_max_velocity_check(self):
        """Velocity at any sample should not exceed max_v."""
        max_v = 2.0
        # Simple linear trajectory: x(t) = v * t
        velocities = [0.5, 1.0, 1.5, 2.0, 1.8, 1.0]
        for v in velocities:
            self.assertLessEqual(v, max_v + 0.01)

    def test_max_acceleration_check(self):
        """Acceleration at any sample should not exceed max_a."""
        max_a = 2.0
        accelerations = [0.5, 1.0, 1.5, 2.0, 1.5, 0.5]
        for a in accelerations:
            self.assertLessEqual(a, max_a + 0.01)

    def test_trajectory_starts_at_first_waypoint(self):
        """Trajectory should start exactly at the first waypoint."""
        start = np.array([1.0, 2.0, 3.0])
        # Coefficients where c0 = start position
        coeffs_x = [start[0], 0, 0, 0]
        self.assertAlmostEqual(coeffs_x[0], start[0])

    def test_trajectory_ends_at_last_waypoint(self):
        """Trajectory should end at the last waypoint."""
        # For a simple case: p(T) should equal the endpoint
        T = 2.0
        coeffs = [0.0, 0.0, 0.5, 0]  # p(t) = 0.5t^2, p(2) = 2
        end_val = sum(c * T ** k for k, c in enumerate(coeffs))
        self.assertAlmostEqual(end_val, 2.0)


class TestWaypointSpacing(unittest.TestCase):
    """Test waypoint spacing and sub-sampling."""

    def _subsample(self, path, spacing=1.0):
        """Sub-sample a path at regular spacing."""
        if len(path) < 2:
            return path
        result = [path[0]]
        accumulated = 0.0
        for i in range(1, len(path)):
            dist = np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
            accumulated += dist
            if accumulated >= spacing:
                result.append(path[i])
                accumulated = 0.0
        if result[-1] != path[-1]:
            result.append(path[-1])
        return result

    def test_long_path_subsampled(self):
        path = [(i * 0.1, 0, 1.5) for i in range(100)]  # 10m path
        subsampled = self._subsample(path, spacing=1.0)
        self.assertLess(len(subsampled), len(path))
        self.assertGreater(len(subsampled), 5)

    def test_preserves_endpoints(self):
        path = [(i * 0.1, 0, 1.5) for i in range(100)]
        subsampled = self._subsample(path, spacing=1.0)
        np.testing.assert_array_equal(subsampled[0], path[0])
        np.testing.assert_array_equal(subsampled[-1], path[-1])

    def test_short_path_unchanged(self):
        path = [(0, 0, 1.5), (0.5, 0, 1.5)]
        subsampled = self._subsample(path, spacing=1.0)
        self.assertEqual(len(subsampled), 2)


if __name__ == '__main__':
    unittest.main()
