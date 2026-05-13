#!/usr/bin/env python3
"""
ANTIGRAVITY — Monte Carlo Localization Unit Tests
===================================================
Tests particle filter initialization, motion model, sensor model,
resampling, and convergence.
"""

import unittest
import numpy as np
from dataclasses import dataclass


@dataclass
class Particle:
    x: float
    y: float
    yaw: float
    weight: float


class MCLCore:
    """Minimal MCL implementation for unit testing."""

    def __init__(self, n_particles=200, x0=0.0, y0=0.0, yaw0=0.0,
                 spread=1.0, alpha=None):
        self.alpha = alpha or [0.2, 0.2, 0.2, 0.2]
        self.particles = []
        for _ in range(n_particles):
            self.particles.append(Particle(
                x=x0 + np.random.normal(0, spread),
                y=y0 + np.random.normal(0, spread),
                yaw=yaw0 + np.random.normal(0, 0.5),
                weight=1.0 / n_particles,
            ))

    def motion_update(self, dx, dy, dyaw):
        """Apply noisy odometry motion model."""
        d_trans = np.sqrt(dx ** 2 + dy ** 2)
        if d_trans < 0.001:
            return
        d_rot1 = np.arctan2(dy, dx)
        d_rot2 = dyaw - d_rot1
        a = self.alpha
        for p in self.particles:
            hat_rot1 = d_rot1 + np.random.normal(0, a[0] * abs(d_rot1) + a[1] * d_trans)
            hat_trans = d_trans + np.random.normal(0, a[2] * d_trans + a[3] * (abs(d_rot1) + abs(d_rot2)))
            hat_rot2 = d_rot2 + np.random.normal(0, a[0] * abs(d_rot2) + a[1] * d_trans)
            p.x += hat_trans * np.cos(p.yaw + hat_rot1)
            p.y += hat_trans * np.sin(p.yaw + hat_rot1)
            p.yaw += hat_rot1 + hat_rot2

    def sensor_update(self, true_x, true_y, sigma=0.5):
        """Gaussian observation model (simplified)."""
        for p in self.particles:
            dx = p.x - true_x
            dy = p.y - true_y
            dist = np.sqrt(dx ** 2 + dy ** 2)
            p.weight *= np.exp(-0.5 * (dist / sigma) ** 2)
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total
        else:
            uniform = 1.0 / len(self.particles)
            for p in self.particles:
                p.weight = uniform

    def neff(self):
        """Effective particle count."""
        wsq = sum(p.weight ** 2 for p in self.particles)
        return 1.0 / wsq if wsq > 0 else 0

    def systematic_resample(self):
        n = len(self.particles)
        weights = np.array([p.weight for p in self.particles])
        positions = (np.arange(n) + np.random.uniform()) / n
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)
        new_particles = []
        for i in indices:
            old = self.particles[i]
            new_particles.append(Particle(
                x=old.x, y=old.y, yaw=old.yaw,
                weight=1.0 / n,
            ))
        self.particles = new_particles

    def estimate(self):
        """Weighted mean estimate."""
        tw = sum(p.weight for p in self.particles)
        if tw == 0:
            return 0, 0, 0
        mx = sum(p.x * p.weight for p in self.particles) / tw
        my = sum(p.y * p.weight for p in self.particles) / tw
        sin_sum = sum(np.sin(p.yaw) * p.weight for p in self.particles) / tw
        cos_sum = sum(np.cos(p.yaw) * p.weight for p in self.particles) / tw
        myaw = np.arctan2(sin_sum, cos_sum)
        return mx, my, myaw

    def spread(self):
        """Standard deviation of particles."""
        tw = sum(p.weight for p in self.particles)
        mx, my, _ = self.estimate()
        vx = sum(p.weight * (p.x - mx) ** 2 for p in self.particles) / tw
        vy = sum(p.weight * (p.y - my) ** 2 for p in self.particles) / tw
        return np.sqrt(vx + vy)


class TestMCLInitialization(unittest.TestCase):
    def test_weights_sum_to_one(self):
        mcl = MCLCore(n_particles=500)
        total = sum(p.weight for p in mcl.particles)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_particle_count(self):
        mcl = MCLCore(n_particles=300)
        self.assertEqual(len(mcl.particles), 300)

    def test_initial_spread(self):
        mcl = MCLCore(n_particles=1000, spread=2.0)
        self.assertGreater(mcl.spread(), 1.0)

    def test_initial_centered(self):
        mcl = MCLCore(n_particles=2000, x0=5.0, y0=3.0, spread=0.5)
        mx, my, _ = mcl.estimate()
        self.assertAlmostEqual(mx, 5.0, delta=0.3)
        self.assertAlmostEqual(my, 3.0, delta=0.3)


class TestMCLMotionModel(unittest.TestCase):
    def test_forward_motion(self):
        mcl = MCLCore(n_particles=500, x0=0, y0=0, yaw0=0, spread=0.1)
        mcl.motion_update(1.0, 0.0, 0.0)
        mx, my, _ = mcl.estimate()
        self.assertGreater(mx, 0.5, "Particles should move forward")

    def test_no_motion_preserves_position(self):
        mcl = MCLCore(n_particles=500, x0=1.0, y0=2.0, spread=0.1)
        mx0, my0, _ = mcl.estimate()
        mcl.motion_update(0.0, 0.0, 0.0)
        mx1, my1, _ = mcl.estimate()
        self.assertAlmostEqual(mx0, mx1, delta=0.01)
        self.assertAlmostEqual(my0, my1, delta=0.01)


class TestMCLSensorModel(unittest.TestCase):
    def test_sensor_update_concentrates_weights(self):
        mcl = MCLCore(n_particles=500, x0=0, y0=0, spread=2.0)
        neff_before = mcl.neff()
        mcl.sensor_update(0, 0, sigma=0.5)
        neff_after = mcl.neff()
        self.assertLess(neff_after, neff_before)

    def test_weights_still_sum_to_one(self):
        mcl = MCLCore(n_particles=500)
        mcl.sensor_update(0, 0, sigma=1.0)
        total = sum(p.weight for p in mcl.particles)
        self.assertAlmostEqual(total, 1.0, places=5)


class TestMCLResampling(unittest.TestCase):
    def test_resample_preserves_count(self):
        mcl = MCLCore(n_particles=500)
        mcl.sensor_update(0, 0, sigma=0.5)
        mcl.systematic_resample()
        self.assertEqual(len(mcl.particles), 500)

    def test_resample_equalizes_weights(self):
        mcl = MCLCore(n_particles=500)
        mcl.sensor_update(0, 0, sigma=0.5)
        mcl.systematic_resample()
        weights = [p.weight for p in mcl.particles]
        self.assertAlmostEqual(max(weights), min(weights), places=5)

    def test_resample_restores_neff(self):
        mcl = MCLCore(n_particles=500)
        mcl.sensor_update(0, 0, sigma=0.3)
        mcl.systematic_resample()
        self.assertAlmostEqual(mcl.neff(), 500, delta=5)


class TestMCLConvergence(unittest.TestCase):
    def test_converges_with_observations(self):
        """MCL should converge to true position after repeated observations."""
        mcl = MCLCore(n_particles=500, x0=0, y0=0, spread=3.0)
        true_x, true_y = 2.0, 1.0
        for _ in range(20):
            mcl.sensor_update(true_x, true_y, sigma=0.5)
            if mcl.neff() < 250:
                mcl.systematic_resample()
        mx, my, _ = mcl.estimate()
        error = np.sqrt((mx - true_x) ** 2 + (my - true_y) ** 2)
        self.assertLess(error, 0.5, f"Should converge within 0.5m, got {error:.3f}m")

    def test_spread_decreases_with_observations(self):
        mcl = MCLCore(n_particles=500, x0=0, y0=0, spread=3.0)
        initial_spread = mcl.spread()
        for _ in range(10):
            mcl.sensor_update(0, 0, sigma=0.3)
            if mcl.neff() < 250:
                mcl.systematic_resample()
        final_spread = mcl.spread()
        self.assertLess(final_spread, initial_spread)


if __name__ == '__main__':
    unittest.main()
