#!/usr/bin/env python3
"""
ANTIGRAVITY — Safety Arbiter Unit Tests
========================================
Tests safety state machine, escalation logic, and command filtering.
"""

import unittest
import numpy as np


class SafetyState:
    NOMINAL = 0
    WARNING = 1
    HOLD = 2
    EMERGENCY_LAND = 3
    RETURN_HOME = 4


class MockSafetyArbiter:
    def __init__(self):
        self.state = SafetyState.NOMINAL
        self.battery_pct = 100.0
        self.min_obstacle_distance = 10.0
        self.slam_tracking_valid = True
        self.geofence_violated = False
        self.battery_warning = 30.0
        self.battery_critical = 15.0
        self.collision_warning_dist = 2.0
        self.collision_stop_dist = 0.5

    def evaluate(self):
        new_state = SafetyState.NOMINAL
        if self.battery_pct < self.battery_critical:
            new_state = max(new_state, SafetyState.EMERGENCY_LAND)
        elif self.battery_pct < self.battery_warning:
            new_state = max(new_state, SafetyState.WARNING)
        if self.min_obstacle_distance < self.collision_stop_dist:
            new_state = max(new_state, SafetyState.HOLD)
        elif self.min_obstacle_distance < self.collision_warning_dist:
            new_state = max(new_state, SafetyState.WARNING)
        if not self.slam_tracking_valid:
            new_state = max(new_state, SafetyState.RETURN_HOME)
        if self.geofence_violated:
            new_state = max(new_state, SafetyState.HOLD)
        self.state = new_state
        return new_state

    def filter_velocity(self, vx, vy, vz, max_override=0.5):
        if self.state == SafetyState.HOLD:
            return 0.0, 0.0, 0.0
        elif self.state == SafetyState.EMERGENCY_LAND:
            return 0.0, 0.0, -0.3
        elif self.state == SafetyState.WARNING:
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            if speed > max_override:
                s = max_override / speed
                return vx * s, vy * s, vz * s
        return vx, vy, vz


class TestSafetyStateMachine(unittest.TestCase):
    def test_nominal(self):
        a = MockSafetyArbiter()
        self.assertEqual(a.evaluate(), SafetyState.NOMINAL)

    def test_battery_warning(self):
        a = MockSafetyArbiter()
        a.battery_pct = 25.0
        self.assertEqual(a.evaluate(), SafetyState.WARNING)

    def test_battery_critical(self):
        a = MockSafetyArbiter()
        a.battery_pct = 10.0
        self.assertEqual(a.evaluate(), SafetyState.EMERGENCY_LAND)

    def test_obstacle_hold(self):
        a = MockSafetyArbiter()
        a.min_obstacle_distance = 0.3
        self.assertEqual(a.evaluate(), SafetyState.HOLD)

    def test_slam_lost(self):
        a = MockSafetyArbiter()
        a.slam_tracking_valid = False
        self.assertEqual(a.evaluate(), SafetyState.RETURN_HOME)

    def test_geofence_hold(self):
        a = MockSafetyArbiter()
        a.geofence_violated = True
        self.assertEqual(a.evaluate(), SafetyState.HOLD)

    def test_highest_severity_wins(self):
        a = MockSafetyArbiter()
        a.slam_tracking_valid = False
        a.geofence_violated = True
        self.assertEqual(a.evaluate(), SafetyState.RETURN_HOME)


class TestCommandFiltering(unittest.TestCase):
    def test_nominal_passthrough(self):
        a = MockSafetyArbiter()
        a.evaluate()
        self.assertEqual(a.filter_velocity(1, 0.5, 0.2), (1, 0.5, 0.2))

    def test_hold_zeroes(self):
        a = MockSafetyArbiter()
        a.min_obstacle_distance = 0.3
        a.evaluate()
        self.assertEqual(a.filter_velocity(2, 1, 0.5), (0.0, 0.0, 0.0))

    def test_emergency_descends(self):
        a = MockSafetyArbiter()
        a.battery_pct = 5.0
        a.evaluate()
        _, _, vz = a.filter_velocity(2, 1, 0.5)
        self.assertLess(vz, 0.0)

    def test_warning_limits_speed(self):
        a = MockSafetyArbiter()
        a.battery_pct = 25.0
        a.evaluate()
        vx, vy, vz = a.filter_velocity(3, 4, 0)
        self.assertLessEqual(np.sqrt(vx**2 + vy**2 + vz**2), 0.51)


class TestGeofence(unittest.TestCase):
    def _inside(self, x, y, z, r=50, max_alt=30, min_alt=0.3):
        return np.sqrt(x**2 + y**2) <= r and min_alt <= z <= max_alt

    def test_inside(self):
        self.assertTrue(self._inside(0, 0, 1))

    def test_outside_radius(self):
        self.assertFalse(self._inside(60, 0, 1))

    def test_above_ceiling(self):
        self.assertFalse(self._inside(0, 0, 35))

    def test_below_floor(self):
        self.assertFalse(self._inside(0, 0, 0.1))


if __name__ == '__main__':
    unittest.main()
