#!/usr/bin/env python3
"""
ANTIGRAVITY — Sensor Diagnostics Unit Tests
=============================================
Tests sensor health monitoring, rate tracking, and health scoring.
"""

import unittest
import time


class SensorHealthTracker:
    """Minimal sensor health tracker for unit testing."""

    def __init__(self, timeout_s=3.0):
        self.timeout = timeout_s
        self.sensors = {}

    def register(self, name, min_hz):
        self.sensors[name] = {
            'last': 0, 'count': 0,
            'window_start': time.time(),
            'min_hz': min_hz,
        }

    def tick(self, name):
        if name in self.sensors:
            self.sensors[name]['last'] = time.time()
            self.sensors[name]['count'] += 1

    def evaluate(self):
        now = time.time()
        report = {}
        healthy = 0
        for name, info in self.sensors.items():
            elapsed = now - info['window_start']
            hz = info['count'] / max(elapsed, 0.01)
            alive = (now - info['last']) < self.timeout if info['last'] > 0 else False
            is_healthy = alive and hz >= info['min_hz'] * 0.5
            report[name] = {'alive': alive, 'hz': hz, 'healthy': is_healthy}
            if is_healthy:
                healthy += 1
        total = len(self.sensors)
        score = healthy / total if total > 0 else 0.0
        return report, score


class TestSensorRegistration(unittest.TestCase):
    def test_register_sensor(self):
        tracker = SensorHealthTracker()
        tracker.register('camera', 30.0)
        self.assertIn('camera', tracker.sensors)
        self.assertEqual(tracker.sensors['camera']['min_hz'], 30.0)

    def test_multiple_sensors(self):
        tracker = SensorHealthTracker()
        tracker.register('camera', 30.0)
        tracker.register('imu', 200.0)
        tracker.register('slam', 20.0)
        self.assertEqual(len(tracker.sensors), 3)


class TestSensorTicking(unittest.TestCase):
    def test_tick_increments_count(self):
        tracker = SensorHealthTracker()
        tracker.register('cam', 30.0)
        tracker.tick('cam')
        tracker.tick('cam')
        tracker.tick('cam')
        self.assertEqual(tracker.sensors['cam']['count'], 3)

    def test_tick_updates_last_time(self):
        tracker = SensorHealthTracker()
        tracker.register('imu', 200.0)
        self.assertEqual(tracker.sensors['imu']['last'], 0)
        tracker.tick('imu')
        self.assertGreater(tracker.sensors['imu']['last'], 0)

    def test_tick_unknown_sensor(self):
        tracker = SensorHealthTracker()
        tracker.tick('nonexistent')  # Should not crash


class TestSensorEvaluation(unittest.TestCase):
    def test_no_ticks_unhealthy(self):
        tracker = SensorHealthTracker(timeout_s=1.0)
        tracker.register('cam', 30.0)
        report, score = tracker.evaluate()
        self.assertFalse(report['cam']['healthy'])
        self.assertEqual(score, 0.0)

    def test_active_sensor_healthy(self):
        tracker = SensorHealthTracker(timeout_s=5.0)
        tracker.register('cam', 1.0)  # Low min Hz for test
        for _ in range(20):
            tracker.tick('cam')
        report, score = tracker.evaluate()
        self.assertTrue(report['cam']['alive'])
        self.assertEqual(score, 1.0)

    def test_mixed_health_score(self):
        tracker = SensorHealthTracker(timeout_s=5.0)
        tracker.register('cam', 1.0)
        tracker.register('imu', 1.0)
        # Only tick camera
        for _ in range(10):
            tracker.tick('cam')
        report, score = tracker.evaluate()
        self.assertTrue(report['cam']['healthy'])
        self.assertFalse(report['imu']['healthy'])
        self.assertAlmostEqual(score, 0.5)

    def test_empty_tracker_score(self):
        tracker = SensorHealthTracker()
        report, score = tracker.evaluate()
        self.assertEqual(score, 0.0)
        self.assertEqual(len(report), 0)


if __name__ == '__main__':
    unittest.main()
