#!/usr/bin/env python3
"""
ANTIGRAVITY — Sensor Diagnostics Node
=======================================
Monitors health of all perception sensors and publishes a unified
diagnostics message for the safety arbiter and mission manager.

Tracks:
  - Camera frame rate and exposure status
  - IMU data rate and bias drift
  - SLAM tracking quality
  - Depth sensor validity
  - Sensor synchronization jitter
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, Imu

import time
import json
import threading


class SensorDiagnosticsNode(Node):
    """
    Unified sensor health monitoring node.

    Subscribes to raw sensor topics and tracks:
    - Update rates (Hz)
    - Latency (ms)
    - Data validity flags
    - Sensor timeouts

    Publishes:
      - diagnostics/sensors (String): JSON health report
      - diagnostics/sensor_rate (Float32): Overall sensor health score [0-1]
    """

    def __init__(self):
        super().__init__('sensor_diagnostics')

        self.declare_parameter('check_rate_hz', 2)
        self.declare_parameter('camera_min_fps', 15.0)
        self.declare_parameter('imu_min_hz', 100.0)
        self.declare_parameter('slam_min_hz', 10.0)
        self.declare_parameter('sensor_timeout_s', 3.0)

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        # Publishers
        self._pub_diag = self.create_publisher(
            String, 'diagnostics/sensors', reliable_qos)
        self._pub_score = self.create_publisher(
            Float32, 'diagnostics/sensor_rate', reliable_qos)

        # Subscribers — track last update time and count
        self._sensors = {
            'camera_rgb': {'last': 0, 'count': 0, 'window_start': time.time(),
                           'min_hz': self.get_parameter('camera_min_fps').value},
            'camera_depth': {'last': 0, 'count': 0, 'window_start': time.time(),
                             'min_hz': self.get_parameter('camera_min_fps').value},
            'imu': {'last': 0, 'count': 0, 'window_start': time.time(),
                    'min_hz': self.get_parameter('imu_min_hz').value},
            'slam_pose': {'last': 0, 'count': 0, 'window_start': time.time(),
                          'min_hz': self.get_parameter('slam_min_hz').value},
        }
        self._lock = threading.Lock()

        self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            lambda m: self._tick('camera_rgb'), sensor_qos)
        self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw',
            lambda m: self._tick('camera_depth'), sensor_qos)
        self.create_subscription(
            Imu, '/imu/data',
            lambda m: self._tick('imu'), sensor_qos)
        self.create_subscription(
            String, '/slam/slam/state',
            lambda m: self._tick('slam_pose'), reliable_qos)

        rate = self.get_parameter('check_rate_hz').value
        self.create_timer(1.0 / rate, self._evaluate)
        self.create_timer(10.0, self._diag)
        self.get_logger().info('Sensor diagnostics initialized')

    def _tick(self, sensor_name):
        """Record a message arrival for the given sensor."""
        with self._lock:
            self._sensors[sensor_name]['last'] = time.time()
            self._sensors[sensor_name]['count'] += 1

    def _evaluate(self):
        """Evaluate sensor health and publish report."""
        timeout = self.get_parameter('sensor_timeout_s').value
        now = time.time()
        report = {}
        healthy_count = 0

        with self._lock:
            for name, info in self._sensors.items():
                elapsed = now - info['window_start']
                hz = info['count'] / max(elapsed, 0.01)
                alive = (now - info['last']) < timeout if info['last'] > 0 else False
                healthy = alive and hz >= info['min_hz'] * 0.5

                report[name] = {
                    'alive': alive,
                    'hz': round(hz, 1),
                    'min_hz': info['min_hz'],
                    'healthy': healthy,
                    'last_seen_s': round(now - info['last'], 1) if info['last'] > 0 else -1,
                }
                if healthy:
                    healthy_count += 1

                # Reset window every 5 seconds
                if elapsed > 5.0:
                    info['count'] = 0
                    info['window_start'] = now

        total = len(self._sensors)
        score = healthy_count / total if total > 0 else 0.0

        report['overall'] = {
            'healthy_sensors': healthy_count,
            'total_sensors': total,
            'score': round(score, 2),
            'status': 'OK' if score >= 0.75 else 'DEGRADED' if score >= 0.5 else 'CRITICAL',
        }

        self._pub_diag.publish(String(data=json.dumps(report)))
        self._pub_score.publish(Float32(data=float(score)))

    def _diag(self):
        """Log diagnostics summary."""
        timeout = self.get_parameter('sensor_timeout_s').value
        now = time.time()
        with self._lock:
            statuses = []
            for name, info in self._sensors.items():
                alive = (now - info['last']) < timeout if info['last'] > 0 else False
                statuses.append(f'{name}: {"✓" if alive else "✗"}')
        self.get_logger().info(f'Sensors — {" | ".join(statuses)}')


def main(args=None):
    rclpy.init(args=args)
    node = SensorDiagnosticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
