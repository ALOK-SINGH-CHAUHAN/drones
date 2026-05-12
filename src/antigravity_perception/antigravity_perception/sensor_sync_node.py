"""
ANTIGRAVITY — Sensor Synchronization Node
==========================================
Synchronizes camera and IMU data streams using message_filters.
Ensures timestamp alignment between visual and inertial data.

Acceptance Criteria:
  - Camera-IMU timestamp sync error < 2ms
  - Publishes synchronized data bundles
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu, CameraInfo

import message_filters
import numpy as np
import time


class SensorSyncNode(Node):
    """
    Sensor synchronization node using approximate time synchronizer.
    
    Subscribes to:
      - camera/image_raw, camera/depth, imu/data
    Publishes:
      - synced/image, synced/depth, synced/imu (time-aligned)
    """

    def __init__(self):
        super().__init__('sensor_synchronizer')

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('sync_tolerance_ms', 2.0)
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('approximate_sync', True)
        self.declare_parameter('camera_topic', 'camera/image_raw')
        self.declare_parameter('depth_topic', 'camera/depth')
        self.declare_parameter('imu_topic', 'imu/data')

        tolerance_ms = self.get_parameter('sync_tolerance_ms').value
        queue_size = self.get_parameter('queue_size').value

        # ─── QoS ────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ─── Publishers ─────────────────────────────────────────────────
        self._pub_image = self.create_publisher(Image, 'synced/image', sensor_qos)
        self._pub_depth = self.create_publisher(Image, 'synced/depth', sensor_qos)
        self._pub_imu = self.create_publisher(Imu, 'synced/imu', sensor_qos)

        # ─── Subscribers with message_filters ────────────────────────────
        self._sub_image = message_filters.Subscriber(
            self, Image, self.get_parameter('camera_topic').value, qos_profile=sensor_qos
        )
        self._sub_depth = message_filters.Subscriber(
            self, Image, self.get_parameter('depth_topic').value, qos_profile=sensor_qos
        )
        self._sub_imu = message_filters.Subscriber(
            self, Imu, self.get_parameter('imu_topic').value, qos_profile=sensor_qos
        )

        # ─── Time Synchronizer ──────────────────────────────────────────
        if self.get_parameter('approximate_sync').value:
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._sub_image, self._sub_depth, self._sub_imu],
                queue_size=queue_size,
                slop=tolerance_ms / 1000.0,  # Convert ms to seconds
            )
        else:
            self._sync = message_filters.TimeSynchronizer(
                [self._sub_image, self._sub_depth, self._sub_imu],
                queue_size=queue_size,
            )

        self._sync.registerCallback(self._sync_callback)

        # ─── Statistics ──────────────────────────────────────────────────
        self._sync_count = 0
        self._dropped_count = 0
        self._max_sync_error_ms = 0.0
        self._start_time = time.time()

        self._diag_timer = self.create_timer(5.0, self._diagnostics_callback)

        self.get_logger().info(
            f'Sensor sync initialized — tolerance: {tolerance_ms}ms, '
            f'queue: {queue_size}'
        )

    def _sync_callback(self, image_msg, depth_msg, imu_msg):
        """
        Callback for synchronized sensor data.
        Validates timestamp alignment and republishes.
        """
        # Calculate sync error
        img_time = self._stamp_to_seconds(image_msg.header.stamp)
        imu_time = self._stamp_to_seconds(imu_msg.header.stamp)
        sync_error_ms = abs(img_time - imu_time) * 1000.0

        self._max_sync_error_ms = max(self._max_sync_error_ms, sync_error_ms)

        # Check tolerance
        tolerance = self.get_parameter('sync_tolerance_ms').value
        if sync_error_ms > tolerance:
            self._dropped_count += 1
            self.get_logger().warn(
                f'Sync error {sync_error_ms:.2f}ms exceeds tolerance {tolerance}ms — dropping'
            )
            return

        # Use unified timestamp
        unified_stamp = self.get_clock().now().to_msg()

        # Republish with unified timestamp
        image_msg.header.stamp = unified_stamp
        depth_msg.header.stamp = unified_stamp
        imu_msg.header.stamp = unified_stamp

        self._pub_image.publish(image_msg)
        self._pub_depth.publish(depth_msg)
        self._pub_imu.publish(imu_msg)

        self._sync_count += 1

    def _stamp_to_seconds(self, stamp):
        """Convert ROS2 Time message to seconds."""
        return stamp.sec + stamp.nanosec * 1e-9

    def _diagnostics_callback(self):
        """Log synchronization statistics."""
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            sync_rate = self._sync_count / elapsed
            self.get_logger().info(
                f'Sensor sync — Rate: {sync_rate:.1f} Hz | '
                f'Synced: {self._sync_count} | '
                f'Dropped: {self._dropped_count} | '
                f'Max error: {self._max_sync_error_ms:.2f}ms'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SensorSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
