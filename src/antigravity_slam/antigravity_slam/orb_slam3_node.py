"""
ANTIGRAVITY — ORB-SLAM3 Integration Node
=========================================
ROS2 wrapper for ORB-SLAM3 visual-inertial SLAM system.
Provides 6-DOF pose estimation, map point cloud, and relocalization.

Acceptance Criteria:
  - Pose updates at >= 20 Hz
  - Drift < 1% of distance over 100m trajectory
  - Relocalization from tracking loss within 2 seconds
  - Operates on Jetson Orin NX with <= 50ms per pose update
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster

import numpy as np
import time
import threading
from enum import IntEnum


class TrackingState(IntEnum):
    """ORB-SLAM3 tracking states."""
    SYSTEM_NOT_READY = -1
    NO_IMAGES_YET = 0
    NOT_INITIALIZED = 1
    OK = 2
    RECENTLY_LOST = 3
    LOST = 4
    OK_KLT = 5


class OrbSlam3Node(Node):
    """
    ORB-SLAM3 ROS2 integration node.
    
    Wraps the ORB-SLAM3 C++ library via Python bindings.
    Falls back to simulated SLAM for development without the actual library.
    
    Subscribes:
      - camera/image_raw (sensor_msgs/Image)
      - camera/depth (sensor_msgs/Image) for RGBD mode
      - imu/data (sensor_msgs/Imu) for inertial modes
    
    Publishes:
      - slam/pose (geometry_msgs/PoseStamped): 6-DOF camera pose
      - slam/map_points (sensor_msgs/PointCloud2): Sparse map points
      - slam/state (std_msgs/String): Tracking state
    
    Broadcasts:
      - TF: map -> camera_link transform
    """

    def __init__(self):
        super().__init__('orb_slam3')

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('slam_mode', 'stereo_inertial')
        self.declare_parameter('vocabulary_path', 'models/ORBvoc.txt')
        self.declare_parameter('settings_path', 'config/slam_camera.yaml')
        self.declare_parameter('publish_rate_hz', 30)
        self.declare_parameter('num_features', 1200)
        self.declare_parameter('scale_factor', 1.2)
        self.declare_parameter('num_levels', 8)
        self.declare_parameter('enable_map_reuse', True)
        self.declare_parameter('map_file', '')
        self.declare_parameter('save_map_on_exit', True)
        self.declare_parameter('relocalization_enabled', True)
        self.declare_parameter('relocalization_timeout_s', 2.0)
        self.declare_parameter('use_viewer', False)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('child_frame_id', 'camera_link')

        self._slam_mode = self.get_parameter('slam_mode').value
        self._frame_id = self.get_parameter('frame_id').value
        self._child_frame_id = self.get_parameter('child_frame_id').value

        # ─── QoS ────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ─── Publishers ─────────────────────────────────────────────────
        self._pub_pose = self.create_publisher(PoseStamped, 'slam/pose', reliable_qos)
        self._pub_map_points = self.create_publisher(PointCloud2, 'slam/map_points', reliable_qos)
        self._pub_state = self.create_publisher(String, 'slam/state', reliable_qos)

        # ─── TF Broadcaster ─────────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ─── Subscribers ────────────────────────────────────────────────
        self._sub_image = self.create_subscription(
            Image, 'camera/image_raw', self._image_callback, sensor_qos
        )
        self._sub_depth = self.create_subscription(
            Image, 'camera/depth', self._depth_callback, sensor_qos
        )
        self._sub_imu = self.create_subscription(
            Imu, 'imu/data', self._imu_callback, sensor_qos
        )

        # ─── Internal State ─────────────────────────────────────────────
        self._bridge = CvBridge()
        self._slam_system = None
        self._tracking_state = TrackingState.SYSTEM_NOT_READY
        self._current_pose = np.eye(4)
        self._map_points = []
        self._imu_buffer = []
        self._imu_lock = threading.Lock()

        # Pose tracking
        self._total_distance = 0.0
        self._last_position = None
        self._pose_count = 0
        self._tracking_lost_time = None

        # Performance tracking
        self._frame_times = []
        self._start_time = time.time()

        # ─── Initialize SLAM ────────────────────────────────────────────
        self._init_slam()

        # ─── Diagnostics ────────────────────────────────────────────────
        self._diag_timer = self.create_timer(5.0, self._diagnostics_callback)

        self.get_logger().info(
            f'ORB-SLAM3 node initialized — mode: {self._slam_mode}'
        )

    def _init_slam(self):
        """Initialize ORB-SLAM3 system."""
        try:
            # Try to import ORB-SLAM3 Python bindings
            # These need to be built from the ORB-SLAM3 source with Python wrapper
            import orbslam3

            vocab_path = self.get_parameter('vocabulary_path').value
            settings_path = self.get_parameter('settings_path').value

            mode_map = {
                'mono': orbslam3.Sensor.MONOCULAR,
                'stereo': orbslam3.Sensor.STEREO,
                'rgbd': orbslam3.Sensor.RGBD,
                'mono_inertial': orbslam3.Sensor.IMU_MONOCULAR,
                'stereo_inertial': orbslam3.Sensor.IMU_STEREO,
            }

            sensor_type = mode_map.get(self._slam_mode, orbslam3.Sensor.IMU_STEREO)
            use_viewer = self.get_parameter('use_viewer').value

            self._slam_system = orbslam3.System(
                vocab_path, settings_path, sensor_type, use_viewer
            )

            # Load pre-built map if available
            map_file = self.get_parameter('map_file').value
            if map_file and self.get_parameter('enable_map_reuse').value:
                self._slam_system.load_map(map_file)
                self.get_logger().info(f'Loaded pre-built SLAM map: {map_file}')

            self._slam_backend = 'orbslam3'
            self._tracking_state = TrackingState.NO_IMAGES_YET
            self.get_logger().info('ORB-SLAM3 system initialized successfully')

        except ImportError:
            self.get_logger().warn(
                'ORB-SLAM3 Python bindings not found. Using simulated SLAM. '
                'Build ORB-SLAM3 with Python wrapper for real operation.'
            )
            self._slam_backend = 'simulated'
            self._tracking_state = TrackingState.OK
            self._sim_time = 0.0

    def _image_callback(self, msg):
        """Process incoming camera frames through SLAM."""
        t_start = time.time()

        if self._slam_backend == 'orbslam3':
            self._process_orbslam3(msg)
        else:
            self._process_simulated(msg)

        # Track processing time
        dt = (time.time() - t_start) * 1000.0
        self._frame_times.append(dt)
        if len(self._frame_times) > 100:
            self._frame_times.pop(0)

    def _depth_callback(self, msg):
        """Store depth frame for RGBD mode."""
        self._last_depth = msg

    def _imu_callback(self, msg):
        """Buffer IMU measurements for visual-inertial modes."""
        with self._imu_lock:
            self._imu_buffer.append({
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'acc': [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ],
                'gyro': [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ],
            })
            # Keep buffer manageable
            if len(self._imu_buffer) > 1000:
                self._imu_buffer = self._imu_buffer[-500:]

    def _process_orbslam3(self, img_msg):
        """Process frame through actual ORB-SLAM3."""
        import orbslam3

        cv_image = self._bridge.imgmsg_to_cv2(img_msg, 'mono8')
        timestamp = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9

        # Get IMU measurements since last frame
        with self._imu_lock:
            imu_data = list(self._imu_buffer)
            self._imu_buffer.clear()

        # Track frame
        if 'inertial' in self._slam_mode and imu_data:
            pose = self._slam_system.process_image_imu(cv_image, imu_data, timestamp)
        else:
            pose = self._slam_system.process_image_mono(cv_image, timestamp)

        # Update tracking state
        state = self._slam_system.get_tracking_state()
        self._tracking_state = TrackingState(state)

        if pose is not None and self._tracking_state == TrackingState.OK:
            self._current_pose = pose
            self._publish_pose(img_msg.header.stamp)
            self._tracking_lost_time = None
        elif self._tracking_state in (TrackingState.RECENTLY_LOST, TrackingState.LOST):
            if self._tracking_lost_time is None:
                self._tracking_lost_time = time.time()
            self._handle_tracking_loss()

        # Publish state
        state_msg = String()
        state_msg.data = self._tracking_state.name
        self._pub_state.publish(state_msg)

    def _process_simulated(self, img_msg):
        """Simulated SLAM for development/testing."""
        self._sim_time += 1.0 / 30.0  # Assume 30 Hz

        # Simulate a circular trajectory
        radius = 5.0
        speed = 0.5  # m/s
        angle = speed * self._sim_time / radius

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.5  # Fixed altitude
        yaw = angle + np.pi / 2  # Tangent direction

        # Build pose matrix
        self._current_pose = np.eye(4)
        self._current_pose[0, 3] = x
        self._current_pose[1, 3] = y
        self._current_pose[2, 3] = z

        # Rotation around Z (yaw)
        cy, sy = np.cos(yaw), np.sin(yaw)
        self._current_pose[0, 0] = cy
        self._current_pose[0, 1] = -sy
        self._current_pose[1, 0] = sy
        self._current_pose[1, 1] = cy

        # Add small noise to simulate drift
        noise = np.random.normal(0, 0.001, 3)
        self._current_pose[0:3, 3] += noise

        self._tracking_state = TrackingState.OK
        self._publish_pose(img_msg.header.stamp)

        # Publish state
        state_msg = String()
        state_msg.data = 'OK'
        self._pub_state.publish(state_msg)

        # Periodically publish simulated map points
        if self._pose_count % 30 == 0:
            self._publish_simulated_map_points(img_msg.header.stamp)

    def _publish_pose(self, stamp):
        """Publish current pose as PoseStamped and TF."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self._frame_id

        # Extract position
        pose_msg.pose.position.x = float(self._current_pose[0, 3])
        pose_msg.pose.position.y = float(self._current_pose[1, 3])
        pose_msg.pose.position.z = float(self._current_pose[2, 3])

        # Extract quaternion from rotation matrix
        q = self._rotation_matrix_to_quaternion(self._current_pose[:3, :3])
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        self._pub_pose.publish(pose_msg)

        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._frame_id
        t.child_frame_id = self._child_frame_id
        t.transform.translation.x = pose_msg.pose.position.x
        t.transform.translation.y = pose_msg.pose.position.y
        t.transform.translation.z = pose_msg.pose.position.z
        t.transform.rotation = pose_msg.pose.orientation
        self._tf_broadcaster.sendTransform(t)

        # Track distance
        current_pos = np.array([
            self._current_pose[0, 3],
            self._current_pose[1, 3],
            self._current_pose[2, 3],
        ])
        if self._last_position is not None:
            self._total_distance += np.linalg.norm(current_pos - self._last_position)
        self._last_position = current_pos
        self._pose_count += 1

    def _publish_simulated_map_points(self, stamp):
        """Publish simulated sparse map points as PointCloud2."""
        num_points = 200
        points = np.random.randn(num_points, 3).astype(np.float32)
        points[:, 0] += float(self._current_pose[0, 3])
        points[:, 1] += float(self._current_pose[1, 3])
        points[:, 2] = np.abs(points[:, 2]) + 0.5

        # Build PointCloud2 message
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = num_points
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * num_points
        msg.data = points.tobytes()
        msg.is_dense = True

        self._pub_map_points.publish(msg)

    def _handle_tracking_loss(self):
        """Handle SLAM tracking loss with relocalization attempt."""
        if not self.get_parameter('relocalization_enabled').value:
            return

        timeout = self.get_parameter('relocalization_timeout_s').value
        lost_duration = time.time() - self._tracking_lost_time

        if lost_duration < timeout:
            self.get_logger().warn(
                f'SLAM tracking lost — attempting relocalization '
                f'({lost_duration:.1f}s / {timeout}s)'
            )
        else:
            self.get_logger().error(
                f'SLAM relocalization failed after {timeout}s — '
                f'falling back to last known pose'
            )

    @staticmethod
    def _rotation_matrix_to_quaternion(R):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return [x, y, z, w]

    def _diagnostics_callback(self):
        """Log SLAM diagnostics."""
        avg_time = np.mean(self._frame_times) if self._frame_times else 0
        elapsed = time.time() - self._start_time
        pose_rate = self._pose_count / elapsed if elapsed > 0 else 0

        self.get_logger().info(
            f'SLAM [{self._slam_backend}] — '
            f'State: {self._tracking_state.name} | '
            f'Rate: {pose_rate:.1f} Hz | '
            f'Avg time: {avg_time:.1f}ms | '
            f'Distance: {self._total_distance:.1f}m | '
            f'Poses: {self._pose_count}'
        )

    def destroy_node(self):
        """Save map on exit if configured."""
        if (self._slam_backend == 'orbslam3' and
                self.get_parameter('save_map_on_exit').value):
            try:
                self._slam_system.save_map('slam_map_autosave.osm')
                self.get_logger().info('SLAM map saved on exit')
            except Exception as e:
                self.get_logger().error(f'Failed to save SLAM map: {e}')

        if self._slam_system:
            self._slam_system.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OrbSlam3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
