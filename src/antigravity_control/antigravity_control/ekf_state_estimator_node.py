"""
ANTIGRAVITY — Extended Kalman Filter State Estimator
=====================================================
Fuses SLAM, IMU, MCL, and barometer into a single optimal state estimate.
Provides the authoritative drone state for all downstream nodes.

Acceptance Criteria (P3-T5):
  - Position accuracy <= 0.1m RMS in mapped areas
  - Handles individual sensor failures gracefully
  - 100 Hz state output with < 5ms latency
  - Covariance output for downstream planners
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String
from nav_msgs.msg import Odometry

import numpy as np
import time
import threading


class EKFStateEstimatorNode(Node):
    """
    Extended Kalman Filter fusing multiple sensor sources.

    State vector [15]: [x, y, z, vx, vy, vz, roll, pitch, yaw, bax, bay, baz, bgx, bgy, bgz]
    - Position (3), Velocity (3), Orientation (3), Accel bias (3), Gyro bias (3)

    Subscribes:
      - /slam/slam/pose (PoseStamped): Visual SLAM pose
      - /localization/localization/pose (PoseStamped): MCL pose
      - /imu/data (Imu): IMU data (prediction step)
      - /baro/altitude (Float32): Barometric altitude

    Publishes:
      - state/pose (PoseWithCovarianceStamped): Fused pose with covariance
      - state/odometry (Odometry): Full odometry (pose + velocity)
      - state/status (String): Estimator status
    """

    STATE_DIM = 15
    # State indices
    PX, PY, PZ = 0, 1, 2
    VX, VY, VZ = 3, 4, 5
    ROLL, PITCH, YAW = 6, 7, 8
    BAX, BAY, BAZ = 9, 10, 11
    BGX, BGY, BGZ = 12, 13, 14

    def __init__(self):
        super().__init__('ekf_state_estimator')

        self.declare_parameter('publish_rate_hz', 100)
        self.declare_parameter('process_noise_accel', 0.5)
        self.declare_parameter('process_noise_gyro', 0.01)
        self.declare_parameter('process_noise_bias', 0.001)
        self.declare_parameter('slam_noise_position', 0.05)
        self.declare_parameter('slam_noise_orientation', 0.02)
        self.declare_parameter('mcl_noise_position', 0.15)
        self.declare_parameter('baro_noise', 0.3)
        self.declare_parameter('imu_noise_accel', 0.1)
        self.declare_parameter('imu_noise_gyro', 0.01)
        self.declare_parameter('outlier_threshold', 3.0)  # Mahalanobis
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('child_frame_id', 'base_link')

        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=5)

        self._pub_pose = self.create_publisher(PoseWithCovarianceStamped, 'state/pose', reliable_qos)
        self._pub_odom = self.create_publisher(Odometry, 'state/odometry', reliable_qos)
        self._pub_status = self.create_publisher(String, 'state/status', reliable_qos)

        self._sub_imu = self.create_subscription(Imu, '/imu/data', self._imu_cb, sensor_qos)
        self._sub_slam = self.create_subscription(PoseStamped, '/slam/slam/pose', self._slam_cb, reliable_qos)
        self._sub_mcl = self.create_subscription(PoseStamped, '/localization/localization/pose',
                                                  self._mcl_cb, reliable_qos)
        self._sub_baro = self.create_subscription(Float32, '/baro/altitude', self._baro_cb, sensor_qos)

        # EKF state
        self._x = np.zeros(self.STATE_DIM)
        self._x[self.PZ] = 1.5  # Initial altitude
        self._P = np.eye(self.STATE_DIM) * 10.0  # Initial covariance
        self._last_imu_time = None
        self._lock = threading.Lock()
        self._initialized = False

        # TF2 broadcaster for map → base_link transform
        try:
            from tf2_ros import TransformBroadcaster
            self._tf_broadcaster = TransformBroadcaster(self)
            self._tf_available = True
        except ImportError:
            self.get_logger().warn('tf2_ros not available — TF broadcasting disabled')
            self._tf_broadcaster = None
            self._tf_available = False

        # Sensor health tracking
        self._sensor_last_update = {'slam': 0, 'mcl': 0, 'imu': 0, 'baro': 0}
        self._sensor_timeout = 2.0
        self._update_count = 0
        self._outlier_count = 0

        rate = self.get_parameter('publish_rate_hz').value
        self.create_timer(1.0 / rate, self._publish_state)
        self.create_timer(5.0, self._diag)
        self.get_logger().info(f'EKF state estimator initialized — {self.STATE_DIM} states, {rate} Hz')

    # ─── Prediction (IMU) ────────────────────────────────────────────────

    def _imu_cb(self, msg):
        """IMU-driven prediction step."""
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        with self._lock:
            if self._last_imu_time is None:
                self._last_imu_time = now
                return

            dt = now - self._last_imu_time
            if dt <= 0 or dt > 0.5:
                self._last_imu_time = now
                return
            self._last_imu_time = now

            # Extract measurements
            ax = msg.linear_acceleration.x - self._x[self.BAX]
            ay = msg.linear_acceleration.y - self._x[self.BAY]
            az = msg.linear_acceleration.z - self._x[self.BAZ] - 9.81
            gx = msg.angular_velocity.x - self._x[self.BGX]
            gy = msg.angular_velocity.y - self._x[self.BGY]
            gz = msg.angular_velocity.z - self._x[self.BGZ]

            # Rotation matrix from current orientation
            R = self._euler_to_rotation(self._x[self.ROLL], self._x[self.PITCH], self._x[self.YAW])
            a_world = R @ np.array([ax, ay, az])

            # State prediction
            self._x[self.PX] += self._x[self.VX] * dt + 0.5 * a_world[0] * dt**2
            self._x[self.PY] += self._x[self.VY] * dt + 0.5 * a_world[1] * dt**2
            self._x[self.PZ] += self._x[self.VZ] * dt + 0.5 * a_world[2] * dt**2
            self._x[self.VX] += a_world[0] * dt
            self._x[self.VY] += a_world[1] * dt
            self._x[self.VZ] += a_world[2] * dt
            self._x[self.ROLL] += gx * dt
            self._x[self.PITCH] += gy * dt
            self._x[self.YAW] += gz * dt

            # Normalize angles
            self._x[self.YAW] = np.arctan2(np.sin(self._x[self.YAW]), np.cos(self._x[self.YAW]))

            # Jacobian F (linearized state transition)
            F = np.eye(self.STATE_DIM)
            F[self.PX, self.VX] = dt
            F[self.PY, self.VY] = dt
            F[self.PZ, self.VZ] = dt

            # Process noise
            q_a = self.get_parameter('process_noise_accel').value
            q_g = self.get_parameter('process_noise_gyro').value
            q_b = self.get_parameter('process_noise_bias').value
            Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
            Q[self.PX, self.PX] = q_a * dt**4 / 4
            Q[self.PY, self.PY] = q_a * dt**4 / 4
            Q[self.PZ, self.PZ] = q_a * dt**4 / 4
            Q[self.VX, self.VX] = q_a * dt**2
            Q[self.VY, self.VY] = q_a * dt**2
            Q[self.VZ, self.VZ] = q_a * dt**2
            Q[self.ROLL, self.ROLL] = q_g * dt**2
            Q[self.PITCH, self.PITCH] = q_g * dt**2
            Q[self.YAW, self.YAW] = q_g * dt**2
            for i in range(self.BAX, self.BGZ + 1):
                Q[i, i] = q_b * dt

            self._P = F @ self._P @ F.T + Q
            self._sensor_last_update['imu'] = time.time()
            self._initialized = True

    # ─── Updates ─────────────────────────────────────────────────────────

    def _slam_cb(self, msg):
        """SLAM position update."""
        z = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        H = np.zeros((3, self.STATE_DIM))
        H[0, self.PX] = 1; H[1, self.PY] = 1; H[2, self.PZ] = 1
        noise = self.get_parameter('slam_noise_position').value
        R = np.eye(3) * noise**2

        with self._lock:
            self._ekf_update(z, H, R, 'slam')
        self._sensor_last_update['slam'] = time.time()

    def _mcl_cb(self, msg):
        """MCL position update."""
        z = np.array([msg.pose.position.x, msg.pose.position.y])
        H = np.zeros((2, self.STATE_DIM))
        H[0, self.PX] = 1; H[1, self.PY] = 1
        noise = self.get_parameter('mcl_noise_position').value
        R = np.eye(2) * noise**2

        with self._lock:
            self._ekf_update(z, H, R, 'mcl')
        self._sensor_last_update['mcl'] = time.time()

    def _baro_cb(self, msg):
        """Barometric altitude update."""
        z = np.array([msg.data])
        H = np.zeros((1, self.STATE_DIM))
        H[0, self.PZ] = 1
        noise = self.get_parameter('baro_noise').value
        R = np.array([[noise**2]])

        with self._lock:
            self._ekf_update(z, H, R, 'baro')
        self._sensor_last_update['baro'] = time.time()

    def _ekf_update(self, z, H, R, source):
        """Generic EKF update step with Mahalanobis outlier rejection."""
        y = z - H @ self._x  # Innovation
        S = H @ self._P @ H.T + R  # Innovation covariance

        # Mahalanobis distance for outlier rejection
        try:
            S_inv = np.linalg.inv(S)
            mahal = float(np.sqrt(y @ S_inv @ y))
            threshold = self.get_parameter('outlier_threshold').value
            if mahal > threshold:
                self._outlier_count += 1
                self.get_logger().debug(f'Outlier rejected from {source}: mahal={mahal:.2f}')
                return
        except np.linalg.LinAlgError:
            return

        # Kalman gain
        K = self._P @ H.T @ S_inv

        # State update
        self._x = self._x + K @ y

        # Joseph form covariance update (numerically stable)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

        self._update_count += 1

    # ─── Publishing ──────────────────────────────────────────────────────

    def _publish_state(self):
        if not self._initialized:
            return

        stamp = self.get_clock().now().to_msg()
        frame = self.get_parameter('frame_id').value
        child = self.get_parameter('child_frame_id').value

        with self._lock:
            x = self._x.copy()
            P = self._P.copy()

        # PoseWithCovariance
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = frame
        pose_msg.pose.pose.position.x = float(x[self.PX])
        pose_msg.pose.pose.position.y = float(x[self.PY])
        pose_msg.pose.pose.position.z = float(x[self.PZ])

        q = self._euler_to_quaternion(x[self.ROLL], x[self.PITCH], x[self.YAW])
        pose_msg.pose.pose.orientation.x = q[0]
        pose_msg.pose.pose.orientation.y = q[1]
        pose_msg.pose.pose.orientation.z = q[2]
        pose_msg.pose.pose.orientation.w = q[3]

        # 6x6 covariance (position + orientation)
        cov = np.zeros(36)
        cov[0] = P[self.PX, self.PX]; cov[7] = P[self.PY, self.PY]; cov[14] = P[self.PZ, self.PZ]
        cov[21] = P[self.ROLL, self.ROLL]; cov[28] = P[self.PITCH, self.PITCH]; cov[35] = P[self.YAW, self.YAW]
        pose_msg.pose.covariance = cov.tolist()
        self._pub_pose.publish(pose_msg)

        # Odometry
        odom = Odometry()
        odom.header = pose_msg.header
        odom.child_frame_id = child
        odom.pose = pose_msg.pose
        odom.twist.twist.linear.x = float(x[self.VX])
        odom.twist.twist.linear.y = float(x[self.VY])
        odom.twist.twist.linear.z = float(x[self.VZ])
        # Velocity covariance from EKF
        vel_cov = np.zeros(36)
        vel_cov[0] = P[self.VX, self.VX]; vel_cov[7] = P[self.VY, self.VY]; vel_cov[14] = P[self.VZ, self.VZ]
        odom.twist.covariance = vel_cov.tolist()
        self._pub_odom.publish(odom)

        # TF2 broadcast: map → base_link
        if self._tf_available and self._tf_broadcaster:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = frame
            t.child_frame_id = child
            t.transform.translation.x = float(x[self.PX])
            t.transform.translation.y = float(x[self.PY])
            t.transform.translation.z = float(x[self.PZ])
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self._tf_broadcaster.sendTransform(t)

        # Status
        active = [s for s, t in self._sensor_last_update.items() if time.time() - t < self._sensor_timeout]
        status = f'FUSING ({",".join(active)}) pos_std=({np.sqrt(P[0,0]):.3f},{np.sqrt(P[1,1]):.3f},{np.sqrt(P[2,2]):.3f})'
        self._pub_status.publish(String(data=status))

    @staticmethod
    def _euler_to_rotation(roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr],
        ])

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        return [
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy,
        ]

    def _diag(self):
        with self._lock:
            pos_std = np.sqrt(np.diag(self._P[:3, :3]))
        active = [s for s, t in self._sensor_last_update.items() if time.time() - t < self._sensor_timeout]
        self.get_logger().info(
            f'EKF — Pos: ({self._x[0]:.2f},{self._x[1]:.2f},{self._x[2]:.2f}) '
            f'σ: ({pos_std[0]:.3f},{pos_std[1]:.3f},{pos_std[2]:.3f}) '
            f'Active: {active} | Updates: {self._update_count} | Outliers: {self._outlier_count}')


def main(args=None):
    rclpy.init(args=args)
    node = EKFStateEstimatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
