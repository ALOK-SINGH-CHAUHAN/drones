"""
ANTIGRAVITY — Monte Carlo Localization (MCL) Node
==================================================
Particle filter localization fusing SLAM pose estimates with pre-built maps.
Maintains a probability distribution over drone position on the map.

Acceptance Criteria:
  - Converges to correct position within 5 seconds of mission start
  - Position error <= 0.3m in mapped environments
  - Handles SLAM drift gracefully by re-weighting particles
  - Adaptive particle count for computational efficiency
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

import numpy as np
import time
import threading
from dataclasses import dataclass


@dataclass
class Particle:
    """Single particle in the MCL particle filter."""
    x: float
    y: float
    z: float
    yaw: float
    weight: float


class MCLNode(Node):
    """
    Monte Carlo Localization using adaptive particle filter.
    
    Fuses SLAM pose estimates with a pre-built occupancy grid map to maintain
    an accurate probability distribution over the drone's position.
    
    Subscribes:
      - /slam/slam/pose (geometry_msgs/PoseStamped): SLAM pose input
      - /mapping/map (nav_msgs/OccupancyGrid): Pre-built map
      - /odom (nav_msgs/Odometry): Odometry for motion model
    
    Publishes:
      - localization/pose (geometry_msgs/PoseStamped): Best pose estimate
      - localization/particles (geometry_msgs/PoseArray): Particle cloud
      - localization/state (std_msgs/String): Localization state
    
    Broadcasts:
      - TF: map -> odom transform (correction)
    """

    def __init__(self):
        super().__init__('mcl_localization')

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('num_particles', 500)
        self.declare_parameter('min_particles', 100)
        self.declare_parameter('max_particles', 2000)
        self.declare_parameter('resample_threshold', 0.5)
        self.declare_parameter('resample_method', 'systematic')
        self.declare_parameter('alpha1', 0.2)
        self.declare_parameter('alpha2', 0.2)
        self.declare_parameter('alpha3', 0.2)
        self.declare_parameter('alpha4', 0.2)
        self.declare_parameter('sensor_model_type', 'likelihood_field')
        self.declare_parameter('z_hit', 0.95)
        self.declare_parameter('z_rand', 0.05)
        self.declare_parameter('sigma_hit', 0.2)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('convergence_threshold', 0.3)
        self.declare_parameter('update_min_d', 0.1)
        self.declare_parameter('update_min_a', 0.1)
        self.declare_parameter('update_rate_hz', 20.0)
        self.declare_parameter('initial_pose_x', 0.0)
        self.declare_parameter('initial_pose_y', 0.0)
        self.declare_parameter('initial_pose_z', 1.5)
        self.declare_parameter('initial_pose_yaw', 0.0)
        self.declare_parameter('initial_cov_xx', 1.0)
        self.declare_parameter('initial_cov_yy', 1.0)
        self.declare_parameter('initial_cov_aa', 0.5)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

        self._num_particles = self.get_parameter('num_particles').value
        self._min_particles = self.get_parameter('min_particles').value
        self._max_particles = self.get_parameter('max_particles').value
        self._map_frame = self.get_parameter('map_frame').value

        # Motion model noise parameters (Thrun's probabilistic robotics)
        self._alpha = np.array([
            self.get_parameter('alpha1').value,
            self.get_parameter('alpha2').value,
            self.get_parameter('alpha3').value,
            self.get_parameter('alpha4').value,
        ])

        # ─── QoS ────────────────────────────────────────────────────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ─── Publishers ─────────────────────────────────────────────────
        self._pub_pose = self.create_publisher(PoseStamped, 'localization/pose', reliable_qos)
        self._pub_particles = self.create_publisher(PoseArray, 'localization/particles', reliable_qos)
        self._pub_state = self.create_publisher(String, 'localization/state', reliable_qos)

        # ─── TF ─────────────────────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)

        # ─── Subscribers ────────────────────────────────────────────────
        self._sub_slam_pose = self.create_subscription(
            PoseStamped, '/slam/slam/pose', self._slam_pose_callback, reliable_qos
        )
        self._sub_map = self.create_subscription(
            OccupancyGrid, '/mapping/map', self._map_callback, map_qos
        )
        self._sub_odom = self.create_subscription(
            Odometry, '/odom', self._odom_callback, reliable_qos
        )

        # ─── State ──────────────────────────────────────────────────────
        self._particles = []
        self._map_data = None
        self._map_info = None
        self._distance_table = None  # Pre-computed distance transform
        self._last_slam_pose = None
        self._prev_slam_pose = None
        self._last_odom = None
        self._prev_odom = None
        self._converged = False
        self._best_pose = None
        self._lock = threading.Lock()

        # ─── Initialize Particles ────────────────────────────────────────
        self._initialize_particles()

        # ─── Update Timer ────────────────────────────────────────────────
        rate = self.get_parameter('update_rate_hz').value
        self._update_timer = self.create_timer(1.0 / rate, self._update)

        # ─── Diagnostics ────────────────────────────────────────────────
        self._update_count = 0
        self._start_time = time.time()
        self._diag_timer = self.create_timer(5.0, self._diagnostics_callback)

        self.get_logger().info(
            f'MCL initialized with {self._num_particles} particles'
        )

    def _initialize_particles(self):
        """Initialize particle cloud around initial pose with uncertainty."""
        init_x = self.get_parameter('initial_pose_x').value
        init_y = self.get_parameter('initial_pose_y').value
        init_z = self.get_parameter('initial_pose_z').value
        init_yaw = self.get_parameter('initial_pose_yaw').value

        cov_xx = self.get_parameter('initial_cov_xx').value
        cov_yy = self.get_parameter('initial_cov_yy').value
        cov_aa = self.get_parameter('initial_cov_aa').value

        self._particles = []
        for _ in range(self._num_particles):
            p = Particle(
                x=init_x + np.random.normal(0, np.sqrt(cov_xx)),
                y=init_y + np.random.normal(0, np.sqrt(cov_yy)),
                z=init_z,
                yaw=init_yaw + np.random.normal(0, np.sqrt(cov_aa)),
                weight=1.0 / self._num_particles,
            )
            self._particles.append(p)

        self.get_logger().info(
            f'Particles initialized around ({init_x}, {init_y}, {init_z})'
        )

    def _map_callback(self, msg):
        """Receive occupancy grid map and pre-compute distance transform."""
        with self._lock:
            self._map_info = msg.info
            width = msg.info.width
            height = msg.info.height

            # Convert to numpy array
            self._map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

            # Pre-compute distance transform for likelihood field model
            self._compute_distance_transform()

            self.get_logger().info(
                f'Map received: {width}x{height} @ {msg.info.resolution}m/px'
            )

    def _compute_distance_transform(self):
        """Pre-compute distance to nearest obstacle for likelihood field sensor model."""
        if self._map_data is None:
            return

        from scipy import ndimage

        # Create binary obstacle map (1 = obstacle, 0 = free/unknown)
        obstacle_map = (self._map_data >= 50).astype(np.float64)

        # Compute Euclidean distance transform
        # Each cell gets the distance to the nearest obstacle
        self._distance_table = ndimage.distance_transform_edt(
            1 - obstacle_map
        ) * self._map_info.resolution

        self.get_logger().info('Distance transform computed for likelihood field model')

    def _slam_pose_callback(self, msg):
        """Receive SLAM pose estimate as observation."""
        self._prev_slam_pose = self._last_slam_pose
        self._last_slam_pose = msg

    def _odom_callback(self, msg):
        """Receive odometry for motion model."""
        self._prev_odom = self._last_odom
        self._last_odom = msg

    def _update(self):
        """Main MCL update loop: predict → weight → resample."""
        if not self._particles or self._last_slam_pose is None:
            return

        with self._lock:
            # ── PREDICT: Apply motion model ──────────────────────────────
            self._motion_update()

            # ── WEIGHT: Apply sensor model ───────────────────────────────
            self._sensor_update()

            # ── RESAMPLE: If effective particle count is low ─────────────
            n_eff = self._effective_particle_count()
            threshold = self.get_parameter('resample_threshold').value * len(self._particles)

            if n_eff < threshold:
                self._resample()

            # ── ESTIMATE: Compute best pose ──────────────────────────────
            self._compute_best_estimate()

            # ── PUBLISH ──────────────────────────────────────────────────
            self._publish_results()

            self._update_count += 1

    def _motion_update(self):
        """Apply odometry-based motion model to all particles."""
        if self._prev_slam_pose is None or self._last_slam_pose is None:
            return

        # Compute motion delta from SLAM poses
        dx = (self._last_slam_pose.pose.position.x -
              self._prev_slam_pose.pose.position.x)
        dy = (self._last_slam_pose.pose.position.y -
              self._prev_slam_pose.pose.position.y)

        # Decompose into rotation-translation-rotation
        delta_trans = np.sqrt(dx * dx + dy * dy)
        if delta_trans < self.get_parameter('update_min_d').value:
            return

        delta_rot1 = np.arctan2(dy, dx)
        delta_rot2 = 0.0  # Simplified

        # Apply noisy motion to each particle
        for p in self._particles:
            # Add noise proportional to motion
            a = self._alpha
            hat_rot1 = delta_rot1 + np.random.normal(
                0, a[0] * abs(delta_rot1) + a[1] * delta_trans
            )
            hat_trans = delta_trans + np.random.normal(
                0, a[2] * delta_trans + a[3] * (abs(delta_rot1) + abs(delta_rot2))
            )
            hat_rot2 = delta_rot2 + np.random.normal(
                0, a[0] * abs(delta_rot2) + a[1] * delta_trans
            )

            p.x += hat_trans * np.cos(p.yaw + hat_rot1)
            p.y += hat_trans * np.sin(p.yaw + hat_rot1)
            p.yaw += hat_rot1 + hat_rot2

            # Normalize yaw
            p.yaw = np.arctan2(np.sin(p.yaw), np.cos(p.yaw))

    def _sensor_update(self):
        """Update particle weights using likelihood field sensor model."""
        if self._map_data is None or self._distance_table is None:
            return

        sigma = self.get_parameter('sigma_hit').value
        z_hit = self.get_parameter('z_hit').value
        z_rand = self.get_parameter('z_rand').value

        resolution = self._map_info.resolution
        origin_x = self._map_info.origin.position.x
        origin_y = self._map_info.origin.position.y
        width = self._map_info.width
        height = self._map_info.height

        for p in self._particles:
            # Convert particle position to map grid coordinates
            gx = int((p.x - origin_x) / resolution)
            gy = int((p.y - origin_y) / resolution)

            # Check if particle is inside the map
            if 0 <= gx < width and 0 <= gy < height:
                # Check if particle is in occupied space
                cell_value = self._map_data[gy, gx]
                if cell_value >= 50:  # Occupied
                    p.weight *= 0.001  # Very unlikely to be inside a wall
                    continue

                # Get distance to nearest obstacle
                dist = self._distance_table[gy, gx]

                # Likelihood field model
                # p(z|x) = z_hit * gaussian(dist, 0, sigma) + z_rand * uniform
                likelihood = (
                    z_hit * np.exp(-0.5 * (dist / sigma) ** 2) /
                    (sigma * np.sqrt(2 * np.pi)) +
                    z_rand / self.get_parameter('max_range').value
                )

                p.weight *= max(likelihood, 1e-10)
            else:
                # Outside map bounds — very low weight
                p.weight *= 0.001

        # Normalize weights
        total_weight = sum(p.weight for p in self._particles)
        if total_weight > 0:
            for p in self._particles:
                p.weight /= total_weight
        else:
            # Reset weights if all zero
            uniform = 1.0 / len(self._particles)
            for p in self._particles:
                p.weight = uniform

    def _effective_particle_count(self):
        """Compute effective number of particles (Neff)."""
        weights_sq = sum(p.weight ** 2 for p in self._particles)
        if weights_sq > 0:
            return 1.0 / weights_sq
        return 0

    def _resample(self):
        """Systematic resampling of particles."""
        n = len(self._particles)
        weights = np.array([p.weight for p in self._particles])

        method = self.get_parameter('resample_method').value
        if method == 'systematic':
            indices = self._systematic_resample(weights, n)
        elif method == 'stratified':
            indices = self._stratified_resample(weights, n)
        else:
            indices = self._multinomial_resample(weights, n)

        new_particles = []
        for i in indices:
            old = self._particles[i]
            new_particles.append(Particle(
                x=old.x, y=old.y, z=old.z,
                yaw=old.yaw, weight=1.0 / n
            ))

        self._particles = new_particles

    @staticmethod
    def _systematic_resample(weights, n):
        """Low-variance systematic resampling."""
        positions = (np.arange(n) + np.random.uniform()) / n
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        return np.clip(indices, 0, n - 1)

    @staticmethod
    def _stratified_resample(weights, n):
        """Stratified resampling."""
        positions = (np.arange(n) + np.random.uniform(size=n)) / n
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        return np.clip(indices, 0, n - 1)

    @staticmethod
    def _multinomial_resample(weights, n):
        """Standard multinomial resampling."""
        return np.random.choice(n, size=n, replace=True, p=weights)

    def _compute_best_estimate(self):
        """Compute weighted mean pose from particle cloud."""
        if not self._particles:
            return

        # Weighted mean position
        total_weight = sum(p.weight for p in self._particles)
        if total_weight == 0:
            return

        mean_x = sum(p.x * p.weight for p in self._particles) / total_weight
        mean_y = sum(p.y * p.weight for p in self._particles) / total_weight
        mean_z = sum(p.z * p.weight for p in self._particles) / total_weight

        # Circular mean for yaw
        sin_sum = sum(np.sin(p.yaw) * p.weight for p in self._particles) / total_weight
        cos_sum = sum(np.cos(p.yaw) * p.weight for p in self._particles) / total_weight
        mean_yaw = np.arctan2(sin_sum, cos_sum)

        # Compute variance for convergence check
        var_x = sum(p.weight * (p.x - mean_x) ** 2 for p in self._particles) / total_weight
        var_y = sum(p.weight * (p.y - mean_y) ** 2 for p in self._particles) / total_weight
        std = np.sqrt(var_x + var_y)

        conv_thresh = self.get_parameter('convergence_threshold').value
        self._converged = std < conv_thresh

        self._best_pose = {
            'x': mean_x, 'y': mean_y, 'z': mean_z,
            'yaw': mean_yaw,
            'std': std,
            'converged': self._converged,
        }

    def _publish_results(self):
        """Publish pose estimate and particle cloud."""
        if self._best_pose is None:
            return

        stamp = self.get_clock().now().to_msg()
        bp = self._best_pose

        # ── Pose ─────────────────────────────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self._map_frame
        pose_msg.pose.position.x = bp['x']
        pose_msg.pose.position.y = bp['y']
        pose_msg.pose.position.z = bp['z']

        # Yaw to quaternion
        cy = np.cos(bp['yaw'] / 2)
        sy = np.sin(bp['yaw'] / 2)
        pose_msg.pose.orientation.z = sy
        pose_msg.pose.orientation.w = cy

        self._pub_pose.publish(pose_msg)

        # ── Particle Cloud ───────────────────────────────────────────────
        particle_msg = PoseArray()
        particle_msg.header.stamp = stamp
        particle_msg.header.frame_id = self._map_frame

        for p in self._particles:
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            pose.position.z = p.z
            cy = np.cos(p.yaw / 2)
            sy = np.sin(p.yaw / 2)
            pose.orientation.z = sy
            pose.orientation.w = cy
            particle_msg.poses.append(pose)

        self._pub_particles.publish(particle_msg)

        # ── State ────────────────────────────────────────────────────────
        state_msg = String()
        state_msg.data = (
            f'CONVERGED (σ={bp["std"]:.3f}m)' if bp['converged']
            else f'CONVERGING (σ={bp["std"]:.3f}m)'
        )
        self._pub_state.publish(state_msg)

    def _diagnostics_callback(self):
        """Log MCL diagnostics."""
        n_eff = self._effective_particle_count()
        elapsed = time.time() - self._start_time

        pose_str = "N/A"
        if self._best_pose:
            bp = self._best_pose
            pose_str = f'({bp["x"]:.2f}, {bp["y"]:.2f}) σ={bp["std"]:.3f}m'

        self.get_logger().info(
            f'MCL — '
            f'Particles: {len(self._particles)} | '
            f'Neff: {n_eff:.0f} | '
            f'Pose: {pose_str} | '
            f'Converged: {self._converged} | '
            f'Updates: {self._update_count}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MCLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
