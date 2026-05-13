"""
ANTIGRAVITY — MPC Local Planner Node
======================================
Model Predictive Control local planner running at 10-50 Hz.
Optimizes trajectory over 1-3 second horizon considering drone dynamics.

Acceptance Criteria (P3-T2):
  - MPC solve time <= 20ms per cycle on Jetson Orin
  - Zero collisions with static obstacles in simulation
  - Respects all drone physical constraints
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from std_msgs.msg import String

import numpy as np
import time
import threading


class MPCLocalPlannerNode(Node):
    """
    MPC local planner generating smooth, collision-free velocity commands.

    Subscribes:
      - planning/global_path (Path): Waypoints from global planner
      - /localization/localization/pose (PoseStamped): Current drone pose
      - /cognition/prediction/tracks (String): Dynamic obstacle tracks

    Publishes:
      - control/setpoint_velocity (TwistStamped): Velocity commands at 20 Hz
      - planning/local_trajectory (Marker): Predicted trajectory visualization
      - planning/local_status (String): Planner status
    """

    def __init__(self):
        super().__init__('local_planner')

        self.declare_parameter('control_rate_hz', 20)
        self.declare_parameter('horizon_s', 2.0)
        self.declare_parameter('horizon_steps', 20)
        self.declare_parameter('max_vel_xy', 2.0)
        self.declare_parameter('max_vel_z', 1.0)
        self.declare_parameter('max_accel', 2.0)
        self.declare_parameter('max_yaw_rate', 1.0)
        self.declare_parameter('obstacle_clearance_m', 0.8)
        self.declare_parameter('goal_weight', 5.0)
        self.declare_parameter('velocity_weight', 0.1)
        self.declare_parameter('acceleration_weight', 1.0)
        self.declare_parameter('obstacle_weight', 50.0)
        self.declare_parameter('lookahead_distance_m', 3.0)
        self.declare_parameter('waypoint_reached_tolerance_m', 0.5)
        self.declare_parameter('use_casadi', False)

        self._rate = self.get_parameter('control_rate_hz').value
        self._N = self.get_parameter('horizon_steps').value
        self._dt = self.get_parameter('horizon_s').value / self._N
        self._max_v_xy = self.get_parameter('max_vel_xy').value
        self._max_v_z = self.get_parameter('max_vel_z').value
        self._max_a = self.get_parameter('max_accel').value

        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)

        self._pub_vel = self.create_publisher(TwistStamped, 'control/setpoint_velocity', reliable_qos)
        self._pub_traj_marker = self.create_publisher(Marker, 'planning/local_trajectory', reliable_qos)
        self._pub_status = self.create_publisher(String, 'planning/local_status', reliable_qos)

        self._sub_path = self.create_subscription(Path, 'planning/global_path', self._path_cb, reliable_qos)
        self._sub_pose = self.create_subscription(PoseStamped, '/localization/localization/pose',
                                                   self._pose_cb, reliable_qos)
        self._sub_obstacles = self.create_subscription(String, '/cognition/prediction/tracks',
                                                        self._obstacles_cb, reliable_qos)

        self._global_path = []
        self._current_waypoint_idx = 0
        self._pose = None
        self._prev_vel = np.zeros(3)
        self._obstacles = []
        self._lock = threading.Lock()
        self._solve_times = []
        self._solver = None

        if self.get_parameter('use_casadi').value:
            self._init_casadi()

        self.create_timer(1.0 / self._rate, self._control_loop)
        self.create_timer(5.0, self._diag)
        self.get_logger().info(f'MPC local planner — rate: {self._rate} Hz, horizon: {self._N} steps')

    def _init_casadi(self):
        """Initialize CasADi-based MPC solver."""
        try:
            import casadi as ca
            self.get_logger().info('CasADi MPC solver initialized')
            self._solver = 'casadi'
        except ImportError:
            self.get_logger().warn('CasADi not available — using gradient descent MPC')
            self._solver = None

    def _path_cb(self, msg):
        with self._lock:
            self._global_path = [(p.pose.position.x, p.pose.position.y, p.pose.position.z)
                                 for p in msg.poses]
            self._current_waypoint_idx = 0
            self.get_logger().info(f'Received global path with {len(self._global_path)} waypoints')

    def _pose_cb(self, msg):
        self._pose = msg

    def _obstacles_cb(self, msg):
        """Parse dynamic obstacle tracks."""
        import json
        try:
            data = json.loads(msg.data)
            self._obstacles = data.get('tracks', [])
        except: pass

    def _control_loop(self):
        """Main MPC control loop."""
        if self._pose is None or not self._global_path:
            return

        t0 = time.time()
        pos = np.array([self._pose.pose.position.x,
                        self._pose.pose.position.y,
                        self._pose.pose.position.z])

        # Advance waypoint index
        self._advance_waypoint(pos)

        if self._current_waypoint_idx >= len(self._global_path):
            self._publish_zero_vel()
            self._pub_status.publish(String(data='GOAL_REACHED'))
            return

        # Get reference trajectory (lookahead on global path)
        ref_trajectory = self._get_reference_trajectory(pos)

        # Solve MPC
        if self._solver == 'casadi':
            optimal_vel = self._solve_casadi(pos, ref_trajectory)
        else:
            optimal_vel = self._solve_gradient(pos, ref_trajectory)

        # Apply acceleration limit
        vel_delta = optimal_vel - self._prev_vel
        accel_mag = np.linalg.norm(vel_delta) * self._rate
        if accel_mag > self._max_a:
            vel_delta = vel_delta / accel_mag * self._max_a / self._rate
        final_vel = self._prev_vel + vel_delta

        # Clamp velocity
        xy_speed = np.linalg.norm(final_vel[:2])
        if xy_speed > self._max_v_xy:
            final_vel[:2] = final_vel[:2] / xy_speed * self._max_v_xy
        final_vel[2] = np.clip(final_vel[2], -self._max_v_z, self._max_v_z)

        # Check obstacle proximity
        clearance = self.get_parameter('obstacle_clearance_m').value
        for obs in self._obstacles:
            obs_pos = np.array(obs.get('position', [0,0,0]))
            dist = np.linalg.norm(pos[:2] - obs_pos[:2])
            if dist < clearance:
                # Emergency slow-down / avoidance
                away = pos[:2] - obs_pos[:2]
                if np.linalg.norm(away) > 0.01:
                    away = away / np.linalg.norm(away)
                final_vel[:2] += away * self._max_v_xy * 0.5
                self.get_logger().warn(f'Obstacle avoidance — dist: {dist:.2f}m')

        self._prev_vel = final_vel

        # Publish velocity
        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = 'map'
        vel_msg.twist.linear.x = float(final_vel[0])
        vel_msg.twist.linear.y = float(final_vel[1])
        vel_msg.twist.linear.z = float(final_vel[2])
        self._pub_vel.publish(vel_msg)

        dt = (time.time() - t0) * 1000
        self._solve_times.append(dt)
        if len(self._solve_times) > 100: self._solve_times.pop(0)

        self._pub_status.publish(String(data=f'TRACKING (wp {self._current_waypoint_idx}/{len(self._global_path)})'))

    def _advance_waypoint(self, pos):
        """Move to next waypoint if close enough."""
        tol = self.get_parameter('waypoint_reached_tolerance_m').value
        while self._current_waypoint_idx < len(self._global_path):
            wp = np.array(self._global_path[self._current_waypoint_idx])
            if np.linalg.norm(pos - wp) < tol:
                self._current_waypoint_idx += 1
            else:
                break

    def _get_reference_trajectory(self, pos):
        """Extract reference trajectory from global path for MPC horizon."""
        ref = []
        idx = self._current_waypoint_idx
        for _ in range(self._N):
            if idx < len(self._global_path):
                ref.append(np.array(self._global_path[idx]))
                # Check if we should advance to next waypoint in reference
                if len(ref) >= 2:
                    if np.linalg.norm(ref[-1] - pos) > self.get_parameter('lookahead_distance_m').value:
                        pass  # Stay at this waypoint
                    else:
                        idx = min(idx + 1, len(self._global_path) - 1)
            else:
                ref.append(np.array(self._global_path[-1]))
        return ref

    def _solve_gradient(self, pos, ref_trajectory):
        """Simple gradient-descent MPC solver (fallback when CasADi unavailable)."""
        w_goal = self.get_parameter('goal_weight').value
        w_vel = self.get_parameter('velocity_weight').value
        w_acc = self.get_parameter('acceleration_weight').value
        w_obs = self.get_parameter('obstacle_weight').value
        clearance = self.get_parameter('obstacle_clearance_m').value

        # Initialize control sequence
        u = np.zeros((self._N, 3))
        learning_rate = 0.5

        for iteration in range(5):  # Few iterations for real-time
            states = self._rollout(pos, u)
            grad = np.zeros_like(u)

            for k in range(self._N):
                # Goal tracking gradient
                error = states[k] - ref_trajectory[min(k, len(ref_trajectory)-1)]
                grad[k] += w_goal * 2 * error * self._dt

                # Velocity smoothness
                grad[k] += w_vel * 2 * u[k]

                # Acceleration penalty
                if k > 0:
                    accel = (u[k] - u[k-1]) / self._dt
                    grad[k] += w_acc * 2 * accel / self._dt
                    grad[k-1] -= w_acc * 2 * accel / self._dt

                # Obstacle avoidance
                for obs in self._obstacles:
                    obs_pos = np.array(obs.get('position', [0,0,0]))
                    diff = states[k] - obs_pos
                    dist = np.linalg.norm(diff)
                    if dist < clearance * 2 and dist > 0.01:
                        grad[k] -= w_obs * diff / (dist ** 3) * self._dt

            u -= learning_rate * grad
            # Clamp
            for k in range(self._N):
                speed = np.linalg.norm(u[k, :2])
                if speed > self._max_v_xy:
                    u[k, :2] *= self._max_v_xy / speed
                u[k, 2] = np.clip(u[k, 2], -self._max_v_z, self._max_v_z)

        return u[0]

    def _rollout(self, pos, controls):
        """Forward-simulate drone states given control inputs."""
        states = []
        state = pos.copy()
        for k in range(self._N):
            state = state + controls[k] * self._dt
            states.append(state.copy())
        return states

    def _solve_casadi(self, pos, ref_trajectory):
        """CasADi-based optimal MPC solver."""
        import casadi as ca

        opti = ca.Opti()
        X = opti.variable(self._N + 1, 3)
        U = opti.variable(self._N, 3)

        # Initial condition
        opti.subject_to(X[0, :] == pos)

        # Dynamics
        for k in range(self._N):
            opti.subject_to(X[k+1, :] == X[k, :] + U[k, :] * self._dt)

        # Velocity constraints
        for k in range(self._N):
            opti.subject_to(U[k, 0]**2 + U[k, 1]**2 <= self._max_v_xy**2)
            opti.subject_to(ca.fabs(U[k, 2]) <= self._max_v_z)

        # Cost function
        cost = 0
        w_g = self.get_parameter('goal_weight').value
        w_v = self.get_parameter('velocity_weight').value
        for k in range(self._N):
            ref = ref_trajectory[min(k, len(ref_trajectory)-1)]
            err = X[k+1, :] - ref
            cost += w_g * ca.dot(err, err)
            cost += w_v * ca.dot(U[k, :], U[k, :])

        opti.minimize(cost)
        opti.solver('ipopt', {'print_time': False}, {'print_level': 0, 'max_iter': 50})

        try:
            sol = opti.solve()
            return np.array(sol.value(U[0, :])).flatten()
        except Exception:
            return self._solve_gradient(pos, ref_trajectory)

    def _publish_zero_vel(self):
        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = 'map'
        self._pub_vel.publish(vel_msg)

    def _diag(self):
        avg = np.mean(self._solve_times) if self._solve_times else 0
        self.get_logger().info(
            f'MPC Local Planner — Avg solve: {avg:.1f}ms | '
            f'Waypoint: {self._current_waypoint_idx}/{len(self._global_path)}')


def main(args=None):
    rclpy.init(args=args)
    node = MPCLocalPlannerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
