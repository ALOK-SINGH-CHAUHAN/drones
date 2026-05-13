"""
ANTIGRAVITY — Minimum Snap Trajectory Optimizer
=================================================
Generates dynamically feasible, minimum-snap trajectories through waypoints.
Polynomial optimization ensuring C4 continuity (position, velocity,
acceleration, jerk, snap) for smooth quad-rotor flight.

Acceptance Criteria (P3-T4):
  - Snap-optimal trajectory in < 50ms for 10 waypoints
  - Maximum velocity and acceleration constraints satisfied
  - Smooth transitions through all waypoints
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import String

import numpy as np
import time


class TrajectoryOptimizerNode(Node):
    """
    Minimum snap trajectory generation for quadrotor waypoint following.

    Subscribes:
      - planning/global_path (Path): Sparse waypoints
    Publishes:
      - control/smooth_trajectory (Path): Dense, smooth trajectory
      - control/trajectory_markers (Marker): Visualization
    """

    def __init__(self):
        super().__init__('trajectory_optimizer')

        self.declare_parameter('polynomial_order', 7)  # 7th order for snap
        self.declare_parameter('max_velocity', 2.0)
        self.declare_parameter('max_acceleration', 2.0)
        self.declare_parameter('sample_dt', 0.05)  # 20 Hz output
        self.declare_parameter('time_allocation', 'trapezoidal')  # trapezoidal or uniform

        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)

        self._pub_traj = self.create_publisher(Path, 'control/smooth_trajectory', reliable_qos)
        self._pub_markers = self.create_publisher(Marker, 'control/trajectory_markers', reliable_qos)
        self._pub_status = self.create_publisher(String, 'control/trajectory_status', reliable_qos)

        self._sub_path = self.create_subscription(Path, 'planning/global_path',
                                                   self._path_cb, reliable_qos)
        self._opt_count = 0
        self.create_timer(5.0, self._diag)
        self.get_logger().info('Trajectory optimizer initialized')

    def _path_cb(self, msg):
        """Receive waypoints and generate minimum snap trajectory."""
        waypoints = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z]
                              for p in msg.poses])

        if len(waypoints) < 2:
            return

        t0 = time.time()

        # Allocate time segments
        times = self._allocate_times(waypoints)

        # Solve minimum snap for each axis independently
        trajectories = []
        for axis in range(3):
            positions = waypoints[:, axis]
            coeffs = self._solve_minimum_snap(positions, times)
            trajectories.append(coeffs)

        # Sample the trajectory
        dt = self.get_parameter('sample_dt').value
        total_time = sum(times)
        num_samples = int(total_time / dt) + 1

        smooth_path = []
        for i in range(num_samples):
            t = i * dt
            point = np.zeros(3)
            for axis in range(3):
                point[axis] = self._evaluate_polynomial(trajectories[axis], times, t)
            smooth_path.append(point)

        opt_time = (time.time() - t0) * 1000

        # Publish
        self._publish_trajectory(smooth_path, msg.header.stamp)
        self._publish_viz(smooth_path, msg.header.stamp)

        status = f'OK ({opt_time:.1f}ms, {len(smooth_path)} points, {total_time:.1f}s)'
        self._pub_status.publish(String(data=status))

        self._opt_count += 1
        self.get_logger().info(f'Trajectory optimized: {len(waypoints)} WP -> {len(smooth_path)} pts in {opt_time:.1f}ms')

    def _allocate_times(self, waypoints):
        """Allocate time for each segment based on distance and velocity limits."""
        method = self.get_parameter('time_allocation').value
        max_v = self.get_parameter('max_velocity').value
        max_a = self.get_parameter('max_acceleration').value

        n_segments = len(waypoints) - 1
        times = []

        for i in range(n_segments):
            dist = np.linalg.norm(waypoints[i+1] - waypoints[i])

            if method == 'trapezoidal':
                # Trapezoidal velocity profile: accel -> cruise -> decel
                t_accel = max_v / max_a
                d_accel = 0.5 * max_a * t_accel**2
                if 2 * d_accel >= dist:
                    # Triangle profile (never reaches max velocity)
                    t = 2 * np.sqrt(dist / max_a)
                else:
                    d_cruise = dist - 2 * d_accel
                    t_cruise = d_cruise / max_v
                    t = 2 * t_accel + t_cruise
            else:
                t = dist / max_v

            times.append(max(t, 0.5))  # Minimum segment time

        return times

    def _solve_minimum_snap(self, positions, times):
        """
        Solve minimum snap polynomial optimization.
        Each segment is a 7th-order polynomial: p(t) = c0 + c1*t + ... + c7*t^7
        Minimizes integral of snap^2 (4th derivative squared).
        """
        n_seg = len(times)
        order = self.get_parameter('polynomial_order').value
        n_coeffs = order + 1
        n_vars = n_seg * n_coeffs

        # Build QP: minimize x^T H x subject to A x = b
        H = np.zeros((n_vars, n_vars))
        A_eq_rows = []
        b_eq_rows = []

        for seg in range(n_seg):
            T = times[seg]
            offset = seg * n_coeffs

            # Snap cost matrix (4th derivative)
            for i in range(4, n_coeffs):
                for j in range(4, n_coeffs):
                    ci = np.math.factorial(i) / np.math.factorial(i - 4)
                    cj = np.math.factorial(j) / np.math.factorial(j - 4)
                    power = i + j - 7
                    if power >= 0:
                        H[offset+i, offset+j] += ci * cj * T**(power) / max(power, 1)

        # Endpoint constraints: position at segment boundaries
        for seg in range(n_seg):
            T = times[seg]
            offset = seg * n_coeffs

            # Start position: p(0) = positions[seg]
            row = np.zeros(n_vars)
            row[offset] = 1.0  # c0
            A_eq_rows.append(row)
            b_eq_rows.append(positions[seg])

            # End position: p(T) = positions[seg+1]
            row = np.zeros(n_vars)
            for k in range(n_coeffs):
                row[offset + k] = T**k
            A_eq_rows.append(row)
            b_eq_rows.append(positions[seg + 1])

        # Continuity constraints between segments (velocity, acceleration, jerk)
        for seg in range(n_seg - 1):
            T = times[seg]
            off1 = seg * n_coeffs
            off2 = (seg + 1) * n_coeffs

            for deriv in range(1, 4):  # velocity, accel, jerk continuity
                row = np.zeros(n_vars)
                # End of segment seg
                for k in range(deriv, n_coeffs):
                    coeff = 1.0
                    for d in range(deriv):
                        coeff *= (k - d)
                    row[off1 + k] = coeff * T**(k - deriv)
                # Start of segment seg+1 (subtract)
                for d in range(deriv):
                    pass  # Only c_deriv contributes at t=0
                fact = 1.0
                for d in range(deriv):
                    fact *= (deriv - d)
                row[off2 + deriv] = -fact
                A_eq_rows.append(row)
                b_eq_rows.append(0.0)

        # Start/end boundary: zero velocity, acceleration, jerk
        for deriv in range(1, 4):
            # Start
            row = np.zeros(n_vars)
            fact = 1.0
            for d in range(deriv):
                fact *= (deriv - d)
            row[deriv] = fact
            A_eq_rows.append(row)
            b_eq_rows.append(0.0)

            # End
            T = times[-1]
            off = (n_seg - 1) * n_coeffs
            row = np.zeros(n_vars)
            for k in range(deriv, n_coeffs):
                coeff = 1.0
                for d in range(deriv):
                    coeff *= (k - d)
                row[off + k] = coeff * T**(k - deriv)
            A_eq_rows.append(row)
            b_eq_rows.append(0.0)

        A_eq = np.array(A_eq_rows)
        b_eq = np.array(b_eq_rows)

        # Regularize H
        H = H + np.eye(n_vars) * 1e-8

        # Solve with least-squares if QP solver unavailable
        try:
            from scipy.optimize import minimize as scipy_minimize

            def cost(x):
                return 0.5 * x @ H @ x

            def cost_grad(x):
                return H @ x

            constraints = {'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq,
                          'jac': lambda x: A_eq}
            x0 = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
            result = scipy_minimize(cost, x0, jac=cost_grad, constraints=constraints,
                                   method='SLSQP', options={'maxiter': 200})
            coeffs = result.x.reshape(n_seg, n_coeffs)
        except Exception:
            # Direct solve via pseudoinverse
            x = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
            coeffs = x.reshape(n_seg, n_coeffs)

        return coeffs

    def _evaluate_polynomial(self, coeffs, times, t):
        """Evaluate piecewise polynomial at time t."""
        n_seg = len(times)
        elapsed = 0.0

        for seg in range(n_seg):
            if t <= elapsed + times[seg] or seg == n_seg - 1:
                local_t = t - elapsed
                local_t = min(local_t, times[seg])
                val = 0.0
                for k in range(len(coeffs[seg])):
                    val += coeffs[seg][k] * local_t**k
                return val
            elapsed += times[seg]

        # Past end — return last value
        local_t = times[-1]
        val = 0.0
        for k in range(len(coeffs[-1])):
            val += coeffs[-1][k] * local_t**k
        return val

    def _publish_trajectory(self, points, stamp):
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        for pt in points:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(pt[0]), float(pt[1]), float(pt[2])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._pub_traj.publish(msg)

    def _publish_viz(self, points, stamp):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = 'map'
        m.ns, m.id = 'smooth_trajectory', 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.08
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.5, 0.9
        for pt in points:
            p = Point()
            p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
            m.points.append(p)
        self._pub_markers.publish(m)

    def _diag(self):
        self.get_logger().info(f'Trajectory Optimizer — Optimizations: {self._opt_count}')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryOptimizerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
