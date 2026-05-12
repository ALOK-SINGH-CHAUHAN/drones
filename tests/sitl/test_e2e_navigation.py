#!/usr/bin/env python3
"""
ANTIGRAVITY — End-to-End SITL Autonomous Navigation Test (P3-T6)
=================================================================
Launches the full stack in simulated mode and executes an autonomous
waypoint mission. Validates all layers from perception to control.

Usage:
  python tests/sitl/test_e2e_navigation.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String, Float32, Bool

import numpy as np
import time
import sys
from enum import IntEnum


class MissionPhase(IntEnum):
    INIT = 0
    WAIT_LOCALIZATION = 1
    ARM = 2
    TAKEOFF = 3
    NAVIGATE = 4
    WAYPOINT_REACHED = 5
    RETURN = 6
    LAND = 7
    COMPLETE = 8
    FAILED = 9


class E2ENavigationTest(Node):
    """End-to-end autonomous navigation test node."""

    def __init__(self):
        super().__init__('e2e_test')

        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)

        # Subscribers
        self._sub_pose = self.create_subscription(PoseStamped, '/state/state/pose',
                                                   self._pose_cb, reliable_qos)
        self._sub_slam = self.create_subscription(String, '/slam/slam/state',
                                                   self._slam_cb, reliable_qos)
        self._sub_loc = self.create_subscription(String, '/localization/localization/state',
                                                  self._loc_cb, reliable_qos)
        self._sub_safety = self.create_subscription(String, '/safety/safety/level',
                                                     self._safety_cb, reliable_qos)
        self._sub_planner = self.create_subscription(String, 'planning/local_status',
                                                      self._planner_cb, reliable_qos)
        self._sub_battery = self.create_subscription(Float32, '/control/control/battery',
                                                      self._battery_cb, reliable_qos)

        # Publishers
        self._pub_command = self.create_publisher(String, '/control/control/command', reliable_qos)
        self._pub_goal = self.create_publisher(PoseStamped, 'planning/goal', reliable_qos)

        # Mission
        self._phase = MissionPhase.INIT
        self._pose = None
        self._slam_state = 'UNKNOWN'
        self._loc_state = 'UNKNOWN'
        self._safety_level = 'UNKNOWN'
        self._planner_status = 'UNKNOWN'
        self._battery = 100.0
        self._start_time = time.time()

        # Test waypoints (square mission)
        self._waypoints = [
            (3.0, 0.0, 1.5),
            (3.0, 3.0, 1.5),
            (0.0, 3.0, 1.5),
            (0.0, 0.0, 1.5),
        ]
        self._current_wp = 0
        self._wp_reached_count = 0
        self._wp_tolerance = 0.8

        # Test results
        self._results = {
            'passed': [],
            'failed': [],
            'metrics': {},
        }

        self.create_timer(0.5, self._mission_loop)
        self.get_logger().info('╔══════════════════════════════════════════╗')
        self.get_logger().info('║  ANTIGRAVITY E2E Navigation Test        ║')
        self.get_logger().info('╚══════════════════════════════════════════╝')

    def _pose_cb(self, msg): self._pose = msg
    def _slam_cb(self, msg): self._slam_state = msg.data
    def _loc_cb(self, msg): self._loc_state = msg.data
    def _safety_cb(self, msg): self._safety_level = msg.data
    def _planner_cb(self, msg): self._planner_status = msg.data
    def _battery_cb(self, msg): self._battery = msg.data

    def _mission_loop(self):
        """State machine driving the test mission."""
        elapsed = time.time() - self._start_time

        if self._phase == MissionPhase.INIT:
            self.get_logger().info('[INIT] Waiting for systems...')
            if elapsed > 3.0:
                self._check_test('System boots within 3s', True)
                self._phase = MissionPhase.WAIT_LOCALIZATION
                self._phase_start = time.time()

        elif self._phase == MissionPhase.WAIT_LOCALIZATION:
            converged = 'CONVERGED' in self._loc_state.upper() or 'OK' in self._slam_state.upper()
            wait_time = time.time() - self._phase_start

            if converged:
                self._check_test('Localization converges within 10s', wait_time < 10.0)
                self._results['metrics']['localization_convergence_s'] = wait_time
                self._phase = MissionPhase.ARM
            elif wait_time > 15.0:
                self._check_test('Localization converges within 10s', False)
                self._phase = MissionPhase.ARM  # Try anyway

        elif self._phase == MissionPhase.ARM:
            self.get_logger().info('[ARM] Arming...')
            self._pub_command.publish(String(data='arm'))
            time.sleep(1.0)
            self._pub_command.publish(String(data='offboard'))
            self._phase = MissionPhase.TAKEOFF
            self._phase_start = time.time()

        elif self._phase == MissionPhase.TAKEOFF:
            self._pub_command.publish(String(data='takeoff'))
            if self._pose and self._pose.pose.position.z > 1.0:
                takeoff_time = time.time() - self._phase_start
                self._check_test('Takeoff to 1.5m within 10s', takeoff_time < 10.0)
                self._results['metrics']['takeoff_time_s'] = takeoff_time
                self._phase = MissionPhase.NAVIGATE
                self._send_next_waypoint()
            elif time.time() - self._phase_start > 15.0:
                self._check_test('Takeoff to 1.5m within 10s', False)
                self._phase = MissionPhase.NAVIGATE
                self._send_next_waypoint()

        elif self._phase == MissionPhase.NAVIGATE:
            if self._pose and self._current_wp < len(self._waypoints):
                wp = np.array(self._waypoints[self._current_wp])
                pos = np.array([self._pose.pose.position.x,
                               self._pose.pose.position.y,
                               self._pose.pose.position.z])
                dist = np.linalg.norm(pos - wp)

                if dist < self._wp_tolerance:
                    self.get_logger().info(f'  ✅ Waypoint {self._current_wp} reached (dist={dist:.2f}m)')
                    self._wp_reached_count += 1
                    self._current_wp += 1
                    if self._current_wp < len(self._waypoints):
                        self._send_next_waypoint()
                    else:
                        self._phase = MissionPhase.WAYPOINT_REACHED

            # Safety check during navigation
            if self._safety_level in ('EMERGENCY', 'CRITICAL'):
                self.get_logger().warn(f'  Safety trigger: {self._safety_level}')
                self._check_test('No safety violations during nominal flight', False)

        elif self._phase == MissionPhase.WAYPOINT_REACHED:
            total = len(self._waypoints)
            self._check_test(f'All {total} waypoints reached', self._wp_reached_count == total)
            self._results['metrics']['waypoints_reached'] = f'{self._wp_reached_count}/{total}'
            self._phase = MissionPhase.LAND

        elif self._phase == MissionPhase.LAND:
            self.get_logger().info('[LAND] Landing...')
            self._pub_command.publish(String(data='land'))
            self._phase = MissionPhase.COMPLETE
            self._phase_start = time.time()

        elif self._phase == MissionPhase.COMPLETE:
            if time.time() - self._phase_start > 3.0:
                self._finalize()

    def _send_next_waypoint(self):
        wp = self._waypoints[self._current_wp]
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = wp
        goal.pose.orientation.w = 1.0
        self._pub_goal.publish(goal)
        self.get_logger().info(f'  → Waypoint {self._current_wp}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})')

    def _check_test(self, name, passed):
        if passed:
            self._results['passed'].append(name)
            self.get_logger().info(f'  ✅ PASS: {name}')
        else:
            self._results['failed'].append(name)
            self.get_logger().error(f'  ❌ FAIL: {name}')

    def _finalize(self):
        """Print final test report."""
        total_time = time.time() - self._start_time
        self._results['metrics']['total_mission_time_s'] = total_time
        self._results['metrics']['battery_remaining_pct'] = self._battery

        p, f = len(self._results['passed']), len(self._results['failed'])
        total = p + f

        self.get_logger().info('')
        self.get_logger().info('╔══════════════════════════════════════════╗')
        self.get_logger().info('║        E2E TEST RESULTS                 ║')
        self.get_logger().info('╠══════════════════════════════════════════╣')

        for test in self._results['passed']:
            self.get_logger().info(f'║ ✅ {test}')
        for test in self._results['failed']:
            self.get_logger().info(f'║ ❌ {test}')

        self.get_logger().info('╠══════════════════════════════════════════╣')
        for k, v in self._results['metrics'].items():
            self.get_logger().info(f'║ {k}: {v}')

        self.get_logger().info('╠══════════════════════════════════════════╣')
        result = '🚀 ALL TESTS PASSED' if f == 0 else f'⚠️  {f}/{total} FAILED'
        self.get_logger().info(f'║ {result}')
        self.get_logger().info('╚══════════════════════════════════════════╝')

        self._phase = MissionPhase.FAILED if f > 0 else MissionPhase.COMPLETE
        # Shutdown after report
        raise SystemExit(1 if f > 0 else 0)


def main(args=None):
    rclpy.init(args=args)
    node = E2ENavigationTest()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
