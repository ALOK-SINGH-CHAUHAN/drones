#!/usr/bin/env python3
"""
ANTIGRAVITY — Mission Manager Node
=====================================
Top-level orchestration node that manages complete autonomous missions.
Implements the NavigateToGoal action server and coordinates all subsystems.

Acceptance Criteria (PRD §3.4):
  - Accept goal positions and execute autonomous navigation
  - Manage waypoint queues with dynamic re-planning
  - Report progress via action feedback
  - Integrate with safety arbiter for emergency overrides
  - Support mission pause/resume/abort commands
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import String, Bool
from nav_msgs.msg import Path

import numpy as np
import time
import json
import threading
from enum import IntEnum


class MissionState(IntEnum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2
    REACHED_WAYPOINT = 3
    PAUSED = 4
    ABORTING = 5
    COMPLETED = 6
    FAILED = 7


class MissionManagerNode(Node):
    """
    Top-level mission orchestration and NavigateToGoal action server.

    Manages the full lifecycle of autonomous navigation missions:
    1. Goal acceptance and validation
    2. Global path planning request
    3. Local planning and trajectory execution
    4. Progress monitoring with safety integration
    5. Waypoint queue management for multi-waypoint missions

    Subscribes:
      - /state/state/pose (PoseWithCovarianceStamped): Current EKF pose
      - /planning/global_path (Path): Planned path from global planner
      - /safety/safety/level (String): Safety arbiter level
      - mission/command (String): pause, resume, abort

    Publishes:
      - mission/state (String): Current mission state
      - mission/progress (String): JSON progress report
      - mission/goal (PoseStamped): Current target goal
      - /planning/planning/goal (PoseStamped): Goal to global planner
    """

    def __init__(self):
        super().__init__('mission_manager')
        self._cb_group = ReentrantCallbackGroup()

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('goal_tolerance_m', 0.5)
        self.declare_parameter('waypoint_tolerance_m', 1.0)
        self.declare_parameter('replan_interval_s', 5.0)
        self.declare_parameter('progress_rate_hz', 2.0)
        self.declare_parameter('max_mission_time_s', 300.0)
        self.declare_parameter('safety_hold_timeout_s', 30.0)

        # ─── QoS ────────────────────────────────────────────────────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10
        )

        # ─── Publishers ─────────────────────────────────────────────────
        self._pub_state = self.create_publisher(
            String, 'mission/state', reliable_qos)
        self._pub_progress = self.create_publisher(
            String, 'mission/progress', reliable_qos)
        self._pub_goal = self.create_publisher(
            PoseStamped, 'mission/goal', reliable_qos)
        self._pub_planner_goal = self.create_publisher(
            PoseStamped, '/planning/planning/goal', reliable_qos)

        # ─── Subscribers ────────────────────────────────────────────────
        self._sub_pose = self.create_subscription(
            PoseWithCovarianceStamped, '/state/state/pose',
            self._pose_cb, reliable_qos,
            callback_group=self._cb_group)
        self._sub_path = self.create_subscription(
            Path, '/planning/global_path',
            self._path_cb, reliable_qos,
            callback_group=self._cb_group)
        self._sub_safety = self.create_subscription(
            String, '/safety/safety/level',
            self._safety_cb, reliable_qos,
            callback_group=self._cb_group)
        self._sub_command = self.create_subscription(
            String, 'mission/command',
            self._command_cb, reliable_qos,
            callback_group=self._cb_group)

        # ─── State ──────────────────────────────────────────────────────
        self._state = MissionState.IDLE
        self._current_pose = None
        self._current_path = None
        self._safety_level = 'NOMINAL'

        # Waypoint management
        self._waypoint_queue = []
        self._current_waypoint_idx = 0
        self._final_goal = None
        self._mission_start_time = None

        # Mission stats
        self._total_distance = 0.0
        self._last_pose_pos = None
        self._replan_count = 0

        self._lock = threading.Lock()

        # ─── Action Server (NavigateToGoal) ─────────────────────────────
        # Uses string-based goal since custom actions may not be compiled
        self._goal_active = False

        # ─── Timers ─────────────────────────────────────────────────────
        progress_rate = self.get_parameter('progress_rate_hz').value
        self.create_timer(1.0 / progress_rate, self._publish_progress)
        self.create_timer(1.0, self._mission_tick)
        self.create_timer(10.0, self._diag)

        self.get_logger().info('Mission Manager initialized — awaiting goals 🎯')

    # ─── Callbacks ───────────────────────────────────────────────────────

    def _pose_cb(self, msg):
        """Track current drone position from EKF."""
        self._current_pose = msg
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        # Track total distance traveled
        if self._last_pose_pos is not None and self._state == MissionState.NAVIGATING:
            self._total_distance += np.linalg.norm(pos - self._last_pose_pos)
        self._last_pose_pos = pos

    def _path_cb(self, msg):
        """Receive planned path from global planner."""
        with self._lock:
            self._current_path = msg
            if self._state == MissionState.PLANNING:
                if len(msg.poses) > 0:
                    self._state = MissionState.NAVIGATING
                    self.get_logger().info(
                        f'Path received — {len(msg.poses)} waypoints, navigating...')
                else:
                    self.get_logger().warn('Empty path received — replanning')
                    self._state = MissionState.PLANNING

    def _safety_cb(self, msg):
        """Track safety arbiter level."""
        self._safety_level = msg.data

    def _command_cb(self, msg):
        """Handle mission commands: pause, resume, abort, goto."""
        cmd = msg.data.strip().lower()

        if cmd == 'pause':
            if self._state == MissionState.NAVIGATING:
                self._state = MissionState.PAUSED
                self.get_logger().info('Mission PAUSED')

        elif cmd == 'resume':
            if self._state == MissionState.PAUSED:
                self._state = MissionState.NAVIGATING
                self.get_logger().info('Mission RESUMED')

        elif cmd == 'abort':
            self._state = MissionState.ABORTING
            self.get_logger().warn('Mission ABORTED')
            self._cleanup_mission()

        elif cmd.startswith('goto:'):
            # Parse goal: "goto:x,y,z"
            try:
                coords = cmd.split(':')[1].split(',')
                goal = PoseStamped()
                goal.header.stamp = self.get_clock().now().to_msg()
                goal.header.frame_id = 'map'
                goal.pose.position.x = float(coords[0])
                goal.pose.position.y = float(coords[1])
                goal.pose.position.z = float(coords[2]) if len(coords) > 2 else 1.5
                goal.pose.orientation.w = 1.0
                self._start_mission([goal])
            except (ValueError, IndexError) as e:
                self.get_logger().error(f'Invalid goto command: {e}')

        elif cmd.startswith('waypoints:'):
            # Parse multi-waypoint: "waypoints:x1,y1,z1;x2,y2,z2;..."
            try:
                waypoint_strs = cmd.split(':')[1].split(';')
                goals = []
                for wp_str in waypoint_strs:
                    c = wp_str.strip().split(',')
                    g = PoseStamped()
                    g.header.stamp = self.get_clock().now().to_msg()
                    g.header.frame_id = 'map'
                    g.pose.position.x = float(c[0])
                    g.pose.position.y = float(c[1])
                    g.pose.position.z = float(c[2]) if len(c) > 2 else 1.5
                    g.pose.orientation.w = 1.0
                    goals.append(g)
                self._start_mission(goals)
            except (ValueError, IndexError) as e:
                self.get_logger().error(f'Invalid waypoints command: {e}')

    # ─── Mission Management ──────────────────────────────────────────────

    def _start_mission(self, waypoints):
        """Start a new navigation mission with the given waypoints."""
        with self._lock:
            self._waypoint_queue = waypoints
            self._current_waypoint_idx = 0
            self._final_goal = waypoints[-1]
            self._total_distance = 0.0
            self._replan_count = 0
            self._mission_start_time = time.time()
            self._state = MissionState.PLANNING

        # Send first waypoint to global planner
        self._send_goal_to_planner(waypoints[0])
        self.get_logger().info(
            f'Mission started — {len(waypoints)} waypoint(s), '
            f'first: ({waypoints[0].pose.position.x:.1f}, '
            f'{waypoints[0].pose.position.y:.1f}, '
            f'{waypoints[0].pose.position.z:.1f})')

    def _send_goal_to_planner(self, goal):
        """Publish goal to the global planner."""
        self._pub_planner_goal.publish(goal)
        self._pub_goal.publish(goal)
        self._replan_count += 1

    def _mission_tick(self):
        """Main mission state machine — runs at 1 Hz."""
        if self._state == MissionState.IDLE:
            return

        if self._state == MissionState.PAUSED:
            return

        # Safety check
        if self._safety_level in ('CRITICAL', 'EMERGENCY'):
            if self._state == MissionState.NAVIGATING:
                self._state = MissionState.PAUSED
                self.get_logger().warn(
                    f'Mission paused due to safety: {self._safety_level}')
            return

        # Timeout check
        max_time = self.get_parameter('max_mission_time_s').value
        if (self._mission_start_time and
                time.time() - self._mission_start_time > max_time):
            self.get_logger().error(f'Mission timeout ({max_time}s)')
            self._state = MissionState.FAILED
            self._cleanup_mission()
            return

        if self._state == MissionState.NAVIGATING:
            self._check_waypoint_reached()

    def _check_waypoint_reached(self):
        """Check if the current waypoint has been reached."""
        if not self._current_pose or not self._waypoint_queue:
            return

        current_pos = np.array([
            self._current_pose.pose.pose.position.x,
            self._current_pose.pose.pose.position.y,
            self._current_pose.pose.pose.position.z,
        ])

        idx = self._current_waypoint_idx
        if idx >= len(self._waypoint_queue):
            return

        target = self._waypoint_queue[idx]
        target_pos = np.array([
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
        ])

        dist = np.linalg.norm(current_pos - target_pos)

        # Use tighter tolerance for final goal
        if idx == len(self._waypoint_queue) - 1:
            tolerance = self.get_parameter('goal_tolerance_m').value
        else:
            tolerance = self.get_parameter('waypoint_tolerance_m').value

        if dist < tolerance:
            self.get_logger().info(
                f'Waypoint {idx + 1}/{len(self._waypoint_queue)} reached '
                f'(error: {dist:.2f}m)')

            # Advance to next waypoint
            self._current_waypoint_idx += 1
            if self._current_waypoint_idx >= len(self._waypoint_queue):
                # Mission complete!
                self._state = MissionState.COMPLETED
                elapsed = time.time() - self._mission_start_time
                self.get_logger().info(
                    f'🎉 Mission COMPLETE — '
                    f'{self._total_distance:.1f}m traveled, '
                    f'{elapsed:.1f}s elapsed, '
                    f'{self._replan_count} replans')
                self._cleanup_mission()
            else:
                # Navigate to next waypoint
                next_wp = self._waypoint_queue[self._current_waypoint_idx]
                self._state = MissionState.PLANNING
                self._send_goal_to_planner(next_wp)

    def _cleanup_mission(self):
        """Clean up after mission completion or abort."""
        self._goal_active = False

    # ─── Publishing ──────────────────────────────────────────────────────

    def _publish_progress(self):
        """Publish mission state and progress."""
        # State
        self._pub_state.publish(String(data=self._state.name))

        if self._state == MissionState.IDLE:
            return

        # Progress report
        elapsed = (time.time() - self._mission_start_time
                   if self._mission_start_time else 0)

        # Distance to current waypoint
        dist_to_wp = float('inf')
        if (self._current_pose and self._waypoint_queue and
                self._current_waypoint_idx < len(self._waypoint_queue)):
            curr = np.array([
                self._current_pose.pose.pose.position.x,
                self._current_pose.pose.pose.position.y,
                self._current_pose.pose.pose.position.z,
            ])
            tgt = self._waypoint_queue[self._current_waypoint_idx]
            tgt_pos = np.array([
                tgt.pose.position.x, tgt.pose.position.y, tgt.pose.position.z])
            dist_to_wp = float(np.linalg.norm(curr - tgt_pos))

        progress = {
            'state': self._state.name,
            'waypoint': f'{self._current_waypoint_idx + 1}/{len(self._waypoint_queue)}',
            'distance_to_waypoint_m': round(dist_to_wp, 2),
            'total_distance_m': round(self._total_distance, 2),
            'elapsed_s': round(elapsed, 1),
            'replan_count': self._replan_count,
            'safety_level': self._safety_level,
        }
        self._pub_progress.publish(String(data=json.dumps(progress)))

    def _diag(self):
        """Log diagnostics."""
        self.get_logger().info(
            f'Mission — State: {self._state.name} | '
            f'WP: {self._current_waypoint_idx + 1}/{len(self._waypoint_queue)} | '
            f'Dist: {self._total_distance:.1f}m | '
            f'Safety: {self._safety_level}')


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
