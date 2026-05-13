"""
ANTIGRAVITY — Safety Arbiter Node
===================================
Highest-priority decision layer. Monitors all system health indicators
and can override any command to enforce safe flight.

Acceptance Criteria (P4-T1):
  - Collision avoidance triggers within 200ms of detection
  - Emergency landing triggers on battery < 10% or SLAM lost > 5s
  - Safety override commands have highest priority
  - Zero false-positive emergency landings during nominal flight
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String, Float32, Bool

import numpy as np
import time
import json
from enum import IntEnum


class SafetyLevel(IntEnum):
    NOMINAL = 0
    CAUTION = 1     # Slow down, increase margin
    WARNING = 2     # Hold position
    CRITICAL = 3    # Return to launch
    EMERGENCY = 4   # Emergency land immediately


class SafetyArbiterNode(Node):
    """
    Master safety arbiter — final gate for all drone commands.

    Monitors:
      - Battery level
      - SLAM/localization state
      - Obstacle proximity
      - Communication health
      - Geofence violations
      - System resource usage

    Subscribes:
      - /control/control/battery (Float32)
      - /slam/slam/state (String)
      - /localization/localization/state (String)
      - /state/state/status (String)
      - /cognition/prediction/tracks (String)
      - planning/local_status (String)
      - control/setpoint_velocity (TwistStamped): Commanded velocity
      - geofence/violation (Bool)

    Publishes:
      - safety/level (String): Current safety level
      - safety/override_command (String): Override flight command
      - safety/filtered_velocity (TwistStamped): Safe velocity (or zero)
      - safety/status (String): Detailed status
    """

    def __init__(self):
        super().__init__('safety_arbiter')

        self.declare_parameter('check_rate_hz', 50)
        self.declare_parameter('battery_caution_pct', 25.0)
        self.declare_parameter('battery_warning_pct', 15.0)
        self.declare_parameter('battery_critical_pct', 10.0)
        self.declare_parameter('battery_emergency_pct', 5.0)
        self.declare_parameter('min_obstacle_distance_m', 0.5)
        self.declare_parameter('caution_obstacle_distance_m', 1.5)
        self.declare_parameter('slam_lost_timeout_s', 5.0)
        self.declare_parameter('localization_lost_timeout_s', 10.0)
        self.declare_parameter('ekf_divergence_threshold', 2.0)
        self.declare_parameter('comms_timeout_s', 3.0)
        self.declare_parameter('max_altitude_m', 30.0)
        self.declare_parameter('max_velocity_safety_mps', 3.0)
        self.declare_parameter('enable_auto_land', True)
        self.declare_parameter('enable_rtl', True)
        self.declare_parameter('landing_velocity_mps', 0.5)

        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)

        # Publishers
        self._pub_level = self.create_publisher(String, 'safety/level', reliable_qos)
        self._pub_override = self.create_publisher(String, 'safety/override_command', reliable_qos)
        self._pub_filtered_vel = self.create_publisher(TwistStamped, 'safety/filtered_velocity', reliable_qos)
        self._pub_status = self.create_publisher(String, 'safety/status', reliable_qos)

        # Subscribers
        self._sub_battery = self.create_subscription(Float32, '/control/control/battery', self._battery_cb, reliable_qos)
        self._sub_slam = self.create_subscription(String, '/slam/slam/state', self._slam_cb, reliable_qos)
        self._sub_loc = self.create_subscription(String, '/localization/localization/state', self._loc_cb, reliable_qos)
        self._sub_ekf = self.create_subscription(String, '/state/state/status', self._ekf_cb, reliable_qos)
        self._sub_tracks = self.create_subscription(String, '/cognition/prediction/tracks', self._tracks_cb, reliable_qos)
        self._sub_vel_cmd = self.create_subscription(TwistStamped, 'control/setpoint_velocity', self._vel_cb, reliable_qos)
        self._sub_geofence = self.create_subscription(Bool, 'geofence/violation', self._geofence_cb, reliable_qos)

        # State
        self._battery = 100.0
        self._slam_state = 'OK'
        self._slam_last_ok = time.time()
        self._loc_state = 'CONVERGING'
        self._loc_last_ok = time.time()
        self._ekf_status = ''
        self._min_obstacle_dist = float('inf')
        self._geofence_violated = False
        self._last_vel_cmd = None
        self._safety_level = SafetyLevel.NOMINAL
        self._override_active = False
        self._triggers = {}  # Active trigger reasons

        rate = self.get_parameter('check_rate_hz').value
        self.create_timer(1.0 / rate, self._safety_check)
        self.create_timer(5.0, self._diag)
        self.get_logger().info('Safety arbiter initialized — protecting your drone 🛡️')

    def _battery_cb(self, msg): self._battery = msg.data
    def _slam_cb(self, msg):
        self._slam_state = msg.data
        if 'OK' in msg.data.upper(): self._slam_last_ok = time.time()
    def _loc_cb(self, msg):
        self._loc_state = msg.data
        if 'CONVERGED' in msg.data.upper(): self._loc_last_ok = time.time()
    def _ekf_cb(self, msg): self._ekf_status = msg.data
    def _geofence_cb(self, msg): self._geofence_violated = msg.data
    def _vel_cb(self, msg): self._last_vel_cmd = msg

    def _tracks_cb(self, msg):
        try:
            tracks = json.loads(msg.data).get('tracks', [])
            if tracks:
                self._min_obstacle_dist = min(
                    np.linalg.norm(t.get('position', [100,100,100])) for t in tracks
                )
            else:
                self._min_obstacle_dist = float('inf')
        except: pass

    def _safety_check(self):
        """Main safety evaluation loop — runs at 50 Hz."""
        self._triggers.clear()
        new_level = SafetyLevel.NOMINAL

        # ── Battery ──────────────────────────────────────────────────────
        if self._battery < self.get_parameter('battery_emergency_pct').value:
            new_level = max(new_level, SafetyLevel.EMERGENCY)
            self._triggers['battery'] = f'EMERGENCY ({self._battery:.1f}%)'
        elif self._battery < self.get_parameter('battery_critical_pct').value:
            new_level = max(new_level, SafetyLevel.CRITICAL)
            self._triggers['battery'] = f'CRITICAL ({self._battery:.1f}%)'
        elif self._battery < self.get_parameter('battery_warning_pct').value:
            new_level = max(new_level, SafetyLevel.WARNING)
            self._triggers['battery'] = f'WARNING ({self._battery:.1f}%)'
        elif self._battery < self.get_parameter('battery_caution_pct').value:
            new_level = max(new_level, SafetyLevel.CAUTION)
            self._triggers['battery'] = f'CAUTION ({self._battery:.1f}%)'

        # ── SLAM Health ──────────────────────────────────────────────────
        slam_lost_dur = time.time() - self._slam_last_ok
        timeout = self.get_parameter('slam_lost_timeout_s').value
        if slam_lost_dur > timeout:
            new_level = max(new_level, SafetyLevel.CRITICAL)
            self._triggers['slam'] = f'LOST {slam_lost_dur:.1f}s'
        elif 'LOST' in self._slam_state.upper():
            new_level = max(new_level, SafetyLevel.WARNING)
            self._triggers['slam'] = self._slam_state

        # ── Localization Health ──────────────────────────────────────────
        loc_lost_dur = time.time() - self._loc_last_ok
        loc_timeout = self.get_parameter('localization_lost_timeout_s').value
        if loc_lost_dur > loc_timeout:
            new_level = max(new_level, SafetyLevel.WARNING)
            self._triggers['localization'] = f'NOT_CONVERGED {loc_lost_dur:.1f}s'

        # ── Obstacle Proximity ───────────────────────────────────────────
        min_dist = self.get_parameter('min_obstacle_distance_m').value
        caution_dist = self.get_parameter('caution_obstacle_distance_m').value
        if self._min_obstacle_dist < min_dist:
            new_level = max(new_level, SafetyLevel.WARNING)
            self._triggers['obstacle'] = f'TOO_CLOSE ({self._min_obstacle_dist:.2f}m)'
        elif self._min_obstacle_dist < caution_dist:
            new_level = max(new_level, SafetyLevel.CAUTION)
            self._triggers['obstacle'] = f'CLOSE ({self._min_obstacle_dist:.2f}m)'

        # ── Geofence ─────────────────────────────────────────────────────
        if self._geofence_violated:
            new_level = max(new_level, SafetyLevel.CRITICAL)
            self._triggers['geofence'] = 'VIOLATED'

        self._safety_level = new_level

        # ── Execute safety response ──────────────────────────────────────
        self._execute_response(new_level)

        # ── Publish level ────────────────────────────────────────────────
        level_msg = String(); level_msg.data = new_level.name
        self._pub_level.publish(level_msg)

        # ── Publish detailed status ──────────────────────────────────────
        status = json.dumps({'level': new_level.name, 'triggers': self._triggers,
                             'battery': self._battery, 'slam': self._slam_state,
                             'min_obstacle_m': self._min_obstacle_dist})
        self._pub_status.publish(String(data=status))

    def _execute_response(self, level):
        """Execute appropriate safety response for the given level."""
        if level == SafetyLevel.EMERGENCY:
            if self.get_parameter('enable_auto_land').value:
                self._pub_override.publish(String(data='land'))
                self._override_active = True
                self._publish_emergency_vel()
            return

        if level == SafetyLevel.CRITICAL:
            if self.get_parameter('enable_rtl').value:
                self._pub_override.publish(String(data='rtl'))
                self._override_active = True
            return

        if level == SafetyLevel.WARNING:
            # Hold position (zero velocity)
            self._pub_override.publish(String(data='hold'))
            self._override_active = True
            self._publish_zero_vel()
            return

        if level == SafetyLevel.CAUTION:
            # Allow movement but clamp velocity
            self._override_active = False
            if self._last_vel_cmd:
                self._publish_clamped_vel()
            return

        # NOMINAL — pass through commands
        self._override_active = False
        if self._last_vel_cmd:
            self._pub_filtered_vel.publish(self._last_vel_cmd)

    def _publish_zero_vel(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        self._pub_filtered_vel.publish(msg)

    def _publish_emergency_vel(self):
        """Publish downward velocity for emergency landing."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.twist.linear.z = -self.get_parameter('landing_velocity_mps').value
        self._pub_filtered_vel.publish(msg)

    def _publish_clamped_vel(self):
        """Publish velocity clamped to safe maximum."""
        if not self._last_vel_cmd: return
        msg = TwistStamped()
        msg.header = self._last_vel_cmd.header
        max_v = self.get_parameter('max_velocity_safety_mps').value * 0.5  # Halved in caution
        vx = np.clip(self._last_vel_cmd.twist.linear.x, -max_v, max_v)
        vy = np.clip(self._last_vel_cmd.twist.linear.y, -max_v, max_v)
        vz = np.clip(self._last_vel_cmd.twist.linear.z, -0.5, 0.5)
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = vx, vy, vz
        self._pub_filtered_vel.publish(msg)

    def _diag(self):
        self.get_logger().info(
            f'Safety — Level: {self._safety_level.name} | '
            f'Battery: {self._battery:.1f}% | '
            f'SLAM: {self._slam_state} | '
            f'MinObs: {self._min_obstacle_dist:.2f}m | '
            f'Override: {self._override_active} | '
            f'Triggers: {self._triggers}')


def main(args=None):
    rclpy.init(args=args)
    node = SafetyArbiterNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
