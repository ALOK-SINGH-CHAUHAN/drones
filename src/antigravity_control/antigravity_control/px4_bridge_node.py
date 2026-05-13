"""
ANTIGRAVITY — PX4 MAVLink Bridge Node
=======================================
ROS2 node bridging the flight control system (PX4/ArduPilot)
via MAVLink protocol. Provides offboard position control and telemetry.

Acceptance Criteria:
  - Position hold error <= 0.15m in still air
  - Step response settling time <= 1.5s for 1m position change
  - No oscillation in hover
  - Heartbeat and failsafe management
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from std_msgs.msg import String, Float32, Bool

import numpy as np
import time
import threading
from enum import IntEnum


class FlightMode(IntEnum):
    """PX4 flight modes."""
    MANUAL = 0
    ALTITUDE = 1
    POSITION = 2
    OFFBOARD = 3
    RETURN = 4
    LAND = 5
    TAKEOFF = 6


class ArmState(IntEnum):
    DISARMED = 0
    ARMED = 1


class PX4BridgeNode(Node):
    """
    PX4/ArduPilot MAVLink bridge for offboard drone control.
    
    Connects to Pixhawk flight controller via MAVLink (pymavlink/MAVSDK).
    Provides ROS2 interface for position setpoints, velocity commands,
    arming, mode switching, and telemetry streaming.
    
    Subscribes:
      - control/setpoint_position (geometry_msgs/PoseStamped)
      - control/setpoint_velocity (geometry_msgs/TwistStamped)
      - control/command (std_msgs/String): arm, disarm, takeoff, land, rtl
    
    Publishes:
      - control/state (std_msgs/String): Current flight mode
      - control/battery (std_msgs/Float32): Battery percentage
      - control/armed (std_msgs/Bool): Armed state
      - control/local_position (geometry_msgs/PoseStamped): Current position
    """

    def __init__(self):
        super().__init__('px4_bridge')

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('fcu_url', '/dev/ttyACM0:921600')
        self.declare_parameter('gcs_url', 'udp://:14550@')
        self.declare_parameter('target_system_id', 1)
        self.declare_parameter('target_component_id', 1)
        self.declare_parameter('simulation_mode', False)
        self.declare_parameter('offboard_mode_enabled', True)
        self.declare_parameter('setpoint_rate_hz', 50)
        self.declare_parameter('offboard_timeout_s', 0.5)
        self.declare_parameter('max_velocity_xy_mps', 2.0)
        self.declare_parameter('max_velocity_z_mps', 1.0)
        self.declare_parameter('max_acceleration_mps2', 2.0)
        self.declare_parameter('position_smoothing_enabled', True)
        self.declare_parameter('smoothing_factor', 0.8)
        self.declare_parameter('auto_arm', False)
        self.declare_parameter('takeoff_altitude_m', 1.5)
        self.declare_parameter('landing_speed_mps', 0.5)
        self.declare_parameter('heartbeat_rate_hz', 2.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('body_frame', 'base_link')

        self._sim_mode = self.get_parameter('simulation_mode').value
        self._setpoint_rate = self.get_parameter('setpoint_rate_hz').value
        self._max_vel_xy = self.get_parameter('max_velocity_xy_mps').value
        self._max_vel_z = self.get_parameter('max_velocity_z_mps').value
        self._takeoff_alt = self.get_parameter('takeoff_altitude_m').value

        # ─── QoS ────────────────────────────────────────────────────────
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ─── Publishers ─────────────────────────────────────────────────
        self._pub_state = self.create_publisher(String, 'control/state', reliable_qos)
        self._pub_battery = self.create_publisher(Float32, 'control/battery', reliable_qos)
        self._pub_armed = self.create_publisher(Bool, 'control/armed', reliable_qos)
        self._pub_position = self.create_publisher(
            PoseStamped, 'control/local_position', reliable_qos
        )

        # ─── Subscribers ────────────────────────────────────────────────
        self._sub_setpoint_pos = self.create_subscription(
            PoseStamped, 'control/setpoint_position',
            self._setpoint_position_callback, reliable_qos
        )
        self._sub_setpoint_vel = self.create_subscription(
            TwistStamped, 'control/setpoint_velocity',
            self._setpoint_velocity_callback, reliable_qos
        )
        self._sub_command = self.create_subscription(
            String, 'control/command',
            self._command_callback, reliable_qos
        )

        # ─── State ──────────────────────────────────────────────────────
        self._mavlink_conn = None
        self._flight_mode = FlightMode.MANUAL
        self._arm_state = ArmState.DISARMED
        self._battery_percent = 100.0
        self._current_position = np.array([0.0, 0.0, 0.0])
        self._current_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion xyzw
        self._target_position = np.array([0.0, 0.0, self._takeoff_alt])
        self._target_yaw = 0.0
        self._last_setpoint_time = time.time()
        self._connected = False

        # Position smoothing
        self._smoothed_setpoint = np.array([0.0, 0.0, self._takeoff_alt])
        self._smoothing = self.get_parameter('smoothing_factor').value

        # ─── Initialize MAVLink ──────────────────────────────────────────
        self._init_mavlink()

        # ─── Timers ──────────────────────────────────────────────────────
        # Setpoint stream (must be continuous for offboard mode)
        self._setpoint_timer = self.create_timer(
            1.0 / self._setpoint_rate, self._send_setpoint
        )

        # Heartbeat
        hb_rate = self.get_parameter('heartbeat_rate_hz').value
        self._heartbeat_timer = self.create_timer(1.0 / hb_rate, self._send_heartbeat)

        # Telemetry polling
        self._telemetry_timer = self.create_timer(0.1, self._poll_telemetry)

        # Diagnostics
        self._diag_timer = self.create_timer(5.0, self._diagnostics_callback)

        self.get_logger().info(
            f'PX4 bridge initialized — '
            f'mode: {"SIM" if self._sim_mode else "HARDWARE"} | '
            f'setpoint rate: {self._setpoint_rate} Hz'
        )

    def _init_mavlink(self):
        """Initialize MAVLink connection to PX4/ArduPilot."""
        fcu_url = self.get_parameter('fcu_url').value

        try:
            from pymavlink import mavutil

            self._mavlink_conn = mavutil.mavlink_connection(fcu_url)
            self._mavlink_conn.wait_heartbeat(timeout=10)
            self._connected = True

            self.get_logger().info(
                f'MAVLink connected to {fcu_url} — '
                f'System: {self._mavlink_conn.target_system}, '
                f'Component: {self._mavlink_conn.target_component}'
            )

        except ImportError:
            self.get_logger().warn(
                'pymavlink not installed. Using simulated flight controller. '
                'Install with: pip install pymavlink'
            )
            self._connected = False

        except Exception as e:
            self.get_logger().error(f'MAVLink connection failed: {e}')
            self._connected = False

    def _setpoint_position_callback(self, msg):
        """Handle position setpoint command."""
        self._target_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

        # Enforce altitude and speed limits
        self._target_position[2] = max(0.3, self._target_position[2])

        # Extract yaw from quaternion
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self._target_yaw = np.arctan2(siny_cosp, cosy_cosp)

        self._last_setpoint_time = time.time()

    def _setpoint_velocity_callback(self, msg):
        """Handle velocity setpoint command."""
        vx = np.clip(msg.twist.linear.x, -self._max_vel_xy, self._max_vel_xy)
        vy = np.clip(msg.twist.linear.y, -self._max_vel_xy, self._max_vel_xy)
        vz = np.clip(msg.twist.linear.z, -self._max_vel_z, self._max_vel_z)

        # Integrate velocity into position target
        dt = 1.0 / self._setpoint_rate
        self._target_position += np.array([vx, vy, vz]) * dt
        self._target_position[2] = max(0.3, self._target_position[2])

        self._last_setpoint_time = time.time()

    def _command_callback(self, msg):
        """Handle high-level flight commands."""
        command = msg.data.lower().strip()

        if command == 'arm':
            self._arm()
        elif command == 'disarm':
            self._disarm()
        elif command == 'takeoff':
            self._takeoff()
        elif command == 'land':
            self._land()
        elif command == 'rtl':
            self._return_to_launch()
        elif command == 'offboard':
            self._set_offboard_mode()
        else:
            self.get_logger().warn(f'Unknown command: {command}')

    def _arm(self):
        """Arm the drone motors."""
        if self._connected and self._mavlink_conn:
            self._mavlink_conn.arducopter_arm()
            self.get_logger().info('ARM command sent')
        self._arm_state = ArmState.ARMED

    def _disarm(self):
        """Disarm the drone motors."""
        if self._connected and self._mavlink_conn:
            self._mavlink_conn.arducopter_disarm()
            self.get_logger().info('DISARM command sent')
        self._arm_state = ArmState.DISARMED

    def _takeoff(self):
        """Execute takeoff to configured altitude."""
        self._target_position[2] = self._takeoff_alt
        self._flight_mode = FlightMode.TAKEOFF
        self.get_logger().info(f'Takeoff to {self._takeoff_alt}m')

        if self._connected and self._mavlink_conn:
            from pymavlink import mavutil
            self._mavlink_conn.mav.command_long_send(
                self._mavlink_conn.target_system,
                self._mavlink_conn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, self._takeoff_alt
            )

    def _land(self):
        """Execute landing."""
        self._flight_mode = FlightMode.LAND
        self.get_logger().info('Landing initiated')

        if self._connected and self._mavlink_conn:
            from pymavlink import mavutil
            self._mavlink_conn.mav.command_long_send(
                self._mavlink_conn.target_system,
                self._mavlink_conn.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0, 0, 0, 0, 0, 0, 0, 0
            )

    def _return_to_launch(self):
        """Return to launch position."""
        self._flight_mode = FlightMode.RETURN
        self.get_logger().info('Return to launch')

    def _set_offboard_mode(self):
        """Switch to offboard control mode."""
        self._flight_mode = FlightMode.OFFBOARD
        self.get_logger().info('Offboard mode enabled')

        if self._connected and self._mavlink_conn:
            from pymavlink import mavutil
            self._mavlink_conn.set_mode_apm(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6)

    def _send_setpoint(self):
        """Send continuous position setpoints (required for offboard mode)."""
        # Apply position smoothing
        if self.get_parameter('position_smoothing_enabled').value:
            alpha = self._smoothing
            self._smoothed_setpoint = (
                alpha * self._smoothed_setpoint +
                (1 - alpha) * self._target_position
            )
            setpoint = self._smoothed_setpoint
        else:
            setpoint = self._target_position

        if self._connected and self._mavlink_conn:
            from pymavlink import mavutil

            self._mavlink_conn.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms
                self._mavlink_conn.target_system,
                self._mavlink_conn.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,  # Position only
                setpoint[0], setpoint[1], -setpoint[2],  # NED (z inverted)
                0, 0, 0,  # velocity
                0, 0, 0,  # acceleration
                self._target_yaw, 0  # yaw, yaw_rate
            )
        else:
            # Simulated position update
            move_speed = self._max_vel_xy / self._setpoint_rate
            delta = self._target_position - self._current_position
            dist = np.linalg.norm(delta)
            if dist > 0.01:
                step = min(move_speed, dist)
                self._current_position += (delta / dist) * step

    def _send_heartbeat(self):
        """Send MAVLink heartbeat to maintain connection."""
        if self._connected and self._mavlink_conn:
            from pymavlink import mavutil
            self._mavlink_conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )

    def _poll_telemetry(self):
        """Poll and publish telemetry data."""
        stamp = self.get_clock().now().to_msg()

        if self._connected and self._mavlink_conn:
            # Non-blocking message receive
            msg = self._mavlink_conn.recv_match(blocking=False)
            if msg:
                msg_type = msg.get_type()

                if msg_type == 'LOCAL_POSITION_NED':
                    self._current_position = np.array([msg.x, msg.y, -msg.z])

                elif msg_type == 'BATTERY_STATUS':
                    self._battery_percent = msg.battery_remaining

                elif msg_type == 'HEARTBEAT':
                    pass  # Could parse flight mode

        # Simulate battery drain
        if not self._connected:
            if self._arm_state == ArmState.ARMED:
                self._battery_percent -= 0.01

        # Publish telemetry
        # State
        state_msg = String()
        state_msg.data = self._flight_mode.name
        self._pub_state.publish(state_msg)

        # Battery
        battery_msg = Float32()
        battery_msg.data = self._battery_percent
        self._pub_battery.publish(battery_msg)

        # Armed
        armed_msg = Bool()
        armed_msg.data = self._arm_state == ArmState.ARMED
        self._pub_armed.publish(armed_msg)

        # Position
        pos_msg = PoseStamped()
        pos_msg.header.stamp = stamp
        pos_msg.header.frame_id = self.get_parameter('map_frame').value
        pos_msg.pose.position.x = float(self._current_position[0])
        pos_msg.pose.position.y = float(self._current_position[1])
        pos_msg.pose.position.z = float(self._current_position[2])
        pos_msg.pose.orientation.w = 1.0
        self._pub_position.publish(pos_msg)

    def _diagnostics_callback(self):
        """Log flight control diagnostics."""
        pos = self._current_position
        tgt = self._target_position
        error = np.linalg.norm(pos - tgt)

        self.get_logger().info(
            f'PX4 Bridge — '
            f'Mode: {self._flight_mode.name} | '
            f'Armed: {self._arm_state.name} | '
            f'Battery: {self._battery_percent:.1f}% | '
            f'Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | '
            f'Error: {error:.3f}m | '
            f'Connected: {self._connected}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = PX4BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
