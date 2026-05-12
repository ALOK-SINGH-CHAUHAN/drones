"""
ANTIGRAVITY — Geofence Enforcement Node
=========================================
Validates drone position against configurable polygon/cylinder geofence.

Acceptance Criteria (P5-T2):
  - Geofence violation detected within 100ms
  - Supports polygon and cylindrical boundaries
  - Configurable soft (warn) and hard (stop) boundaries
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker

import numpy as np
import json


class GeofenceNode(Node):
    def __init__(self):
        super().__init__('geofence')

        self.declare_parameter('check_rate_hz', 20)
        self.declare_parameter('geofence_type', 'cylinder')  # cylinder or polygon
        self.declare_parameter('radius_m', 50.0)
        self.declare_parameter('max_altitude_m', 30.0)
        self.declare_parameter('min_altitude_m', 0.3)
        self.declare_parameter('center_x', 0.0)
        self.declare_parameter('center_y', 0.0)
        self.declare_parameter('soft_margin_m', 5.0)
        self.declare_parameter('polygon_json', '[]')

        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)

        self._pub_violation = self.create_publisher(Bool, 'geofence/violation', reliable_qos)
        self._pub_status = self.create_publisher(String, 'geofence/status', reliable_qos)
        self._pub_viz = self.create_publisher(Marker, 'geofence/boundary', reliable_qos)

        self._sub_pose = self.create_subscription(
            PoseStamped, '/state/state/pose', self._pose_cb, reliable_qos)

        self._pose = None
        self._polygon = []
        poly_str = self.get_parameter('polygon_json').value
        if poly_str and poly_str != '[]':
            try: self._polygon = json.loads(poly_str)
            except: pass

        rate = self.get_parameter('check_rate_hz').value
        self.create_timer(1.0 / rate, self._check)
        self.create_timer(2.0, self._publish_boundary_viz)
        self.get_logger().info(f'Geofence — type: {self.get_parameter("geofence_type").value}, '
                               f'radius: {self.get_parameter("radius_m").value}m')

    def _pose_cb(self, msg): self._pose = msg

    def _check(self):
        if not self._pose: return
        x = self._pose.pose.position.x
        y = self._pose.pose.position.y
        z = self._pose.pose.position.z

        violated = False
        soft_warn = False
        reason = ''

        # Altitude check
        max_alt = self.get_parameter('max_altitude_m').value
        min_alt = self.get_parameter('min_altitude_m').value
        if z > max_alt:
            violated = True; reason = f'ALT_MAX ({z:.1f}>{max_alt})'
        elif z < min_alt:
            violated = True; reason = f'ALT_MIN ({z:.1f}<{min_alt})'

        # Horizontal check
        fence_type = self.get_parameter('geofence_type').value
        if fence_type == 'cylinder':
            cx = self.get_parameter('center_x').value
            cy = self.get_parameter('center_y').value
            radius = self.get_parameter('radius_m').value
            margin = self.get_parameter('soft_margin_m').value
            dist = np.sqrt((x-cx)**2 + (y-cy)**2)

            if dist > radius:
                violated = True; reason = f'RADIUS ({dist:.1f}>{radius})'
            elif dist > radius - margin:
                soft_warn = True; reason = f'NEAR_BOUNDARY ({dist:.1f}m / {radius}m)'

        elif fence_type == 'polygon' and self._polygon:
            inside = self._point_in_polygon(x, y, self._polygon)
            if not inside:
                violated = True; reason = 'OUTSIDE_POLYGON'

        self._pub_violation.publish(Bool(data=violated))

        if violated:
            status = f'VIOLATED: {reason}'
        elif soft_warn:
            status = f'SOFT_WARNING: {reason}'
        else:
            status = 'OK'
        self._pub_status.publish(String(data=status))

    @staticmethod
    def _point_in_polygon(px, py, polygon):
        """Ray-casting point-in-polygon test."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj-xi)*(py-yi)/(yj-yi) + xi):
                inside = not inside
            j = i
        return inside

    def _publish_boundary_viz(self):
        """Publish geofence boundary for RViz."""
        fence_type = self.get_parameter('geofence_type').value
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns, m.id = 'geofence', 0
        m.action = Marker.ADD

        if fence_type == 'cylinder':
            m.type = Marker.CYLINDER
            m.pose.position.x = self.get_parameter('center_x').value
            m.pose.position.y = self.get_parameter('center_y').value
            max_alt = self.get_parameter('max_altitude_m').value
            m.pose.position.z = max_alt / 2
            m.pose.orientation.w = 1.0
            radius = self.get_parameter('radius_m').value
            m.scale.x = m.scale.y = radius * 2
            m.scale.z = max_alt
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.1
        else:
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.3
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.8
            for px, py in self._polygon:
                p = Point(); p.x, p.y, p.z = px, py, 1.0
                m.points.append(p)
            if self._polygon:
                p = Point(); p.x, p.y, p.z = self._polygon[0][0], self._polygon[0][1], 1.0
                m.points.append(p)

        self._pub_viz.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = GeofenceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
