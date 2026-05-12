"""
ANTIGRAVITY — OctoMap World Model Node
========================================
Live 3D voxel map using log-odds octree representation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import time
import threading


class OctoMapWorldModelNode(Node):
    def __init__(self):
        super().__init__('world_model')
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('min_range', 0.3)
        self.declare_parameter('prob_hit', 0.7)
        self.declare_parameter('prob_miss', 0.4)
        self.declare_parameter('max_memory_mb', 512)
        self.declare_parameter('publish_rate_hz', 10)
        self.declare_parameter('publish_free_space', True)

        self._res = self.get_parameter('resolution').value
        self._frame_id = self.get_parameter('frame_id').value
        p_hit = self.get_parameter('prob_hit').value
        p_miss = self.get_parameter('prob_miss').value
        self._l_hit = np.log(p_hit / (1.0 - p_hit))
        self._l_miss = np.log(p_miss / (1.0 - p_miss))
        self._l_min, self._l_max = -5.0, 5.0

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)

        self._pub_occ = self.create_publisher(PointCloud2, 'world_model/octomap', reliable_qos)
        self._pub_free = self.create_publisher(PointCloud2, 'world_model/free_space', reliable_qos)
        self._sub_depth = self.create_subscription(Image, '/camera/depth', self._depth_cb, sensor_qos)
        self._sub_pose = self.create_subscription(PoseStamped, '/slam/slam/pose', self._pose_cb, sensor_qos)

        self._bridge = CvBridge()
        self._voxel_map = {}
        self._semantic_map = {}
        self._pose = None
        self._lock = threading.Lock()
        self._fx, self._fy, self._cx, self._cy = 615.67, 615.96, 326.07, 240.25
        self._count = 0

        pub_rate = self.get_parameter('publish_rate_hz').value
        self.create_timer(1.0 / pub_rate, self._publish)
        self.create_timer(5.0, self._diag)
        self.get_logger().info(f'World model: res={self._res}m')

    def _pose_cb(self, msg): self._pose = msg

    def _depth_cb(self, msg):
        if not self._pose: return
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, 'passthrough')
        except: return
        p = self._pose.pose
        cam = np.array([p.position.x, p.position.y, p.position.z])
        yaw = 2.0 * np.arctan2(p.orientation.z, p.orientation.w)
        h, w = depth.shape[:2]
        step = 4
        cy, sy = np.cos(yaw), np.sin(yaw)
        with self._lock:
            for v in range(0, h, step):
                for u in range(0, w, step):
                    d = depth[v, u]
                    d_m = d / 1000.0 if depth.dtype == np.uint16 else float(d)
                    if d_m < 0.3 or d_m > 8.0 or np.isnan(d_m): continue
                    xc = (u - self._cx) * d_m / self._fx
                    yc = (v - self._cy) * d_m / self._fy
                    xm = cam[0] + cy * d_m - sy * xc
                    ym = cam[1] + sy * d_m + cy * xc
                    zm = cam[2] - yc
                    k = self._to_key(xm, ym, zm)
                    self._update(k, True)
                    for t in np.linspace(0, 0.7, 3):
                        fk = self._to_key(cam[0]+t*(xm-cam[0]), cam[1]+t*(ym-cam[1]), cam[2]+t*(zm-cam[2]))
                        if fk != k: self._update(fk, False)
        self._count += 1

    def _to_key(self, x, y, z):
        r = self._res
        return (int(np.floor(x/r)), int(np.floor(y/r)), int(np.floor(z/r)))

    def _to_world(self, ix, iy, iz):
        r = self._res
        return ((ix+0.5)*r, (iy+0.5)*r, (iz+0.5)*r)

    def _update(self, key, hit):
        c = self._voxel_map.get(key, 0.0)
        c += self._l_hit if hit else self._l_miss
        self._voxel_map[key] = np.clip(c, self._l_min, self._l_max)

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        with self._lock:
            occ = [self._to_world(*k) for k, v in self._voxel_map.items() if 1.0/(1+np.exp(-v)) > 0.65]
            free = [self._to_world(*k) for k, v in self._voxel_map.items() if 1.0/(1+np.exp(-v)) < 0.35]
        if occ: self._pub_occ.publish(self._make_pc2(occ, stamp))
        if free and self.get_parameter('publish_free_space').value:
            self._pub_free.publish(self._make_pc2(free, stamp))

    def _make_pc2(self, pts, stamp):
        msg = PointCloud2()
        msg.header.stamp, msg.header.frame_id = stamp, self._frame_id
        msg.height, msg.width = 1, len(pts)
        msg.fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
        msg.is_bigendian, msg.point_step, msg.row_step = False, 12, 12*len(pts)
        msg.data = np.array(pts, dtype=np.float32).flatten().tobytes()
        msg.is_dense = True
        return msg

    def _diag(self):
        n = len(self._voxel_map)
        self.get_logger().info(f'World Model — Voxels: {n} ({n*32/1048576:.1f}MB) | Updates: {self._count}')


def main(args=None):
    rclpy.init(args=args)
    node = OctoMapWorldModelNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
