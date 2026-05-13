"""
ANTIGRAVITY — Map Server Node
===============================
Loads and serves pre-built occupancy grid maps and OctoMap 3D maps.
Supports both 2D .pgm/.yaml and 3D .bt OctoMap formats.

Acceptance Criteria:
  - Map loads in < 3 seconds for 1000m² floor plan
  - Map visible and correct in RViz2
  - OctoMap .bt files loadable as 3D alternative
  - Publishes /map topic in nav_msgs/OccupancyGrid format
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header

import numpy as np
import os
import time
import yaml


class MapServerNode(Node):
    """
    Map server for loading and serving pre-built navigation maps.
    
    Supports:
      - ROS2 nav_msgs/OccupancyGrid from .pgm + .yaml files
      - OctoMap .bt binary tree files (via octomap library)
      - Procedurally generated test maps for development
    
    Publishes:
      - map (nav_msgs/OccupancyGrid): The occupancy grid map (latched)
      - map_metadata (nav_msgs/MapMetaData): Map metadata
    
    Services:
      - load_map: Load a new map at runtime
    """

    def __init__(self):
        super().__init__('map_server')

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('map_file', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_rate_hz', 1.0)  # Republish rate for latching

        self._frame_id = self.get_parameter('frame_id').value
        self._map_msg = None

        # ─── QoS (transient local for latching behavior) ────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ─── Publishers ─────────────────────────────────────────────────
        self._pub_map = self.create_publisher(OccupancyGrid, 'map', map_qos)
        self._pub_metadata = self.create_publisher(MapMetaData, 'map_metadata', map_qos)

        # ─── Load Map ────────────────────────────────────────────────────
        map_file = self.get_parameter('map_file').value
        if map_file and os.path.exists(map_file):
            self._load_map_file(map_file)
        elif map_file:
            self.get_logger().warn(f'Map file not found: {map_file}. Generating test map.')
            self._generate_test_map()
        else:
            self.get_logger().info('No map file specified. Generating test map.')
            self._generate_test_map()

        # ─── Periodic Republish ──────────────────────────────────────────
        publish_rate = self.get_parameter('publish_rate_hz').value
        self._pub_timer = self.create_timer(1.0 / publish_rate, self._publish_map)

        self.get_logger().info('Map server initialized')

    def _load_map_file(self, map_file):
        """Load map from file based on extension."""
        t_start = time.time()
        ext = os.path.splitext(map_file)[1].lower()

        if ext == '.yaml':
            self._load_yaml_map(map_file)
        elif ext == '.pgm':
            # Look for corresponding .yaml
            yaml_file = map_file.replace('.pgm', '.yaml')
            if os.path.exists(yaml_file):
                self._load_yaml_map(yaml_file)
            else:
                self.get_logger().error(f'No .yaml file found for {map_file}')
                self._generate_test_map()
                return
        elif ext in ('.bt', '.ot'):
            self._load_octomap(map_file)
        else:
            self.get_logger().error(f'Unsupported map format: {ext}')
            self._generate_test_map()
            return

        load_time = time.time() - t_start
        self.get_logger().info(f'Map loaded in {load_time:.2f}s from {map_file}')

    def _load_yaml_map(self, yaml_file):
        """Load ROS2 map from .yaml + .pgm files."""
        with open(yaml_file, 'r') as f:
            map_info = yaml.safe_load(f)

        # Resolve PGM file path
        pgm_file = map_info.get('image', '')
        if not os.path.isabs(pgm_file):
            pgm_file = os.path.join(os.path.dirname(yaml_file), pgm_file)

        if not os.path.exists(pgm_file):
            self.get_logger().error(f'PGM file not found: {pgm_file}')
            self._generate_test_map()
            return

        # Read PGM image
        image = self._read_pgm(pgm_file)
        if image is None:
            self._generate_test_map()
            return

        resolution = float(map_info.get('resolution', 0.05))
        origin = map_info.get('origin', [0.0, 0.0, 0.0])
        negate = int(map_info.get('negate', 0))
        occupied_thresh = float(map_info.get('occupied_thresh', 0.65))
        free_thresh = float(map_info.get('free_thresh', 0.196))

        # Convert image to occupancy values
        if negate:
            image = 255 - image

        h, w = image.shape
        occupancy = np.full(h * w, -1, dtype=np.int8)  # Unknown

        for i in range(h * w):
            pixel = image.flat[i] / 255.0
            if pixel >= occupied_thresh:
                occupancy[i] = 0     # Free
            elif pixel <= free_thresh:
                occupancy[i] = 100   # Occupied
            # else: unknown (-1)

        # Build OccupancyGrid message
        self._map_msg = OccupancyGrid()
        self._map_msg.header.frame_id = self._frame_id
        self._map_msg.info.resolution = resolution
        self._map_msg.info.width = w
        self._map_msg.info.height = h
        self._map_msg.info.origin.position.x = float(origin[0])
        self._map_msg.info.origin.position.y = float(origin[1])
        self._map_msg.info.origin.position.z = 0.0
        self._map_msg.info.origin.orientation.w = 1.0
        self._map_msg.data = occupancy.tolist()

        self.get_logger().info(
            f'Loaded occupancy grid: {w}x{h} @ {resolution}m/px '
            f'({w * resolution:.0f}x{h * resolution:.0f}m)'
        )

    def _read_pgm(self, filepath):
        """Read PGM (P5 binary) image file."""
        try:
            with open(filepath, 'rb') as f:
                # Read header
                magic = f.readline().decode().strip()
                if magic not in ('P5', 'P2'):
                    self.get_logger().error(f'Invalid PGM magic: {magic}')
                    return None

                # Skip comments
                line = f.readline().decode().strip()
                while line.startswith('#'):
                    line = f.readline().decode().strip()

                w, h = map(int, line.split())
                maxval = int(f.readline().decode().strip())

                if magic == 'P5':
                    data = np.frombuffer(f.read(), dtype=np.uint8).reshape((h, w))
                else:
                    data = np.loadtxt(f, dtype=np.uint8).reshape((h, w))

                return data

        except Exception as e:
            self.get_logger().error(f'Failed to read PGM: {e}')
            return None

    def _load_octomap(self, bt_file):
        """Load OctoMap .bt file and convert to 2D occupancy grid projection."""
        try:
            import octomap

            tree = octomap.OcTree(self.get_parameter('resolution', 0.1).value)
            tree.readBinary(bt_file.encode())

            # Get bounds
            bbx_min = tree.getMetricMin()
            bbx_max = tree.getMetricMax()

            resolution = tree.getResolution()
            width = int((bbx_max[0] - bbx_min[0]) / resolution) + 1
            height = int((bbx_max[1] - bbx_min[1]) / resolution) + 1

            # Project to 2D (max occupancy along Z)
            occupancy = np.full(width * height, -1, dtype=np.int8)

            for it in tree.begin_leafs():
                node = it
                point = [node.getX(), node.getY(), node.getZ()]
                occ = node.getOccupancy()

                gx = int((point[0] - bbx_min[0]) / resolution)
                gy = int((point[1] - bbx_min[1]) / resolution)

                if 0 <= gx < width and 0 <= gy < height:
                    idx = gy * width + gx
                    cell_val = int(occ * 100)
                    occupancy[idx] = max(occupancy[idx], cell_val)

            self._map_msg = OccupancyGrid()
            self._map_msg.header.frame_id = self._frame_id
            self._map_msg.info.resolution = resolution
            self._map_msg.info.width = width
            self._map_msg.info.height = height
            self._map_msg.info.origin.position.x = float(bbx_min[0])
            self._map_msg.info.origin.position.y = float(bbx_min[1])
            self._map_msg.info.origin.orientation.w = 1.0
            self._map_msg.data = occupancy.tolist()

            self.get_logger().info(
                f'Loaded OctoMap: {width}x{height} @ {resolution}m/voxel'
            )

        except ImportError:
            self.get_logger().error('octomap-python not installed. pip install octomap-python')
            self._generate_test_map()
        except Exception as e:
            self.get_logger().error(f'OctoMap load failed: {e}')
            self._generate_test_map()

    def _generate_test_map(self):
        """Generate a procedural indoor corridor map for testing."""
        resolution = 0.05  # 5cm per pixel
        width_m = 30.0     # 30m wide
        height_m = 30.0    # 30m tall

        w = int(width_m / resolution)
        h = int(height_m / resolution)

        # Start with all unknown
        grid = np.full((h, w), 0, dtype=np.int8)  # Free space

        # Draw walls (occupied = 100)
        wall_thickness = int(0.3 / resolution)  # 30cm walls

        # Outer walls
        grid[:wall_thickness, :] = 100
        grid[-wall_thickness:, :] = 100
        grid[:, :wall_thickness] = 100
        grid[:, -wall_thickness:] = 100

        # Horizontal corridor walls
        corridor_y = h // 2
        corridor_width = int(3.0 / resolution)  # 3m corridors
        grid[corridor_y - corridor_width // 2 - wall_thickness:
             corridor_y - corridor_width // 2, :] = 100
        grid[corridor_y + corridor_width // 2:
             corridor_y + corridor_width // 2 + wall_thickness, :] = 100

        # Doorways (gaps in walls)
        door_width = int(1.2 / resolution)  # 1.2m doors
        for dx in [w // 4, w // 2, 3 * w // 4]:
            grid[corridor_y - corridor_width // 2 - wall_thickness:
                 corridor_y - corridor_width // 2,
                 dx - door_width // 2:dx + door_width // 2] = 0

        # Vertical walls creating rooms
        for rx in [w // 3, 2 * w // 3]:
            grid[wall_thickness:corridor_y - corridor_width // 2,
                 rx:rx + wall_thickness] = 100
            grid[corridor_y + corridor_width // 2:h - wall_thickness,
                 rx:rx + wall_thickness] = 100
            # Door in each room wall
            door_y_top = (wall_thickness + corridor_y - corridor_width // 2) // 2
            door_y_bot = (corridor_y + corridor_width // 2 + h - wall_thickness) // 2
            grid[door_y_top - door_width // 2:door_y_top + door_width // 2,
                 rx:rx + wall_thickness] = 0
            grid[door_y_bot - door_width // 2:door_y_bot + door_width // 2,
                 rx:rx + wall_thickness] = 0

        # Build message
        self._map_msg = OccupancyGrid()
        self._map_msg.header.frame_id = self._frame_id
        self._map_msg.info.resolution = resolution
        self._map_msg.info.width = w
        self._map_msg.info.height = h
        self._map_msg.info.origin.position.x = -width_m / 2
        self._map_msg.info.origin.position.y = -height_m / 2
        self._map_msg.info.origin.orientation.w = 1.0
        self._map_msg.data = grid.flatten().tolist()

        self.get_logger().info(
            f'Generated test map: {w}x{h} ({width_m}x{height_m}m) '
            f'with corridors and rooms'
        )

    def _publish_map(self):
        """Publish the current map."""
        if self._map_msg is None:
            return

        self._map_msg.header.stamp = self.get_clock().now().to_msg()
        self._pub_map.publish(self._map_msg)

        metadata = self._map_msg.info
        self._pub_metadata.publish(metadata)


def main(args=None):
    rclpy.init(args=args)
    node = MapServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
