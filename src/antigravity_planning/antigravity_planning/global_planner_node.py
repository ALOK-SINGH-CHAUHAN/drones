"""
ANTIGRAVITY — Global Planner Node (A* / RRT*)
===============================================
Computes optimal waypoint paths on occupancy grid or OctoMap.
Supports A* for 2D grids and RRT* for complex 3D spaces.

Acceptance Criteria (P3-T1):
  - Always finds a path if one exists
  - Path length within 10% of optimal
  - Replan triggered and complete within 2 seconds on map update
  - Compute time <= 2 seconds for 1000m² map
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String

import numpy as np
import heapq
import time
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass(order=True)
class AStarNode:
    """Priority queue node for A*."""
    f_cost: float
    g_cost: float = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    parent: Optional['AStarNode'] = field(compare=False, default=None)


class GlobalPlannerNode(Node):
    """
    A* / RRT* global path planner operating on occupancy grids.

    Subscribes:
      - /mapping/map (OccupancyGrid)
      - /localization/localization/pose (PoseStamped): Current position
      - planning/goal (PoseStamped): Navigation goal

    Publishes:
      - planning/global_path (Path): Waypoint sequence
      - planning/path_markers (MarkerArray): Visualization
      - planning/status (String): Planner status
    """

    # 8-connected grid neighbors
    NEIGHBORS_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    DIAG_COST = 1.414
    STRAIGHT_COST = 1.0

    def __init__(self):
        super().__init__('global_planner')

        self.declare_parameter('algorithm', 'astar')       # astar or rrtstar
        self.declare_parameter('inflation_radius_m', 0.5)  # Obstacle inflation
        self.declare_parameter('path_simplification', True)
        self.declare_parameter('replan_on_map_change', True)
        self.declare_parameter('max_compute_time_s', 2.0)
        self.declare_parameter('rrt_max_iterations', 5000)
        self.declare_parameter('rrt_step_size', 0.5)
        self.declare_parameter('goal_tolerance_m', 0.5)
        self.declare_parameter('waypoint_spacing_m', 1.0)
        self.declare_parameter('altitude_m', 1.5)

        self._algorithm = self.get_parameter('algorithm').value
        self._inflation = self.get_parameter('inflation_radius_m').value
        self._altitude = self.get_parameter('altitude_m').value

        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)

        self._pub_path = self.create_publisher(Path, 'planning/global_path', reliable_qos)
        self._pub_markers = self.create_publisher(MarkerArray, 'planning/path_markers', reliable_qos)
        self._pub_status = self.create_publisher(String, 'planning/status', reliable_qos)

        self._sub_map = self.create_subscription(OccupancyGrid, '/mapping/map', self._map_cb, map_qos)
        self._sub_pose = self.create_subscription(PoseStamped, '/localization/localization/pose',
                                                   self._pose_cb, reliable_qos)
        self._sub_goal = self.create_subscription(PoseStamped, 'planning/goal', self._goal_cb, reliable_qos)

        self._map_data = None
        self._map_info = None
        self._inflated_map = None
        self._current_pose = None
        self._current_goal = None
        self._current_path = None
        self._lock = threading.Lock()
        self._plan_count = 0

        self.create_timer(5.0, self._diag)
        self.get_logger().info(f'Global planner initialized — algorithm: {self._algorithm}')

    def _map_cb(self, msg):
        """Receive and inflate the occupancy grid."""
        with self._lock:
            self._map_info = msg.info
            w, h = msg.info.width, msg.info.height
            self._map_data = np.array(msg.data, dtype=np.int8).reshape((h, w))
            self._inflate_obstacles()

        if self._current_goal and self.get_parameter('replan_on_map_change').value:
            self.get_logger().info('Map updated — replanning...')
            self._plan_path()

    def _inflate_obstacles(self):
        """Inflate obstacles by the safety radius."""
        from scipy.ndimage import binary_dilation

        obstacles = (self._map_data >= 50).astype(bool)
        radius_cells = int(self._inflation / self._map_info.resolution)
        if radius_cells > 0:
            struct = np.ones((2*radius_cells+1, 2*radius_cells+1), dtype=bool)
            y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
            struct = (x**2 + y**2) <= radius_cells**2
            inflated = binary_dilation(obstacles, structure=struct)
            self._inflated_map = np.where(inflated, 100, self._map_data)
        else:
            self._inflated_map = self._map_data.copy()

    def _pose_cb(self, msg):
        self._current_pose = msg

    def _goal_cb(self, msg):
        """Receive navigation goal and plan path."""
        self._current_goal = msg
        self.get_logger().info(
            f'Goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')
        self._plan_path()

    def _plan_path(self):
        """Execute path planning."""
        if self._inflated_map is None or self._current_pose is None or self._current_goal is None:
            self._publish_status('NO_MAP_OR_POSE')
            return

        t0 = time.time()
        start = (self._current_pose.pose.position.x, self._current_pose.pose.position.y)
        goal = (self._current_goal.pose.position.x, self._current_goal.pose.position.y)

        if self._algorithm == 'rrtstar':
            path = self._plan_rrtstar(start, goal)
        else:
            path = self._plan_astar(start, goal)

        dt = time.time() - t0

        if path:
            if self.get_parameter('path_simplification').value:
                path = self._simplify_path(path)
            self._current_path = path
            self._publish_path(path)
            self._publish_markers(path)
            self._publish_status(f'SUCCESS ({dt:.3f}s, {len(path)} waypoints)')
            self.get_logger().info(f'Path found: {len(path)} waypoints in {dt:.3f}s')
        else:
            self._publish_status(f'FAILED ({dt:.3f}s)')
            self.get_logger().error('No path found!')

        self._plan_count += 1

    # ─── A* Implementation ───────────────────────────────────────────────

    def _plan_astar(self, start_world, goal_world):
        """A* search on inflated occupancy grid."""
        info = self._map_info
        start_grid = self._world_to_grid(start_world[0], start_world[1])
        goal_grid = self._world_to_grid(goal_world[0], goal_world[1])

        h, w = self._inflated_map.shape
        if not (0 <= start_grid[0] < w and 0 <= start_grid[1] < h):
            self.get_logger().error('Start position outside map!')
            return None
        if not (0 <= goal_grid[0] < w and 0 <= goal_grid[1] < h):
            self.get_logger().error('Goal position outside map!')
            return None
        if self._inflated_map[start_grid[1], start_grid[0]] >= 50:
            self.get_logger().warn('Start is in occupied space — finding nearest free cell')
            start_grid = self._find_nearest_free(start_grid)
            if start_grid is None:
                return None

        open_set = []
        h_cost = self._heuristic(start_grid, goal_grid)
        start_node = AStarNode(f_cost=h_cost, g_cost=0.0, position=start_grid)
        heapq.heappush(open_set, start_node)

        came_from: Dict[Tuple[int,int], Tuple[int,int]] = {}
        g_scores: Dict[Tuple[int,int], float] = {start_grid: 0.0}
        closed_set = set()

        max_time = self.get_parameter('max_compute_time_s').value
        t0 = time.time()
        iterations = 0

        while open_set:
            if time.time() - t0 > max_time:
                self.get_logger().warn(f'A* timeout after {iterations} iterations')
                return None

            current = heapq.heappop(open_set)
            pos = current.position
            iterations += 1

            if pos == goal_grid:
                return self._reconstruct_path(came_from, pos)

            if pos in closed_set:
                continue
            closed_set.add(pos)

            for dx, dy in self.NEIGHBORS_8:
                nx, ny = pos[0] + dx, pos[1] + dy
                neighbor = (nx, ny)

                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if self._inflated_map[ny, nx] >= 50:
                    continue
                if neighbor in closed_set:
                    continue

                move_cost = self.DIAG_COST if (dx != 0 and dy != 0) else self.STRAIGHT_COST
                new_g = current.g_cost + move_cost

                if neighbor not in g_scores or new_g < g_scores[neighbor]:
                    g_scores[neighbor] = new_g
                    f = new_g + self._heuristic(neighbor, goal_grid)
                    came_from[neighbor] = pos
                    heapq.heappush(open_set, AStarNode(f_cost=f, g_cost=new_g, position=neighbor))

        return None  # No path

    def _heuristic(self, a, b):
        """Octile distance heuristic for 8-connected grid."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (self.DIAG_COST - 1) * min(dx, dy)

    def _reconstruct_path(self, came_from, current):
        """Trace back from goal to start, converting to world coordinates."""
        path_grid = [current]
        while current in came_from:
            current = came_from[current]
            path_grid.append(current)
        path_grid.reverse()

        spacing = self.get_parameter('waypoint_spacing_m').value
        res = self._map_info.resolution
        spacing_cells = max(1, int(spacing / res))

        # Subsample waypoints
        world_path = []
        for i in range(0, len(path_grid), spacing_cells):
            gx, gy = path_grid[i]
            wx, wy = self._grid_to_world(gx, gy)
            world_path.append((wx, wy, self._altitude))

        # Always include the final point
        gx, gy = path_grid[-1]
        wx, wy = self._grid_to_world(gx, gy)
        last = (wx, wy, self._altitude)
        if not world_path or world_path[-1] != last:
            world_path.append(last)

        return world_path

    # ─── RRT* Implementation ────────────────────────────────────────────

    def _plan_rrtstar(self, start, goal):
        """RRT* for complex or 3D environments."""
        max_iter = self.get_parameter('rrt_max_iterations').value
        step = self.get_parameter('rrt_step_size').value
        goal_tol = self.get_parameter('goal_tolerance_m').value
        info = self._map_info
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        w, h = info.width, info.height
        x_range = (ox, ox + w * res)
        y_range = (oy, oy + h * res)

        nodes = [{'pos': np.array(start), 'parent': None, 'cost': 0.0}]
        best_goal_idx = None
        best_goal_cost = float('inf')

        for i in range(max_iter):
            # Bias toward goal 10% of time
            if np.random.random() < 0.1:
                sample = np.array(goal)
            else:
                sample = np.array([
                    np.random.uniform(*x_range),
                    np.random.uniform(*y_range),
                ])

            # Find nearest node
            dists = [np.linalg.norm(n['pos'] - sample) for n in nodes]
            nearest_idx = int(np.argmin(dists))
            nearest = nodes[nearest_idx]

            # Steer toward sample
            direction = sample - nearest['pos']
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
            direction = direction / dist
            new_pos = nearest['pos'] + direction * min(step, dist)

            # Collision check along segment
            if not self._collision_free_line(nearest['pos'], new_pos):
                continue

            new_cost = nearest['cost'] + np.linalg.norm(new_pos - nearest['pos'])

            # RRT* rewire: find nearby nodes
            rewire_radius = min(step * 3, 2.0)
            nearby = []
            for j, n in enumerate(nodes):
                if np.linalg.norm(n['pos'] - new_pos) < rewire_radius:
                    nearby.append(j)

            # Choose best parent
            best_parent = nearest_idx
            best_cost = new_cost
            for j in nearby:
                alt_cost = nodes[j]['cost'] + np.linalg.norm(nodes[j]['pos'] - new_pos)
                if alt_cost < best_cost and self._collision_free_line(nodes[j]['pos'], new_pos):
                    best_parent = j
                    best_cost = alt_cost

            new_node = {'pos': new_pos, 'parent': best_parent, 'cost': best_cost}
            new_idx = len(nodes)
            nodes.append(new_node)

            # Rewire neighbors
            for j in nearby:
                alt_cost = best_cost + np.linalg.norm(nodes[j]['pos'] - new_pos)
                if alt_cost < nodes[j]['cost'] and self._collision_free_line(new_pos, nodes[j]['pos']):
                    nodes[j]['parent'] = new_idx
                    nodes[j]['cost'] = alt_cost

            # Check if goal reached
            if np.linalg.norm(new_pos - np.array(goal)) < goal_tol:
                if best_cost < best_goal_cost:
                    best_goal_idx = new_idx
                    best_goal_cost = best_cost

        if best_goal_idx is None:
            return None

        # Extract path
        path = []
        idx = best_goal_idx
        while idx is not None:
            pos = nodes[idx]['pos']
            path.append((float(pos[0]), float(pos[1]), self._altitude))
            idx = nodes[idx]['parent']
        path.reverse()
        return path

    def _collision_free_line(self, p1, p2):
        """Check if a line segment is collision-free on the map."""
        res = self._map_info.resolution
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        steps = max(2, int(dist / (res * 0.5)))

        for t in np.linspace(0, 1, steps):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            gx, gy = self._world_to_grid(x, y)
            h, w = self._inflated_map.shape
            if not (0 <= gx < w and 0 <= gy < h):
                return False
            if self._inflated_map[gy, gx] >= 50:
                return False
        return True

    # ─── Path Simplification ────────────────────────────────────────────

    def _simplify_path(self, path):
        """Remove redundant waypoints using line-of-sight checks."""
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._collision_free_line(
                    (path[i][0], path[i][1]),
                    (path[j][0], path[j][1])
                ):
                    break
                j -= 1
            simplified.append(path[j])
            i = j
        return simplified

    # ─── Grid Conversions ────────────────────────────────────────────────

    def _world_to_grid(self, wx, wy):
        res = self._map_info.resolution
        ox = self._map_info.origin.position.x
        oy = self._map_info.origin.position.y
        return (int((wx - ox) / res), int((wy - oy) / res))

    def _grid_to_world(self, gx, gy):
        res = self._map_info.resolution
        ox = self._map_info.origin.position.x
        oy = self._map_info.origin.position.y
        return (ox + (gx + 0.5) * res, oy + (gy + 0.5) * res)

    def _find_nearest_free(self, pos):
        """Find nearest free cell to a position."""
        h, w = self._inflated_map.shape
        for r in range(1, 50):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    nx, ny = pos[0]+dx, pos[1]+dy
                    if 0 <= nx < w and 0 <= ny < h and self._inflated_map[ny, nx] < 50:
                        return (nx, ny)
        return None

    # ─── Publishing ──────────────────────────────────────────────────────

    def _publish_path(self, path):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for wx, wy, wz in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = wz
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._pub_path.publish(msg)

    def _publish_markers(self, path):
        ma = MarkerArray()
        # Path line
        line = Marker()
        line.header.stamp = self.get_clock().now().to_msg()
        line.header.frame_id = 'map'
        line.ns, line.id = 'global_path', 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.15
        line.color.r, line.color.g, line.color.b, line.color.a = 0.0, 0.8, 1.0, 0.9
        for wx, wy, wz in path:
            p = Point(); p.x, p.y, p.z = wx, wy, wz
            line.points.append(p)
        ma.markers.append(line)
        # Waypoint spheres
        for i, (wx, wy, wz) in enumerate(path):
            m = Marker()
            m.header = line.header
            m.ns, m.id = 'waypoints', i + 100
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = wx, wy, wz
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.3, 0.0, 1.0
            ma.markers.append(m)
        self._pub_markers.publish(ma)

    def _publish_status(self, status):
        msg = String(); msg.data = status
        self._pub_status.publish(msg)

    def _diag(self):
        self.get_logger().info(
            f'Global Planner [{self._algorithm}] — Plans: {self._plan_count} | '
            f'Map: {"loaded" if self._map_data is not None else "none"}')


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
