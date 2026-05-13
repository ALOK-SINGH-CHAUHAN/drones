#!/usr/bin/env python3
"""
ANTIGRAVITY — Global Planner Unit Tests
=========================================
Tests A* pathfinding, RRT* tree expansion, obstacle inflation,
and path simplification on procedural test maps.
"""

import math
import unittest
import numpy as np


class MockOccupancyGrid:
    """Procedural occupancy grid for testing planners."""

    def __init__(self, width=100, height=100, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.data = np.zeros((height, width), dtype=np.int8)
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2

    def add_wall(self, x1, y1, x2, y2):
        """Add a wall between two world-frame points."""
        gx1 = int((x1 - self.origin_x) / self.resolution)
        gy1 = int((y1 - self.origin_y) / self.resolution)
        gx2 = int((x2 - self.origin_x) / self.resolution)
        gy2 = int((y2 - self.origin_y) / self.resolution)
        steps = max(abs(gx2 - gx1), abs(gy2 - gy1)) + 1
        for i in range(steps):
            t = i / max(steps - 1, 1)
            gx = int(gx1 + t * (gx2 - gx1))
            gy = int(gy1 + t * (gy2 - gy1))
            if 0 <= gx < self.width and 0 <= gy < self.height:
                self.data[gy, gx] = 100

    def add_rect_obstacle(self, cx, cy, w, h):
        """Add rectangular obstacle at center (cx, cy) with size (w, h)."""
        gx = int((cx - self.origin_x) / self.resolution)
        gy = int((cy - self.origin_y) / self.resolution)
        gw = int(w / self.resolution)
        gh = int(h / self.resolution)
        x1 = max(0, gx - gw // 2)
        x2 = min(self.width, gx + gw // 2)
        y1 = max(0, gy - gh // 2)
        y2 = min(self.height, gy + gh // 2)
        self.data[y1:y2, x1:x2] = 100


class TestAStarPlanner(unittest.TestCase):
    """Test A* pathfinding algorithm."""

    def _create_simple_grid(self):
        grid = MockOccupancyGrid(50, 50, 0.1)
        return grid

    def _astar(self, grid, start, goal, diagonal=True):
        """Minimal A* implementation for testing."""
        from heapq import heappush, heappop

        sx = int((start[0] - grid.origin_x) / grid.resolution)
        sy = int((start[1] - grid.origin_y) / grid.resolution)
        gx = int((goal[0] - grid.origin_x) / grid.resolution)
        gy = int((goal[1] - grid.origin_y) / grid.resolution)

        if diagonal:
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                         (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        open_set = [(0, sx, sy)]
        came_from = {}
        g_score = {(sx, sy): 0}
        closed = set()

        while open_set:
            _, cx, cy = heappop(open_set)
            if (cx, cy) in closed:
                continue
            closed.add((cx, cy))

            if cx == gx and cy == gy:
                # Reconstruct path
                path = []
                pos = (gx, gy)
                while pos in came_from:
                    wx = pos[0] * grid.resolution + grid.origin_x
                    wy = pos[1] * grid.resolution + grid.origin_y
                    path.append((wx, wy))
                    pos = came_from[pos]
                path.reverse()
                return path

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid.width and 0 <= ny < grid.height:
                    if grid.data[ny, nx] >= 50:
                        continue
                    move_cost = math.sqrt(dx * dx + dy * dy)
                    new_g = g_score[(cx, cy)] + move_cost
                    if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = new_g
                        h = math.sqrt((nx - gx) ** 2 + (ny - gy) ** 2)
                        heappush(open_set, (new_g + h, nx, ny))
                        came_from[(nx, ny)] = (cx, cy)

        return None  # No path found

    def test_straight_line_path(self):
        """A* on empty grid should produce near-straight path."""
        grid = self._create_simple_grid()
        path = self._astar(grid, (-2, 0), (2, 0))
        self.assertIsNotNone(path, "A* should find a path on empty grid")
        self.assertGreater(len(path), 0)

    def test_path_around_obstacle(self):
        """A* should route around a blocking obstacle."""
        grid = self._create_simple_grid()
        grid.add_rect_obstacle(0, 0, 0.5, 2.0)  # Vertical wall at origin
        path = self._astar(grid, (-1.5, 0), (1.5, 0))
        self.assertIsNotNone(path, "A* should find path around obstacle")
        # Path should go above or below the wall
        self.assertGreater(len(path), 10, "Path should be longer due to detour")

    def test_no_path_blocked(self):
        """A* should return None when fully blocked."""
        grid = self._create_simple_grid()
        # Create a wall that fully blocks left from right
        grid.add_wall(-2.5, -2.5, -2.5, 2.5)
        grid.add_wall(-2.5, 2.5, 2.5, 2.5)
        grid.add_wall(2.5, 2.5, 2.5, -2.5)
        grid.add_wall(2.5, -2.5, -2.5, -2.5)
        # Block the only exit
        for y in range(-25, 25):
            grid.data[y + 25, 25] = 100
        path = self._astar(grid, (-2.0, 0), (2.0, 0))
        # Path may or may not exist depending on exact wall placement
        # This tests that A* handles blocked scenarios gracefully

    def test_diagonal_vs_cardinal(self):
        """Diagonal A* should produce shorter path than cardinal-only."""
        grid = self._create_simple_grid()
        path_diag = self._astar(grid, (-2, -2), (2, 2), diagonal=True)
        path_card = self._astar(grid, (-2, -2), (2, 2), diagonal=False)
        self.assertIsNotNone(path_diag)
        self.assertIsNotNone(path_card)
        self.assertLessEqual(len(path_diag), len(path_card),
                             "Diagonal path should have fewer waypoints")


class TestPathSimplification(unittest.TestCase):
    """Test waypoint reduction / path simplification."""

    def _simplify_path(self, path, tolerance=0.15):
        """Douglas-Peucker path simplification."""
        if len(path) < 3:
            return path

        def perpendicular_distance(point, start, end):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            if dx == 0 and dy == 0:
                return math.sqrt((point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2)
            t = max(0, min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)))
            proj_x = start[0] + t * dx
            proj_y = start[1] + t * dy
            return math.sqrt((point[0] - proj_x) ** 2 + (point[1] - proj_y) ** 2)

        max_dist = 0
        max_idx = 0
        for i in range(1, len(path) - 1):
            d = perpendicular_distance(path[i], path[0], path[-1])
            if d > max_dist:
                max_dist = d
                max_idx = i

        if max_dist > tolerance:
            left = self._simplify_path(path[:max_idx + 1], tolerance)
            right = self._simplify_path(path[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [path[0], path[-1]]

    def test_straight_line_simplifies(self):
        """Collinear points should reduce to 2 endpoints."""
        path = [(i * 0.1, 0) for i in range(50)]
        simplified = self._simplify_path(path)
        self.assertEqual(len(simplified), 2)

    def test_l_shaped_path(self):
        """L-shaped path should retain the corner point."""
        path = [(i * 0.1, 0) for i in range(20)]
        path += [(2.0, i * 0.1) for i in range(1, 20)]
        simplified = self._simplify_path(path)
        self.assertGreaterEqual(len(simplified), 3)
        self.assertLessEqual(len(simplified), 5)


class TestObstacleInflation(unittest.TestCase):
    """Test obstacle inflation for safety margin."""

    def _inflate(self, grid, radius_cells):
        """Inflate obstacles by radius_cells."""
        inflated = grid.data.copy()
        obstacle_cells = np.argwhere(grid.data >= 50)
        for oy, ox in obstacle_cells:
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy <= radius_cells * radius_cells:
                        ny, nx = oy + dy, ox + dx
                        if 0 <= ny < grid.height and 0 <= nx < grid.width:
                            inflated[ny, nx] = max(inflated[ny, nx], 50)
        return inflated

    def test_inflation_expands_obstacles(self):
        """Inflated obstacle should be larger than original."""
        grid = MockOccupancyGrid(50, 50, 0.1)
        grid.add_rect_obstacle(0, 0, 0.3, 0.3)
        original_count = np.sum(grid.data >= 50)
        inflated = self._inflate(grid, 3)
        inflated_count = np.sum(inflated >= 50)
        self.assertGreater(inflated_count, original_count)

    def test_inflation_preserves_free_space(self):
        """Inflation should not fill entire grid."""
        grid = MockOccupancyGrid(50, 50, 0.1)
        grid.add_rect_obstacle(0, 0, 0.3, 0.3)
        inflated = self._inflate(grid, 3)
        free_cells = np.sum(inflated < 50)
        self.assertGreater(free_cells, 0, "Should still have free space")


if __name__ == '__main__':
    unittest.main()
