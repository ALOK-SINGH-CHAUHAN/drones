"""
ANTIGRAVITY — PPO Navigation Training Environment
====================================================
Custom Gymnasium environment wrapping simulated drone navigation.
Used with Stable-Baselines3 PPO/SAC for training navigation policies.

Usage:
  python training/train_ppo.py --env DroneNav-v0 --timesteps 1000000
"""

import numpy as np
import time

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class DroneNavEnv(gym.Env):
    """
    Gymnasium environment for drone navigation training.

    Observation (19):
      [x, y, z, vx, vy, vz, dx, dy, dz, ray0..ray7, battery, slam_conf]

    Action (3):
      [vx_cmd, vy_cmd, vz_cmd] — continuous velocity commands

    Reward:
      + Progress toward goal
      - Collision penalty
      - Energy penalty
      + Goal reached bonus
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None, map_size=30.0, max_steps=500):
        super().__init__()

        self.map_size = map_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.dt = 0.1  # 10 Hz

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-2, -2, -1], dtype=np.float32),
            high=np.array([2, 2, 1], dtype=np.float32))

        # Environment state
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.goal = np.zeros(3)
        self.obstacles = []
        self.walls = []
        self.step_count = 0
        self.battery = 100.0
        self.cumulative_reward = 0.0
        self.prev_goal_dist = 0.0
        self.prev_action = np.zeros(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random start and goal
        margin = 3.0
        half = self.map_size / 2 - margin
        self.drone_pos = self.np_random.uniform([-half, -half, 1.0], [half, half, 2.0])
        self.goal = self.np_random.uniform([-half, -half, 1.0], [half, half, 2.0])

        # Ensure start and goal aren't too close
        while np.linalg.norm(self.goal[:2] - self.drone_pos[:2]) < 5.0:
            self.goal = self.np_random.uniform([-half, -half, 1.0], [half, half, 2.0])

        self.drone_vel = np.zeros(3)
        self.battery = 100.0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.prev_action = np.zeros(3)

        # Generate random obstacles
        n_obs = self.np_random.integers(5, 20)
        self.obstacles = []
        for _ in range(n_obs):
            ox = self.np_random.uniform(-half, half)
            oy = self.np_random.uniform(-half, half)
            r = self.np_random.uniform(0.3, 1.5)
            obs_pos = np.array([ox, oy, 1.5])
            if np.linalg.norm(obs_pos[:2] - self.drone_pos[:2]) > r + 1.0 and \
               np.linalg.norm(obs_pos[:2] - self.goal[:2]) > r + 1.0:
                self.obstacles.append({'pos': obs_pos, 'radius': r})

        # Walls
        half_map = self.map_size / 2
        self.walls = [
            {'min': np.array([-half_map, -half_map]), 'max': np.array([half_map, -half_map + 0.3])},
            {'min': np.array([-half_map, half_map - 0.3]), 'max': np.array([half_map, half_map])},
            {'min': np.array([-half_map, -half_map]), 'max': np.array([-half_map + 0.3, half_map])},
            {'min': np.array([half_map - 0.3, -half_map]), 'max': np.array([half_map, half_map])},
        ]

        obs = self._get_observation()
        info = {'distance_to_goal': np.linalg.norm(self.goal - self.drone_pos)}
        self.prev_goal_dist = info['distance_to_goal']
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Simple dynamics with drag
        drag = 0.1
        self.drone_vel = self.drone_vel * (1 - drag) + action * self.dt
        self.drone_pos += self.drone_vel * self.dt

        # Altitude constraint
        self.drone_pos[2] = np.clip(self.drone_pos[2], 0.3, 30.0)

        self.battery -= 0.02 * np.linalg.norm(action)
        self.step_count += 1

        # Check collisions
        collision = False
        for obs in self.obstacles:
            dist = np.linalg.norm(self.drone_pos[:2] - obs['pos'][:2])
            if dist < obs['radius']:
                collision = True
                break

        # Wall collision
        half_map = self.map_size / 2
        if abs(self.drone_pos[0]) > half_map or abs(self.drone_pos[1]) > half_map:
            collision = True

        # Goal check
        goal_dist = np.linalg.norm(self.goal - self.drone_pos)
        goal_reached = goal_dist < 0.5

        # Reward
        reward = 0.0
        reward -= 0.01  # Time penalty
        
        # Potential-based progress reward
        progress = self.prev_goal_dist - goal_dist
        self.prev_goal_dist = goal_dist
        reward += progress * 10.0  # Dense progress reward
        
        reward -= np.linalg.norm(action) * 0.005  # Energy penalty
        
        # Smoothness / jerk penalty
        jerk = np.linalg.norm(action - self.prev_action)
        self.prev_action = action.copy()
        reward -= jerk * 0.01

        if collision:
            reward -= 50.0  # Harsh penalty
        if goal_reached:
            reward += 100.0  # High bonus
        if self.battery <= 0:
            reward -= 10.0

        # Termination conditions
        terminated = goal_reached or collision or self.battery <= 0
        truncated = self.step_count >= self.max_steps

        self.cumulative_reward += reward

        obs = self._get_observation()
        info = {
            'distance_to_goal': goal_dist,
            'collision': collision,
            'goal_reached': goal_reached,
            'battery': self.battery,
            'cumulative_reward': self.cumulative_reward,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        obs = np.zeros(19, dtype=np.float32)
        obs[0:3] = self.drone_pos
        obs[3:6] = self.drone_vel
        obs[6:9] = self.goal - self.drone_pos  # Relative goal

        # 8-directional obstacle raycasting
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        max_range = 10.0
        for i, angle in enumerate(angles):
            min_dist = max_range
            ray_dir = np.array([np.cos(angle), np.sin(angle)])

            for ob in self.obstacles:
                diff = ob['pos'][:2] - self.drone_pos[:2]
                proj = np.dot(diff, ray_dir)
                if proj > 0:
                    perp = np.linalg.norm(diff - proj * ray_dir)
                    if perp < ob['radius']:
                        hit_dist = proj - np.sqrt(max(0, ob['radius']**2 - perp**2))
                        min_dist = min(min_dist, max(0, hit_dist))

            # Wall distances
            half = self.map_size / 2
            for t in np.linspace(0.1, max_range, 20):
                px = self.drone_pos[0] + ray_dir[0] * t
                py = self.drone_pos[1] + ray_dir[1] * t
                if abs(px) >= half or abs(py) >= half:
                    min_dist = min(min_dist, t)
                    break

            obs[9 + i] = min_dist / max_range

        obs[17] = self.battery / 100.0
        obs[18] = 1.0  # SLAM confidence (always 1 in sim)

        return obs

    def render(self):
        if self.render_mode == 'human':
            print(f'Step {self.step_count} | Pos: {self.drone_pos} | '
                  f'Goal dist: {np.linalg.norm(self.goal-self.drone_pos):.2f} | '
                  f'Battery: {self.battery:.1f}%')

    def close(self):
        pass
