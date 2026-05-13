import pytest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

class DummyEnv:
    def __init__(self, *args, **kwargs): pass
    def reset(self, seed=None, options=None):
        import numpy as np
        self.np_random = np.random.default_rng(seed)
    def step(self, action): pass

class DummyBox:
    def __init__(self, low, high, shape=None, dtype=None):
        import numpy as np
        if isinstance(low, np.ndarray):
            self.low = low
            self.high = high
        else:
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)

gym_mock = MagicMock()
gym_mock.Env = DummyEnv

sys.modules['gymnasium'] = gym_mock
sys.modules['gymnasium.spaces'] = MagicMock()
sys.modules['gymnasium.spaces'].Box = DummyBox
sys.modules['gym'] = gym_mock
sys.modules['gym'].spaces = MagicMock()
sys.modules['gym'].spaces.Box = DummyBox

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from training.drone_nav_env import DroneNavEnv

class TestRLPlanningEnvironment:
    def test_environment_reset(self):
        """Test proper initialization of RL environment state."""
        env = DroneNavEnv(map_size=30.0, max_steps=500)
        obs, info = env.reset(seed=42)
        
        assert obs.shape == (19,)
        assert 'distance_to_goal' in info
        assert env.prev_goal_dist == info['distance_to_goal']
        assert np.array_equal(env.prev_action, np.zeros(3))
        
        # Ensure goal and start are not colliding with obstacles
        for ob in env.obstacles:
            assert np.linalg.norm(ob['pos'][:2] - env.drone_pos[:2]) >= ob['radius']
            assert np.linalg.norm(ob['pos'][:2] - env.goal[:2]) >= ob['radius']

    def test_environment_step_progress_reward(self):
        """Test potential-based reward shaping."""
        env = DroneNavEnv(map_size=30.0, max_steps=500)
        obs, info = env.reset(seed=42)
        
        initial_dist = env.prev_goal_dist
        
        # Action moving towards the goal
        direction = env.goal - env.drone_pos
        action = (direction / np.linalg.norm(direction)) * 2.0
        
        # Take step
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        
        new_dist = next_info['distance_to_goal']
        
        # Drone should be closer to goal
        assert new_dist < initial_dist
        
        progress = initial_dist - new_dist
        assert progress > 0

    def test_environment_collision_penalty(self):
        """Test collision penalization."""
        env = DroneNavEnv(map_size=30.0, max_steps=500)
        env.reset(seed=42)
        
        # Force a collision by placing an obstacle at drone pos
        env.obstacles = [{'pos': env.drone_pos.copy(), 'radius': 1.0}]
        
        next_obs, reward, terminated, truncated, next_info = env.step(np.array([0.0, 0.0, 0.0]))
        
        assert next_info['collision'] is True
        assert terminated is True
        assert reward < -10.0  # Harsh penalty
