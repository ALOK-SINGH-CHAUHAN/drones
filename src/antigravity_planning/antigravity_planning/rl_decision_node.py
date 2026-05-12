"""
ANTIGRAVITY — RL Decision Layer (PPO Policy Inference)
=======================================================
Loads a pre-trained PPO policy (Stable-Baselines3) and publishes
high-level navigation decisions. Acts as an intelligent supervisor
that can override MPC commands during ambiguous situations.

Acceptance Criteria (P3-T3):
  - Policy inference <= 5ms per step
  - Sim2real transfer success rate >= 80%
  - Handles unseen obstacle configurations
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image

import numpy as np
import time
import json
import threading


class RLDecisionNode(Node):
    """
    PPO reinforcement learning decision layer.

    Observation space (continuous):
      - Drone position (x, y, z)           [3]
      - Drone velocity (vx, vy, vz)        [3]
      - Goal position relative (dx, dy, dz) [3]
      - Nearest obstacle distances (8 rays) [8]
      - Battery level                       [1]
      - SLAM confidence                     [1]
      Total: 19-dimensional observation

    Action space:
      - Desired velocity (vx, vy, vz)       [3]
      - Or: high-level decisions (continue, replan, hold, land)

    Subscribes:
      - /localization/localization/pose (PoseStamped)
      - /cognition/prediction/tracks (String)
      - /control/control/battery (Float32)
      - /slam/slam/state (String)
      - planning/goal (PoseStamped)

    Publishes:
      - planning/rl_action (TwistStamped)
      - planning/rl_decision (String)
      - planning/rl_observation (Float32MultiArray)
    """

    DECISIONS = ['continue', 'replan', 'hold', 'avoid_left', 'avoid_right', 'ascend', 'land']

    def __init__(self):
        super().__init__('rl_decision')

        self.declare_parameter('policy_path', 'models/ppo_navigation.zip')
        self.declare_parameter('inference_rate_hz', 10)
        self.declare_parameter('observation_dim', 19)
        self.declare_parameter('action_mode', 'hybrid')  # velocity or decision
        self.declare_parameter('rl_weight', 0.3)  # Blend weight vs MPC
        self.declare_parameter('exploration_noise', 0.0)  # 0 = deterministic
        self.declare_parameter('enabled', True)

        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)

        self._pub_action = self.create_publisher(TwistStamped, 'planning/rl_action', reliable_qos)
        self._pub_decision = self.create_publisher(String, 'planning/rl_decision', reliable_qos)
        self._pub_obs = self.create_publisher(Float32MultiArray, 'planning/rl_observation', reliable_qos)

        self._sub_pose = self.create_subscription(PoseStamped, '/localization/localization/pose',
                                                   self._pose_cb, reliable_qos)
        self._sub_tracks = self.create_subscription(String, '/cognition/prediction/tracks',
                                                     self._tracks_cb, reliable_qos)
        self._sub_goal = self.create_subscription(PoseStamped, 'planning/goal',
                                                   self._goal_cb, reliable_qos)
        self._sub_battery = self.create_subscription(String, '/control/control/battery',
                                                      self._battery_cb, reliable_qos)
        self._sub_slam_state = self.create_subscription(String, '/slam/slam/state',
                                                         self._slam_cb, reliable_qos)

        self._pose = None
        self._goal = None
        self._prev_pose = None
        self._obstacles = []
        self._battery = 100.0
        self._slam_confidence = 1.0
        self._policy = None
        self._inference_times = []

        self._init_policy()

        rate = self.get_parameter('inference_rate_hz').value
        self.create_timer(1.0 / rate, self._inference_loop)
        self.create_timer(5.0, self._diag)
        self.get_logger().info('RL decision node initialized')

    def _init_policy(self):
        """Load pre-trained PPO policy."""
        try:
            from stable_baselines3 import PPO
            path = self.get_parameter('policy_path').value
            self._policy = PPO.load(path)
            self._backend = 'sb3'
            self.get_logger().info(f'PPO policy loaded: {path}')
        except Exception as e:
            self.get_logger().warn(f'Policy load failed ({e}). Using heuristic fallback.')
            self._policy = None
            self._backend = 'heuristic'

    def _pose_cb(self, msg):
        self._prev_pose = self._pose
        self._pose = msg

    def _goal_cb(self, msg):
        self._goal = msg

    def _tracks_cb(self, msg):
        try:
            self._obstacles = json.loads(msg.data).get('tracks', [])
        except: pass

    def _battery_cb(self, msg):
        try: self._battery = float(msg.data)
        except: pass

    def _slam_cb(self, msg):
        state = msg.data.upper()
        if state == 'OK': self._slam_confidence = 1.0
        elif state == 'RECENTLY_LOST': self._slam_confidence = 0.5
        elif state == 'LOST': self._slam_confidence = 0.1
        else: self._slam_confidence = 0.8

    def _build_observation(self):
        """Construct observation vector from current state."""
        obs = np.zeros(self.get_parameter('observation_dim').value, dtype=np.float32)

        if self._pose:
            obs[0] = self._pose.pose.position.x
            obs[1] = self._pose.pose.position.y
            obs[2] = self._pose.pose.position.z

        # Estimated velocity from pose difference
        if self._pose and self._prev_pose:
            dt = 0.1  # Approximate
            obs[3] = (self._pose.pose.position.x - self._prev_pose.pose.position.x) / dt
            obs[4] = (self._pose.pose.position.y - self._prev_pose.pose.position.y) / dt
            obs[5] = (self._pose.pose.position.z - self._prev_pose.pose.position.z) / dt

        # Relative goal
        if self._goal and self._pose:
            obs[6] = self._goal.pose.position.x - self._pose.pose.position.x
            obs[7] = self._goal.pose.position.y - self._pose.pose.position.y
            obs[8] = self._goal.pose.position.z - self._pose.pose.position.z

        # Obstacle distances (8 directions: N,NE,E,SE,S,SW,W,NW)
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        max_range = 10.0
        for i, angle in enumerate(angles):
            min_dist = max_range
            if self._pose:
                px = self._pose.pose.position.x
                py = self._pose.pose.position.y
                for obs_data in self._obstacles:
                    op = obs_data.get('position', [0,0,0])
                    dx = op[0] - px
                    dy = op[1] - py
                    obs_angle = np.arctan2(dy, dx)
                    angle_diff = abs(np.arctan2(np.sin(obs_angle-angle), np.cos(obs_angle-angle)))
                    if angle_diff < np.pi / 4:
                        dist = np.sqrt(dx**2 + dy**2)
                        min_dist = min(min_dist, dist)
            obs[9 + i] = min_dist / max_range  # Normalized

        obs[17] = self._battery / 100.0
        obs[18] = self._slam_confidence

        return obs

    def _inference_loop(self):
        """Run policy inference and publish decisions."""
        if not self.get_parameter('enabled').value:
            return
        if self._pose is None or self._goal is None:
            return

        t0 = time.time()
        observation = self._build_observation()

        # Publish observation
        obs_msg = Float32MultiArray()
        obs_msg.data = observation.tolist()
        self._pub_obs.publish(obs_msg)

        if self._backend == 'sb3' and self._policy:
            action, _ = self._policy.predict(observation, deterministic=True)
            noise = self.get_parameter('exploration_noise').value
            if noise > 0:
                action += np.random.normal(0, noise, size=action.shape)
        else:
            action = self._heuristic_policy(observation)

        dt_ms = (time.time() - t0) * 1000
        self._inference_times.append(dt_ms)
        if len(self._inference_times) > 100: self._inference_times.pop(0)

        # Determine high-level decision
        decision = self._action_to_decision(observation, action)
        dec_msg = String(); dec_msg.data = decision
        self._pub_decision.publish(dec_msg)

        # Publish velocity action
        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = 'map'
        if len(action) >= 3:
            vel_msg.twist.linear.x = float(np.clip(action[0], -2, 2))
            vel_msg.twist.linear.y = float(np.clip(action[1], -2, 2))
            vel_msg.twist.linear.z = float(np.clip(action[2], -1, 1))
        self._pub_action.publish(vel_msg)

    def _heuristic_policy(self, obs):
        """Rule-based fallback when no trained policy is available."""
        # Simple proportional controller toward goal
        goal_rel = obs[6:9]
        dist = np.linalg.norm(goal_rel[:2])
        vel = np.zeros(3)

        if dist > 0.3:
            direction = goal_rel / max(dist, 0.01)
            speed = min(dist * 0.8, 2.0)
            vel[:3] = direction * speed

        # Obstacle avoidance
        min_obs_dist = min(obs[9:17]) * 10.0
        if min_obs_dist < 1.5:
            min_idx = np.argmin(obs[9:17])
            avoid_angle = np.linspace(0, 2*np.pi, 8, endpoint=False)[min_idx] + np.pi
            vel[0] += np.cos(avoid_angle) * 0.5
            vel[1] += np.sin(avoid_angle) * 0.5

        # Low battery — descend
        if obs[17] < 0.15:
            vel[2] = -0.5

        return vel

    def _action_to_decision(self, obs, action):
        """Convert continuous action + observation to high-level decision."""
        if obs[17] < 0.1:
            return 'land'
        if obs[18] < 0.3:
            return 'hold'
        min_obs = min(obs[9:17]) * 10.0
        if min_obs < 0.5:
            return 'avoid_left' if action[1] > 0 else 'avoid_right'
        goal_dist = np.linalg.norm(obs[6:9])
        if goal_dist > 20.0:
            return 'replan'
        return 'continue'

    def _diag(self):
        avg = np.mean(self._inference_times) if self._inference_times else 0
        self.get_logger().info(f'RL Decision [{self._backend}] — Avg inference: {avg:.1f}ms')


def main(args=None):
    rclpy.init(args=args)
    node = RLDecisionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
