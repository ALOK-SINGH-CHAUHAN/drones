"""
ANTIGRAVITY — Prediction Engine Node (Kalman Filter + LSTM)
============================================================
Multi-object tracker with Kalman filter prediction.
Predicts future positions up to 2 seconds ahead with uncertainty ellipses.
Uses Hungarian algorithm for detection-to-track assignment.

Acceptance Criteria:
  - Prediction error <= 0.5m at t+2s for straight-line pedestrians
  - Hungarian algorithm for track assignment
  - Handles track birth, death, and occlusion correctly
  - LSTM fallback for nonlinear trajectories (optional)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import time
import json
import threading
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KalmanTrack:
    """Single object track with Kalman filter state."""
    track_id: int
    class_name: str = 'unknown'
    # State: [x, y, z, vx, vy, vz]
    state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6) * 10.0)
    hits: int = 0
    misses: int = 0
    age: int = 0
    confirmed: bool = False
    last_detection_time: float = 0.0
    trajectory_history: list = field(default_factory=list)


class PredictionEngineNode(Node):
    """
    Multi-object tracker with Kalman filter and optional LSTM prediction.
    
    Subscribes:
      - /detection/detections/json (std_msgs/String): Detection data
    
    Publishes:
      - prediction/tracks (std_msgs/String): Active tracks JSON
      - prediction/trajectories (std_msgs/String): Predicted trajectories
      - prediction/visualization (visualization_msgs/MarkerArray): RViz markers
    """

    def __init__(self):
        super().__init__('prediction_engine')

        self.declare_parameter('max_track_age', 30)
        self.declare_parameter('min_hits_to_confirm', 3)
        self.declare_parameter('iou_threshold', 0.3)
        self.declare_parameter('process_noise_pos', 1.0)
        self.declare_parameter('process_noise_vel', 1.0)
        self.declare_parameter('measurement_noise', 1.0)
        self.declare_parameter('prediction_horizon_s', 2.0)
        self.declare_parameter('prediction_steps', 20)
        self.declare_parameter('max_assignment_cost', 50.0)
        self.declare_parameter('publish_rate_hz', 20)
        self.declare_parameter('use_lstm', False)

        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)

        self._pub_tracks = self.create_publisher(String, 'prediction/tracks', reliable_qos)
        self._pub_traj = self.create_publisher(String, 'prediction/trajectories', reliable_qos)
        self._pub_viz = self.create_publisher(MarkerArray, 'prediction/visualization', reliable_qos)

        self._sub_det = self.create_subscription(
            String, '/detection/detections/json', self._detection_cb, reliable_qos)

        self._tracks: List[KalmanTrack] = []
        self._next_id = 0
        self._dt = 1.0 / 30.0  # Assume 30 FPS detections
        self._lock = threading.Lock()
        self._count = 0

        # Kalman filter matrices
        q_pos = self.get_parameter('process_noise_pos').value
        q_vel = self.get_parameter('process_noise_vel').value
        r = self.get_parameter('measurement_noise').value
        dt = self._dt

        # State transition: constant velocity model
        self._F = np.eye(6)
        self._F[0, 3] = dt; self._F[1, 4] = dt; self._F[2, 5] = dt

        # Process noise
        self._Q = np.diag([q_pos*dt**2, q_pos*dt**2, q_pos*dt**2,
                           q_vel*dt, q_vel*dt, q_vel*dt])

        # Measurement matrix (observe position only)
        self._H = np.zeros((3, 6))
        self._H[0, 0] = 1; self._H[1, 1] = 1; self._H[2, 2] = 1

        # Measurement noise
        self._R = np.eye(3) * r

        pub_rate = self.get_parameter('publish_rate_hz').value
        self.create_timer(1.0 / pub_rate, self._publish)
        self.create_timer(5.0, self._diag)
        self.get_logger().info('Prediction engine initialized')

    def _detection_cb(self, msg):
        """Process new detections and update tracks."""
        try:
            data = json.loads(msg.data)
            detections = data.get('detections', [])
        except json.JSONDecodeError:
            return

        with self._lock:
            # Predict all tracks forward
            for track in self._tracks:
                self._kalman_predict(track)
                track.age += 1

            # Associate detections to tracks using Hungarian algorithm
            if detections and self._tracks:
                matches, unmatched_det, unmatched_trk = self._associate(detections)

                # Update matched tracks
                for d_idx, t_idx in matches:
                    det = detections[d_idx]
                    pos = det.get('position_3d')
                    if pos:
                        z = np.array([pos['x'], pos['y'], pos['z']])
                    else:
                        cx = (det['x_min'] + det['x_max']) / 2
                        cy = (det['y_min'] + det['y_max']) / 2
                        z = np.array([cx / 100, cy / 100, 2.0])  # Rough estimate

                    self._kalman_update(self._tracks[t_idx], z)
                    self._tracks[t_idx].hits += 1
                    self._tracks[t_idx].misses = 0
                    self._tracks[t_idx].class_name = det['class_name']
                    self._tracks[t_idx].last_detection_time = time.time()

                    if not self._tracks[t_idx].confirmed and self._tracks[t_idx].hits >= self.get_parameter('min_hits_to_confirm').value:
                        self._tracks[t_idx].confirmed = True
                        self.get_logger().info(f'Track {self._tracks[t_idx].track_id} CONFIRMED (class: {self._tracks[t_idx].class_name})')

                # Create new tracks for unmatched detections
                for d_idx in unmatched_det:
                    self._create_track(detections[d_idx])

                # Mark unmatched tracks as missed
                for t_idx in unmatched_trk:
                    self._tracks[t_idx].misses += 1
            elif detections:
                for det in detections:
                    self._create_track(det)

            # Remove dead tracks
            max_age = self.get_parameter('max_track_age').value
            alive_tracks = []
            for t in self._tracks:
                if t.misses < max_age:
                    alive_tracks.append(t)
                else:
                    self.get_logger().info(f'Track {t.track_id} DEAD (age: {t.age}, hits: {t.hits})')
            self._tracks = alive_tracks

            self._count += 1

    def _kalman_predict(self, track):
        """Kalman filter prediction step."""
        track.state = self._F @ track.state
        track.covariance = self._F @ track.covariance @ self._F.T + self._Q

    def _kalman_update(self, track, measurement):
        """Kalman filter update step."""
        y = measurement - self._H @ track.state  # Innovation
        S = self._H @ track.covariance @ self._H.T + self._R  # Innovation covariance
        K = track.covariance @ self._H.T @ np.linalg.inv(S)  # Kalman gain
        track.state = track.state + K @ y
        track.covariance = (np.eye(6) - K @ self._H) @ track.covariance

        # Store trajectory history
        track.trajectory_history.append(track.state[:3].tolist())
        if len(track.trajectory_history) > 100:
            track.trajectory_history.pop(0)

    def _associate(self, detections):
        """Hungarian algorithm for detection-to-track assignment."""
        n_det = len(detections)
        n_trk = len(self._tracks)
        cost_matrix = np.zeros((n_det, n_trk))

        for d, det in enumerate(detections):
            pos = det.get('position_3d')
            if pos:
                d_pos = np.array([pos['x'], pos['y'], pos['z']])
            else:
                d_pos = np.array([(det['x_min']+det['x_max'])/200,
                                  (det['y_min']+det['y_max'])/200, 2.0])

            for t, trk in enumerate(self._tracks):
                t_pos = trk.state[:3]
                cost_matrix[d, t] = np.linalg.norm(d_pos - t_pos)

        # Simple greedy assignment (replace with scipy.optimize.linear_sum_assignment for optimal)
        max_cost = self.get_parameter('max_assignment_cost').value
        matches = []
        used_det = set()
        used_trk = set()

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < max_cost:
                    matches.append((r, c))
                    used_det.add(r)
                    used_trk.add(c)
        except ImportError:
            # Greedy fallback
            for _ in range(min(n_det, n_trk)):
                if cost_matrix.size == 0: break
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                if cost_matrix[min_idx] < max_cost:
                    matches.append(min_idx)
                    used_det.add(min_idx[0])
                    used_trk.add(min_idx[1])
                    cost_matrix[min_idx[0], :] = np.inf
                    cost_matrix[:, min_idx[1]] = np.inf

        unmatched_det = [i for i in range(n_det) if i not in used_det]
        unmatched_trk = [i for i in range(n_trk) if i not in used_trk]

        return matches, unmatched_det, unmatched_trk

    def _create_track(self, detection):
        """Create a new track from an unmatched detection."""
        track = KalmanTrack(track_id=self._next_id)
        self._next_id += 1
        track.class_name = detection['class_name']
        pos = detection.get('position_3d')
        if pos:
            track.state[:3] = [pos['x'], pos['y'], pos['z']]
        else:
            track.state[:3] = [(detection['x_min']+detection['x_max'])/200,
                               (detection['y_min']+detection['y_max'])/200, 2.0]
        track.hits = 1
        track.last_detection_time = time.time()
        self._tracks.append(track)
        self.get_logger().info(f'Track {track.track_id} BORN (class: {track.class_name})')

    def _predict_trajectory(self, track):
        """Predict future trajectory using Kalman state propagation."""
        horizon = self.get_parameter('prediction_horizon_s').value
        steps = self.get_parameter('prediction_steps').value
        dt = horizon / steps

        F_pred = np.eye(6)
        F_pred[0, 3] = dt; F_pred[1, 4] = dt; F_pred[2, 5] = dt

        predictions = []
        state = track.state.copy()
        cov = track.covariance.copy()

        for i in range(steps):
            state = F_pred @ state
            cov = F_pred @ cov @ F_pred.T + self._Q * (dt / self._dt)
            predictions.append({
                'time_offset': (i + 1) * dt,
                'position': state[:3].tolist(),
                'covariance_diag': np.diag(cov[:3, :3]).tolist(),
            })

        return predictions

    def _publish(self):
        """Publish tracks, trajectories, and visualization markers."""
        stamp = self.get_clock().now().to_msg()

        with self._lock:
            confirmed = [t for t in self._tracks if t.confirmed]

            # Tracks JSON
            tracks_data = []
            trajectories_data = []
            markers = MarkerArray()

            for i, track in enumerate(confirmed):
                tracks_data.append({
                    'track_id': track.track_id,
                    'class_name': track.class_name,
                    'position': track.state[:3].tolist(),
                    'velocity': track.state[3:6].tolist(),
                    'hits': track.hits,
                    'age': track.age,
                })

                # Predict trajectory
                predictions = self._predict_trajectory(track)
                trajectories_data.append({
                    'track_id': track.track_id,
                    'predictions': predictions,
                    'source': 'kalman',
                })

                # Visualization marker (sphere at current position)
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = 'map'
                m.ns = 'tracked_objects'
                m.id = track.track_id
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(track.state[0])
                m.pose.position.y = float(track.state[1])
                m.pose.position.z = float(track.state[2])
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.3
                # Color by class
                if track.class_name == 'person':
                    m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
                elif track.class_name == 'vehicle':
                    m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
                else:
                    m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
                m.color.a = 0.8
                m.lifetime.sec = 1
                markers.markers.append(m)

                # Predicted trajectory line
                if predictions:
                    line = Marker()
                    line.header.stamp = stamp
                    line.header.frame_id = 'map'
                    line.ns = 'predicted_paths'
                    line.id = track.track_id + 10000
                    line.type = Marker.LINE_STRIP
                    line.action = Marker.ADD
                    line.scale.x = 0.05
                    line.color.r, line.color.g, line.color.b = 1.0, 0.5, 0.0
                    line.color.a = 0.6
                    line.lifetime.sec = 1
                    p0 = Point()
                    p0.x, p0.y, p0.z = float(track.state[0]), float(track.state[1]), float(track.state[2])
                    line.points.append(p0)
                    for pred in predictions:
                        p = Point()
                        p.x, p.y, p.z = pred['position']
                        line.points.append(p)
                    markers.markers.append(line)

        # Publish
        t_msg = String(); t_msg.data = json.dumps({'tracks': tracks_data})
        self._pub_tracks.publish(t_msg)

        tj_msg = String(); tj_msg.data = json.dumps({'trajectories': trajectories_data})
        self._pub_traj.publish(tj_msg)

        if markers.markers:
            self._pub_viz.publish(markers)

    def _diag(self):
        confirmed = sum(1 for t in self._tracks if t.confirmed)
        self.get_logger().info(
            f'Prediction — Tracks: {len(self._tracks)} (confirmed: {confirmed}) | Updates: {self._count}')


def main(args=None):
    rclpy.init(args=args)
    node = PredictionEngineNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
