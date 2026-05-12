# ANTIGRAVITY — API Reference

## Custom Message Types

### Detection Messages

#### `Detection.msg`
Single object detection from YOLOv8.
```
Header header
float32 x_min, y_min, x_max, y_max    # Bounding box (pixels)
float32 position_x, position_y, position_z  # 3D position (meters)
string class_name                       # "person", "vehicle", etc.
float32 confidence                      # 0.0-1.0
uint32 track_id                         # For association
```

#### `DetectionArray.msg`
Array of detections from a single frame.
```
Header header
Detection[] detections
uint32 frame_id
float32 inference_time_ms
```

### Tracking Messages

#### `TrackedObject.msg`
Single tracked object with Kalman filter state.
```
Header header
uint32 track_id
string class_name
float32 confidence
Point position            # Current position
Vector3 velocity          # Current velocity
float32[9] covariance     # Position covariance
float32 time_since_update # Seconds since last detection
uint32 hit_count          # Total detection associations
```

#### `TrackedObjectArray.msg`
```
Header header
TrackedObject[] tracks
uint32 active_tracks
uint32 total_tracks_created
```

#### `PredictedTrajectory.msg`
Future position predictions with uncertainty ellipses.
```
Header header
uint32 track_id
string class_name
Point[] predicted_positions
float64[] timestamps
float32[] confidence_radii
float32 prediction_horizon_s
```

### Planning Messages

#### `WaypointPath.msg`
Global planner output path.
```
Header header
PoseStamped[] waypoints
float32 total_distance_m
float32 planning_time_ms
string planner_type          # "astar" or "rrt_star"
bool is_valid
uint32 num_replans
```

#### `TrajectoryPoint.msg`
Single trajectory setpoint.
```
Header header
Point position
Vector3 velocity
Vector3 acceleration
Vector3 jerk
float64 yaw
float64 yaw_rate
float64 time_from_start
```

#### `Trajectory.msg`
Full smooth trajectory.
```
Header header
TrajectoryPoint[] points
float32 total_duration_s
float32 max_velocity_mps
float32 max_acceleration_mps2
bool is_feasible
string optimizer_type
```

#### `RLAction.msg`
RL decision layer output.
```
Header header
uint8 action                    # 0-7 (CONTINUE, REPLAN, etc.)
float32 confidence
float32[] action_probabilities
float32 value_estimate
bool using_heuristic
string policy_version
```

### State Estimation Messages

#### `EKFState.msg`
Fused state from Extended Kalman Filter.
```
Header header
PoseWithCovariance pose
TwistWithCovariance twist
Vector3 accel_bias, gyro_bias
float32 slam_weight, mcl_weight, imu_weight, baro_weight
float32 nees
bool is_converged
uint32 outliers_rejected
float32 update_rate_hz
```

#### `LocalizationState.msg`
MCL particle filter output.
```
Header header
PoseWithCovariance pose
uint32 num_particles
float32 effective_particle_count
float32 convergence_score
bool is_converged
float32 best_particle_weight
```

### Safety Messages

#### `SafetyStatus.msg`
Safety arbiter state with trigger flags.
```
Header header
uint8 state                     # 0=NOMINAL, 1=WARNING, 2=HOLD, 3=LAND, 4=RTH
bool battery_critical, slam_lost
bool obstacle_close, geofence_violated
bool system_unhealthy
float32 battery_pct
float32 min_obstacle_distance_m
string active_override_reason
```

#### `GeofenceStatus.msg`
```
Header header
uint8 fence_type
bool inside_fence, soft_margin_violated, hard_margin_violated
float32 distance_to_boundary_m
float32 soft_margin_m, hard_margin_m
Point[] fence_vertices
float32 max_altitude_m, min_altitude_m
```

#### `SystemHealth.msg`
```
Header header
string[] node_names
bool[] node_alive
float32[] node_frequencies_hz
float32 cpu_usage_pct, memory_usage_pct
float32 gpu_usage_pct, temperature_c
float32 disk_usage_pct
```

---

## Services

#### `LoadMap.srv`
```
# Request
string map_file_path
string map_type          # "occupancy_grid" or "octomap"
float32 resolution
---
# Response
bool success
string message
```

#### `SetGoal.srv`
```
PoseStamped goal_pose
float32 tolerance_m
float32 timeout_s
bool allow_replan
---
bool accepted
string message
```

#### `GetLocalization.srv`
```
bool include_particle_cloud
---
LocalizationState state
ParticleCloud particles
```

#### `SafetyOverride.srv`
```
uint8 command              # 0=RESUME, 1=HOLD, 2=LAND, 3=RTH
string reason
string authorization_code
---
bool accepted
string message
```

---

## Actions

#### `NavigateToGoal.action`
```
# Goal
PoseStamped goal_pose
float32 speed_limit_mps
bool avoid_dynamic_obstacles
---
# Result
bool success
float32 total_distance_m
float32 total_time_s
string final_status
---
# Feedback
PoseStamped current_pose
float32 distance_remaining_m
float32 estimated_time_remaining_s
uint8 navigation_state
```
