# ANTIGRAVITY 🚁

## GPS-Independent Autonomous Vision-Based Drone Navigation

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://python.org)
[![PX4](https://img.shields.io/badge/PX4-v1.14-orange.svg)](https://px4.io)

**ANTIGRAVITY** is a complete autonomous drone navigation software stack that operates without GPS. The drone reads a pre-built map, localizes itself using vision and IMU sensors, understands its environment semantically, plans optimal routes, and executes safe flight — all without satellite signal.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  SAFETY          Safety Arbiter → Geofence → System Monitor     │
├──────────────────────────────────────────────────────────────────┤
│  PLANNING        Global (A*/RRT*) → Local (MPC) → RL (PPO)     │
├──────────────────────────────────────────────────────────────────┤
│  CONTROL         EKF (100Hz) → Trajectory (min-snap) → PX4     │
├──────────────────────────────────────────────────────────────────┤
│  COGNITION       OctoMap → Semantic Seg (SAM) → Prediction      │
├──────────────────────────────────────────────────────────────────┤
│  LOCALIZATION    MCL Particle Filter (100-2000 particles)        │
├──────────────────────────────────────────────────────────────────┤
│  SLAM            ORB-SLAM3 (stereo-inertial, 20 Hz)             │
├──────────────────────────────────────────────────────────────────┤
│  PERCEPTION      Camera (30 Hz) + IMU (200 Hz) + Sync + YOLOv8  │
├──────────────────────────────────────────────────────────────────┤
│  MAP             Map Server (PGM/YAML, OctoMap .bt)              │
└──────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
drone/
├── src/
│   ├── antigravity_interfaces/      # 17 msgs, 4 srvs, 1 action
│   ├── antigravity_bringup/         # Launch files, 9 YAML configs, RViz, URDF, Worlds
│   ├── antigravity_perception/      # Camera + IMU drivers + sync
│   ├── antigravity_slam/            # ORB-SLAM3 integration
│   ├── antigravity_detection/       # YOLOv8 TensorRT detection
│   ├── antigravity_mapping/         # Map server (PGM/OctoMap)
│   ├── antigravity_localization/    # MCL particle filter
│   ├── antigravity_cognition/       # World model + semantics + prediction
│   ├── antigravity_planning/        # A*/RRT* + MPC + PPO RL decision
│   ├── antigravity_control/         # EKF + trajectory optimizer + PX4 bridge
│   └── antigravity_safety/          # Safety arbiter + geofence + monitor
├── docs/                            # Architecture, API, deployment guides
├── tests/                           # Unit, SITL, and benchmark tests
├── training/                        # PPO/SAC RL training environment
├── docker/                          # Jetson Orin NX deployment
├── maps/                            # Pre-built environment maps
├── models/                          # ML model weights
└── scripts/                         # Utilities (health check, TensorRT export)
```

## 11 ROS2 Packages · 20 Nodes · 17 Message Types

| Package | Nodes | Key Capabilities |
|---------|-------|-----------------|
| **antigravity_perception** | `camera_node`, `imu_node`, `sensor_sync_node` | RealSense/ZED @ 30Hz, IMU @ 200Hz, sync <2ms |
| **antigravity_slam** | `orb_slam3_node` | Stereo-inertial SLAM, relocalization <2s |
| **antigravity_detection** | `yolo_detection_node` | YOLOv8 TensorRT FP16, 4 classes, 3D positions |
| **antigravity_mapping** | `map_server_node` | PGM/YAML + OctoMap .bt formats |
| **antigravity_localization** | `mcl_node` | Adaptive particle filter, likelihood field |
| **antigravity_cognition** | `octomap_world_model_node`, `semantic_segmentation_node`, `prediction_engine_node` | Log-odds voxel map, SAM + classifier, Kalman tracker |
| **antigravity_planning** | `global_planner_node`, `local_planner_node`, `rl_decision_node` | A*/RRT*, MPC @ 20Hz, PPO policy inference |
| **antigravity_control** | `px4_bridge_node`, `trajectory_optimizer_node`, `ekf_state_estimator_node` | MAVLink offboard, min-snap C4 trajectories, 15-state EKF @ 100Hz |
| **antigravity_safety** | `safety_arbiter_node`, `geofence_node`, `system_monitor_node` | 5-level escalation, cylinder/polygon geofence, CPU/GPU/temp monitoring |

## Quick Start

```bash
# Build
cd drone
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Health check
python3 scripts/health_check.py

# Simulation (Gazebo SITL)
ros2 launch antigravity_bringup simulation.launch.py

# Hardware (full autonomous stack)
ros2 launch antigravity_bringup full_stack.launch.py \
    map_file:=maps/test_corridor.yaml

# Run tests (32 unit tests)
python3 -m pytest tests/unit/ -v
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Middleware** | ROS2 Humble |
| **SLAM** | ORB-SLAM3 (stereo-inertial) |
| **Detection** | YOLOv8 + TensorRT |
| **Segmentation** | SAM + custom classifier |
| **ML Framework** | PyTorch |
| **RL Training** | Stable-Baselines3 (PPO/SAC) |
| **Simulation** | Gazebo + PX4 SITL |
| **Autopilot** | PX4 (MAVLink) |
| **Companion** | NVIDIA Jetson Orin NX |
| **Flight Controller** | Pixhawk 6C |

## Documentation

- [Architecture Guide](docs/architecture.md) — Layer diagram, data flows, topic map
- [API Reference](docs/api_reference.md) — All 17 msgs, 4 srvs, 1 action
- [Deployment Guide](docs/deployment.md) — Build, simulate, deploy, train RL

## License

MIT License — see [LICENSE](LICENSE) for details.
