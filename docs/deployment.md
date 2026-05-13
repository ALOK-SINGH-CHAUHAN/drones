# ANTIGRAVITY — Deployment Guide

## Prerequisites

### Software
- Ubuntu 22.04 LTS (Jammy Jellyfish)
- ROS2 Humble Hawksbill
- Python 3.10+
- PX4 Autopilot v1.14+
- Gazebo 11 (for simulation)

### Python Dependencies
```bash
pip3 install numpy scipy opencv-python-headless \
    torch torchvision ultralytics \
    pymavlink stable-baselines3 gymnasium \
    psutil open3d
```

## Quick Start

### 1. Build the Workspace
```bash
cd drone
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 2. Health Check
```bash
python3 scripts/health_check.py
```

### 3. Run in Simulation
```bash
# Full stack in Gazebo SITL
ros2 launch antigravity_bringup simulation.launch.py

# With RL decision layer enabled
ros2 launch antigravity_bringup simulation.launch.py use_rl:=true

# Without RViz (headless CI)
ros2 launch antigravity_bringup simulation.launch.py use_rviz:=false
```

### 4. Run on Hardware
```bash
# Full stack on physical drone
ros2 launch antigravity_bringup hardware.launch.py \
    map_file:=maps/test_corridor.yaml \
    camera_type:=realsense

# Minimal (perception + SLAM only)
ros2 launch antigravity_bringup hardware.launch.py \
    use_planning:=false use_cognition:=false use_safety:=false

# Full autonomous with safety
ros2 launch antigravity_bringup full_stack.launch.py \
    map_file:=maps/test_corridor.yaml \
    use_rl:=true
```

## Jetson Orin NX Deployment

### Docker Build
```bash
cd docker
docker build -f Dockerfile.jetson -t antigravity:latest ..
```

### Docker Run
```bash
docker run --runtime nvidia --privileged \
    --device /dev/ttyACM0 \
    --device /dev/video0 \
    -v /dev/bus/usb:/dev/bus/usb \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    antigravity:latest \
    ros2 launch antigravity_bringup full_stack.launch.py
```

### TensorRT Model Export
```bash
# Export YOLOv8 to TensorRT engine (run on Jetson)
python3 scripts/export_tensorrt.py \
    --model models/yolov8n.pt \
    --output models/yolov8n_drone.engine \
    --fp16
```

## RL Policy Training

### Train PPO Navigation Policy
```bash
python3 training/train_ppo.py \
    --timesteps 1000000 \
    --algorithm ppo \
    --output models/ppo_nav_policy.zip
```

### Train with SAC (alternative)
```bash
python3 training/train_ppo.py \
    --timesteps 500000 \
    --algorithm sac \
    --output models/sac_nav_policy.zip
```

### Monitor Training
```bash
tensorboard --logdir training/logs/
```

## Testing

### Unit Tests
```bash
# Run all tests
colcon test
colcon test-result --verbose

# Run specific package tests
colcon test --packages-select antigravity_planning
```

### SITL End-to-End Test
```bash
python3 tests/sitl/test_e2e_navigation.py
```

### Benchmarks
```bash
python3 tests/benchmarks/benchmark_pipeline.py
```

## Monitoring & Debugging

### Live Topic Monitoring
```bash
# View all ANTIGRAVITY topics
ros2 topic list | grep -E "(perception|slam|detection|localization|cognition|planning|control|safety)"

# Monitor safety status
ros2 topic echo /safety/status

# Monitor EKF state
ros2 topic echo /control/ekf_state

# Monitor system health
ros2 topic echo /safety/system_health
```

### RViz2 Visualization
The custom RViz config (`rviz/antigravity.rviz`) displays:
- Camera image feed
- SLAM trajectory + map points
- Detection bounding boxes
- Occupancy grid map
- MCL particle cloud
- Global path (waypoints)
- Trajectory (smooth curve)
- Geofence boundary
- Predicted object trajectories

### Safety Override (Emergency)
```bash
# Force landing
ros2 service call /safety/override antigravity_interfaces/srv/SafetyOverride \
    "{command: 2, reason: 'manual_override', authorization_code: 'ANTIGRAVITY_AUTH'}"

# Resume from hold
ros2 service call /safety/override antigravity_interfaces/srv/SafetyOverride \
    "{command: 0, reason: 'clear', authorization_code: 'ANTIGRAVITY_AUTH'}"
```

## Launch Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `use_rviz` | `true` | Launch RViz2 visualization |
| `camera_type` | `realsense` | Camera: `realsense` or `zed` |
| `map_file` | `""` | Path to pre-built map |
| `use_slam` | `true` | Enable ORB-SLAM3 |
| `use_detection` | `true` | Enable YOLOv8 |
| `use_localization` | `true` | Enable MCL |
| `use_cognition` | `true` | Enable cognition layer |
| `use_planning` | `true` | Enable A* + MPC |
| `use_safety` | `true` | Enable safety arbiter |
| `use_rl` | `false` | Enable RL decision layer |
| `simulation_mode` | `false` | Run in SITL mode |
| `px4_fcu_url` | `/dev/ttyACM0:921600` | PX4 connection URL |
