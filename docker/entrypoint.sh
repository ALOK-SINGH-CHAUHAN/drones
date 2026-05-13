#!/bin/bash
set -e

# Source ROS2
source /opt/ros/humble/setup.bash
source /antigravity/install/setup.bash

echo "╔══════════════════════════════════════════╗"
echo "║      ANTIGRAVITY Drone Navigation        ║"
echo "║      GPS-Free Autonomous Flight          ║"
echo "╚══════════════════════════════════════════╝"

exec "$@"
