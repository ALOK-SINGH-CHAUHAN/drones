# Contributing to ANTIGRAVITY

Thank you for your interest in contributing to ANTIGRAVITY! This document provides
guidelines for contributing to the project.

## Development Setup

### Prerequisites
- ROS2 Humble (or Iron)
- Python 3.10+
- NVIDIA GPU with CUDA (for TensorRT inference)
- colcon build system

### Quick Setup
```bash
cd drone_navigation
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
python3 scripts/health_check.py
```

## Code Style

### Python
- Follow PEP 8
- Use type hints for all function signatures
- Docstrings for all public classes and methods
- Maximum line length: 100 characters

### ROS2 Conventions
- Node names: `snake_case`
- Topic names: `namespace/topic_name`
- Service names: `namespace/service_name`
- Parameter names: `snake_case`
- Message types: `PascalCase`

## Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: Add new SLAM relocalization strategy
fix: Correct particle weight normalization in MCL
perf: Optimize OctoMap voxel insertion by 40%
docs: Add PID tuning guide for outdoor flights
test: Add SITL corridor navigation test
```

## Pull Request Process
1. Create a feature branch from `main`
2. Write or update tests for your changes
3. Ensure all tests pass: `python3 -m pytest tests/unit/ -v`
4. CodeRabbit will automatically review your PR
5. Request review from at least one maintainer

## Architecture Guidelines
- Each ROS2 node should be self-contained with graceful fallbacks
- All hardware-dependent code must have a simulated fallback
- Safety arbiter has absolute veto authority — no exceptions
- All timing-critical paths must log performance diagnostics

## Testing
- **Unit tests**: `tests/unit/` — Pure logic, no ROS2 dependencies
- **SITL tests**: `tests/sitl/` — Full simulation integration
- **Benchmarks**: `tests/benchmarks/` — Performance profiling

## Reporting Issues
Include:
- ROS2 distribution and version
- Hardware (Jetson model, camera, flight controller)
- Steps to reproduce
- Relevant log output
