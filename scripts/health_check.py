#!/usr/bin/env python3
"""
ANTIGRAVITY — System Health Check Script
Validates that all required dependencies and hardware are available.
Run this before launching the full stack.
"""

import sys
import subprocess
import importlib


def check_python_version():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    print(f"{'✅' if ok else '❌'} Python {v.major}.{v.minor}.{v.micro} (need >= 3.10)")
    return ok


def check_ros2():
    try:
        result = subprocess.run(['ros2', 'doctor', '--report'], capture_output=True, text=True, timeout=30)
        ok = result.returncode == 0
        print(f"{'✅' if ok else '❌'} ROS2 installation")
        return ok
    except Exception:
        print("❌ ROS2 not found in PATH")
        return False


def check_package(name, display=None):
    display = display or name
    try:
        importlib.import_module(name)
        print(f"✅ {display}")
        return True
    except ImportError:
        print(f"❌ {display} — pip install {name}")
        return False


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU: {gpu_name}")
            return True
        else:
            print("⚠️  No CUDA GPU detected (CPU-only mode)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def main():
    print("=" * 60)
    print("  ANTIGRAVITY — System Health Check")
    print("=" * 60)
    print()

    results = []
    print("── Core ─────────────────────────────────")
    results.append(check_python_version())
    results.append(check_ros2())

    print("\n── Python Packages ──────────────────────")
    results.append(check_package('numpy', 'NumPy'))
    results.append(check_package('cv2', 'OpenCV (cv2)'))
    results.append(check_package('yaml', 'PyYAML'))
    results.append(check_package('scipy', 'SciPy'))

    print("\n── ML / Vision ──────────────────────────")
    results.append(check_gpu())
    check_package('torch', 'PyTorch')
    check_package('ultralytics', 'Ultralytics (YOLOv8)')
    check_package('segment_anything', 'Segment Anything (SAM)')

    print("\n── Robotics ─────────────────────────────")
    check_package('rclpy', 'rclpy (ROS2 Python)')
    check_package('cv_bridge', 'cv_bridge')
    check_package('message_filters', 'message_filters')
    check_package('pymavlink', 'pymavlink (MAVLink)')

    print("\n── Hardware Drivers ─────────────────────")
    check_package('pyrealsense2', 'Intel RealSense SDK')
    check_package('serial', 'pyserial')

    print("\n── Optional ─────────────────────────────")
    check_package('octomap', 'octomap-python')
    check_package('open3d', 'Open3D')

    print()
    passed = sum(results)
    total = len(results)
    print(f"{'=' * 60}")
    print(f"  Result: {passed}/{total} critical checks passed")
    if all(results):
        print("  🚀 System ready for ANTIGRAVITY!")
    else:
        print("  ⚠️  Some dependencies missing — check above")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
