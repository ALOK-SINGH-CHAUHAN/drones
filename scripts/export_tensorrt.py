#!/usr/bin/env python3
"""
ANTIGRAVITY — TensorRT Model Export Script
Exports YOLOv8 PyTorch model to TensorRT engine for Jetson deployment.

Usage:
  python3 scripts/export_tensorrt.py --model models/yolov8n.pt --output models/yolov8n_drone.engine
"""

import argparse
import sys


def export_model(model_path, output_path, img_size=640, half=True, int8=False):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Exporting to TensorRT...")
    print(f"  Image size: {img_size}")
    print(f"  FP16: {half}")
    print(f"  INT8: {int8}")

    model.export(
        format='engine',
        imgsz=img_size,
        half=half,
        int8=int8,
        device=0,
        workspace=4,  # GB
        verbose=True,
    )

    print(f"\n✅ TensorRT engine exported successfully")
    print(f"   Output: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export YOLOv8 to TensorRT')
    parser.add_argument('--model', default='models/yolov8n.pt', help='Input PyTorch model')
    parser.add_argument('--output', default='models/yolov8n_drone.engine', help='Output TensorRT engine')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--no-half', action='store_true', help='Disable FP16')
    parser.add_argument('--int8', action='store_true', help='Enable INT8 quantization')
    args = parser.parse_args()

    export_model(args.model, args.output, args.img_size, not args.no_half, args.int8)
