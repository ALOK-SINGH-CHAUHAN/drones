"""
ANTIGRAVITY — YOLOv8 Object Detection Node
============================================
Real-time object detection using YOLOv8 with TensorRT optimization.
Detects: person, vehicle, animal, unknown_moving objects.

Acceptance Criteria:
  - Inference latency <= 30ms per frame at 640x480
  - mAP >= 0.75 on drone-view test dataset
  - No dropped frames during simultaneous SLAM operation
  - Publishes DetectionArray at >= 30 Hz
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import numpy as np
import time
import threading


class YoloDetectionNode(Node):
    """
    YOLOv8 object detection node with TensorRT acceleration.
    
    Subscribes:
      - camera/image_raw (sensor_msgs/Image): Input camera frames
      - camera/depth (sensor_msgs/Image): Depth for 3D position estimation
      - camera/camera_info (sensor_msgs/CameraInfo): Camera intrinsics
    
    Publishes:
      - detections (antigravity_interfaces/DetectionArray): Detected objects
      - detection/annotated_image (sensor_msgs/Image): Annotated visualization
    """

    # Target class mapping for drone navigation
    DRONE_CLASSES = {
        0: 'person',
        1: 'vehicle',
        2: 'animal',
        3: 'unknown_moving',
    }

    # COCO class mapping to our custom classes
    COCO_TO_DRONE = {
        0: 0,    # person -> person
        1: 1,    # bicycle -> vehicle
        2: 1,    # car -> vehicle
        3: 1,    # motorcycle -> vehicle
        5: 1,    # bus -> vehicle
        7: 1,    # truck -> vehicle
        14: 2,   # bird -> animal
        15: 2,   # cat -> animal
        16: 2,   # dog -> animal
        17: 2,   # horse -> animal
    }

    def __init__(self):
        super().__init__('yolo_detector')

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('model_path', 'models/yolov8n_drone.engine')
        self.declare_parameter('model_type', 'yolov8')
        self.declare_parameter('fallback_model_path', 'models/yolov8n_drone.pt')
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 480)
        self.declare_parameter('confidence_threshold', 0.45)
        self.declare_parameter('nms_threshold', 0.5)
        self.declare_parameter('max_detections', 20)
        self.declare_parameter('num_classes', 4)
        self.declare_parameter('use_tensorrt', True)
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('enable_temporal_smoothing', True)
        self.declare_parameter('smoothing_window', 3)
        self.declare_parameter('min_consecutive_detections', 2)
        self.declare_parameter('estimate_3d_position', True)
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('detection_frame_id', 'camera_optical_frame')
        self.declare_parameter('class_names', ['person', 'vehicle', 'animal', 'unknown_moving'])

        self._conf_threshold = self.get_parameter('confidence_threshold').value
        self._nms_threshold = self.get_parameter('nms_threshold').value
        self._max_detections = self.get_parameter('max_detections').value
        self._frame_id = self.get_parameter('detection_frame_id').value
        self._estimate_3d = self.get_parameter('estimate_3d_position').value
        self._publish_annotated = self.get_parameter('publish_annotated_image').value

        # ─── QoS ────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ─── Publishers ─────────────────────────────────────────────────
        # Use generic message since antigravity_interfaces may not be built yet
        from std_msgs.msg import String
        self._pub_detections_json = self.create_publisher(String, 'detections/json', reliable_qos)
        self._pub_annotated = self.create_publisher(Image, 'detection/annotated_image', sensor_qos)

        # Try to use custom messages
        self._use_custom_msgs = False
        try:
            from antigravity_interfaces.msg import Detection, DetectionArray
            self._pub_detections = self.create_publisher(DetectionArray, 'detections', reliable_qos)
            self._use_custom_msgs = True
        except ImportError:
            self.get_logger().warn(
                'antigravity_interfaces not found — publishing JSON detections only'
            )

        # ─── Subscribers ────────────────────────────────────────────────
        self._sub_image = self.create_subscription(
            Image, 'camera/image_raw', self._image_callback, sensor_qos
        )
        self._sub_depth = self.create_subscription(
            Image, 'camera/depth', self._depth_callback, sensor_qos
        )
        self._sub_camera_info = self.create_subscription(
            CameraInfo, 'camera/camera_info', self._camera_info_callback, sensor_qos
        )

        # ─── State ──────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._model = None
        self._last_depth = None
        self._camera_matrix = None
        self._detection_id_counter = 0

        # Temporal smoothing buffers
        self._detection_history = []
        self._smoothing_window = self.get_parameter('smoothing_window').value

        # Performance tracking
        self._inference_times = []
        self._frame_count = 0
        self._start_time = time.time()
        self._processing_lock = threading.Lock()

        # ─── Initialize Model ───────────────────────────────────────────
        self._init_model()

        # ─── Diagnostics ────────────────────────────────────────────────
        self._diag_timer = self.create_timer(5.0, self._diagnostics_callback)

        self.get_logger().info('YOLOv8 detection node initialized')

    def _init_model(self):
        """Initialize YOLOv8 model with TensorRT or PyTorch fallback."""
        model_path = self.get_parameter('model_path').value
        fallback_path = self.get_parameter('fallback_model_path').value
        device = self.get_parameter('device').value

        try:
            from ultralytics import YOLO

            # Try TensorRT engine first
            if self.get_parameter('use_tensorrt').value:
                try:
                    self._model = YOLO(model_path, task='detect')
                    self._model_backend = 'tensorrt'
                    self.get_logger().info(f'Loaded TensorRT engine: {model_path}')
                    return
                except Exception as e:
                    self.get_logger().warn(f'TensorRT load failed: {e}')

            # Fallback to PyTorch
            self._model = YOLO(fallback_path)
            self._model_backend = 'pytorch'
            self.get_logger().info(f'Loaded PyTorch model: {fallback_path}')

        except ImportError:
            self.get_logger().warn(
                'ultralytics not installed. Using simulated detections. '
                'Install with: pip install ultralytics'
            )
            self._model = None
            self._model_backend = 'simulated'

    def _image_callback(self, msg):
        """Process incoming camera frame for object detection."""
        if not self._processing_lock.acquire(blocking=False):
            return  # Skip frame if still processing previous

        try:
            t_start = time.time()

            cv_image = self._bridge.imgmsg_to_cv2(msg, 'bgr8')

            if self._model_backend == 'simulated':
                detections = self._detect_simulated(cv_image)
            else:
                detections = self._detect_yolo(cv_image)

            # 3D position estimation
            if self._estimate_3d and self._last_depth is not None and self._camera_matrix is not None:
                detections = self._estimate_3d_positions(detections)

            # Temporal smoothing
            if self.get_parameter('enable_temporal_smoothing').value:
                detections = self._temporal_filter(detections)

            # Calculate inference time
            inference_time_ms = (time.time() - t_start) * 1000.0
            self._inference_times.append(inference_time_ms)
            if len(self._inference_times) > 100:
                self._inference_times.pop(0)

            # Publish detections
            self._publish_detections(detections, msg.header.stamp, inference_time_ms)

            # Publish annotated image
            if self._publish_annotated:
                annotated = self._draw_detections(cv_image, detections)
                ann_msg = self._bridge.cv2_to_imgmsg(annotated, 'bgr8')
                ann_msg.header = msg.header
                self._pub_annotated.publish(ann_msg)

            self._frame_count += 1

        finally:
            self._processing_lock.release()

    def _detect_yolo(self, image):
        """Run YOLOv8 inference on image."""
        results = self._model(
            image,
            conf=self._conf_threshold,
            iou=self._nms_threshold,
            max_det=self._max_detections,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()

                    # Map COCO class to drone class
                    drone_cls = self.COCO_TO_DRONE.get(cls_id, 3)  # Default: unknown_moving
                    cls_name = self.DRONE_CLASSES[drone_cls]

                    detections.append({
                        'x_min': float(xyxy[0]),
                        'y_min': float(xyxy[1]),
                        'x_max': float(xyxy[2]),
                        'y_max': float(xyxy[3]),
                        'class_id': drone_cls,
                        'class_name': cls_name,
                        'confidence': conf,
                        'detection_id': self._detection_id_counter,
                        'position_3d': None,
                    })
                    self._detection_id_counter += 1

        return detections

    def _detect_simulated(self, image):
        """Generate simulated detections for testing."""
        detections = []
        h, w = image.shape[:2]
        t = time.time()

        # Simulate 0-3 random detections
        num_dets = np.random.choice([0, 1, 1, 2, 2, 3])
        for _ in range(num_dets):
            cx = np.random.uniform(100, w - 100)
            cy = np.random.uniform(100, h - 100)
            bw = np.random.uniform(40, 120)
            bh = np.random.uniform(60, 180)
            cls_id = np.random.choice([0, 0, 0, 1, 2, 3])  # Bias toward person

            detections.append({
                'x_min': max(0, cx - bw / 2),
                'y_min': max(0, cy - bh / 2),
                'x_max': min(w, cx + bw / 2),
                'y_max': min(h, cy + bh / 2),
                'class_id': cls_id,
                'class_name': self.DRONE_CLASSES[cls_id],
                'confidence': np.random.uniform(0.5, 0.95),
                'detection_id': self._detection_id_counter,
                'position_3d': None,
            })
            self._detection_id_counter += 1

        return detections

    def _estimate_3d_positions(self, detections):
        """Estimate 3D positions from depth image and camera intrinsics."""
        try:
            depth_image = self._bridge.imgmsg_to_cv2(self._last_depth, 'passthrough')
            fx = self._camera_matrix[0]
            fy = self._camera_matrix[4]
            cx = self._camera_matrix[2]
            cy = self._camera_matrix[5]

            for det in detections:
                # Get center pixel
                u = int((det['x_min'] + det['x_max']) / 2)
                v = int((det['y_min'] + det['y_max']) / 2)

                # Get depth at center (handle out-of-bounds)
                h, w = depth_image.shape[:2]
                u = np.clip(u, 0, w - 1)
                v = np.clip(v, 0, h - 1)

                depth_val = depth_image[v, u]
                if depth_image.dtype == np.uint16:
                    depth_m = depth_val / 1000.0  # mm to m
                else:
                    depth_m = float(depth_val)

                if 0.3 < depth_m < 10.0:
                    # Back-project to 3D
                    x = (u - cx) * depth_m / fx
                    y = (v - cy) * depth_m / fy
                    z = depth_m

                    det['position_3d'] = {'x': x, 'y': y, 'z': z}

        except Exception as e:
            self.get_logger().debug(f'3D estimation error: {e}')

        return detections

    def _temporal_filter(self, detections):
        """Apply temporal smoothing to reduce spurious detections."""
        self._detection_history.append(detections)
        if len(self._detection_history) > self._smoothing_window:
            self._detection_history.pop(0)

        # For now, pass through (full NMS-over-time would be more complex)
        return detections

    def _publish_detections(self, detections, stamp, inference_time_ms):
        """Publish detections in both custom message and JSON formats."""
        import json
        from std_msgs.msg import String

        # JSON format (always available)
        json_msg = String()
        json_msg.data = json.dumps({
            'stamp': {'sec': stamp.sec, 'nanosec': stamp.nanosec},
            'frame_id': self._frame_id,
            'inference_time_ms': inference_time_ms,
            'detections': detections,
        })
        self._pub_detections_json.publish(json_msg)

        # Custom message format
        if self._use_custom_msgs:
            from antigravity_interfaces.msg import Detection, DetectionArray

            det_array = DetectionArray()
            det_array.header.stamp = stamp
            det_array.header.frame_id = self._frame_id
            det_array.frame_id = self._frame_count
            det_array.inference_time_ms = inference_time_ms

            for det in detections:
                d = Detection()
                d.header.stamp = stamp
                d.header.frame_id = self._frame_id
                d.x_min = det['x_min']
                d.y_min = det['y_min']
                d.x_max = det['x_max']
                d.y_max = det['y_max']
                d.class_id = det['class_id']
                d.class_name = det['class_name']
                d.confidence = det['confidence']
                d.detection_id = det['detection_id']

                if det['position_3d'] is not None:
                    d.has_3d_position = True
                    d.position_3d.x = det['position_3d']['x']
                    d.position_3d.y = det['position_3d']['y']
                    d.position_3d.z = det['position_3d']['z']
                else:
                    d.has_3d_position = False

                det_array.detections.append(d)

            self._pub_detections.publish(det_array)

    def _draw_detections(self, image, detections):
        """Draw detection bounding boxes on image."""
        import cv2

        annotated = image.copy()
        colors = {
            'person': (0, 255, 0),
            'vehicle': (255, 0, 0),
            'animal': (0, 165, 255),
            'unknown_moving': (0, 0, 255),
        }

        for det in detections:
            x1, y1 = int(det['x_min']), int(det['y_min'])
            x2, y2 = int(det['x_max']), int(det['y_max'])
            color = colors.get(det['class_name'], (255, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det['class_name']} {det['confidence']:.2f}"
            if det['position_3d']:
                label += f" ({det['position_3d']['z']:.1f}m)"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return annotated

    def _depth_callback(self, msg):
        """Store latest depth frame."""
        self._last_depth = msg

    def _camera_info_callback(self, msg):
        """Store camera intrinsics."""
        self._camera_matrix = list(msg.k)

    def _diagnostics_callback(self):
        """Log detection diagnostics."""
        avg_time = np.mean(self._inference_times) if self._inference_times else 0
        elapsed = time.time() - self._start_time
        fps = self._frame_count / elapsed if elapsed > 0 else 0

        self.get_logger().info(
            f'Detection [{self._model_backend}] — '
            f'FPS: {fps:.1f} | '
            f'Avg inference: {avg_time:.1f}ms | '
            f'Frames: {self._frame_count}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
