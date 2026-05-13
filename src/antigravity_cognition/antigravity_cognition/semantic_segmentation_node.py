"""
ANTIGRAVITY — Semantic Segmentation Node
==========================================
SAM (Segment Anything Model) + custom classifier for environment semantics.
Classes: doorway, corridor, wall, window, road, landing_zone, obstacle.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import time
import threading


CLASS_NAMES = ['unknown','doorway','corridor','wall','window','road','landing_zone','obstacle','floor','ceiling']
CLASS_COLORS = [(128,128,128),(0,255,0),(255,255,0),(0,0,255),(0,255,255),(128,128,0),(255,0,255),(255,0,0),(200,200,200),(100,100,100)]


class SemanticSegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')
        self.declare_parameter('sam_model_path', 'models/sam_vit_b.pt')
        self.declare_parameter('classifier_model_path', 'models/semantic_classifier.pt')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('inference_rate_hz', 10)
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('num_classes', 10)
        self.declare_parameter('points_per_side', 16)

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)

        self._pub_semantic = self.create_publisher(Image, 'semantics/segmented_image', reliable_qos)
        self._pub_semantic_raw = self.create_publisher(Image, 'semantics/raw_labels', reliable_qos)
        self._pub_labels = self.create_publisher(String, 'semantics/labels_json', reliable_qos)
        self._sub_image = self.create_subscription(Image, '/camera/image_raw', self._image_cb, sensor_qos)

        self._bridge = CvBridge()
        self._sam_model = None
        self._classifier = None
        self._last_image = None
        self._lock = threading.Lock()
        self._count = 0
        self._times = []

        self._init_models()
        rate = self.get_parameter('inference_rate_hz').value
        self.create_timer(1.0 / rate, self._process)
        self.create_timer(5.0, self._diag)
        self.get_logger().info('Semantic segmentation node initialized')

    def _init_models(self):
        """Initialize SAM and classifier models."""
        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            device = self.get_parameter('device').value
            sam_path = self.get_parameter('sam_model_path').value
            sam = sam_model_registry['vit_b'](checkpoint=sam_path)
            sam.to(device)
            self._sam_model = SamAutomaticMaskGenerator(
                sam, points_per_side=self.get_parameter('points_per_side').value,
                pred_iou_thresh=0.86, stability_score_thresh=0.92,
            )
            self._model_backend = 'sam'
            self.get_logger().info('SAM model loaded')
        except Exception as e:
            self.get_logger().warn(f'SAM not available ({e}). Using simulated segmentation.')
            self._model_backend = 'simulated'

        try:
            import torch
            cls_path = self.get_parameter('classifier_model_path').value
            self._classifier = torch.load(cls_path, map_location='cpu')
            self._classifier.eval()
            self.get_logger().info('Custom classifier loaded')
        except Exception:
            self._classifier = None

    def _image_cb(self, msg):
        self._last_image = msg

    def _process(self):
        if self._last_image is None: return
        if not self._lock.acquire(blocking=False): return
        try:
            t0 = time.time()
            cv_img = self._bridge.imgmsg_to_cv2(self._last_image, 'bgr8')
            if self._model_backend == 'sam':
                labels = self._segment_sam(cv_img)
            else:
                labels = self._segment_simulated(cv_img)
            dt = (time.time() - t0) * 1000
            self._times.append(dt)
            if len(self._times) > 50: self._times.pop(0)
            seg_img = self._colorize(cv_img, labels)
            msg = self._bridge.cv2_to_imgmsg(seg_img, 'bgr8')
            msg.header = self._last_image.header
            self._pub_semantic.publish(msg)

            # Publish raw labels for world model
            msg_raw = self._bridge.cv2_to_imgmsg(labels, 'mono8')
            msg_raw.header = self._last_image.header
            self._pub_semantic_raw.publish(msg_raw)

            import json
            lbl_msg = String()
            unique = np.unique(labels)
            lbl_msg.data = json.dumps({
                'classes': [CLASS_NAMES[i] for i in unique if i < len(CLASS_NAMES)],
                'inference_ms': dt
            })
            self._pub_labels.publish(lbl_msg)
            self._count += 1
        finally:
            self._lock.release()

    def _segment_sam(self, image):
        """Run SAM + classifier pipeline."""
        import torch
        rgb = image[:, :, ::-1]  # BGR to RGB
        masks = self._sam_model.generate(rgb)
        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.uint8)
        conf_thresh = self.get_parameter('confidence_threshold').value
        for mask_data in sorted(masks, key=lambda x: x['area'], reverse=True):
            mask = mask_data['segmentation']
            if self._classifier:
                crop = image[mask]
                if len(crop) > 0:
                    cls_id = self._classify_region(image, mask)
                    labels[mask] = cls_id
            else:
                labels[mask] = self._heuristic_classify(mask_data, h, w)
        return labels

    def _classify_region(self, image, mask):
        """Classify a segmented region using the custom classifier."""
        import torch, cv2
        ys, xs = np.where(mask)
        if len(ys) == 0: return 0
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        crop = image[y1:y2+1, x1:x2+1]
        crop = cv2.resize(crop, (64, 64))
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            output = self._classifier(tensor)
            return int(torch.argmax(output, dim=1).item())

    def _heuristic_classify(self, mask_data, h, w):
        """Heuristic classification when no classifier is available."""
        mask = mask_data['segmentation']
        area_ratio = mask_data['area'] / (h * w)
        ys, xs = np.where(mask)
        cy = ys.mean() / h
        if area_ratio > 0.3 and cy > 0.7: return 8  # floor
        if area_ratio > 0.3 and cy < 0.3: return 9  # ceiling
        if area_ratio > 0.1: return 3  # wall
        bbox = mask_data.get('bbox', [0,0,0,0])
        aspect = (bbox[3]+1) / (bbox[2]+1) if bbox[2] > 0 else 1
        if aspect > 1.5 and area_ratio < 0.05: return 1  # doorway
        return 7  # obstacle

    def _segment_simulated(self, image):
        """Generate simulated semantic labels."""
        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.uint8)
        labels[:h//5, :] = 9       # ceiling
        labels[4*h//5:, :] = 8     # floor
        labels[h//5:4*h//5, :w//8] = 3  # left wall
        labels[h//5:4*h//5, 7*w//8:] = 3  # right wall
        labels[h//3:2*h//3, 3*w//8:5*w//8] = 2  # corridor
        dw = w // 10
        labels[h//4:3*h//4, w//2-dw:w//2+dw] = 1  # doorway
        return labels

    def _colorize(self, image, labels):
        """Overlay semantic colors on image."""
        overlay = image.copy()
        for cls_id in range(len(CLASS_COLORS)):
            mask = labels == cls_id
            if mask.any():
                color = CLASS_COLORS[cls_id]
                overlay[mask] = (np.array(overlay[mask], dtype=np.float32) * 0.5 +
                                np.array(color, dtype=np.float32) * 0.5).astype(np.uint8)
        return overlay

    def _diag(self):
        avg = np.mean(self._times) if self._times else 0
        self.get_logger().info(f'Semantics [{self._model_backend}] — Avg: {avg:.1f}ms | Frames: {self._count}')


def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegmentationNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
