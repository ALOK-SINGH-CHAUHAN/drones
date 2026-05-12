"""
ANTIGRAVITY — Camera Driver Node
=================================
ROS2 node for Intel RealSense D435i / ZED 2 camera integration.
Publishes synchronized color, depth, and infrared streams.

Acceptance Criteria:
  - Camera publishes at >= 30 Hz
  - Supports RealSense D435i and ZED 2 cameras
  - Publishes camera_info for calibration data
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge

import numpy as np
import time
import threading


class CameraNode(Node):
    """
    Camera driver node supporting Intel RealSense D435i and ZED 2.
    
    Publishes:
      - camera/image_raw (sensor_msgs/Image): Color image at target FPS
      - camera/depth (sensor_msgs/Image): Depth image aligned to color
      - camera/camera_info (sensor_msgs/CameraInfo): Camera calibration
      - camera/infra1 (sensor_msgs/Image): Left infrared (RealSense only)
      - camera/infra2 (sensor_msgs/Image): Right infrared (RealSense only)
    """

    def __init__(self):
        super().__init__('camera_driver')

        # ─── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('camera_type', 'realsense')
        self.declare_parameter('serial_number', '')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('color_format', 'bgr8')
        self.declare_parameter('depth_enabled', True)
        self.declare_parameter('depth_width', 640)
        self.declare_parameter('depth_height', 480)
        self.declare_parameter('depth_fps', 30)
        self.declare_parameter('min_depth_m', 0.3)
        self.declare_parameter('max_depth_m', 10.0)
        self.declare_parameter('enable_infra1', True)
        self.declare_parameter('enable_infra2', True)
        self.declare_parameter('emitter_enabled', True)
        self.declare_parameter('enable_auto_exposure', True)
        self.declare_parameter('exposure_us', 8000)
        self.declare_parameter('gain', 16)
        self.declare_parameter('camera_frame_id', 'camera_link')
        self.declare_parameter('optical_frame_id', 'camera_optical_frame')

        self._camera_type = self.get_parameter('camera_type').value
        self._width = self.get_parameter('image_width').value
        self._height = self.get_parameter('image_height').value
        self._fps = self.get_parameter('fps').value
        self._frame_id = self.get_parameter('optical_frame_id').value
        self._depth_enabled = self.get_parameter('depth_enabled').value

        # ─── QoS Profile ────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # ─── Publishers ─────────────────────────────────────────────────
        self._pub_color = self.create_publisher(Image, 'camera/image_raw', sensor_qos)
        self._pub_depth = self.create_publisher(Image, 'camera/depth', sensor_qos)
        self._pub_info = self.create_publisher(CameraInfo, 'camera/camera_info', sensor_qos)
        self._pub_infra1 = self.create_publisher(Image, 'camera/infra1', sensor_qos)
        self._pub_infra2 = self.create_publisher(Image, 'camera/infra2', sensor_qos)

        self._bridge = CvBridge()
        self._pipeline = None
        self._camera_info_msg = None
        self._frame_count = 0
        self._start_time = time.time()

        # ─── Initialize Camera ──────────────────────────────────────────
        self._init_camera()

        # ─── Capture Timer ──────────────────────────────────────────────
        timer_period = 1.0 / self._fps
        self._capture_timer = self.create_timer(timer_period, self._capture_callback)

        # ─── Diagnostics Timer ───────────────────────────────────────────
        self._diag_timer = self.create_timer(5.0, self._diagnostics_callback)

        self.get_logger().info(
            f'Camera node initialized: {self._camera_type} @ {self._width}x{self._height} {self._fps}fps'
        )

    def _init_camera(self):
        """Initialize camera hardware based on camera_type parameter."""
        if self._camera_type == 'realsense':
            self._init_realsense()
        elif self._camera_type == 'zed':
            self._init_zed()
        else:
            self.get_logger().warn(
                f'Unknown camera type: {self._camera_type}. Using simulated camera.'
            )
            self._init_simulated()

    def _init_realsense(self):
        """Initialize Intel RealSense D435i pipeline."""
        try:
            import pyrealsense2 as rs

            self._pipeline = rs.pipeline()
            config = rs.config()

            serial = self.get_parameter('serial_number').value
            if serial:
                config.enable_device(serial)

            # Enable color stream
            config.enable_stream(
                rs.stream.color,
                self._width, self._height,
                rs.format.bgr8, self._fps
            )

            # Enable depth stream
            if self._depth_enabled:
                config.enable_stream(
                    rs.stream.depth,
                    self.get_parameter('depth_width').value,
                    self.get_parameter('depth_height').value,
                    rs.format.z16,
                    self.get_parameter('depth_fps').value
                )

            # Enable infrared streams
            if self.get_parameter('enable_infra1').value:
                config.enable_stream(rs.stream.infrared, 1, self._width, self._height, rs.format.y8, self._fps)
            if self.get_parameter('enable_infra2').value:
                config.enable_stream(rs.stream.infrared, 2, self._width, self._height, rs.format.y8, self._fps)

            # Start pipeline
            profile = self._pipeline.start(config)

            # Configure sensor settings
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()

            if self.get_parameter('emitter_enabled').value:
                depth_sensor.set_option(rs.option.emitter_enabled, 1)

            # Align depth to color
            self._align = rs.align(rs.stream.color)

            # Build camera info from intrinsics
            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            self._camera_info_msg = self._build_camera_info(intrinsics)

            self._camera_backend = 'realsense'
            self.get_logger().info('RealSense D435i initialized successfully')

        except ImportError:
            self.get_logger().error('pyrealsense2 not installed. Falling back to simulated camera.')
            self._init_simulated()
        except Exception as e:
            self.get_logger().error(f'RealSense initialization failed: {e}. Falling back to simulated camera.')
            self._init_simulated()

    def _init_zed(self):
        """Initialize ZED 2 camera via pyzed SDK."""
        try:
            import pyzed.sl as sl

            self._zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.VGA
            init_params.camera_fps = self._fps
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_minimum_distance = self.get_parameter('min_depth_m').value

            err = self._zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f'ZED open failed: {err}')

            # Pre-allocate mats
            self._zed_image = sl.Mat()
            self._zed_depth = sl.Mat()

            # Build camera info
            calib = self._zed.get_camera_information().camera_configuration.calibration_parameters
            self._camera_info_msg = self._build_camera_info_from_zed(calib)

            self._camera_backend = 'zed'
            self.get_logger().info('ZED 2 camera initialized successfully')

        except ImportError:
            self.get_logger().error('pyzed not installed. Falling back to simulated camera.')
            self._init_simulated()
        except Exception as e:
            self.get_logger().error(f'ZED initialization failed: {e}. Falling back to simulated camera.')
            self._init_simulated()

    def _init_simulated(self):
        """Initialize simulated camera for development/testing."""
        self._camera_backend = 'simulated'
        self._camera_info_msg = CameraInfo()
        self._camera_info_msg.header.frame_id = self._frame_id
        self._camera_info_msg.width = self._width
        self._camera_info_msg.height = self._height
        # RealSense D435i typical intrinsics
        fx, fy = 615.6707, 615.9621
        cx, cy = 326.0741, 240.2530
        self._camera_info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self._camera_info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._camera_info_msg.distortion_model = 'plumb_bob'

        self.get_logger().warn('Using SIMULATED camera — no real sensor data')

    def _capture_callback(self):
        """Main capture loop — grab frames and publish."""
        stamp = self.get_clock().now().to_msg()

        if self._camera_backend == 'realsense':
            self._capture_realsense(stamp)
        elif self._camera_backend == 'zed':
            self._capture_zed(stamp)
        else:
            self._capture_simulated(stamp)

        self._frame_count += 1

    def _capture_realsense(self, stamp):
        """Capture and publish frames from RealSense."""
        import pyrealsense2 as rs

        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        aligned = self._align.process(frames)

        # Color frame
        color_frame = aligned.get_color_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            color_msg = self._bridge.cv2_to_imgmsg(color_image, 'bgr8')
            color_msg.header.stamp = stamp
            color_msg.header.frame_id = self._frame_id
            self._pub_color.publish(color_msg)

        # Depth frame
        if self._depth_enabled:
            depth_frame = aligned.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_msg = self._bridge.cv2_to_imgmsg(depth_image, '16UC1')
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = self._frame_id
                self._pub_depth.publish(depth_msg)

        # Infrared frames
        infra1 = frames.get_infrared_frame(1)
        if infra1:
            infra1_image = np.asanyarray(infra1.get_data())
            infra1_msg = self._bridge.cv2_to_imgmsg(infra1_image, 'mono8')
            infra1_msg.header.stamp = stamp
            infra1_msg.header.frame_id = self._frame_id
            self._pub_infra1.publish(infra1_msg)

        infra2 = frames.get_infrared_frame(2)
        if infra2:
            infra2_image = np.asanyarray(infra2.get_data())
            infra2_msg = self._bridge.cv2_to_imgmsg(infra2_image, 'mono8')
            infra2_msg.header.stamp = stamp
            infra2_msg.header.frame_id = self._frame_id
            self._pub_infra2.publish(infra2_msg)

        # Camera info
        self._camera_info_msg.header.stamp = stamp
        self._pub_info.publish(self._camera_info_msg)

    def _capture_zed(self, stamp):
        """Capture and publish frames from ZED 2."""
        import pyzed.sl as sl

        if self._zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Color
            self._zed.retrieve_image(self._zed_image, sl.VIEW.LEFT)
            color_image = self._zed_image.get_data()[:, :, :3]  # Remove alpha
            color_msg = self._bridge.cv2_to_imgmsg(color_image, 'bgr8')
            color_msg.header.stamp = stamp
            color_msg.header.frame_id = self._frame_id
            self._pub_color.publish(color_msg)

            # Depth
            if self._depth_enabled:
                self._zed.retrieve_measure(self._zed_depth, sl.MEASURE.DEPTH)
                depth_data = self._zed_depth.get_data()
                depth_mm = (depth_data * 1000).astype(np.uint16)
                depth_msg = self._bridge.cv2_to_imgmsg(depth_mm, '16UC1')
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = self._frame_id
                self._pub_depth.publish(depth_msg)

            # Camera info
            self._camera_info_msg.header.stamp = stamp
            self._pub_info.publish(self._camera_info_msg)

    def _capture_simulated(self, stamp):
        """Generate simulated camera frames for testing."""
        # Simulated color image (gradient pattern)
        color_image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        t = time.time()
        for i in range(3):
            color_image[:, :, i] = (
                np.sin(np.linspace(0, 2 * np.pi, self._width) + t * (i + 1)) * 127 + 128
            ).astype(np.uint8)

        color_msg = self._bridge.cv2_to_imgmsg(color_image, 'bgr8')
        color_msg.header.stamp = stamp
        color_msg.header.frame_id = self._frame_id
        self._pub_color.publish(color_msg)

        # Simulated depth (flat plane at 2m)
        if self._depth_enabled:
            depth_image = np.full((self._height, self._width), 2000, dtype=np.uint16)
            depth_msg = self._bridge.cv2_to_imgmsg(depth_image, '16UC1')
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = self._frame_id
            self._pub_depth.publish(depth_msg)

        self._camera_info_msg.header.stamp = stamp
        self._pub_info.publish(self._camera_info_msg)

    def _build_camera_info(self, intrinsics):
        """Build CameraInfo message from RealSense intrinsics."""
        msg = CameraInfo()
        msg.header.frame_id = self._frame_id
        msg.width = intrinsics.width
        msg.height = intrinsics.height
        msg.distortion_model = 'plumb_bob'
        msg.d = list(intrinsics.coeffs)
        msg.k = [
            intrinsics.fx, 0.0, intrinsics.ppx,
            0.0, intrinsics.fy, intrinsics.ppy,
            0.0, 0.0, 1.0
        ]
        msg.p = [
            intrinsics.fx, 0.0, intrinsics.ppx, 0.0,
            0.0, intrinsics.fy, intrinsics.ppy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        return msg

    def _build_camera_info_from_zed(self, calib):
        """Build CameraInfo from ZED calibration parameters."""
        left = calib.left_cam
        msg = CameraInfo()
        msg.header.frame_id = self._frame_id
        msg.width = int(left.image_size.width)
        msg.height = int(left.image_size.height)
        msg.distortion_model = 'plumb_bob'
        msg.d = list(left.disto)
        msg.k = [
            left.fx, 0.0, left.cx,
            0.0, left.fy, left.cy,
            0.0, 0.0, 1.0
        ]
        msg.p = [
            left.fx, 0.0, left.cx, 0.0,
            0.0, left.fy, left.cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        return msg

    def _diagnostics_callback(self):
        """Log camera diagnostics periodically."""
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            actual_fps = self._frame_count / elapsed
            self.get_logger().info(
                f'Camera [{self._camera_backend}] — '
                f'FPS: {actual_fps:.1f}/{self._fps} | '
                f'Frames: {self._frame_count} | '
                f'Resolution: {self._width}x{self._height}'
            )

    def destroy_node(self):
        """Clean up camera resources."""
        if self._camera_backend == 'realsense' and self._pipeline:
            self._pipeline.stop()
        elif self._camera_backend == 'zed':
            self._zed.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
