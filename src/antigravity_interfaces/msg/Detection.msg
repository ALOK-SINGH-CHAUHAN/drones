# Single object detection from YOLOv8
# Represents a bounding box detection with class and confidence

std_msgs/Header header

# Bounding box in image coordinates (pixels)
float32 x_min
float32 y_min
float32 x_max
float32 y_max

# 3D position estimate in camera frame (meters)
geometry_msgs/Point position_3d
bool has_3d_position

# Classification
string class_name          # person, vehicle, animal, unknown_moving
uint8 class_id
float32 confidence         # 0.0 to 1.0

# Unique detection ID for tracking
uint32 detection_id
