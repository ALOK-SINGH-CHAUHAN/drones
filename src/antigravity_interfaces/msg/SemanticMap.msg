# Semantic map with labeled voxels/regions
std_msgs/Header header

# Labeled regions in the current frame
geometry_msgs/Point[] positions         # 3D positions of labeled voxels
antigravity_interfaces/SemanticLabel[] labels  # Corresponding labels
uint32 num_labels

# Image segmentation mask (optional)
sensor_msgs/Image segmentation_mask
bool has_mask

float32 inference_time_ms
