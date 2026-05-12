# Geofence enforcement status
std_msgs/Header header

# Fence type
uint8 FENCE_CYLINDER = 0
uint8 FENCE_POLYGON = 1
uint8 fence_type

# Violation state
bool inside_fence              # True if drone is within the geofence boundary
bool soft_margin_violated      # True if within soft warning margin
bool hard_margin_violated      # True if within hard stop margin

# Distances
float32 distance_to_boundary_m   # Distance to nearest boundary edge [m]
float32 soft_margin_m            # Soft margin distance [m]
float32 hard_margin_m            # Hard margin distance [m]

# Fence geometry (for RViz visualization)
geometry_msgs/Point[] fence_vertices   # 2D/3D polygon vertices or circle params
float32 max_altitude_m                 # Maximum allowed altitude [m]
float32 min_altitude_m                 # Minimum allowed altitude [m]
