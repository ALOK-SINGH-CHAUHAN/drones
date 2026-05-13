# Tracked object with Kalman filter state
std_msgs/Header header

uint32 track_id
string class_name
float32 confidence

# Current state
geometry_msgs/Point position           # Current position (meters)
geometry_msgs/Vector3 velocity         # Current velocity (m/s)
geometry_msgs/Vector3 acceleration     # Current acceleration (m/s^2)

# Uncertainty (covariance diagonal)
float64[3] position_covariance        # x, y, z
float64[3] velocity_covariance        # vx, vy, vz

# Track status
uint8 STATUS_TENTATIVE = 0
uint8 STATUS_CONFIRMED = 1
uint8 STATUS_OCCLUDED = 2
uint8 STATUS_LOST = 3
uint8 status

uint32 age_frames                     # Number of frames tracked
uint32 hits                           # Number of detection matches
uint32 misses                         # Consecutive missed detections
