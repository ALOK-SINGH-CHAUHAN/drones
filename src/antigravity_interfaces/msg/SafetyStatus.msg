# Safety arbiter status
std_msgs/Header header

# Overall safety state
uint8 STATE_NOMINAL = 0
uint8 STATE_WARNING = 1
uint8 STATE_HOLD = 2
uint8 STATE_EMERGENCY_LAND = 3
uint8 STATE_RETURN_HOME = 4
uint8 state

# Individual trigger flags
bool battery_low              # Battery < 20%
bool sensor_failure           # Critical sensor offline
bool geofence_breach          # Outside allowed operating volume
bool obstacle_proximity       # Object within 0.5m
bool comms_loss               # Communication link lost
bool slam_tracking_lost       # SLAM lost tracking

# Diagnostics
float32 battery_percent
float32 nearest_obstacle_distance_m
float32 geofence_margin_m
string active_trigger_description
builtin_interfaces/Time last_trigger_time
