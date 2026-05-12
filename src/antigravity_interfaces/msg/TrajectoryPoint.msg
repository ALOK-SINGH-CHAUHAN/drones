# Single trajectory setpoint from the trajectory optimizer
std_msgs/Header header

# Position reference
geometry_msgs/Point position            # [m] Target position in world frame
geometry_msgs/Vector3 velocity          # [m/s] Desired velocity
geometry_msgs/Vector3 acceleration      # [m/s^2] Desired acceleration
geometry_msgs/Vector3 jerk              # [m/s^3] Desired jerk (for minimum snap)

# Orientation
float64 yaw                             # [rad] Desired yaw angle
float64 yaw_rate                        # [rad/s] Desired yaw rate

# Timing
float64 time_from_start                 # [s] Time offset from trajectory start
