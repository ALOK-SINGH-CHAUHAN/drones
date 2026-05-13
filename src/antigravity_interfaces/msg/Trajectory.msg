# Full trajectory — array of setpoints from optimizer
std_msgs/Header header

antigravity_interfaces/TrajectoryPoint[] points

# Trajectory metadata
float32 total_duration_s        # Total trajectory duration
float32 max_velocity_mps        # Maximum velocity in trajectory
float32 max_acceleration_mps2   # Maximum acceleration in trajectory
bool is_feasible                # True if dynamics constraints satisfied
string optimizer_type           # "minimum_snap" or "mpc"
