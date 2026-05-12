# Global planner output — ordered waypoints forming a path
std_msgs/Header header

# Ordered waypoints from start to goal
geometry_msgs/PoseStamped[] waypoints

# Path metadata
float32 total_distance_m        # Total path distance in meters
float32 planning_time_ms        # Time taken to compute the path
string planner_type             # "astar" or "rrt_star"
bool is_valid                   # True if path was found
uint32 num_replans              # Number of replanning attempts
