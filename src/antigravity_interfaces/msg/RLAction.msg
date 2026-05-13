# RL Decision Layer action output
std_msgs/Header header

# High-level discrete action
uint8 ACTION_CONTINUE = 0
uint8 ACTION_REPLAN = 1
uint8 ACTION_SLOW_DOWN = 2
uint8 ACTION_HOVER = 3
uint8 ACTION_AVOID_LEFT = 4
uint8 ACTION_AVOID_RIGHT = 5
uint8 ACTION_AVOID_UP = 6
uint8 ACTION_LAND = 7
uint8 action

# Policy metadata
float32 confidence              # Policy confidence [0.0-1.0]
float32[] action_probabilities  # Full action probability distribution
float32 value_estimate          # State value estimate from critic network
bool using_heuristic            # True if RL model unavailable, using heuristic
string policy_version           # Model checkpoint identifier
