# Localization state from particle filter
std_msgs/Header header

# Best estimate pose on the map
geometry_msgs/PoseWithCovariance pose

# Particle filter diagnostics
uint32 num_particles
float32 effective_particle_count    # Neff for resampling decision
float32 convergence_score           # 0.0 = dispersed, 1.0 = converged

# Localization quality
uint8 QUALITY_LOST = 0
uint8 QUALITY_POOR = 1
uint8 QUALITY_FAIR = 2
uint8 QUALITY_GOOD = 3
uint8 QUALITY_EXCELLENT = 4
uint8 quality

# Map alignment error
float32 map_match_score             # Scan-to-map match quality 0.0-1.0
bool is_relocalized                 # True after successful relocalization
