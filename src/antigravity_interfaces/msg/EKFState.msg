# Extended Kalman Filter fused state estimate
std_msgs/Header header

# Fused pose (authoritative drone state)
geometry_msgs/PoseWithCovariance pose

# Fused velocity
geometry_msgs/TwistWithCovariance twist

# Accelerometer bias estimate
geometry_msgs/Vector3 accel_bias

# Gyroscope bias estimate
geometry_msgs/Vector3 gyro_bias

# Source contributions (0.0-1.0 innovation weights)
float32 slam_weight
float32 mcl_weight
float32 imu_weight
float32 baro_weight

# Quality metrics
float32 nees                    # Normalized Estimation Error Squared
bool is_converged               # Filter convergence status
uint32 outliers_rejected        # Number of outlier measurements rejected
float32 update_rate_hz          # Actual EKF update rate
