# System health monitoring
std_msgs/Header header

# Node health
string[] node_names
bool[] node_alive
float32[] node_frequencies_hz    # Actual publish rates
float32[] node_expected_hz       # Expected publish rates

# Hardware health
float32 cpu_usage_percent
float32 gpu_usage_percent
float32 ram_usage_mb
float32 temperature_celsius
float32 battery_voltage
float32 battery_current_a

# Communication latency
float32 mavlink_latency_ms
float32 slam_latency_ms
float32 detection_latency_ms
