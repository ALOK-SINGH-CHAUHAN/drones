# Predicted trajectory for a tracked object
std_msgs/Header header

uint32 track_id
string class_name

# Predicted positions at future timesteps
geometry_msgs/Point[] predicted_positions      # Predicted positions
float64[] timestamps                           # Time offset from now (seconds)
float64[] confidence_scores                    # Confidence at each timestep

# Uncertainty ellipses at each predicted position
float64[] covariance_xx
float64[] covariance_yy
float64[] covariance_xy

# Prediction source
uint8 SOURCE_KALMAN = 0
uint8 SOURCE_LSTM = 1
uint8 prediction_source
float32 model_confidence
