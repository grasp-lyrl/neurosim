# Custom IMU message with only the fields we use
# Header with timestamp and frame info

std_msgs/Header header

# Angular velocity (gyro) in rad/s
geometry_msgs/Vector3 gyro

# Linear acceleration (accel) in m/s^2
geometry_msgs/Vector3 accel
