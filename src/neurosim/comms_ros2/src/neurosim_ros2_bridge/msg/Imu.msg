# IMU reading published by the neurosim simulator.
# Mirrors the DictMessage payload on cortex "imu/<uuid>" topics.
std_msgs/Header header
string  uuid
float64 timestamp
uint64  simsteps
geometry_msgs/Vector3 accel
geometry_msgs/Vector3 gyro
