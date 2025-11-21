# State message containing position, orientation, velocity, and angular velocity
# Header with timestamp and frame info

std_msgs/Header header

# Position (x, y, z) in meters
geometry_msgs/Vector3 x

# Orientation quaternion (x, y, z, w)
geometry_msgs/Quaternion q

# Linear velocity (vx, vy, vz) in m/s
geometry_msgs/Vector3 v

# Angular velocity (wx, wy, wz) in rad/s
geometry_msgs/Vector3 w

# Simulation timestamp in seconds
float64 timestamp

# Simulation step count
uint64 simsteps
