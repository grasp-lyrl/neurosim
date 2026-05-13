# Vehicle state published by the neurosim simulator.
# Mirrors the DictMessage payload on the cortex "state" topic.
std_msgs/Header header
float64 timestamp
uint64  simsteps
geometry_msgs/Vector3    x   # position (world frame)
geometry_msgs/Quaternion q   # orientation [qx, qy, qz, qw]
geometry_msgs/Vector3    v   # linear velocity
geometry_msgs/Vector3    w   # angular velocity
