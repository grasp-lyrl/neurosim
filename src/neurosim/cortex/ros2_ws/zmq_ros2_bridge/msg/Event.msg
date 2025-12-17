# Event message for neuromorphic camera events
# Arrays of x, y, t (timestamp), p (polarity)

std_msgs/Header header

uint16[] x
uint16[] y
uint64[] t
uint8[] p
