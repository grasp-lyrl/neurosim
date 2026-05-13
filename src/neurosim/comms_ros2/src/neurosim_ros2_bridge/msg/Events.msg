# Event camera packet published by the neurosim simulator.
# Mirrors the MultiArrayMessage payload on cortex "events/<uuid>" topics.
# All four arrays are equal length; one element per event.
std_msgs/Header header
uint16[] x   # pixel column
uint16[] y   # pixel row
uint64[] t   # timestamp (nanoseconds, monotonically increasing within a packet)
uint8[]  p   # polarity (0 or 1)
