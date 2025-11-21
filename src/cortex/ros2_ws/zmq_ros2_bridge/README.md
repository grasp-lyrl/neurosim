# ZMQ ↔ ROS2 Bridge

A bidirectional bridge for converting between ZMQ messages and ROS2 messages.

## Table of Contents

- [ZMQ ↔ ROS2 Bridge](#zmq--ros2-bridge)
  - [Table of Contents](#table-of-contents)
  - [Bridges](#bridges)
    - [1. ZMQ to ROS2 Bridge (`zmq_ros2_bridge_node`)](#1-zmq-to-ros2-bridge-zmq_ros2_bridge_node)
    - [2. ROS2 to ZMQ Bridge (`ros2_zmq_bridge_node`)](#2-ros2-to-zmq-bridge-ros2_zmq_bridge_node)
  - [Features](#features)
  - [Dependencies](#dependencies)
  - [Building](#building)
  - [ZMQ to ROS2 Bridge Usage](#zmq-to-ros2-bridge-usage)
    - [Configuration](#configuration)
    - [Topics](#topics)
    - [Parameters](#parameters)
    - [Message Formats](#message-formats)
    - [Viewing the Data](#viewing-the-data)
  - [ROS2 to ZMQ Bridge Usage](#ros2-to-zmq-bridge-usage)
    - [Launch the bridge](#launch-the-bridge)
    - [Configuration (ros2\_zmq\_bridge.yaml)](#configuration-ros2_zmq_bridgeyaml)
    - [Message Format](#message-format)
  - [Custom Message Definitions](#custom-message-definitions)
    - [`zmq_ros2_bridge/msg/Imu`](#zmq_ros2_bridgemsgimu)
    - [`zmq_ros2_bridge/msg/Event`](#zmq_ros2_bridgemsgevent)
    - [`zmq_ros2_bridge/msg/State`](#zmq_ros2_bridgemsgstate)
    - [`zmq_ros2_bridge/msg/Control`](#zmq_ros2_bridgemsgcontrol)
  - [Compilation Issues](#compilation-issues)

## Bridges

### 1. ZMQ to ROS2 Bridge (`zmq_ros2_bridge_node`)
Converts ZMQ messages (numpy arrays, JSON, and event data) to ROS2 messages.

### 2. ROS2 to ZMQ Bridge (`ros2_zmq_bridge_node`)
Converts ROS2 control messages to ZMQ messages.

## Features

- **Multi-threaded Architecture**: Separate threads for each sensor type (IMU, Color, Depth, Event, State)
- **Flexible Data Types**: 
  - Numpy arrays → `sensor_msgs/Image` messages (Color, Depth)
  - JSON data → Custom IMU and State messages
  - Dictionary of numpy arrays → Custom Event messages
- **Individual Topic Control**: Enable/disable each sensor independently
- **Per-Sensor Configuration**: Each sensor can connect to different ZMQ addresses and topics
- **Configurable via YAML**: Easy configuration through YAML parameter files

## Dependencies

- ROS2 (Humble)
- ZeroMQ (`libzmq3-dev`)
- cppzmq headers
- nlohmann_json

## Building

```bash
# Navigate to the ROS2 workspace where the package is located
cd ~/ros2_ws

# Build the package
colcon build --packages-select zmq_ros2_bridge

# Source the workspace
source install/setup.bash
```

## ZMQ to ROS2 Bridge Usage

```bash
# With default config
ros2 launch zmq_ros2_bridge zmq_ros2_bridge.launch.py

# With custom config file
ros2 launch zmq_ros2_bridge zmq_ros2_bridge.launch.py \
  config_file:=/path/to/your/config.yaml
```


### Configuration

Create a YAML file with the following structure: (See `config/zmq_ros2_bridge.yaml` for an example)

```yaml
zmq_ros2_bridge:
  ros__parameters:
    # Enable/disable flags for each sensor
    enable_imu: true
    enable_color: true
    enable_depth: true
    enable_event: true
    enable_state: true
    
    # IMU parameters (JSON data)
    imu_zmq_address: "ipc:///tmp/0"
    imu_zmq_topic: "imu"
    imu_ros2_topic: "imu"
    
    # Color image parameters (numpy array)
    color_zmq_address: "ipc:///tmp/0"
    color_zmq_topic: "color"
    color_ros2_topic: "color"
    
    # Depth image parameters (numpy array)
    depth_zmq_address: "ipc:///tmp/0"
    depth_zmq_topic: "depth"
    depth_ros2_topic: "depth"
    
    # Event parameters (dict of numpy arrays)
    event_zmq_address: "ipc:///tmp/0"
    event_zmq_topic: "events"
    event_ros2_topic: "events"
    
    # State parameters (JSON data)
    state_zmq_address: "ipc:///tmp/0"
    state_zmq_topic: "state"
    state_ros2_topic: "state"
```

### Topics

Topics are configured individually for each sensor type:

- `/imu` (`zmq_ros2_bridge/msg/Imu`) - IMU data with gyro and acceleration
- `/color` (`sensor_msgs/msg/Image`) - Color camera images
- `/depth` (`sensor_msgs/msg/Image`) - Depth camera images  
- `/events` (`zmq_ros2_bridge/msg/Event`) - Event camera data
- `/state` (`zmq_ros2_bridge/msg/State`) - Robot/drone state (position, orientation, velocities)

**Note:** Topic names are configurable via `*_ros2_topic` parameters in the config file.

### Parameters

### Enable/Disable Flags

- `enable_imu` (bool, default: false) - Enable IMU data stream
- `enable_color` (bool, default: false) - Enable color image stream
- `enable_depth` (bool, default: false) - Enable depth image stream
- `enable_event` (bool, default: false) - Enable event camera stream
- `enable_state` (bool, default: false) - Enable state data stream

### Per-Sensor Parameters

For each sensor type, you can configure:
- `<sensor>_zmq_address` (string) - ZMQ address to connect to (e.g., "ipc:///tmp/0" or "tcp://localhost:5555")
- `<sensor>_zmq_topic` (string) - ZMQ topic to subscribe to
- `<sensor>_ros2_topic` (string) - ROS2 topic name to publish to

Where `<sensor>` is one of: `imu`, `color`, `depth`, `event`, `state`

### Message Formats

### 1. Image Messages (Numpy Arrays) - Color & Depth

ZMQ multipart message format (5 parts):
1. **Topic** (string) - e.g., "color", "depth"
2. **ndim** (uint8) - Number of dimensions
3. **dtype** (string) - Data type (e.g., "uint8", "|u1", "float32", "<f4")
4. **shape** (uint32[] in network byte order) - Array dimensions [H, W] or [H, W, C]
5. **data** (raw bytes) - Image pixel data

**Supported image encodings:**
- **uint8**: mono8 (1ch), rgb8 (3ch), rgba8 (4ch)
- **uint16**: mono16 (1ch), rgb16 (3ch)
- **float32**: 32FC1 (1ch), 32FC3 (3ch), 32FC4 (4ch)
- **float64**: 64FC1 (1ch), 64FC3 (3ch), 64FC4 (4ch)

### 2. IMU Messages (JSON)

ZMQ multipart message format (2 parts):
1. **Topic** (string) - e.g., "imu"
2. **JSON data** (string):

```json
{
  "timestamp": 1234567890.123,
  "gyro": [0.0, 0.0, 0.0],      // Angular velocity [rad/s]
  "accel": [0.0, 0.0, 9.81]     // Linear acceleration [m/s²]
}
```

**ROS2 Message:** `zmq_ros2_bridge/msg/Imu`
```
std_msgs/Header header
geometry_msgs/Vector3 gyro   # Angular velocity in rad/s
geometry_msgs/Vector3 accel  # Linear acceleration in m/s²
```

### 3. State Messages (JSON)

ZMQ multipart message format (2 parts):
1. **Topic** (string) - e.g., "state"
2. **JSON data** (string):

```json
{
  "timestamp": 1234567890.123,
  "x": [1.0, 2.0, 3.0],           // Position [m]
  "q": [0.0, 0.0, 0.0, 1.0],      // Orientation quaternion [x,y,z,w]
  "v": [0.5, 0.0, 0.0],           // Linear velocity [m/s]
  "w": [0.0, 0.0, 0.1],           // Angular velocity [rad/s]
  "simsteps": 12345               // Simulation step count
}
```

**ROS2 Message:** `zmq_ros2_bridge/msg/State`
```
std_msgs/Header header
geometry_msgs/Vector3 x      # Position [m]
geometry_msgs/Quaternion q   # Orientation quaternion
geometry_msgs/Vector3 v      # Linear velocity [m/s]
geometry_msgs/Vector3 w      # Angular velocity [rad/s]
float64 timestamp            # Simulation timestamp [s]
uint64 simsteps              # Simulation step count
```

### 4. Event Messages (Dict of Numpy Arrays)

ZMQ multipart message format (variable parts):
1. **Topic** (string) - e.g., "events"
2. **key1** (string) - "x"
3. **ndim1** (uint8) - 1
4. **dtype1** (string) - "uint16" or "<u2"
5. **shape1** (uint32[] in network byte order) - [N]
6. **data1** (raw bytes) - x coordinates
7. **key2** (string) - "y"
8. ... (repeat for y, t, p)

**Required keys:**
- **x** (uint16[]) - Pixel x coordinates
- **y** (uint16[]) - Pixel y coordinates
- **t** (uint64[]) - Timestamps in microseconds
- **p** (uint8[]) - Polarity (0 or 1)

**ROS2 Message:** `zmq_ros2_bridge/msg/Event`
```
std_msgs/Header header
uint16[] x  # X coordinates
uint16[] y  # Y coordinates
uint64[] t  # Timestamps
uint8[] p   # Polarities
```

### Viewing the Data

```bash
# List all topics
ros2 topic list

# View image in RViz2
rviz2

# Or use rqt_image_view for specific topic
ros2 run rqt_image_view rqt_image_view /color

# Echo IMU data
ros2 topic echo /imu

# Echo State data
ros2 topic echo /state

# Monitor topic frequency
ros2 topic hz /color
ros2 topic hz /events

# Get topic info
ros2 topic info /imu
ros2 topic info /state
```

### Custom Message Definitions

The bridge provides four custom message types:

### `zmq_ros2_bridge/msg/Imu`
Simplified IMU message with only gyroscope and accelerometer data.

### `zmq_ros2_bridge/msg/Event`  
Event camera message containing arrays of x, y coordinates, timestamps, and polarities.

### `zmq_ros2_bridge/msg/State`
Complete robot/drone state including position, orientation, linear velocity, angular velocity, timestamp, and simulation step count.

### `zmq_ros2_bridge/msg/Control`
Control message containing motor commands (4-element float32 array for cmd_motor_speeds).

---

## ROS2 to ZMQ Bridge Usage

The ROS2 to ZMQ bridge (`ros2_zmq_bridge_node`) subscribes to ROS2 control messages and publishes them to ZMQ.

### Launch the bridge

```bash
# With default config
ros2 launch zmq_ros2_bridge ros2_zmq_bridge.launch.py

# With custom config file
ros2 launch zmq_ros2_bridge ros2_zmq_bridge.launch.py \
  config_file:=/path/to/your/config.yaml
```

### Configuration (ros2_zmq_bridge.yaml)

```yaml
/**:
  ros2__parameters:
    # Enable/disable control bridge
    enable_control: true
    
    # Control parameters
    control_zmq_address: "ipc:///tmp/1"
    control_zmq_topic: "control"
    control_ros2_topic: "control"
```

### Message Format

The bridge converts ROS2 Control messages to JSON format:

**ROS2 Message:**
```
header:
  stamp: ...
  frame_id: base_link
cmd_motor_speeds: [100.0, 100.0, 100.0, 100.0]
```

**ZMQ Message:**
```json
{
  "timestamp": 1234567890.123,
  "cmd_motor_speeds": [100.0, 100.0, 100.0, 100.0]
}
```

## Compilation Issues

If you encounter compilation issues, you may need to install specific versions of the following Python packages:

```bash
pip install empy==3.3.4 lark==1.3.1 catkin_pkg==1.1.0
```

These dependencies are required for ROS2 message generation and parsing.
