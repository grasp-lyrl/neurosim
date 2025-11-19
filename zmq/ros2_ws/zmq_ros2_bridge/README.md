# ZMQ to ROS2 Bridge

A lightweight bridge for converting ZMQ messages (numpy arrays and JSON) to ROS2 messages (Image and IMU).

## Features

- **Multiple ZMQ Subscribers**: Run multiple ZMQ subscribers on separate threads, each connecting to different ZMQ publishers
- **Thread-Safe Publishing**: Dynamic creation of ROS2 publishers with thread-safe access
- **Flexible Topic Configuration**: Configure multiple ZMQ topics per subscriber with custom ROS2 topic prefixes
- **Multi-Format Support**: 
  - Numpy arrays → `sensor_msgs/Image` messages
  - JSON data → `sensor_msgs/Imu` messages
- **Configurable via YAML**: Easy configuration through YAML parameter files

## Dependencies

- ROS2 (Humble or later)
- ZeroMQ (`libzmq3-dev`)
- cppzmq headers
- nlohmann_json

## Building

```bash
# Navigate to your ROS2 workspace
cd ~/ros2_ws

# Build the package
colcon build --packages-select zmq_ros2_bridge

# Source the workspace
source install/setup.bash
```

## Usage

### Using launch file with config file (recommended)

```bash
# With default single subscriber config
ros2 launch zmq_ros2_bridge zmq_ros2_bridge.launch.py

# With multi-subscriber config
ros2 launch zmq_ros2_bridge zmq_ros2_bridge.launch.py \
  config_file:=/path/to/your/config.yaml

# With the example multi-subscriber config
ros2 launch zmq_ros2_bridge zmq_ros2_bridge.launch.py \
  config_file:=$(ros2 pkg prefix zmq_ros2_bridge)/share/zmq_ros2_bridge/config/multi_subscriber_example.yaml
```

### Running directly with parameters

```bash
ros2 run zmq_ros2_bridge zmq_ros2_bridge_node \
  --ros-args \
  --params-file /path/to/config.yaml
```

## Configuration

### Configuration File Format

Create a YAML file with the following structure:

```yaml
/**:
  ros__parameters:
    # Number of ZMQ subscribers to create
    num_subscribers: 2
    
    # Configuration for subscriber 0
    subscriber_0:
      zmq_address: "tcp://localhost:5555"
      zmq_topics: ["color", "depth", "imu"]
      ros2_topic_prefix: "drone1"
    
    # Configuration for subscriber 1
    subscriber_1:
      zmq_address: "tcp://localhost:5556"
      zmq_topics: ["color", "depth", "imu"]
      ros2_topic_prefix: "drone2"
```

### Example Configurations

Two example configuration files are provided:

1. **`config/single_subscriber.yaml`** - Single subscriber (backward compatible)
2. **`config/multi_subscriber_example.yaml`** - Two subscribers example

## Topics

### Published Topics

Topics are dynamically created based on configuration. Format: `<ros2_topic_prefix>/<zmq_topic>`

**Example with config above:**
- `/drone1/color` (`sensor_msgs/Image`)
- `/drone1/depth` (`sensor_msgs/Image`)
- `/drone1/imu` (`sensor_msgs/Imu`)
- `/drone2/color` (`sensor_msgs/Image`)
- `/drone2/depth` (`sensor_msgs/Image`)
- `/drone2/imu` (`sensor_msgs/Imu`)

## Parameters

### Per-Node Parameters

- `num_subscribers` (int, default: 1) - Number of ZMQ subscribers to create

### Per-Subscriber Parameters

For each subscriber `i` (where i = 0, 1, 2, ...):

- `subscriber_<i>.zmq_address` (string) - ZMQ publisher address to connect to
- `subscriber_<i>.zmq_topics` (string array) - List of ZMQ topics to subscribe to
- `subscriber_<i>.ros2_topic_prefix` (string) - Prefix for ROS2 topic names

## Message Formats

### Image Messages (Numpy Arrays)

The bridge expects ZMQ messages in the following 5-part multipart format:
1. Topic (string) - e.g., "color", "depth"
2. Header (4 uint32_t in network byte order: ndim, dtype_len, shape_len, nbytes)
3. Dtype (string) - e.g., "uint8", "|u1"
4. Shape (array of uint32_t in network byte order) - e.g., [480, 640, 3]
5. Array data (raw bytes)

**Supported image formats:**
- RGB8 (HxWx3, uint8)
- RGBA8 (HxWx4, uint8)
- MONO8 (HxWx1, uint8)

### IMU Messages (JSON)

The bridge expects ZMQ messages in the following 2-part multipart format:
1. Topic (string) - e.g., "imu"
2. JSON data (string) with the following structure:

```json
{
  "timestamp": 1234567890.123,
  "orientation": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "w": 1.0
  },
  "angular_velocity": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0
  },
  "linear_acceleration": {
    "x": 0.0,
    "y": 9.81,
    "z": 0.0
  }
}
```

## Architecture

The bridge creates one thread per ZMQ subscriber. Each thread:
1. Connects to its configured ZMQ address
2. Subscribes to multiple ZMQ topics
3. Receives and decodes messages (numpy arrays or JSON)
4. Publishes to dynamically-created ROS2 topics

**Thread Safety:** Publishers are created on-demand with mutex protection to ensure thread-safe operation.

## Use Cases

### Single Camera System
```yaml
num_subscribers: 1
subscriber_0:
  zmq_address: "tcp://localhost:5555"
  zmq_topics: ["color", "depth"]
  ros2_topic_prefix: "camera"
```
**Output topics:** `/camera/color`, `/camera/depth`

### Multi-Robot System
```yaml
num_subscribers: 3
subscriber_0:
  zmq_address: "tcp://robot1:5555"
  zmq_topics: ["color", "imu"]
  ros2_topic_prefix: "robot1"
subscriber_1:
  zmq_address: "tcp://robot2:5555"
  zmq_topics: ["color", "imu"]
  ros2_topic_prefix: "robot2"
subscriber_2:
  zmq_address: "tcp://robot3:5555"
  zmq_topics: ["color", "imu"]
  ros2_topic_prefix: "robot3"
```
**Output topics:** `/robot1/color`, `/robot1/imu`, `/robot2/color`, `/robot2/imu`, `/robot3/color`, `/robot3/imu`

## Viewing the Data

```bash
# List all topics
ros2 topic list

# View image in RViz2
rviz2

# Or use rqt_image_view for specific topic
ros2 run rqt_image_view rqt_image_view /drone1/color

# Echo IMU data
ros2 topic echo /drone1/imu

# Monitor topic frequency
ros2 topic hz /drone1/color
```
