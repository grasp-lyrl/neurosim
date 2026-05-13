# neurosim_ros2_bridge

A composable ROS 2 bridge for the neurosim simulator's Cortex pub/sub topics.

```
┌───────────────────────────────────┐                 ┌──────────────────────┐
│ simulator_node (Python, cortex)   │                 │ rviz2 / consumers    │
│   state, imu/*, color/*, depth/*, │                 │ /neurosim/state      │
│   events/*                        │                 │ /neurosim/camera/... │
└───────────────┬───────────────────┘                 └──────────▲───────────┘
                │                                                │
                │ ZMQ multipart (cortex wire: header+msgpack+OOB)│
                ▼                                                │
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ComposableNodeContainer                               │
│                  ┌─────────────────────────────┐                            │
│                  │   NeurosimRos2Bridge        │                            │
│                  │   one SUB thread per topic  │                            │
│                  │   one PUB socket per topic  │                            │
│                  └─────────────────────────────┘                            │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              │  discovery REQ/REP (ipc:///tmp/cortex/discovery.sock)
                              ▼
                       cortex-discovery (Python daemon)
```

This package is intentionally narrow: it understands the exact set of Cortex
messages the neurosim simulator emits (and consumes), not arbitrary Cortex
types. Each `(payload tag) -> (ROS 2 type)` mapping is a single C++ decoder
function. No type-erased adapter registry, no factory pattern — see
[`include/neurosim_ros2_bridge/decoders.hpp`](include/neurosim_ros2_bridge/decoders.hpp).

All ZMQ + cortex protocol machinery (SUB thread, fingerprint check, multipart
encode/decode, discovery register/unregister, ipc:// endpoint slugification)
lives in `cortex_wire_cpp` —
the bridge just instantiates `cortex_wire::Subscriber` per inbound entry and
`cortex_wire::Publisher` per outbound entry. See
`cortex_wire_cpp/DOCS.md` for the underlying client's feature surface.

For a generic Cortex<->ROS 2 bridge with a pluggable adapter system, see
`deps/cortex/ros2_bridge`.

## Streams supported

| Direction | Cortex topic / type | ROS 2 topic / type |
| --- | --- | --- |
| cortex → ROS 2 | `state` (DictMessage) | `/neurosim/state` (`neurosim_ros2_bridge/msg/State`) |
| cortex → ROS 2 | `imu/<uuid>` (DictMessage) | `/neurosim/imu/<uuid>` (`neurosim_ros2_bridge/msg/Imu`) |
| cortex → ROS 2 | `color/<uuid>` (ArrayMessage `u1` HxWx3) | `/neurosim/camera/color/image_raw` (`sensor_msgs/msg/Image`, `rgb8`) |
| cortex → ROS 2 | `depth/<uuid>` (ArrayMessage `f4` HxW) | `/neurosim/camera/depth/image_raw` (`sensor_msgs/msg/Image`, `32FC1`) |
| cortex → ROS 2 | `events/<uuid>` (MultiArrayMessage) | `/neurosim/camera/events/<uuid>` (`neurosim_ros2_bridge/msg/Events`) |
| ROS 2 → cortex | `/neurosim/control` (`std_msgs/Float64MultiArray`) | `control` (DictMessage) |

Color/depth become `sensor_msgs/Image` directly so rviz2's Image display
renders them with no extra plumbing. State / IMU / Events use custom message
types because the wire payloads (especially events' four parallel arrays)
don't fit any stdlib type cleanly.

## Build

Dependencies:

- ROS 2 Humble (`ros-humble-desktop`)
- `libzmq3-dev`, `cppzmq` (header-only), `libmsgpack-dev`, `libyaml-cpp-dev`
- `cortex_wire_cpp` — at `deps/cortex/cortex_wire_cpp/`.
  Build and install it before this package
  (`cmake -S deps/cortex/cortex_wire_cpp -B build && cmake --build build && sudo cmake --install build`).
  The bridge's CMakeLists uses plain `find_package(cortex_wire_cpp REQUIRED)`
  and fails loudly if it isn't on `CMAKE_PREFIX_PATH`.

From a colcon workspace that contains this package under `src/`:

```bash
colcon build --packages-select neurosim_ros2_bridge \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
               -DPYTHON_EXECUTABLE=/usr/bin/python3
```

## Configure

A bridge YAML enumerates every Cortex<->ROS 2 mapping. Schema:

```yaml
version: 1
discovery_address: "ipc:///tmp/cortex/discovery.sock"
node_name_prefix: "neurosim_bridge"

cortex_to_ros2:
  - name: state                       # human-readable; used in logs
    cortex_topic: state               # cortex topic name
    ros2_topic: /neurosim/state       # ROS 2 topic name
    payload: state                    # which decoder to use (see below)
    frame_id: world                   # stamped on the outbound ROS msg header
    qos:
      reliability: reliable           # reliable | best_effort
      depth: 10

ros2_to_cortex:
  - name: control
    ros2_topic: /neurosim/control
    cortex_topic: control
    payload: control
    cortex_type: DictMessage          # must match the simulator's expected type
```

Valid `payload` values: `state`, `imu`, `events`, `color_image`, `depth_image`,
`control`. Anything else fails fast at load time.

A working full-coverage example for the `apartment_1` simulation ships at
[`config/apartment_1.yaml`](config/apartment_1.yaml).

## Run

```bash
# 1. discovery daemon
cortex-discovery

# 2. the simulator
python -m neurosim.sims.asynchronous_simulator.simulator_node \
    --settings configs/apartment_1-settings.yaml

# 3. the bridge as a composable node
ros2 launch neurosim_ros2_bridge bridge.launch.py \
    config:=$(ros2 pkg prefix neurosim_ros2_bridge)/share/neurosim_ros2_bridge/config/apartment_1.yaml

# 4. visualize
rviz2  # Fixed Frame: world, add Image displays for color + depth, State message inspector for /neurosim/state
```

Drive the simulator from a ROS 2 publisher (loop closure through ROS 2):

```bash
ros2 topic pub --once /neurosim/control std_msgs/msg/Float64MultiArray \
  "{data: [1500.0, 1500.0, 1500.0, 1500.0]}"
```

## Zero-copy notes

- ZMQ OOB frames travel through the bridge as `std::shared_ptr<zmq::message_t>`
  views (`cortex_wire::OobBuffer<T>`) — no copy from socket buffer to decoder.
- The single unavoidable copy per message is the `memcpy` from the OOB frame
  into the destination ROS 2 message's `std::vector` body (sensor_msgs::Image
  and the Events arrays). True zero-copy of the body would require a
  loaned-message RMW (Iceoryx) or custom intra-process types, both out of
  scope for v1.
- Published messages are `std::unique_ptr<Msg>`, so colocated subscribers
  loaded into the same `component_container_mt` receive them via rclcpp's
  intra-process path without DDS serialisation.

## Smoke test

A minimal end-to-end test lives in [`test/`](test/):

```bash
# In one terminal
cortex-discovery
python test/smoke_state_publisher.py

# In another
ros2 run neurosim_ros2_bridge neurosim_bridge --ros-args \
  -p config_path:=$(pwd)/test/smoke_state_only.yaml
ros2 topic hz /neurosim/state  # expect ~10 Hz
```

## Adding a new payload

1. Add a Cortex-side `.msg` definition (if a custom ROS type is needed).
2. Extend the `Payload` enum in [`config.hpp`](include/neurosim_ros2_bridge/config.hpp) and the YAML parser in [`config.cpp`](src/config.cpp).
3. Add a `decode_*` function in [`decoders.hpp`](include/neurosim_ros2_bridge/decoders.hpp) / [`decoders.cpp`](src/decoders.cpp).
4. Hook it into the `switch(e.payload)` in `wire_inbound` ([`bridge_node.cpp`](src/bridge_node.cpp)).

No registry, no factory, no plugin discovery — keep it simple.
