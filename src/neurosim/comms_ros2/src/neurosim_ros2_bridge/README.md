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
| cortex → ROS 2 | `state` (DictMessage) | `/neurosim/odom` (`nav_msgs/msg/Odometry`) + TF `world` → `base_link` |
| cortex → ROS 2 | `state` (DictMessage) | `/clock` (`rosgraph_msgs/msg/Clock`), the simulation time |
| cortex → ROS 2 | `imu/<uuid>` (DictMessage) | `/neurosim/imu/<uuid>` (`neurosim_ros2_bridge/msg/Imu`) |
| cortex → ROS 2 | `imu/<uuid>` (DictMessage) | `/neurosim/imu/data` (`sensor_msgs/msg/Imu`) |
| cortex → ROS 2 | `color/<uuid>` (ArrayMessage `u1` HxWx3) | `/neurosim/camera/color/image_raw` (`sensor_msgs/msg/Image`, `rgb8`) |
| cortex → ROS 2 | `depth/<uuid>` (ArrayMessage `f4` HxW) | `/neurosim/camera/depth/image_raw` (`sensor_msgs/msg/Image`, `32FC1`) |
| cortex → ROS 2 | `events/<uuid>` (MultiArrayMessage) | `/neurosim/camera/events/<uuid>` (`neurosim_ros2_bridge/msg/Events`) |
| cortex → ROS 2 | `events/<uuid>` (MultiArrayMessage) | `/neurosim/camera/events/image` (`sensor_msgs/msg/Image`, `rgb8`, ON blue, OFF red) |
| cortex → ROS 2 | `camera_info/<uuid>` (DictMessage) | `/neurosim/camera/<color\|depth\|events>/camera_info` (`sensor_msgs/msg/CameraInfo`, `plumb_bob`) |
| ROS 2 → cortex | `/neurosim/control` (`std_msgs/Float64MultiArray`) | `control` (DictMessage) |

Color/depth become `sensor_msgs/Image` directly so rviz2's Image display
renders them with no extra plumbing. State / IMU / Events use custom message
types because the wire payloads (especially events' four parallel arrays)
don't fit any stdlib type cleanly; the `odometry`, `sensor_imu` and
`events_image` payloads republish the same streams as standard types for
rviz2 and other ROS 2 tools. Every `header.stamp` is the simulation time the
sample was taken at, counted from the simulator's start, on the same clock as
`Events.t` (microseconds), so samples of one simulation step share one stamp.
An events packet is stamped with its last render. The simulator publishes each
camera's CameraInfo with every image or events packet, under the same stamp, from
`neurosim.core.coord_trans.calibration.camera_info`. rviz2 runs on `/clock`
(`use_sim_time`).

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

Valid `payload` values: `state`, `odometry`, `imu`, `sensor_imu`, `events`,
`events_image`, `color_image`, `depth_image`, `control`. Anything else fails
fast at load time. `odometry` also needs `child_frame_id`; `events_image`
needs the sensor `width` and `height`.

A working full-coverage example for the `apartment_1` simulation ships at
[`config/apartment_1.yaml`](config/apartment_1.yaml).

## Run

One process per terminal, from the neurosim repo root:

```bash
# 1. discovery daemon
cortex-discovery

# 2. the simulator; wait for "SimulatorNode initialized successfully"
python -m neurosim.sims.asynchronous_simulator.simulator_node \
    --settings configs/apartment_1-settings.yaml

# 3. optional: fly the random trajectory so the cameras see motion
python -m neurosim.sims.asynchronous_simulator.controller_node \
    --settings configs/apartment_1-settings.yaml

# 4. the bridge, plus rviz2 with rviz/neurosim.rviz
#    (events, color, depth images; odometry trail, TF, IMU acceleration)
ros2 launch neurosim_ros2_bridge bridge.launch.py
```

`config:=<yaml>` picks another bridge config (default `config/apartment_1.yaml`);
`rviz:=false` skips rviz2 on a headless machine.

Start the bridge after the simulator has finished initializing: each
`cortex_to_ros2` entry looks its topic up once, and an entry whose topic is
not registered yet logs `wire_inbound: ... is not registered` and stays dead.

In the `neurosim:ros` image the bridge is prebuilt and sourced in every shell.
Run all four in one container (e.g. tmux panes): cortex topics are sockets under
its `/tmp/cortex`. If another `--net=host` ROS 2 container is on the same host,
`export ROS_DOMAIN_ID=42` (any id it doesn't use) before starting tmux, or the
`ros2` CLI talks to that container's daemon.

Drive the simulator from a ROS 2 publisher (loop closure through ROS 2):
uncomment `ros2_to_cortex` in the config and do not run `controller_node`
(a cortex topic has one publisher). The simulator waits 60 s after start for
a `control` publisher, and keeps applying the last command it received.
Crazyflie hover is 1788.5 rad/s per rotor:

```bash
ros2 topic pub --once /neurosim/control std_msgs/msg/Float64MultiArray \
  "{data: [1788.5, 1788.5, 1788.5, 1788.5]}"
```

## Record flights

`record:=true` loads
[rosbag2_composable_recorder](https://github.com/berndpfrommer/rosbag2_composable_recorder)
into the bridge's component container and records the topics in
[`config/recorder.yaml`](config/recorder.yaml) to MCAP, starting immediately:

```bash
ros2 launch neurosim_ros2_bridge bridge.launch.py record:=true
```

The bag goes to `$NEUROSIM_BAG_DIR/flight_<date-time>/`, or
`outputs/bags/flight_<date-time>/` under the directory you launch from when the
variable is unset (the launch prints the absolute path); run it from the neurosim
repo root so relative paths land on the mounted repo and not inside the container.
`bag_prefix:=` changes the location, `record_config:=` the topic list. Ctrl-C
closes the bag; `ros2 service call /stop_recording std_srvs/srv/Trigger` closes it
without stopping the bridge.

Beside the MCAP, `python -m neurosim.core.coord_trans.calibration` writes the calibration of
every camera and IMU the bridge config publishes, from the simulator settings
(`settings:=`, default `configs/apartment_1-settings.yaml`; pass the file the
simulator runs with):

- `<uuid>.xml`: OpenCV `FileStorage` intrinsics (`camera_matrix`,
  `distortion_coefficients`, `image_width`, `image_height`), pixel centres at
  integer coordinates.
- `camchain-imucam.yaml`: Kalibr camchain, pinhole-radtan intrinsics, `T_cam_imu`
  (IMU to OpenCV camera coordinates), `T_cn_cnm1`, `rostopic` of each camera.
- `<imu uuid>.yaml`: Kalibr's imu.yaml (`rostopic`, `update_rate`, noise densities
  and random walks); all noise is zero, as the simulated IMU has none.

Humble's recorder stamps bag time with the wall clock, whatever `use_sim_time`
says, so a live bag's bag time differs from its header stamps. Bags converted
from H5 (below) have both on the simulation clock.

### From H5

The synchronous simulator writes every sample, without lag or drops, to H5
(`python test_sim.py --settings <yaml> --log-h5 flight.h5`). Compress the file, then
make it the bag the bridge and recorder would have made, with the calibration beside:

```bash
python scripts/compress_h5.py flight.h5         # LZF + shuffle in place, about 20% of raw
python scripts/h5_to_mcap.py flight.h5 --out outputs/bags/flight
```

The converter reads the same two configs (`--config` for what each topic carries,
`--record-config` for which topics go in), publishes on the same cadence
(`SimulationConfig`), stamps header and bag time with the simulation time, and needs
ROS 2 and this package sourced, so run it in `neurosim:ros`. Play with
`ros2 bag play --clock`.

The IMU frame is rotorpy's body frame, `base_link` in TF. Cameras are Habitat
sensors mounted on it at their configured `position` (Habitat's forward is `-z`,
so `[0, 0, 0.05]` is 5 cm behind the IMU) and `orientation` (about the parent's x,
then y, then z). `HabitatWrapper` renders the drone `agent_height` straight above
the dynamics state, so `world`'s origin sits `agent_height` above the Habitat
scene's; relative poses and `T_cam_imu` do not depend on that offset.
`tests/test_calibration.py` checks the result against Habitat's sensor nodes and
renders.

A flying apartment_1 run records about 150 MB/s: raw events 114, depth 25, color
18, everything else under 0.1. The MCAP stores the message definitions, so
`mcap_ros2` decodes `neurosim_ros2_bridge/msg/Events` without a ROS install.

Copies: rosbag2 records serialized messages through generic subscriptions,
which rclcpp's intra-process path does not serve, so each recorded message is
serialized once by the bridge's publisher. Composing the recorder removes the
separate `ros2 bag record` process and the DDS transport into it, not that
serialization.

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
