// Copyright (c) 2026, Neurosim contributors. Apache-2.0.
#ifndef NEUROSIM_ROS2_BRIDGE__CONFIG_HPP_
#define NEUROSIM_ROS2_BRIDGE__CONFIG_HPP_

#include <optional>
#include <string>
#include <vector>

namespace neurosim_ros2_bridge
{

// Payload tags: which neurosim message type is on the wire, and how to map it
// into a ROS 2 message. The bridge keeps a fixed, small set of payloads (one
// per relevant cortex stream) rather than a generic adapter registry — see
// README for the type table.
enum class Payload
{
  State,        // DictMessage{x,q,v,w,timestamp,simsteps}  -> neurosim_msgs/State
  Imu,          // DictMessage{accel,gyro,timestamp,uuid}   -> neurosim_msgs/Imu
  Events,       // MultiArrayMessage{x,y,t,p}               -> neurosim_msgs/Events
  ColorImage,   // ArrayMessage uint8 HxWx3                 -> sensor_msgs/Image (rgb8)
  DepthImage,   // ArrayMessage float32 HxW                 -> sensor_msgs/Image (32FC1)
  Control,      // ROS 2 std_msgs/Float64MultiArray         -> cortex DictMessage{cmd_motor_speeds,timestamp}
};

// One entry in the YAML config. Direction is implied by which list the entry
// lives in (cortex_to_ros2 vs ros2_to_cortex).
struct Entry
{
  std::string name;            // human-readable; appears in logs
  std::string cortex_topic;    // cortex topic name (no prefix)
  std::string ros2_topic;      // ROS 2 topic name (use a leading slash)
  std::string frame_id;        // header.frame_id stamped on outbound ROS msgs
  Payload payload;
  std::uint32_t depth = 10;    // ROS 2 QoS depth (history KEEP_LAST)
  bool best_effort = false;    // KEEP_LAST/RELIABLE by default; opt-in to best-effort for cameras

  // Outbound (ros2_to_cortex) only. Allowed values today: "DictMessage".
  // For inbound entries this is set automatically from `payload`.
  std::string cortex_type;
};

struct BridgeConfig
{
  std::string discovery_address = "ipc:///tmp/cortex/discovery.sock";
  std::string node_name_prefix = "neurosim_bridge";

  std::vector<Entry> cortex_to_ros2;
  std::vector<Entry> ros2_to_cortex;
};

// Parse a neurosim_ros2_bridge YAML file. Throws std::runtime_error on schema
// violations with a message that points at the offending entry.
BridgeConfig load_config(const std::string & path);

// Convert a payload tag <-> the YAML string the user writes ("state", "imu",
// "events", "color_image", "depth_image", "control").
std::optional<Payload> parse_payload(const std::string & s);
std::string payload_to_string(Payload p);

}  // namespace neurosim_ros2_bridge

#endif  // NEUROSIM_ROS2_BRIDGE__CONFIG_HPP_
