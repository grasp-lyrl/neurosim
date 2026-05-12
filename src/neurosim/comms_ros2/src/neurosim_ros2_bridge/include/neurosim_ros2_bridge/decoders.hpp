// Per-payload decoders. Each function takes a fully-decoded Cortex multipart
// message (header + DecodedMetadata + OOB frames) and returns a heap-allocated
// ROS 2 message ready for rclcpp::Publisher::publish (which moves it into the
// intra-process zero-copy path).
//
// Decoders touch the OOB frames at most once via a single memcpy into the
// destination ROS 2 message body. They never copy small inline metadata
// (state/imu/control are entirely inline, no OOB).
#ifndef NEUROSIM_ROS2_BRIDGE__DECODERS_HPP_
#define NEUROSIM_ROS2_BRIDGE__DECODERS_HPP_

#include <cortex_wire/header.hpp>
#include <cortex_wire/metadata.hpp>
#include <cortex_wire/oob_buffer.hpp>

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "neurosim_ros2_bridge/msg/events.hpp"
#include "neurosim_ros2_bridge/msg/imu.hpp"
#include "neurosim_ros2_bridge/msg/state.hpp"

namespace neurosim_ros2_bridge::decoders
{

// All inbound decoders take this — a thin bundle of the wire pieces plus the
// configured frame_id. Lifetime of `metadata` and `oob_frames` extends through
// the call.
struct Inbound
{
  cortex_wire::MessageHeader header;
  const cortex_wire::DecodedMetadata & metadata;
  const std::vector<cortex_wire::ZmqFramePtr> & oob_frames;
  const std::string & frame_id;
};

// Cortex DictMessage{x,q,v,w,timestamp,simsteps} -> neurosim_msgs/State.
// Everything is inline msgpack (cortex publishes the positions as lists via
// .tolist()); zero OOB frames are consumed.
std::unique_ptr<neurosim_ros2_bridge::msg::State> decode_state(const Inbound & in);

// Cortex DictMessage{uuid,accel,gyro,timestamp,simsteps} -> neurosim_msgs/Imu.
// accel and gyro are length-3 numpy OOB frames; one tiny memcpy each (24 B).
std::unique_ptr<neurosim_ros2_bridge::msg::Imu> decode_imu(const Inbound & in);

// Cortex MultiArrayMessage{x,y,t,p} -> neurosim_msgs/Events.
// One memcpy per array into the ROS message's std::vector backing.
std::unique_ptr<neurosim_ros2_bridge::msg::Events> decode_events(const Inbound & in);

// Cortex ArrayMessage(uint8 HxWx3) -> sensor_msgs/Image(rgb8). One memcpy
// from the OOB frame into Image::data. Channels are inferred from shape[2];
// the bridge errors loudly if it's not 3.
std::unique_ptr<sensor_msgs::msg::Image> decode_color_image(const Inbound & in);

// Cortex ArrayMessage(float32 HxW) -> sensor_msgs/Image(32FC1).
std::unique_ptr<sensor_msgs::msg::Image> decode_depth_image(const Inbound & in);

// ---- ROS 2 -> Cortex --------------------------------------------------------

// std_msgs/Float64MultiArray -> cortex DictMessage{cmd_motor_speeds, timestamp}.
// Returns (metadata_bytes, oob_buffers). Currently no OOB — control is small
// enough to inline as a msgpack array.
struct OutboundFrames
{
  std::vector<std::uint8_t> metadata;
  std::vector<std::vector<std::uint8_t>> oob_buffers;
};

OutboundFrames encode_control(const std_msgs::msg::Float64MultiArray & msg);

}  // namespace neurosim_ros2_bridge::decoders

#endif  // NEUROSIM_ROS2_BRIDGE__DECODERS_HPP_
