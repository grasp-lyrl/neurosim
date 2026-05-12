// Copyright (c) 2026, Neurosim contributors. Apache-2.0.
//
// One composable rclcpp::Node carrying both directions. Each YAML entry spins
// either a ZMQ SUB recv thread (cortex_to_ros2) or installs an rclcpp
// subscription that publishes on a ZMQ PUB socket (ros2_to_cortex). The bridge
// is a closed system — adapters/decoders are baked in.
#ifndef NEUROSIM_ROS2_BRIDGE__BRIDGE_NODE_HPP_
#define NEUROSIM_ROS2_BRIDGE__BRIDGE_NODE_HPP_

#include <cortex_wire/discovery_client.hpp>
#include <rclcpp/rclcpp.hpp>
#include <zmq.hpp>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "neurosim_ros2_bridge/config.hpp"

namespace neurosim_ros2_bridge
{

// Abstract per-entry handle. The bridge node holds these polymorphically so
// it can stop everything in destruction order.
class BridgeWorker
{
public:
  virtual ~BridgeWorker() = default;
  virtual void stop() = 0;
};

class NeurosimRos2Bridge : public rclcpp::Node
{
public:
  explicit NeurosimRos2Bridge(const rclcpp::NodeOptions & options);
  ~NeurosimRos2Bridge() override;

  // Number of entries actually wired up (skipping ones that failed discovery
  // lookup, fingerprint mismatch, etc). Useful for the launch smoke test.
  std::size_t num_active() const noexcept {return workers_.size();}

private:
  void initialize();
  void wire_inbound(const Entry & e);
  void wire_outbound(const Entry & e);

  BridgeConfig config_;
  std::shared_ptr<zmq::context_t> ctx_;
  std::unique_ptr<cortex_wire::DiscoveryClient> discovery_;

  std::vector<std::unique_ptr<BridgeWorker>> workers_;
  std::vector<std::string> registered_topics_;
};

}  // namespace neurosim_ros2_bridge

#endif  // NEUROSIM_ROS2_BRIDGE__BRIDGE_NODE_HPP_
