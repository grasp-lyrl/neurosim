// One composable rclcpp::Node carrying both directions. Each YAML entry
// becomes either a cortex_wire::Subscriber + payload decoder (cortex_to_ros2)
// or a cortex_wire::Publisher + rclcpp::Subscription (ros2_to_cortex). All
// ZMQ / cortex protocol machinery lives in cortex_wire_cpp.
#ifndef NEUROSIM_ROS2_BRIDGE__BRIDGE_NODE_HPP_
#define NEUROSIM_ROS2_BRIDGE__BRIDGE_NODE_HPP_

#include <cortex_wire/fingerprint_table.hpp>
#include <cortex_wire/publisher.hpp>
#include <cortex_wire/subscriber.hpp>
#include <cortex_wire/context.hpp>
#include <cortex_wire/discovery_client.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <memory>
#include <vector>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "neurosim_ros2_bridge/config.hpp"
#include "neurosim_ros2_bridge/decoders.hpp"

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

  // Visible for tests: number of bridge entries that wired up successfully.
  std::size_t num_active() const noexcept {return workers_.size();}

private:
  void initialize();
  void wire_inbound(const Entry & e);
  void wire_outbound(const Entry & e);

  BridgeConfig config_;
  cortex_wire::Context ctx_;
  std::unique_ptr<cortex_wire::DiscoveryClient> discovery_;
  std::vector<std::unique_ptr<BridgeWorker>> workers_;
};

}  // namespace neurosim_ros2_bridge

#endif  // NEUROSIM_ROS2_BRIDGE__BRIDGE_NODE_HPP_
