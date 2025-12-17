#ifndef ROS2_ZMQ_BRIDGE_HPP
#define ROS2_ZMQ_BRIDGE_HPP

#include "zmq_ros2_bridge/msg/control.hpp"
#include <atomic>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <zmq.hpp>
#include <zmq_addon.hpp>

namespace ros2_zmq_bridge {

class ROS2ZMQBridge : public rclcpp::Node {
  public:
    ROS2ZMQBridge();
    ~ROS2ZMQBridge();

  private:
    void declare_parameters();
    zmq::socket_t create_zmq_publisher(const std::string &address);

    // ROS2 callback for control messages
    void control_callback(const zmq_ros2_bridge::msg::Control::SharedPtr msg);

    // Function to publish control message to ZMQ
    bool
    publish_control_zmq(const zmq_ros2_bridge::msg::Control::SharedPtr &msg);

    // ROS2 subscriber
    rclcpp::Subscription<zmq_ros2_bridge::msg::Control>::SharedPtr control_sub_;

    // ZMQ context and publisher socket
    zmq::context_t zmq_context_;
    zmq::socket_t zmq_pub_;

    // Parameters
    bool control_enable_;
    std::string control_zmq_address_;
    std::string control_zmq_topic_;
    std::string control_ros2_topic_;

    std::atomic<bool> running_;
};

} // namespace ros2_zmq_bridge

#endif // ROS2_ZMQ_BRIDGE_HPP
