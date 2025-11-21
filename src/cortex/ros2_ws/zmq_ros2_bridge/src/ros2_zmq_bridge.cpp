#include "zmq_ros2_bridge/ros2_zmq_bridge.hpp"

namespace ros2_zmq_bridge {

void ROS2ZMQBridge::declare_parameters() {
    // Enable/disable flag for control
    this->declare_parameter("enable_control", true);

    // Parameters for Control
    this->declare_parameter("control_zmq_address", "ipc:///tmp/1");
    this->declare_parameter("control_zmq_topic", "control");
    this->declare_parameter("control_ros2_topic", "control");
}

ROS2ZMQBridge::ROS2ZMQBridge()
    : Node("ros2_zmq_bridge"), zmq_context_(1),
      zmq_pub_(zmq_context_, zmq::socket_type::pub), running_(true) {
    declare_parameters();

    control_enable_ = this->get_parameter("enable_control").as_bool();

    if (control_enable_) {
        control_zmq_address_ =
            this->get_parameter("control_zmq_address").as_string();
        control_zmq_topic_ =
            this->get_parameter("control_zmq_topic").as_string();
        control_ros2_topic_ =
            this->get_parameter("control_ros2_topic").as_string();

        // Create ZMQ publisher
        zmq_pub_ = create_zmq_publisher(control_zmq_address_);

        // Create ROS2 subscriber
        control_sub_ = this->create_subscription<zmq_ros2_bridge::msg::Control>(
            control_ros2_topic_, 10,
            std::bind(&ROS2ZMQBridge::control_callback, this,
                      std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Control: ROS2[%s] -> ZMQ[%s:%s]",
                    control_ros2_topic_.c_str(), control_zmq_address_.c_str(),
                    control_zmq_topic_.c_str());
    }

    RCLCPP_INFO(this->get_logger(), "ROS2 to ZMQ bridge started");
}

ROS2ZMQBridge::~ROS2ZMQBridge() {
    running_ = false;
    zmq_pub_.close();
}

zmq::socket_t ROS2ZMQBridge::create_zmq_publisher(const std::string &address) {
    zmq::socket_t publisher(zmq_context_, zmq::socket_type::pub);
    publisher.bind(address);

    // Set socket options for better performance
    publisher.set(zmq::sockopt::sndhwm, 1000);
    publisher.set(zmq::sockopt::linger, 0);
    publisher.set(zmq::sockopt::immediate, 1);

    return publisher;
}

void ROS2ZMQBridge::control_callback(
    const zmq_ros2_bridge::msg::Control::SharedPtr msg) {
    if (!control_enable_ || !running_) {
        return;
    }

    if (!publish_control_zmq(msg)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Failed to publish control message to ZMQ");
    }
}

bool ROS2ZMQBridge::publish_control_zmq(
    const zmq_ros2_bridge::msg::Control::SharedPtr &msg) {
    try {
        nlohmann::json control_json;

        control_json["cmd_motor_speeds"] = msg->cmd_motor_speeds;

        std::string json_str = control_json.dump();

        // Send multipart message: [topic, json_data]
        std::vector<zmq::message_t> messages;

        // Topic
        messages.emplace_back(control_zmq_topic_.data(),
                              control_zmq_topic_.size());

        // JSON data
        messages.emplace_back(json_str.data(), json_str.size());

        auto result =
            zmq::send_multipart(zmq_pub_, messages, zmq::send_flags::dontwait);

        if (!result) {
            RCLCPP_ERROR(this->get_logger(),
                         "Failed to send control message to ZMQ");
            return false;
        }

        RCLCPP_DEBUG(this->get_logger(), "Published control message to ZMQ");
        return true;

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Error publishing control to ZMQ: %s",
                     e.what());
        return false;
    }
}

} // namespace ros2_zmq_bridge

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ros2_zmq_bridge::ROS2ZMQBridge>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
