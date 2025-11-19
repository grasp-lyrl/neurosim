#ifndef ZMQ_ROS2_BRIDGE_HPP
#define ZMQ_ROS2_BRIDGE_HPP

#include <arpa/inet.h>
#include <atomic>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>
#include <zmq_addon.hpp>

namespace zmq_ros2_bridge {

struct NumpyArrayMetadata {
    uint32_t ndim;
    uint32_t dtype_len;
    uint32_t shape_len;
    uint32_t nbytes;
    std::string dtype;
    std::vector<uint32_t> shape;
};

class ZMQROS2Bridge : public rclcpp::Node {
  public:
    ZMQROS2Bridge();
    ~ZMQROS2Bridge();

  private:
    void zmq_thread_func();
    bool decode_numpy_array(const std::vector<zmq::message_t> &messages,
                            std::string &topic, NumpyArrayMetadata &metadata,
                            const uint8_t *&array_data);
    bool decode_imu_json(const std::vector<zmq::message_t> &messages,
                         std::string &topic, nlohmann::json &imu_json);
    void publish_image(const NumpyArrayMetadata &metadata,
                       const uint8_t *array_data);
    void publish_imu(const nlohmann::json &imu_json);

    // ROS2 publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;

    // ZMQ thread
    std::thread zmq_thread_;
    std::atomic<bool> running_;

    // Parameters
    std::string zmq_address_;
    std::string zmq_topic_;
};

} // namespace zmq_ros2_bridge

#endif // ZMQ_ROS2_BRIDGE_HPP
