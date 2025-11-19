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
    void declare_parameters();
    zmq::socket_t create_zmq_subscriber(const std::string &address,
                                        const std::string &topic);
    void imu_sub_pub();
    void color_sub_pub();
    bool decode_imu_json(const std::vector<zmq::message_t> &messages,
                         nlohmann::json &imu_json);
    bool decode_numpy_array(const std::vector<zmq::message_t> &messages,
                            NumpyArrayMetadata &metadata,
                            const uint8_t *&array_data);
    void publish_imu(const nlohmann::json &imu_json);
    void publish_image(const NumpyArrayMetadata &metadata,
                       const uint8_t *array_data);

    // ROS2 publishers
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_pub_;

    // ZMQ context (shared across all threads)
    zmq::context_t zmq_context_;

    // ZMQ threads
    std::vector<std::thread> zmq_threads_;
    std::atomic<bool> running_;

    // Parameters
    bool imu_enable_;
    std::string imu_zmq_address_;
    std::string imu_zmq_topic_;
    std::string imu_ros2_topic_;
    bool color_enable_;
    std::string color_zmq_address_;
    std::string color_zmq_topic_;
    std::string color_ros2_topic_;
};

} // namespace zmq_ros2_bridge

#endif // ZMQ_ROS2_BRIDGE_HPP
