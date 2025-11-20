#ifndef ZMQ_ROS2_BRIDGE_HPP
#define ZMQ_ROS2_BRIDGE_HPP

#include "zmq_ros2_bridge/msg/event.hpp"
#include "zmq_ros2_bridge/msg/imu.hpp"
#include <arpa/inet.h>
#include <atomic>
#include <map>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <zmq.hpp>
#include <zmq_addon.hpp>

namespace zmq_ros2_bridge {

struct NumpyArray {
    uint8_t ndim;
    std::string dtype;
    std::vector<uint32_t> shape;
    const uint8_t *data; // Pointer to the raw data buffer, not owned
};

using DictofNumpyArray = std::unordered_map<std::string, NumpyArray>;

// Image encoding information
struct ImageEncodingInfo {
    std::string encoding;
    uint32_t dtype_size;
};

// Key: (dtype, channels) -> Value: (encoding, dtype_size)
using ImageEncodingMap =
    std::map<std::pair<std::string, uint32_t>, ImageEncodingInfo>;

class ZMQROS2Bridge : public rclcpp::Node {
  public:
    ZMQROS2Bridge();
    ~ZMQROS2Bridge();

  private:
    void declare_parameters();
    zmq::socket_t create_zmq_subscriber(const std::string &address,
                                        const std::string &topic);

    // Functions running on separate threads to sub ZMQ and pub ROS2
    void imu_sub_pub();
    void color_sub_pub();
    void depth_sub_pub();
    void event_sub_pub();

    // Different decode functions from ZMQ messages to C++ structures
    bool decode_imu_json(const std::vector<zmq::message_t> &messages,
                         nlohmann::json &imu_json);
    bool decode_numpy_array(const std::vector<zmq::message_t> &messages,
                            NumpyArray &numpy_array);
    bool
    decode_dict_of_numpy_arrays(const std::vector<zmq::message_t> &messages,
                                DictofNumpyArray &event_data);

    // Functions to form and publish ROS2 messages
    void publish_imu(const nlohmann::json &imu_json);
    void publish_image(const NumpyArray &numpy_array,
                       const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr &publisher,
                       const std::string &frame_id);
    void publish_events(const DictofNumpyArray &event_data);

    // Image encoding lookup map
    inline static const ImageEncodingMap image_encoding_map_ = {
        // uint8 encodings
        {{"uint8", 1}, {"mono8", 1}},
        {{"|u1", 1}, {"mono8", 1}},
        {{"uint8", 3}, {"rgb8", 1}},
        {{"|u1", 3}, {"rgb8", 1}},
        {{"uint8", 4}, {"rgba8", 1}},
        {{"|u1", 4}, {"rgba8", 1}},

        // uint16 encodings
        {{"uint16", 1}, {"mono16", 2}},
        {{"<u2", 1}, {"mono16", 2}},
        {{">u2", 1}, {"mono16", 2}},
        {{"uint16", 3}, {"rgb16", 2}},
        {{"<u2", 3}, {"rgb16", 2}},
        {{">u2", 3}, {"rgb16", 2}},

        // float32 encodings
        {{"float32", 1}, {"32FC1", 4}},
        {{"<f4", 1}, {"32FC1", 4}},
        {{">f4", 1}, {"32FC1", 4}},
        {{"float32", 3}, {"32FC3", 4}},
        {{"<f4", 3}, {"32FC3", 4}},
        {{">f4", 3}, {"32FC3", 4}},
        {{"float32", 4}, {"32FC4", 4}},
        {{"<f4", 4}, {"32FC4", 4}},
        {{">f4", 4}, {"32FC4", 4}},

        // float64 encodings
        {{"float64", 1}, {"64FC1", 8}},
        {{"<f8", 1}, {"64FC1", 8}},
        {{">f8", 1}, {"64FC1", 8}},
        {{"float64", 3}, {"64FC3", 8}},
        {{"<f8", 3}, {"64FC3", 8}},
        {{">f8", 3}, {"64FC3", 8}},
        {{"float64", 4}, {"64FC4", 8}},
        {{"<f8", 4}, {"64FC4", 8}},
        {{">f8", 4}, {"64FC4", 8}}};

    // ROS2 publishers
    rclcpp::Publisher<zmq_ros2_bridge::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<zmq_ros2_bridge::msg::Event>::SharedPtr event_pub_;

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
    bool depth_enable_;
    std::string depth_zmq_address_;
    std::string depth_zmq_topic_;
    std::string depth_ros2_topic_;
    bool event_enable_;
    std::string event_zmq_address_;
    std::string event_zmq_topic_;
    std::string event_ros2_topic_;
};

} // namespace zmq_ros2_bridge

#endif // ZMQ_ROS2_BRIDGE_HPP
