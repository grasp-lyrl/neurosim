#include "zmq_ros2_bridge/zmq_ros2_bridge.hpp"

namespace zmq_ros2_bridge {

ZMQROS2Bridge::ZMQROS2Bridge() : Node("zmq_ros2_bridge"), running_(true) {
    // Declare and get parameters
    this->declare_parameter("zmq_address", "tcp://localhost:5555");
    this->declare_parameter("zmq_topic", "color");

    zmq_address_ = this->get_parameter("zmq_address").as_string();
    zmq_topic_ = this->get_parameter("zmq_topic").as_string();

    RCLCPP_INFO(this->get_logger(), "ZMQ Address: %s", zmq_address_.c_str());
    RCLCPP_INFO(this->get_logger(), "ZMQ Topic: %s", zmq_topic_.c_str());

    // Create ROS2 publishers
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("image", 10);
    imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);

    // Start ZMQ subscriber thread
    zmq_thread_ = std::thread(&ZMQROS2Bridge::zmq_thread_func, this);

    RCLCPP_INFO(this->get_logger(), "ZMQ to ROS2 bridge started");
}

ZMQROS2Bridge::~ZMQROS2Bridge() {
    running_ = false;
    if (zmq_thread_.joinable()) {
        zmq_thread_.join();
    }
}

bool ZMQROS2Bridge::decode_numpy_array(
    const std::vector<zmq::message_t> &messages, std::string &topic,
    NumpyArrayMetadata &metadata, const uint8_t *&array_data) {

    if (messages.size() != 5) {
        RCLCPP_ERROR(this->get_logger(), "Expected 5 parts, got %zu",
                     messages.size());
        return false;
    }

    // Extract topic
    topic = std::string(static_cast<const char *>(messages[0].data()),
                        messages[0].size());

    // Extract header (4 uint32_t values in network byte order)
    if (messages[1].size() != 16) {
        RCLCPP_ERROR(this->get_logger(), "Invalid header size");
        return false;
    }

    const uint32_t *header = static_cast<const uint32_t *>(messages[1].data());
    metadata.ndim = ntohl(header[0]);
    metadata.dtype_len = ntohl(header[1]);
    metadata.shape_len = ntohl(header[2]);
    metadata.nbytes = ntohl(header[3]);

    // Extract dtype
    metadata.dtype = std::string(static_cast<const char *>(messages[2].data()),
                                 messages[2].size());

    // Extract shape
    const uint32_t *shape_data =
        static_cast<const uint32_t *>(messages[3].data());
    metadata.shape.clear();
    for (uint32_t i = 0; i < metadata.ndim; ++i) {
        metadata.shape.push_back(ntohl(shape_data[i]));
    }

    // Get pointer to array data
    array_data = static_cast<const uint8_t *>(messages[4].data());

    return true;
}

bool ZMQROS2Bridge::decode_imu_json(const std::vector<zmq::message_t> &messages,
                                    std::string &topic,
                                    nlohmann::json &imu_json) {
    if (messages.size() < 2) {
        RCLCPP_ERROR(this->get_logger(),
                     "Expected at least 2 parts for IMU, got %zu",
                     messages.size());
        return false;
    }

    // Extract topic
    topic = std::string(static_cast<const char *>(messages[0].data()),
                        messages[0].size());

    // Extract JSON data from second part
    std::string json_str(static_cast<const char *>(messages[1].data()),
                         messages[1].size());

    try {
        imu_json = nlohmann::json::parse(json_str);
        return true;
    } catch (const nlohmann::json::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to parse IMU JSON: %s",
                     e.what());
        return false;
    }
}

void ZMQROS2Bridge::publish_imu(const nlohmann::json &imu_json) {

    auto msg = sensor_msgs::msg::Imu();

    try {
        // Set timestamp from the message if available
        if (imu_json.contains("timestamp")) {
            double timestamp = imu_json["timestamp"].get<double>();
            msg.header.stamp =
                rclcpp::Time(static_cast<int64_t>(timestamp * 1e9));
        } else {
            msg.header.stamp = this->now();
        }

        msg.header.frame_id = zmq_topic_;

        // Extract angular velocity
        if (imu_json.contains("gyro")) {
            auto &ang_vel = imu_json["gyro"];
            msg.angular_velocity.x = ang_vel[0].get<double>();
            msg.angular_velocity.y = ang_vel[1].get<double>();
            msg.angular_velocity.z = ang_vel[2].get<double>();
        }

        // Extract linear acceleration
        if (imu_json.contains("accel")) {
            auto &lin_acc = imu_json["accel"];
            msg.linear_acceleration.x = lin_acc[0].get<double>();
            msg.linear_acceleration.y = lin_acc[1].get<double>();
            msg.linear_acceleration.z = lin_acc[2].get<double>();
        }

        imu_pub_->publish(msg);

        RCLCPP_DEBUG(this->get_logger(), "Published IMU message");

    } catch (const nlohmann::json::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to extract IMU data: %s",
                     e.what());
    }
}

void ZMQROS2Bridge::publish_image(const NumpyArrayMetadata &metadata,
                                  const uint8_t *array_data) {
    auto msg = sensor_msgs::msg::Image();

    // Set timestamp
    msg.header.stamp = this->now();
    msg.header.frame_id = zmq_topic_;

    // Set image dimensions (assuming HxWxC format from numpy)
    if (metadata.shape.size() == 3) {
        msg.height = metadata.shape[0];
        msg.width = metadata.shape[1];
        uint32_t channels = metadata.shape[2];

        // Set encoding based on dtype and channels
        if (metadata.dtype == "uint8" || metadata.dtype == "|u1") {
            if (channels == 3) {
                msg.encoding = "rgb8";
            } else if (channels == 1) {
                msg.encoding = "mono8";
            } else if (channels == 4) {
                msg.encoding = "rgba8";
            }
        }

        msg.is_bigendian = false;
        msg.step = msg.width * channels;

        // Copy image data
        msg.data.assign(array_data, array_data + metadata.nbytes);

        image_pub_->publish(msg);

        RCLCPP_DEBUG(this->get_logger(), "Published image %ux%u, %u channels",
                     msg.height, msg.width, channels);
    } else {
        RCLCPP_WARN(this->get_logger(), "Unexpected shape dimensions: %zu",
                    metadata.shape.size());
    }
}

void ZMQROS2Bridge::zmq_thread_func() {
    try {
        // Create ZMQ context and subscriber socket
        zmq::context_t context(1);
        zmq::socket_t subscriber(context, zmq::socket_type::sub);

        subscriber.connect(zmq_address_);
        subscriber.set(zmq::sockopt::subscribe, zmq_topic_);

        RCLCPP_INFO(this->get_logger(), "Connected to ZMQ publisher");

        while (running_ && rclcpp::ok()) {
            std::vector<zmq::message_t> messages;

            subscriber.set(zmq::sockopt::rcvtimeo, 1000); // 1 second timeout
            auto result =
                zmq::recv_multipart(subscriber, std::back_inserter(messages));

            if (!result || messages.empty()) {
                continue;
            }

            std::string topic;

            // Check if this is a numpy array (5 parts) or JSON (2+ parts)
            if (messages.size() == 5) {
                // Decode as numpy array (color image)
                NumpyArrayMetadata metadata;
                const uint8_t *array_data;

                if (decode_numpy_array(messages, topic, metadata, array_data)) {
                    if (topic == "color") {
                        publish_image(metadata, array_data);
                    }
                }
            } else {
                // Decode as JSON (IMU)
                nlohmann::json imu_json;

                if (decode_imu_json(messages, topic, imu_json)) {
                    if (topic == "imu") {
                        publish_imu(imu_json);
                    }
                }
            }
        }

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "ZMQ thread error: %s", e.what());
    }
}

} // namespace zmq_ros2_bridge

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<zmq_ros2_bridge::ZMQROS2Bridge>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
