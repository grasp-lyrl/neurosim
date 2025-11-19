#include "zmq_ros2_bridge/zmq_ros2_bridge.hpp"

namespace zmq_ros2_bridge {

void ZMQROS2Bridge::declare_parameters() {
    // Enable/disable flags
    this->declare_parameter("enable_imu", false);
    this->declare_parameter("enable_color", false);

    // Parameters for IMU
    this->declare_parameter("imu_zmq_address", "ipc:///tmp/0");
    this->declare_parameter("imu_zmq_topic", "imu");
    this->declare_parameter("imu_ros2_topic", "imu");

    // Parameters for Color
    this->declare_parameter("color_zmq_address", "ipc:///tmp/0");
    this->declare_parameter("color_zmq_topic", "color");
    this->declare_parameter("color_ros2_topic", "image");
}

ZMQROS2Bridge::ZMQROS2Bridge()
    : Node("zmq_ros2_bridge"), zmq_context_(1), running_(true) {
    declare_parameters();

    imu_enable_ = this->get_parameter("enable_imu").as_bool();
    color_enable_ = this->get_parameter("enable_color").as_bool();

    // Configure IMU subscriber and publisher
    if (imu_enable_) {
        imu_zmq_address_ = this->get_parameter("imu_zmq_address").as_string();
        imu_zmq_topic_ = this->get_parameter("imu_zmq_topic").as_string();
        imu_ros2_topic_ = this->get_parameter("imu_ros2_topic").as_string();

        imu_pub_ =
            this->create_publisher<sensor_msgs::msg::Imu>(imu_ros2_topic_, 10);
        zmq_threads_.emplace_back(&ZMQROS2Bridge::imu_sub_pub, this);
        RCLCPP_INFO(this->get_logger(), "IMU: ZMQ[%s:%s] -> ROS2[%s]",
                    imu_zmq_address_.c_str(), imu_zmq_topic_.c_str(),
                    imu_ros2_topic_.c_str());
    }

    // Configure Color subscriber and publisher
    if (color_enable_) {
        color_zmq_address_ =
            this->get_parameter("color_zmq_address").as_string();
        color_zmq_topic_ = this->get_parameter("color_zmq_topic").as_string();
        color_ros2_topic_ = this->get_parameter("color_ros2_topic").as_string();

        color_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            color_ros2_topic_, 10);
        zmq_threads_.emplace_back(&ZMQROS2Bridge::color_sub_pub, this);
        RCLCPP_INFO(this->get_logger(), "Color: ZMQ[%s:%s] -> ROS2[%s]",
                    color_zmq_address_.c_str(), color_zmq_topic_.c_str(),
                    color_ros2_topic_.c_str());
    }

    RCLCPP_INFO(this->get_logger(), "ZMQ to ROS2 bridge started");
}

ZMQROS2Bridge::~ZMQROS2Bridge() {
    running_ = false;
    for (auto &thread : zmq_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

zmq::socket_t ZMQROS2Bridge::create_zmq_subscriber(const std::string &address,
                                                   const std::string &topic) {
    zmq::socket_t subscriber(zmq_context_, zmq::socket_type::sub);
    subscriber.connect(address);
    subscriber.set(zmq::sockopt::subscribe, topic);
    return subscriber;
}

bool ZMQROS2Bridge::decode_imu_json(const std::vector<zmq::message_t> &messages,
                                    nlohmann::json &imu_json) {
    if (messages.size() < 2) {
        RCLCPP_ERROR(this->get_logger(),
                     "Expected at least 2 parts for IMU, got %zu",
                     messages.size());
        return false;
    }

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

bool ZMQROS2Bridge::decode_numpy_array(
    const std::vector<zmq::message_t> &messages, NumpyArrayMetadata &metadata,
    const uint8_t *&array_data) {

    if (messages.size() != 5) {
        RCLCPP_ERROR(this->get_logger(), "Expected 5 parts, got %zu",
                     messages.size());
        return false;
    }

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

void ZMQROS2Bridge::publish_imu(const nlohmann::json &imu_json) {
    if (!imu_pub_)
        return;

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

        msg.header.frame_id = imu_zmq_topic_;

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
    if (!color_pub_)
        return;

    auto msg = sensor_msgs::msg::Image();

    // Set timestamp
    msg.header.stamp = this->now();
    msg.header.frame_id = color_zmq_topic_;

    // Set image dimensions (assuming HxWxC format from numpy)
    if (metadata.shape.size() == 3) {
        msg.height = metadata.shape[0];
        msg.width = metadata.shape[1];
        uint32_t channels = metadata.shape[2];
        uint32_t dtype_size = 0;

        // Set encoding based on dtype and channels
        if (metadata.dtype == "uint8" || metadata.dtype == "|u1") {
            dtype_size = 1;
            switch (channels) {
            case 3:
                msg.encoding = "rgb8";
                break;
            case 1:
                msg.encoding = "mono8";
                break;
            case 4:
                msg.encoding = "rgba8";
                break;
            default:
                RCLCPP_ERROR(this->get_logger(),
                             "Unsupported number of channels: %u", channels);
                return;
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unsupported dtype: %s",
                         metadata.dtype.c_str());
            return;
        }

        msg.is_bigendian = false;
        msg.step = msg.width * channels * dtype_size;
        auto nbytes = msg.height * msg.step;

        // Copy image data
        msg.data.assign(array_data, array_data + nbytes);

        color_pub_->publish(msg);

        RCLCPP_DEBUG(this->get_logger(), "Published image %ux%u, %u channels",
                     msg.height, msg.width, channels);
    } else {
        RCLCPP_WARN(this->get_logger(), "Unexpected shape dimensions: %zu",
                    metadata.shape.size());
    }
}

void ZMQROS2Bridge::imu_sub_pub() {
    if (!imu_enable_)
        return;

    try {
        auto subscriber =
            create_zmq_subscriber(imu_zmq_address_, imu_zmq_topic_);

        RCLCPP_INFO(this->get_logger(), "IMU ZMQ thread connected");

        while (running_ && rclcpp::ok()) {
            std::vector<zmq::message_t> messages;

            subscriber.set(zmq::sockopt::rcvtimeo, 1000);
            auto result =
                zmq::recv_multipart(subscriber, std::back_inserter(messages));

            if (!result || messages.empty()) {
                continue;
            }

            nlohmann::json imu_json;

            if (decode_imu_json(messages, imu_json))
                publish_imu(imu_json);
        }

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "IMU ZMQ thread error: %s", e.what());
    }
}

void ZMQROS2Bridge::color_sub_pub() {
    if (!color_enable_)
        return;

    try {
        auto subscriber =
            create_zmq_subscriber(color_zmq_address_, color_zmq_topic_);

        RCLCPP_INFO(this->get_logger(), "Color ZMQ thread connected");

        while (running_ && rclcpp::ok()) {
            std::vector<zmq::message_t> messages;

            subscriber.set(zmq::sockopt::rcvtimeo, 1000);
            auto result =
                zmq::recv_multipart(subscriber, std::back_inserter(messages));

            if (!result || messages.empty()) {
                continue;
            }

            NumpyArrayMetadata metadata;
            const uint8_t *array_data;

            if (decode_numpy_array(messages, metadata, array_data))
                publish_image(metadata, array_data);
        }

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Color ZMQ thread error: %s",
                     e.what());
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
