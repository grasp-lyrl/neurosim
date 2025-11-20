#include "zmq_ros2_bridge/zmq_ros2_bridge.hpp"

namespace zmq_ros2_bridge {

void ZMQROS2Bridge::declare_parameters() {
    // Enable/disable flags
    this->declare_parameter("enable_imu", false);
    this->declare_parameter("enable_color", false);
    this->declare_parameter("enable_event", false);

    // Parameters for IMU
    this->declare_parameter("imu_zmq_address", "ipc:///tmp/0");
    this->declare_parameter("imu_zmq_topic", "imu");
    this->declare_parameter("imu_ros2_topic", "imu");

    // Parameters for Color
    this->declare_parameter("color_zmq_address", "ipc:///tmp/0");
    this->declare_parameter("color_zmq_topic", "color");
    this->declare_parameter("color_ros2_topic", "image");

    // Parameters for Event
    this->declare_parameter("event_zmq_address", "ipc:///tmp/0");
    this->declare_parameter("event_zmq_topic", "events");
    this->declare_parameter("event_ros2_topic", "events");
}

ZMQROS2Bridge::ZMQROS2Bridge()
    : Node("zmq_ros2_bridge"), zmq_context_(1), running_(true) {
    declare_parameters();

    imu_enable_ = this->get_parameter("enable_imu").as_bool();
    color_enable_ = this->get_parameter("enable_color").as_bool();
    event_enable_ = this->get_parameter("enable_event").as_bool();

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

    // Configure Event subscriber and publisher
    if (event_enable_) {
        event_zmq_address_ =
            this->get_parameter("event_zmq_address").as_string();
        event_zmq_topic_ = this->get_parameter("event_zmq_topic").as_string();
        event_ros2_topic_ = this->get_parameter("event_ros2_topic").as_string();

        event_pub_ = this->create_publisher<zmq_ros2_bridge::msg::Event>(
            event_ros2_topic_, 10);
        zmq_threads_.emplace_back(&ZMQROS2Bridge::event_sub_pub, this);
        RCLCPP_INFO(this->get_logger(), "Event: ZMQ[%s:%s] -> ROS2[%s]",
                    event_zmq_address_.c_str(), event_zmq_topic_.c_str(),
                    event_ros2_topic_.c_str());
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
    const std::vector<zmq::message_t> &messages, NumpyArray &numpy_array) {

    if (messages.size() != 5) {
        RCLCPP_ERROR(this->get_logger(), "Expected 5 parts, got %zu",
                     messages.size());
        return false;
    }

    // Extract number of dimensions
    numpy_array.ndim = *(static_cast<const uint8_t *>(messages[1].data()));

    // Extract dtype - assign directly without intermediate construction because
    // ascii
    numpy_array.dtype.assign(static_cast<const char *>(messages[2].data()),
                             messages[2].size());

    // Extract shape - reserve capacity to avoid reallocation
    numpy_array.shape.resize(numpy_array.ndim);
    const uint32_t *shape_data =
        static_cast<const uint32_t *>(messages[3].data());
    for (uint8_t i = 0; i < numpy_array.ndim; ++i) {
        numpy_array.shape[i] = ntohl(shape_data[i]);
    }

    // Get pointer to array data
    numpy_array.data = static_cast<const uint8_t *>(messages[4].data());

    return true;
}

bool ZMQROS2Bridge::decode_dict_of_numpy_arrays(
    const std::vector<zmq::message_t> &messages, DictofNumpyArray &event_data) {

    // Message format: [topic, key1, ndim1, dtype1, shape1, array1, key2, ...]
    // Each array has 5 parts: key, ndim, dtype, shape, data
    if (messages.size() < 6) {
        RCLCPP_ERROR(
            this->get_logger(),
            "Expected at least 6 parts for dict of numpy arrays, got %zu",
            messages.size());
        return false;
    }

    try {
        // Start at index 1 (skip topic), iterate through array groups
        for (size_t i = 1; i < messages.size(); i += 5) {
            NumpyArray numpy_array;

            // Key is ascii string so decode directly
            std::string key(static_cast<const char *>(messages[i].data()),
                            messages[i].size());

            // Extract ndim
            numpy_array.ndim =
                *(static_cast<const uint8_t *>(messages[i + 1].data()));

            // Extract dtype
            numpy_array.dtype.assign(
                static_cast<const char *>(messages[i + 2].data()),
                messages[i + 2].size());

            // Extract shape
            numpy_array.shape.resize(numpy_array.ndim);
            const uint32_t *shape_data =
                static_cast<const uint32_t *>(messages[i + 3].data());
            for (uint8_t j = 0; j < numpy_array.ndim; ++j) {
                numpy_array.shape[j] = ntohl(shape_data[j]);
            }

            // Get pointer to array data
            numpy_array.data =
                static_cast<const uint8_t *>(messages[i + 4].data());

            event_data[key] = std::move(numpy_array);
        }

        return true;

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to decode dict of arrays: %s",
                     e.what());
        return false;
    }
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

void ZMQROS2Bridge::publish_image(const NumpyArray &numpy_array) {
    if (!color_pub_)
        return;

    auto msg = sensor_msgs::msg::Image();

    // Set timestamp
    msg.header.stamp = this->now();
    msg.header.frame_id = color_zmq_topic_;

    // Set image dimensions (assuming HxWxC format from numpy)
    if (numpy_array.shape.size() != 3) {
        RCLCPP_WARN(this->get_logger(), "Unexpected shape dimensions: %zu",
                    numpy_array.shape.size());
        return;
    }

    msg.height = numpy_array.shape[0];
    msg.width = numpy_array.shape[1];
    uint32_t channels = numpy_array.shape[2];

    // Look up encoding and dtype_size from the map
    auto key = std::make_pair(numpy_array.dtype, channels);
    auto it = image_encoding_map_.find(key);

    if (it == image_encoding_map_.end()) {
        RCLCPP_ERROR(this->get_logger(),
                     "Unsupported dtype '%s' with %u channels",
                     numpy_array.dtype.c_str(), channels);
        return;
    }

    const auto &encoding_info = it->second;
    msg.encoding = encoding_info.encoding;
    uint32_t dtype_size = encoding_info.dtype_size;

    msg.is_bigendian = false;
    msg.step = msg.width * channels * dtype_size;
    auto nbytes = msg.height * msg.step;

    // Copy image data as raw bytes
    msg.data.assign(numpy_array.data, numpy_array.data + nbytes);

    color_pub_->publish(msg);

    RCLCPP_DEBUG(this->get_logger(),
                 "Published image %ux%u, %u channels, encoding=%s", msg.height,
                 msg.width, channels, msg.encoding.c_str());
}

void ZMQROS2Bridge::publish_events(const DictofNumpyArray &event_data) {
    if (!event_pub_)
        return;

    const std::vector<std::string> required_keys = {"x", "y", "t", "p"};

    // Check if all required keys exist
    for (const auto &key : required_keys) {
        if (event_data.find(key) == event_data.end()) {
            RCLCPP_ERROR(this->get_logger(),
                         "Missing required key '%s' in event data",
                         key.c_str());
            return;
        }
    }

    const auto &x_array = event_data.at("x");
    const auto &y_array = event_data.at("y");
    const auto &t_array = event_data.at("t");
    const auto &p_array = event_data.at("p");

    // Validate that all arrays are 1D
    if (x_array.ndim != 1 || y_array.ndim != 1 || t_array.ndim != 1 ||
        p_array.ndim != 1) {
        RCLCPP_ERROR(this->get_logger(),
                     "All event arrays must be 1-dimensional");
        return;
    }

    // Get the number of events
    uint32_t num_events = x_array.shape[0];

    // Validate that all arrays have the same length
    if (y_array.shape[0] != num_events || t_array.shape[0] != num_events ||
        p_array.shape[0] != num_events) {
        RCLCPP_ERROR(this->get_logger(),
                     "All event arrays must have the same length");
        return;
    }

    if (!(x_array.dtype == "<u2" || x_array.dtype == ">u2" ||
          x_array.dtype == "uint16" || y_array.dtype == "<u2" ||
          y_array.dtype == ">u2" || y_array.dtype == "uint16" ||
          t_array.dtype == "<u8" || t_array.dtype == ">u8" ||
          t_array.dtype == "uint64" || p_array.dtype == "|u1" ||
          p_array.dtype == "uint8")) {
        RCLCPP_ERROR(
            this->get_logger(),
            "Unsupported dtypes in event arrays: x=%s, y=%s, t=%s, p=%s",
            x_array.dtype.c_str(), y_array.dtype.c_str(), t_array.dtype.c_str(),
            p_array.dtype.c_str());
        return;
    }

    auto msg = zmq_ros2_bridge::msg::Event();

    // Set timestamp
    msg.header.stamp = this->now();
    msg.header.frame_id = event_zmq_topic_;

    const uint16_t *x_data =
        static_cast<const uint16_t *>(static_cast<const void *>(x_array.data));
    const uint16_t *y_data =
        static_cast<const uint16_t *>(static_cast<const void *>(y_array.data));
    const uint64_t *t_data =
        static_cast<const uint64_t *>(static_cast<const void *>(t_array.data));
    const uint8_t *p_data =
        static_cast<const uint8_t *>(static_cast<const void *>(p_array.data));

    msg.x.assign(x_data, x_data + num_events);
    msg.y.assign(y_data, y_data + num_events);
    msg.t.assign(t_data, t_data + num_events);
    msg.p.assign(p_data, p_data + num_events);

    event_pub_->publish(msg);

    RCLCPP_DEBUG(this->get_logger(), "Published %u events", num_events);
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

            if (decode_imu_json(messages, imu_json)) {
                publish_imu(imu_json);
            }
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

            NumpyArray numpy_array;

            if (decode_numpy_array(messages, numpy_array)) {
                publish_image(numpy_array);
            }
        }

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Color ZMQ thread error: %s",
                     e.what());
    }
}

void ZMQROS2Bridge::event_sub_pub() {
    if (!event_enable_)
        return;

    try {
        auto subscriber =
            create_zmq_subscriber(event_zmq_address_, event_zmq_topic_);

        RCLCPP_INFO(this->get_logger(), "Event ZMQ thread connected");

        while (running_ && rclcpp::ok()) {
            std::vector<zmq::message_t> messages;

            subscriber.set(zmq::sockopt::rcvtimeo, 1000);
            auto result =
                zmq::recv_multipart(subscriber, std::back_inserter(messages));

            if (!result || messages.empty()) {
                continue;
            }

            DictofNumpyArray event_data;

            if (decode_dict_of_numpy_arrays(messages, event_data)) {
                publish_events(event_data);
            }
        }

    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Event ZMQ thread error: %s",
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
