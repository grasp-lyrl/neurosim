#include "neurosim_ros2_bridge/bridge_node.hpp"

namespace neurosim_ros2_bridge
{

namespace
{

rclcpp::QoS make_qos(const Entry & e)
{
  rclcpp::QoS q(e.depth);
  if (e.best_effort) {q.best_effort();} else {q.reliable();}
  return q;
}

// Resolve "DictMessage" / "ArrayMessage" / etc. to the wire fingerprint that
// cortex.messages.standard generated for them. Throws on unknown names so
// the bridge fails loudly on config typos.
std::uint64_t fingerprint_for(const std::string & cortex_type)
{
  const auto * entry = cortex_wire::find_by_name(cortex_type);
  if (!entry) {
    throw std::runtime_error("unknown cortex type: " + cortex_type);
  }
  return entry->fingerprint;
}

// ---- InboundWorker -------------------------------------------------------
//
// Owns a cortex_wire::Subscriber (which owns the SUB socket + recv thread)
// and a payload-specific dispatch lambda that decodes the inbound message
// and forwards it via rclcpp.
class InboundWorker : public BridgeWorker
{
public:
  using Dispatch = std::function<void (
    const cortex_wire::MessageHeader &,
    const cortex_wire::DecodedMetadata &,
    const std::vector<cortex_wire::ZmqFramePtr> &)>;

  InboundWorker(
    rclcpp::Node * node, cortex_wire::Context ctx,
    cortex_wire::DiscoveryClient & disc,
    std::string name, std::string cortex_topic,
    std::uint64_t fp, Dispatch dispatch)
  : name_(std::move(name)),
    dispatch_(std::move(dispatch)),
    sub_(cortex_wire::Subscriber::connect(
        std::move(ctx), disc, std::move(cortex_topic), fp,
        [this](const cortex_wire::Subscriber::Inbound & in) {
          dispatch_(in.header, in.metadata, in.oob_frames);
        },
        [node, n = name_](std::string_view what) {
          RCLCPP_WARN(
            node->get_logger(), "[%s] %.*s",
            n.c_str(), static_cast<int>(what.size()), what.data());
        }))
  {
  }

  // Subscriber's destructor joins the recv thread; no extra stop work needed.
  void stop() override {}

private:
  std::string name_;
  Dispatch dispatch_;
  cortex_wire::Subscriber sub_;
};

// ---- OutboundWorker ------------------------------------------------------
//
// Owns a cortex_wire::Publisher (which owns the PUB socket and the discovery
// registration) plus an rclcpp::Subscription that encodes each inbound ROS
// message and forwards via Publisher::publish().
class OutboundWorker : public BridgeWorker
{
public:
  OutboundWorker(
    rclcpp::Node * node, cortex_wire::Context ctx,
    cortex_wire::DiscoveryClient & disc,
    const Entry & cfg, std::uint64_t fp)
  : name_(cfg.name),
    pub_(std::move(ctx), disc,
      cfg.cortex_topic, cfg.cortex_type, fp,
      node->get_fully_qualified_name())
  {
    // Today the only supported outbound is Control via Float64MultiArray.
    sub_ = node->create_subscription<std_msgs::msg::Float64MultiArray>(
      cfg.ros2_topic, make_qos(cfg),
      [this, node](std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
        try {
          pub_.publish(decoders::encode_control(*msg));
        } catch (const std::exception & e) {
          RCLCPP_WARN(
            node->get_logger(), "[%s] publish: %s",
            name_.c_str(), e.what());
        }
      });
  }

  void stop() override
  {
    sub_.reset();   // drop the subscription so no more callbacks race the dtor
  }

private:
  std::string name_;
  cortex_wire::Publisher pub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
};

}  // namespace

// ---- NeurosimRos2Bridge --------------------------------------------------

NeurosimRos2Bridge::NeurosimRos2Bridge(const rclcpp::NodeOptions & options)
: rclcpp::Node("neurosim_ros2_bridge", options)
{
  this->declare_parameter<std::string>("config_path", "");
  initialize();
}

NeurosimRos2Bridge::~NeurosimRos2Bridge()
{
  for (auto & w : workers_) {
    if (w) {w->stop();}
  }
  workers_.clear();        // joins recv threads / unregisters publishers
  discovery_.reset();
}

void NeurosimRos2Bridge::initialize()
{
  const auto path = this->get_parameter("config_path").as_string();
  if (path.empty()) {
    throw std::runtime_error(
            "neurosim_ros2_bridge: required parameter 'config_path' is empty");
  }
  config_ = load_config(path);

  discovery_ = std::make_unique<cortex_wire::DiscoveryClient>(
    ctx_.raw(), config_.discovery_address);

  for (const auto & e : config_.cortex_to_ros2) {
    try {
      wire_inbound(e);
    } catch (const std::exception & ex) {
      RCLCPP_ERROR(get_logger(), "[%s] wire_inbound: %s", e.name.c_str(), ex.what());
    }
  }
  for (const auto & e : config_.ros2_to_cortex) {
    try {
      wire_outbound(e);
    } catch (const std::exception & ex) {
      RCLCPP_ERROR(get_logger(), "[%s] wire_outbound: %s", e.name.c_str(), ex.what());
    }
  }
}

void NeurosimRos2Bridge::wire_inbound(const Entry & e)
{
  const std::uint64_t fp = fingerprint_for(e.cortex_type);

  // Build the payload-specific dispatch lambda: decode the cortex message and
  // publish on the right ROS 2 topic. The unique_ptr passed to publish() hits
  // the move overload (no copy, intra-process zero-copy when colocated).
  InboundWorker::Dispatch dispatch;
  switch (e.payload) {
    case Payload::State: {
        auto pub = create_publisher<msg::State>(e.ros2_topic, make_qos(e));
        const auto frame = e.frame_id;
        dispatch = [pub, frame](
          const cortex_wire::MessageHeader & h,
          const cortex_wire::DecodedMetadata & m,
          const std::vector<cortex_wire::ZmqFramePtr> & o) {
            pub->publish(decoders::decode_state({h, m, o, frame}));
          };
        break;
      }
    case Payload::Imu: {
        auto pub = create_publisher<msg::Imu>(e.ros2_topic, make_qos(e));
        const auto frame = e.frame_id;
        dispatch = [pub, frame](
          const cortex_wire::MessageHeader & h,
          const cortex_wire::DecodedMetadata & m,
          const std::vector<cortex_wire::ZmqFramePtr> & o) {
            pub->publish(decoders::decode_imu({h, m, o, frame}));
          };
        break;
      }
    case Payload::Events: {
        auto pub = create_publisher<msg::Events>(e.ros2_topic, make_qos(e));
        const auto frame = e.frame_id;
        dispatch = [pub, frame](
          const cortex_wire::MessageHeader & h,
          const cortex_wire::DecodedMetadata & m,
          const std::vector<cortex_wire::ZmqFramePtr> & o) {
            pub->publish(decoders::decode_events({h, m, o, frame}));
          };
        break;
      }
    case Payload::ColorImage: {
        auto pub = create_publisher<sensor_msgs::msg::Image>(e.ros2_topic, make_qos(e));
        const auto frame = e.frame_id;
        dispatch = [pub, frame](
          const cortex_wire::MessageHeader & h,
          const cortex_wire::DecodedMetadata & m,
          const std::vector<cortex_wire::ZmqFramePtr> & o) {
            pub->publish(decoders::decode_color_image({h, m, o, frame}));
          };
        break;
      }
    case Payload::DepthImage: {
        auto pub = create_publisher<sensor_msgs::msg::Image>(e.ros2_topic, make_qos(e));
        const auto frame = e.frame_id;
        dispatch = [pub, frame](
          const cortex_wire::MessageHeader & h,
          const cortex_wire::DecodedMetadata & m,
          const std::vector<cortex_wire::ZmqFramePtr> & o) {
            pub->publish(decoders::decode_depth_image({h, m, o, frame}));
          };
        break;
      }
    case Payload::Control:
      throw std::runtime_error("Payload::Control is not an inbound payload");
  }

  workers_.push_back(
    std::make_unique<InboundWorker>(
      this, ctx_, *discovery_, e.name, e.cortex_topic, fp, std::move(dispatch)));

  RCLCPP_INFO(
    get_logger(), "[%s] cortex(%s) -> ros2(%s) [%s]",
    e.name.c_str(), e.cortex_topic.c_str(), e.ros2_topic.c_str(),
    payload_to_string(e.payload).c_str());
}

void NeurosimRos2Bridge::wire_outbound(const Entry & e)
{
  if (e.payload != Payload::Control) {
    throw std::runtime_error(
            "outbound payload '" + payload_to_string(e.payload) + "' not implemented");
  }
  const std::uint64_t fp = fingerprint_for(e.cortex_type);

  workers_.push_back(
    std::make_unique<OutboundWorker>(this, ctx_, *discovery_, e, fp));

  RCLCPP_INFO(
    get_logger(), "[%s] ros2(%s) -> cortex(%s)",
    e.name.c_str(), e.ros2_topic.c_str(), e.cortex_topic.c_str());
}

}  // namespace neurosim_ros2_bridge

RCLCPP_COMPONENTS_REGISTER_NODE(neurosim_ros2_bridge::NeurosimRos2Bridge)
