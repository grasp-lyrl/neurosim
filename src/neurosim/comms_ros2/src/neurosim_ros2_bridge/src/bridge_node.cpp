// Copyright (c) 2026, Neurosim contributors. Apache-2.0.
#include "neurosim_ros2_bridge/bridge_node.hpp"

#include <cortex_wire/header.hpp>
#include <cortex_wire/metadata.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

#include "neurosim_ros2_bridge/decoders.hpp"

// Fingerprint constants. Hard-coded so we don't depend on
// cortex_wire::find_by_name (which exists, but introduces an enum dependency
// for three values). If the fingerprint table changes upstream, update here.
//
// These come from deps/cortex/cortex_wire_cpp/include/cortex_wire/fingerprint_table.hpp;
// the bridge logs an error and refuses to wire up topics whose registered
// fingerprint mismatches what we expect here.
#include <cortex_wire/fingerprint_table.hpp>

namespace neurosim_ros2_bridge
{

namespace
{

// Build the QoS profile from an Entry. KEEP_LAST + RELIABLE is the default;
// best_effort opts into lossy-but-low-latency, which is what camera streams
// generally want.
rclcpp::QoS make_qos(const Entry & e)
{
  rclcpp::QoS q(e.depth);
  if (e.best_effort) {
    q.best_effort();
  } else {
    q.reliable();
  }
  return q;
}

// Slug for ipc:// endpoint names. Bridges multiple outbound entries by
// picking a deterministic path under /tmp/cortex/topics/.
std::string slugify(std::string s)
{
  for (auto & c : s) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
      c = '_';
    }
  }
  return s;
}

// Resolve the wire fingerprint for a cortex type name (e.g. "DictMessage").
std::uint64_t fingerprint_for(const std::string & cortex_type)
{
  const auto * entry = cortex_wire::find_by_name(cortex_type);
  if (!entry) {
    throw std::runtime_error("unknown cortex type: " + cortex_type);
  }
  return entry->fingerprint;
}

}  // namespace

// ---- InboundWorker -------------------------------------------------------
//
// One SUB socket + recv thread per inbound entry. Dispatch table is set at
// construction time so the recv loop doesn't branch on payload type.

class InboundWorker : public BridgeWorker
{
public:
  using Publisher = std::function<void(
        const cortex_wire::MessageHeader &,
        const cortex_wire::DecodedMetadata &,
        const std::vector<cortex_wire::ZmqFramePtr> &)>;

  InboundWorker(
    rclcpp::Node * node, zmq::context_t * ctx,
    Entry cfg, cortex_wire::TopicInfo topic,
    std::uint64_t expected_fp, Publisher publish)
  : node_(node), ctx_(ctx), cfg_(std::move(cfg)), topic_(std::move(topic)),
    expected_fp_(expected_fp), publish_(std::move(publish))
  {
    running_.store(true);
    thread_ = std::thread(&InboundWorker::recv_loop, this);
  }

  ~InboundWorker() override {stop();}

  void stop() override
  {
    if (!running_.exchange(false)) {return;}
    if (thread_.joinable()) {thread_.join();}
  }

private:
  void recv_loop()
  {
    zmq::socket_t sub(*ctx_, zmq::socket_type::sub);
    sub.set(zmq::sockopt::linger, 0);
    sub.set(zmq::sockopt::rcvtimeo, 100);  // wake every 100 ms to check running_
    try {
      sub.connect(topic_.address);
    } catch (const zmq::error_t & e) {
      RCLCPP_ERROR(
        node_->get_logger(), "[%s] connect(%s) failed: %s",
        cfg_.name.c_str(), topic_.address.c_str(), e.what());
      return;
    }
    sub.set(zmq::sockopt::subscribe, cfg_.cortex_topic);

    std::vector<zmq::message_t> frames;
    while (running_.load()) {
      frames.clear();

      // Recv the full multipart message.
      bool got_one = false;
      while (true) {
        zmq::message_t f;
        zmq::recv_result_t r;
        try {
          r = sub.recv(f, zmq::recv_flags::none);
        } catch (const zmq::error_t & e) {
          if (e.num() == ETERM) {return;}
          RCLCPP_WARN(node_->get_logger(), "[%s] recv: %s", cfg_.name.c_str(), e.what());
          break;
        }
        if (!r) {break;}  // timeout
        const bool more = f.more();
        frames.emplace_back(std::move(f));
        got_one = true;
        if (!more) {break;}
      }
      if (!got_one) {continue;}
      if (frames.size() < 3) {
        RCLCPP_WARN(
          node_->get_logger(), "[%s] short message: %zu frames",
          cfg_.name.c_str(), frames.size());
        continue;
      }

      try {
        const auto header = cortex_wire::MessageHeader::from_bytes(
          frames[1].data(), frames[1].size());
        if (header.fingerprint != expected_fp_) {
          RCLCPP_WARN(
            node_->get_logger(),
            "[%s] fingerprint mismatch on wire: header=0x%016lx expected=0x%016lx — dropping",
            cfg_.name.c_str(),
            static_cast<unsigned long>(header.fingerprint),
            static_cast<unsigned long>(expected_fp_));
          continue;
        }
        const auto metadata = cortex_wire::DecodedMetadata::from_bytes(
          frames[2].data(), frames[2].size());

        std::vector<cortex_wire::ZmqFramePtr> oob;
        oob.reserve(frames.size() > 3 ? frames.size() - 3 : 0);
        for (std::size_t i = 3; i < frames.size(); ++i) {
          oob.push_back(cortex_wire::make_owned(std::move(frames[i])));
        }

        publish_(header, metadata, oob);
      } catch (const std::exception & e) {
        RCLCPP_WARN(
          node_->get_logger(), "[%s] decode/publish: %s",
          cfg_.name.c_str(), e.what());
      }
    }
  }

  rclcpp::Node * node_;
  zmq::context_t * ctx_;
  Entry cfg_;
  cortex_wire::TopicInfo topic_;
  std::uint64_t expected_fp_;
  Publisher publish_;
  std::atomic<bool> running_{false};
  std::thread thread_;
};

// ---- OutboundWorker ------------------------------------------------------

class OutboundWorker : public BridgeWorker
{
public:
  OutboundWorker(
    rclcpp::Node * node, zmq::context_t * ctx,
    Entry cfg, std::string pub_endpoint, std::uint64_t fingerprint)
  : node_(node), ctx_(ctx), cfg_(std::move(cfg)),
    pub_endpoint_(std::move(pub_endpoint)),
    fingerprint_(fingerprint),
    pub_socket_(*ctx_, zmq::socket_type::pub)
  {
    pub_socket_.set(zmq::sockopt::linger, 0);
    pub_socket_.bind(pub_endpoint_);

    // Today the only supported outbound is Control via Float64MultiArray.
    sub_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
      cfg_.ros2_topic, make_qos(cfg_),
      [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {on_msg(*msg);});
  }

  ~OutboundWorker() override {stop();}

  void stop() override
  {
    sub_.reset();  // drop the subscription so no more callbacks race us.
  }

  const std::string & pub_endpoint() const {return pub_endpoint_;}

private:
  void on_msg(const std_msgs::msg::Float64MultiArray & msg)
  {
    try {
      auto out = decoders::encode_control(msg);

      const std::uint64_t now_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count());
      cortex_wire::MessageHeader hdr{
        fingerprint_, now_ns, sequence_.fetch_add(1)};

      std::array<std::uint8_t, cortex_wire::MessageHeader::kSize> hdr_bytes{};
      hdr.to_bytes(hdr_bytes.data());

      // Multipart layout: [topic, header, metadata, *oob]
      const std::size_t frame_count = 3 + out.oob_buffers.size();
      std::size_t i = 0;
      auto send = [&](const void * data, std::size_t size) {
          zmq::message_t m(size);
          std::memcpy(m.data(), data, size);
          const auto flags = (i + 1 < frame_count) ?
            zmq::send_flags::sndmore : zmq::send_flags::none;
          (void)pub_socket_.send(m, flags);
          ++i;
        };

      send(cfg_.cortex_topic.data(), cfg_.cortex_topic.size());
      send(hdr_bytes.data(), hdr_bytes.size());
      send(out.metadata.data(), out.metadata.size());
      for (const auto & b : out.oob_buffers) {
        send(b.data(), b.size());
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN(node_->get_logger(), "[%s] publish: %s", cfg_.name.c_str(), e.what());
    }
  }

  rclcpp::Node * node_;
  zmq::context_t * ctx_;
  Entry cfg_;
  std::string pub_endpoint_;
  std::uint64_t fingerprint_;
  zmq::socket_t pub_socket_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
  std::atomic<std::uint64_t> sequence_{0};
};

// ---- NeurosimRos2Bridge --------------------------------------------------

NeurosimRos2Bridge::NeurosimRos2Bridge(const rclcpp::NodeOptions & options)
: rclcpp::Node("neurosim_ros2_bridge", options),
  ctx_(std::make_shared<zmq::context_t>(1))
{
  this->declare_parameter<std::string>("config_path", "");
  initialize();
}

NeurosimRos2Bridge::~NeurosimRos2Bridge()
{
  for (auto & w : workers_) {
    if (w) {w->stop();}
  }
  workers_.clear();

  if (discovery_) {
    for (const auto & t : registered_topics_) {
      try {
        discovery_->unregister_topic(t);
      } catch (const std::exception & e) {
        RCLCPP_WARN(
          get_logger(), "discovery unregister('%s') failed: %s",
          t.c_str(), e.what());
      }
    }
  }
  registered_topics_.clear();
  discovery_.reset();

  if (ctx_) {
    ctx_->shutdown();
    ctx_->close();
  }
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
    *ctx_, config_.discovery_address);

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
  const std::uint64_t expected_fp = fingerprint_for(e.cortex_type);

  auto lookup = discovery_->lookup(e.cortex_topic);
  if (!lookup) {
    RCLCPP_ERROR(
      get_logger(),
      "[%s] discovery: topic '%s' not registered — make sure the simulator is running first",
      e.name.c_str(), e.cortex_topic.c_str());
    return;
  }
  if (lookup->fingerprint != expected_fp) {
    RCLCPP_ERROR(
      get_logger(),
      "[%s] fingerprint mismatch: daemon=0x%016lx expected=0x%016lx for type '%s'",
      e.name.c_str(),
      static_cast<unsigned long>(lookup->fingerprint),
      static_cast<unsigned long>(expected_fp),
      e.cortex_type.c_str());
    return;
  }

  InboundWorker::Publisher pub_fn;
  switch (e.payload) {
    // Each branch passes the decoded unique_ptr straight to publish() so we
    // hit the move overload — no extra copy, and intra-process zero-copy is
    // unlocked when a colocated subscriber loads into the same container.
    case Payload::State: {
        auto pub = this->create_publisher<msg::State>(e.ros2_topic, make_qos(e));
        const auto frame = e.frame_id;
        pub_fn = [pub, frame](
          const cortex_wire::MessageHeader & h,
          const cortex_wire::DecodedMetadata & m,
          const std::vector<cortex_wire::ZmqFramePtr> & o) {
            pub->publish(decoders::decode_state({h, m, o, frame}));
          };
        break;
      }
    case Payload::Imu: {
        auto pub = this->create_publisher<msg::Imu>(e.ros2_topic, make_qos(e));
        const auto frame = e.frame_id;
        pub_fn = [pub, frame](
          const cortex_wire::MessageHeader & h,
          const cortex_wire::DecodedMetadata & m,
          const std::vector<cortex_wire::ZmqFramePtr> & o) {
            pub->publish(decoders::decode_imu({h, m, o, frame}));
          };
        break;
      }
    case Payload::Events: {
        auto pub = this->create_publisher<msg::Events>(e.ros2_topic, make_qos(e));
        const auto frame = e.frame_id;
        pub_fn = [pub, frame](
          const cortex_wire::MessageHeader & h,
          const cortex_wire::DecodedMetadata & m,
          const std::vector<cortex_wire::ZmqFramePtr> & o) {
            pub->publish(decoders::decode_events({h, m, o, frame}));
          };
        break;
      }
    case Payload::ColorImage: {
        auto pub = this->create_publisher<sensor_msgs::msg::Image>(e.ros2_topic, make_qos(e));
        const auto frame = e.frame_id;
        pub_fn = [pub, frame](
          const cortex_wire::MessageHeader & h,
          const cortex_wire::DecodedMetadata & m,
          const std::vector<cortex_wire::ZmqFramePtr> & o) {
            pub->publish(decoders::decode_color_image({h, m, o, frame}));
          };
        break;
      }
    case Payload::DepthImage: {
        auto pub = this->create_publisher<sensor_msgs::msg::Image>(e.ros2_topic, make_qos(e));
        const auto frame = e.frame_id;
        pub_fn = [pub, frame](
          const cortex_wire::MessageHeader & h,
          const cortex_wire::DecodedMetadata & m,
          const std::vector<cortex_wire::ZmqFramePtr> & o) {
            pub->publish(decoders::decode_depth_image({h, m, o, frame}));
          };
        break;
      }
    case Payload::Control:
      throw std::runtime_error(
              "Payload::Control is not an inbound payload");
  }

  auto worker = std::make_unique<InboundWorker>(
    this, ctx_.get(), e, *lookup, expected_fp, std::move(pub_fn));
  workers_.push_back(std::move(worker));
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

  const std::string endpoint =
    "ipc:///tmp/cortex/topics/" +
    slugify(config_.node_name_prefix) + "__" + slugify(e.name);

  // Make sure the parent directory exists; ZMQ won't create it.
  constexpr std::string_view kIpc = "ipc://";
  if (endpoint.rfind(kIpc, 0) == 0) {
    std::filesystem::path p(endpoint.substr(kIpc.size()));
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    if (ec) {
      throw std::runtime_error(
              "cannot create parent dir for " + endpoint + ": " + ec.message());
    }
  }

  auto worker = std::make_unique<OutboundWorker>(
    this, ctx_.get(), e, endpoint, fp);

  cortex_wire::TopicInfo info{
    e.cortex_topic,
    endpoint,
    e.cortex_type,
    fp,
    this->get_fully_qualified_name(),
  };
  try {
    discovery_->register_topic(info);
    registered_topics_.push_back(e.cortex_topic);
  } catch (const std::exception & ex) {
    throw std::runtime_error(
            "discovery register('" + e.cortex_topic + "') failed: " + ex.what());
  }

  workers_.push_back(std::move(worker));
  RCLCPP_INFO(
    get_logger(), "[%s] ros2(%s) -> cortex(%s @ %s)",
    e.name.c_str(), e.ros2_topic.c_str(), e.cortex_topic.c_str(), endpoint.c_str());
}

}  // namespace neurosim_ros2_bridge

RCLCPP_COMPONENTS_REGISTER_NODE(neurosim_ros2_bridge::NeurosimRos2Bridge)
