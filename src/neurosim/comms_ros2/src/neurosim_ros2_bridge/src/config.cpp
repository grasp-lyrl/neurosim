#include "neurosim_ros2_bridge/config.hpp"

namespace neurosim_ros2_bridge
{

namespace
{

const std::unordered_map<std::string, Payload> kPayloadFromString = {
  {"state", Payload::State},
  {"imu", Payload::Imu},
  {"events", Payload::Events},
  {"color_image", Payload::ColorImage},
  {"depth_image", Payload::DepthImage},
  {"control", Payload::Control},
};

const char * payload_default_cortex_type(Payload p)
{
  switch (p) {
    case Payload::State: return "DictMessage";
    case Payload::Imu: return "DictMessage";
    case Payload::Events: return "MultiArrayMessage";
    case Payload::ColorImage:
    case Payload::DepthImage: return "ArrayMessage";
    case Payload::Control: return "DictMessage";
  }
  return "";
}

Entry parse_entry(const YAML::Node & node, bool is_inbound, std::size_t idx)
{
  if (!node.IsMap()) {
    throw std::runtime_error(
            "entry #" + std::to_string(idx) + " is not a map");
  }
  Entry e;
  e.name = node["name"] ? node["name"].as<std::string>()
                        : ("entry_" + std::to_string(idx));

  const auto payload_str = node["payload"]
    ? node["payload"].as<std::string>()
    : throw std::runtime_error("[" + e.name + "] missing 'payload'");
  const auto p = parse_payload(payload_str);
  if (!p) {
    throw std::runtime_error(
            "[" + e.name + "] unknown payload '" + payload_str + "'");
  }
  e.payload = *p;

  // Direction sanity check: control is the only outbound payload today.
  const bool payload_is_outbound = (e.payload == Payload::Control);
  if (is_inbound && payload_is_outbound) {
    throw std::runtime_error(
            "[" + e.name + "] payload '" + payload_str +
            "' is outbound (ros2_to_cortex); move the entry");
  }
  if (!is_inbound && !payload_is_outbound) {
    throw std::runtime_error(
            "[" + e.name + "] payload '" + payload_str +
            "' is inbound (cortex_to_ros2); move the entry");
  }

  e.cortex_topic = node["cortex_topic"]
    ? node["cortex_topic"].as<std::string>()
    : throw std::runtime_error("[" + e.name + "] missing 'cortex_topic'");
  e.ros2_topic = node["ros2_topic"]
    ? node["ros2_topic"].as<std::string>()
    : throw std::runtime_error("[" + e.name + "] missing 'ros2_topic'");
  e.frame_id = node["frame_id"] ? node["frame_id"].as<std::string>() : "";
  e.cortex_type = node["cortex_type"]
    ? node["cortex_type"].as<std::string>()
    : payload_default_cortex_type(e.payload);

  if (auto q = node["qos"]; q && q.IsMap()) {
    if (q["depth"]) {e.depth = q["depth"].as<std::uint32_t>();}
    if (q["reliability"]) {
      const auto r = q["reliability"].as<std::string>();
      e.best_effort = (r == "best_effort");
    }
  }
  return e;
}

}  // namespace

std::optional<Payload> parse_payload(const std::string & s)
{
  auto it = kPayloadFromString.find(s);
  if (it == kPayloadFromString.end()) {return std::nullopt;}
  return it->second;
}

std::string payload_to_string(Payload p)
{
  for (const auto & [k, v] : kPayloadFromString) {
    if (v == p) {return k;}
  }
  return "?";
}

BridgeConfig load_config(const std::string & path)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception & e) {
    throw std::runtime_error("cannot read config '" + path + "': " + e.what());
  }
  if (!root.IsMap()) {
    throw std::runtime_error("config root must be a map: " + path);
  }
  if (root["version"] && root["version"].as<int>() != 1) {
    throw std::runtime_error("unsupported config version (expected 1)");
  }

  BridgeConfig cfg;
  if (root["discovery_address"]) {
    cfg.discovery_address = root["discovery_address"].as<std::string>();
  }
  if (root["node_name_prefix"]) {
    cfg.node_name_prefix = root["node_name_prefix"].as<std::string>();
  }

  if (auto n = root["cortex_to_ros2"]; n && n.IsSequence()) {
    cfg.cortex_to_ros2.reserve(n.size());
    for (std::size_t i = 0; i < n.size(); ++i) {
      cfg.cortex_to_ros2.push_back(parse_entry(n[i], /*is_inbound=*/true, i));
    }
  }
  if (auto n = root["ros2_to_cortex"]; n && n.IsSequence()) {
    cfg.ros2_to_cortex.reserve(n.size());
    for (std::size_t i = 0; i < n.size(); ++i) {
      cfg.ros2_to_cortex.push_back(parse_entry(n[i], /*is_inbound=*/false, i));
    }
  }
  return cfg;
}

}  // namespace neurosim_ros2_bridge
