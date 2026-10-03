#include "neurosim_ros2_bridge/decoders.hpp"

#include <tf2/LinearMath/Quaternion.h>

#include <cmath>
#include <string>

namespace neurosim_ros2_bridge::decoders
{

namespace
{

using cortex_wire::DecodedMetadata;
using cortex_wire::OobBuffer;
using cortex_wire::OobDescriptor;
using cortex_wire::WireDecodeError;
using cortex_wire::ZmqFramePtr;

// --------------------------------------------------------------------------
// OOB helpers — wrappers around cortex_wire::OobBuffer<T> that add the
// bridge's validation conventions (dtype, element count, frame-bounds).
// --------------------------------------------------------------------------

inline std::size_t shape_elements(const std::vector<std::int64_t> & shape)
{
  std::size_t n = 1;
  for (auto d : shape) {
    if (d < 0) {return 0;}
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

// Resolve a descriptor into a typed view over its ZMQ frame. The OobBuffer
// owns a shared_ptr to the frame so the bytes outlive any caller holding
// the view — no raw-pointer lifetime games.
template<typename T>
OobBuffer<T> oob_view(
  const OobDescriptor & desc,
  const std::vector<ZmqFramePtr> & frames,
  std::string_view expected_dtype = {})
{
  if (!expected_dtype.empty() && desc.dtype != expected_dtype) {
    throw WireDecodeError(
            "dtype mismatch (got '" + desc.dtype +
            "', expected '" + std::string(expected_dtype) + "')");
  }
  if (desc.buffer_index >= frames.size()) {
    throw WireDecodeError(
            "OOB buffer index " + std::to_string(desc.buffer_index) +
            " out of range");
  }
  const auto & frame = frames[desc.buffer_index];
  const std::size_t n = shape_elements(desc.shape);
  if (frame->size() < n * sizeof(T)) {
    throw WireDecodeError(
            "OOB frame too small (" + std::to_string(frame->size()) +
            " < " + std::to_string(n * sizeof(T)) + ")");
  }
  return OobBuffer<T>(frame, n);
}

// Look up an OOB descriptor under a map key, materialise a typed view.
// Single call covers the {key -> descriptor -> buffer} traversal.
template<typename T>
OobBuffer<T> map_oob_view(
  const msgpack::object & obj, std::string_view key,
  const std::vector<ZmqFramePtr> & frames,
  std::string_view expected_dtype = {})
{
  const auto & v = map_require(obj, key);
  auto desc = DecodedMetadata::as_oob(v);
  if (!desc) {
    throw WireDecodeError(
            "key '" + std::string(key) + "' is not an OOB descriptor");
  }
  return oob_view<T>(*desc, frames, expected_dtype);
}

// ArrayMessage fields are [data (OOB), name (str), frame_id (str)].
// Return both the typed view and the descriptor's shape so callers don't
// have to re-walk the metadata.
template<typename T>
struct ArrayView
{
  OobBuffer<T> data;
  std::vector<std::int64_t> shape;
};

template<typename T>
ArrayView<T> array_message_view(
  const Inbound & in, std::string_view expected_dtype)
{
  if (in.metadata.field_count() != 3) {
    throw WireDecodeError("expected 3 metadata fields");
  }
  auto desc = DecodedMetadata::as_oob(in.metadata.field(0));
  if (!desc) {
    throw WireDecodeError("field 0 is not an OOB descriptor");
  }
  auto shape = desc->shape;
  return ArrayView<T>{oob_view<T>(*desc, in.oob_frames, expected_dtype),
    std::move(shape)};
}

// Headers carry the simulation time the sample was taken at, not the wall clock
// cortex stamps at publish. Array messages pack it into their cortex frame_id as
// "uuid|seconds|simsteps" (cortex_io.sensor_frame_id).
double frame_id_seconds(const msgpack::object & frame_id)
{
  const auto s = as_str(frame_id);
  const auto first = s.find('|');
  const auto second = s.find('|', first + 1);
  if (first == std::string_view::npos || second == std::string_view::npos) {
    throw WireDecodeError("frame_id is not uuid|seconds|simsteps: " + std::string(s));
  }
  return std::stod(std::string(s.substr(first + 1, second - first - 1)));
}

void stamp_header(
  std_msgs::msg::Header & h, const double sim_seconds, const std::string & frame_id)
{
  const auto ns = std::llround(sim_seconds * 1e9);
  h.stamp.sec = static_cast<std::int32_t>(ns / 1'000'000'000LL);
  h.stamp.nanosec = static_cast<std::uint32_t>(ns % 1'000'000'000LL);
  h.frame_id = frame_id;
}

void set_vec3(geometry_msgs::msg::Vector3 & out, const double (&v)[3])
{
  out.x = v[0];
  out.y = v[1];
  out.z = v[2];
}

// Copy a length-3 vector out of an OOB frame whose dtype may be f4 or f8.
// The simulator's IMU executor sometimes emits one, sometimes the other,
// depending on whether the source was a torch tensor or a numpy array.
void unpack_vec3_oob(
  const OobDescriptor & desc, const std::vector<ZmqFramePtr> & frames,
  geometry_msgs::msg::Vector3 & out)
{
  if (desc.dtype == "<f8") {
    auto v = oob_view<double>(desc, frames, "<f8");
    if (v.size() < 3) {throw WireDecodeError("expected at least 3 elements");}
    out.x = v[0]; out.y = v[1]; out.z = v[2];
  } else if (desc.dtype == "<f4") {
    auto v = oob_view<float>(desc, frames, "<f4");
    if (v.size() < 3) {throw WireDecodeError("expected at least 3 elements");}
    out.x = v[0]; out.y = v[1]; out.z = v[2];
  } else {
    throw WireDecodeError("unsupported dtype '" + desc.dtype + "'");
  }
}

// memcpy from a typed OobBuffer<T> into a destination std::vector<T>.
// One contiguous copy; the destination is resized to view.size().
template<typename T>
void copy_into_vector(const OobBuffer<T> & view, std::vector<T> & out)
{
  out.resize(view.size());
  std::memcpy(out.data(), view.data(), view.size_bytes());
}

struct EventViews
{
  OobBuffer<std::uint16_t> x;
  OobBuffer<std::uint16_t> y;
  OobBuffer<std::uint64_t> t;
  OobBuffer<std::uint8_t> p;
};

// dtype contract is fixed by the simulator's EventBuffer; we enforce it
// here so a mismatch is loud instead of silently misinterpreted bytes.
EventViews event_views(const Inbound & in)
{
  if (in.metadata.field_count() != 2) {
    throw WireDecodeError("events: expected 2 metadata fields");
  }
  const auto & arrays = in.metadata.field(0);
  if (arrays.type != msgpack::type::MAP) {
    throw WireDecodeError("events: arrays field is not a map");
  }
  EventViews ev{
    map_oob_view<std::uint16_t>(arrays, "x", in.oob_frames, "<u2"),
    map_oob_view<std::uint16_t>(arrays, "y", in.oob_frames, "<u2"),
    map_oob_view<std::uint64_t>(arrays, "t", in.oob_frames, "<u8"),
    map_oob_view<std::uint8_t>(arrays, "p", in.oob_frames, "|u1")};
  const std::size_t n = ev.x.size();
  if (ev.y.size() != n || ev.t.size() != n || ev.p.size() != n) {
    throw WireDecodeError("events: array lengths mismatch");
  }
  return ev;
}

}  // namespace

// ---- State ----------------------------------------------------------------

std::unique_ptr<msg::State> decode_state(const Inbound & in)
{
  if (in.metadata.field_count() != 1) {
    throw WireDecodeError("state: expected 1 metadata field");
  }
  const auto & data = in.metadata.field(0);
  if (data.type != msgpack::type::MAP) {
    throw WireDecodeError("state: top-level field is not a map");
  }

  double x[3], q[4], v[3], w[3];
  read_doubles(map_require(data, "x"), x);
  read_doubles(map_require(data, "q"), q);
  read_doubles(map_require(data, "v"), v);
  read_doubles(map_require(data, "w"), w);

  auto out = std::make_unique<msg::State>();
  out->timestamp = as_double(map_require(data, "timestamp"));
  stamp_header(out->header, out->timestamp, in.frame_id);
  if (auto * s = map_get(data, "simsteps")) {out->simsteps = as_uint(*s);}
  set_vec3(out->x, x);
  out->q.x = q[0];
  out->q.y = q[1];
  out->q.z = q[2];
  out->q.w = q[3];
  set_vec3(out->v, v);
  set_vec3(out->w, w);
  return out;
}

std::unique_ptr<nav_msgs::msg::Odometry> decode_odometry(
  const Inbound & in, const std::string & child_frame_id)
{
  const auto state = decode_state(in);
  auto out = std::make_unique<nav_msgs::msg::Odometry>();
  out->header = state->header;
  out->child_frame_id = child_frame_id;
  out->pose.pose.position.x = state->x.x;
  out->pose.pose.position.y = state->x.y;
  out->pose.pose.position.z = state->x.z;
  out->pose.pose.orientation = state->q;
  // State.v is in the world frame; Odometry's twist is in the child frame.
  const tf2::Quaternion q(state->q.x, state->q.y, state->q.z, state->q.w);
  const auto v_body = tf2::quatRotate(
    q.inverse(), tf2::Vector3(state->v.x, state->v.y, state->v.z));
  out->twist.twist.linear.x = v_body.x();
  out->twist.twist.linear.y = v_body.y();
  out->twist.twist.linear.z = v_body.z();
  out->twist.twist.angular = state->w;
  return out;
}

geometry_msgs::msg::TransformStamped transform_from_odometry(
  const nav_msgs::msg::Odometry & odom)
{
  geometry_msgs::msg::TransformStamped t;
  t.header = odom.header;
  t.child_frame_id = odom.child_frame_id;
  t.transform.translation.x = odom.pose.pose.position.x;
  t.transform.translation.y = odom.pose.pose.position.y;
  t.transform.translation.z = odom.pose.pose.position.z;
  t.transform.rotation = odom.pose.pose.orientation;
  return t;
}

// ---- IMU ------------------------------------------------------------------

std::unique_ptr<msg::Imu> decode_imu(const Inbound & in)
{
  if (in.metadata.field_count() != 1) {
    throw WireDecodeError("imu: expected 1 metadata field");
  }
  const auto & data = in.metadata.field(0);
  if (data.type != msgpack::type::MAP) {
    throw WireDecodeError("imu: top-level field is not a map");
  }

  // Resolve the descriptors here, then let unpack_vec3_oob pick float vs
  // double based on the wire dtype.
  auto accel_desc = DecodedMetadata::as_oob(map_require(data, "accel"));
  auto gyro_desc = DecodedMetadata::as_oob(map_require(data, "gyro"));
  if (!accel_desc || !gyro_desc) {
    throw WireDecodeError("imu: accel/gyro fields are not OOB descriptors");
  }

  auto out = std::make_unique<msg::Imu>();
  out->timestamp = as_double(map_require(data, "timestamp"));
  stamp_header(out->header, out->timestamp, in.frame_id);
  if (auto * s = map_get(data, "simsteps")) {out->simsteps = as_uint(*s);}
  if (auto * u = map_get(data, "uuid"); u && u->type == msgpack::type::STR) {
    out->uuid.assign(u->via.str.ptr, u->via.str.size);
  }
  unpack_vec3_oob(*accel_desc, in.oob_frames, out->accel);
  unpack_vec3_oob(*gyro_desc, in.oob_frames, out->gyro);
  return out;
}

std::unique_ptr<sensor_msgs::msg::Imu> decode_sensor_imu(const Inbound & in)
{
  const auto imu = decode_imu(in);
  auto out = std::make_unique<sensor_msgs::msg::Imu>();
  out->header = imu->header;
  out->orientation_covariance[0] = -1.0;
  out->angular_velocity = imu->gyro;
  out->linear_acceleration = imu->accel;
  return out;
}

// ---- Events ---------------------------------------------------------------

std::unique_ptr<msg::Events> decode_events(const Inbound & in)
{
  const auto ev = event_views(in);
  auto out = std::make_unique<msg::Events>();
  stamp_header(out->header, frame_id_seconds(in.metadata.field(1)), in.frame_id);
  copy_into_vector(ev.x, out->x);
  copy_into_vector(ev.y, out->y);
  copy_into_vector(ev.t, out->t);
  copy_into_vector(ev.p, out->p);
  return out;
}

std::unique_ptr<sensor_msgs::msg::Image> decode_events_image(
  const Inbound & in, std::uint32_t width, std::uint32_t height)
{
  const auto ev = event_views(in);
  auto out = std::make_unique<sensor_msgs::msg::Image>();
  stamp_header(out->header, frame_id_seconds(in.metadata.field(1)), in.frame_id);
  out->height = height;
  out->width = width;
  out->encoding = "rgb8";
  out->is_bigendian = 0;
  out->step = width * 3;
  out->data.assign(static_cast<std::size_t>(out->step) * height, 0);
  for (std::size_t i = 0; i < ev.x.size(); ++i) {
    if (ev.x[i] >= width || ev.y[i] >= height) {
      throw WireDecodeError("events_image: event outside the configured width x height");
    }
    out->data[static_cast<std::size_t>(ev.y[i]) * out->step + ev.x[i] * 3u +
      (ev.p[i] ? 2u : 0u)] = 255;
  }
  return out;
}

// ---- Color / Depth Image --------------------------------------------------

std::unique_ptr<sensor_msgs::msg::Image> decode_color_image(const Inbound & in)
{
  auto av = array_message_view<std::uint8_t>(in, "|u1");
  if (av.shape.size() != 3 || av.shape[2] != 3) {
    throw WireDecodeError("color: expected HxWx3 uint8 array");
  }
  const auto height = static_cast<std::uint32_t>(av.shape[0]);
  const auto width = static_cast<std::uint32_t>(av.shape[1]);

  auto out = std::make_unique<sensor_msgs::msg::Image>();
  stamp_header(out->header, frame_id_seconds(in.metadata.field(2)), in.frame_id);
  out->height = height;
  out->width = width;
  out->encoding = "rgb8";
  out->is_bigendian = 0;
  out->step = width * 3;
  copy_into_vector(av.data, out->data);    // single memcpy, size_bytes = H*W*3
  return out;
}

std::unique_ptr<sensor_msgs::msg::Image> decode_depth_image(const Inbound & in)
{
  auto av = array_message_view<float>(in, "<f4");
  std::uint32_t height = 0, width = 0;
  if (av.shape.size() == 2) {
    height = static_cast<std::uint32_t>(av.shape[0]);
    width = static_cast<std::uint32_t>(av.shape[1]);
  } else if (av.shape.size() == 3 && av.shape[2] == 1) {
    height = static_cast<std::uint32_t>(av.shape[0]);
    width = static_cast<std::uint32_t>(av.shape[1]);
  } else {
    throw WireDecodeError("depth: expected HxW float32 array");
  }

  auto out = std::make_unique<sensor_msgs::msg::Image>();
  stamp_header(out->header, frame_id_seconds(in.metadata.field(2)), in.frame_id);
  out->height = height;
  out->width = width;
  out->encoding = "32FC1";
  out->is_bigendian = 0;
  out->step = width * sizeof(float);
  // sensor_msgs::Image::data is bytes; reinterpret the float view for the
  // memcpy. Single contiguous copy of H*W*4 bytes.
  out->data.resize(av.data.size_bytes());
  std::memcpy(out->data.data(), av.data.data(), av.data.size_bytes());
  return out;
}

// ---- CameraInfo / Range / Clock -------------------------------------------

std::unique_ptr<sensor_msgs::msg::CameraInfo> decode_camera_info(const Inbound & in)
{
  if (in.metadata.field_count() != 1) {
    throw WireDecodeError("camera_info: expected 1 metadata field");
  }
  const auto & data = in.metadata.field(0);
  auto out = std::make_unique<sensor_msgs::msg::CameraInfo>();
  stamp_header(out->header, as_double(map_require(data, "timestamp")), in.frame_id);
  out->width = static_cast<std::uint32_t>(as_uint(map_require(data, "width")));
  out->height = static_cast<std::uint32_t>(as_uint(map_require(data, "height")));
  out->distortion_model = std::string(as_str(map_require(data, "distortion_model")));
  out->d = read_double_vector(map_require(data, "d"));
  read_doubles(map_require(data, "k"), out->k);
  read_doubles(map_require(data, "r"), out->r);
  read_doubles(map_require(data, "p"), out->p);
  return out;
}

std::unique_ptr<sensor_msgs::msg::Range> decode_range(const Inbound & in)
{
  if (in.metadata.field_count() != 1) {
    throw WireDecodeError("range: expected 1 metadata field");
  }
  const auto & data = in.metadata.field(0);
  auto out = std::make_unique<sensor_msgs::msg::Range>();
  stamp_header(out->header, as_double(map_require(data, "timestamp")), in.frame_id);
  out->radiation_type = sensor_msgs::msg::Range::INFRARED;
  out->field_of_view = static_cast<float>(as_double(map_require(data, "field_of_view")));
  out->min_range = static_cast<float>(as_double(map_require(data, "min_range")));
  out->max_range = static_cast<float>(as_double(map_require(data, "max_range")));
  out->range = static_cast<float>(as_double(map_require(data, "range")));
  return out;
}

std::unique_ptr<rosgraph_msgs::msg::Clock> decode_clock(const Inbound & in)
{
  auto out = std::make_unique<rosgraph_msgs::msg::Clock>();
  out->clock = decode_state(in)->header.stamp;
  return out;
}

// ---- Control (ROS 2 -> Cortex) --------------------------------------------

cortex_wire::MetadataBuilder::Frames encode_control(
  const std_msgs::msg::Float64MultiArray & msg)
{
  // Cortex DictMessage: 1 field, a msgpack MAP. The simulator's
  // receive_control expects {"cmd_motor_speeds": [...], ...}. We pack the
  // motor speeds as an inline msgpack array (they're tiny — 4 floats), and a
  // monotonic timestamp so the simulator's logging stays sensible.
  cortex_wire::MetadataBuilder b(1);

  auto & p = b.packer();
  const std::size_t map_size = 2;        // cmd_motor_speeds + timestamp
  p.pack_map(static_cast<std::uint32_t>(map_size));

  p.pack_str(static_cast<std::uint32_t>(std::string_view("cmd_motor_speeds").size()));
  p.pack_str_body("cmd_motor_speeds", static_cast<std::uint32_t>(16));
  p.pack_array(static_cast<std::uint32_t>(msg.data.size()));
  for (const double v : msg.data) {
    p.pack_double(v);
  }

  const double now_s = static_cast<double>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count()) * 1e-9;
  p.pack_str(static_cast<std::uint32_t>(std::string_view("timestamp").size()));
  p.pack_str_body("timestamp", static_cast<std::uint32_t>(9));
  p.pack_double(now_s);

  return std::move(b).finish();
}

}  // namespace neurosim_ros2_bridge::decoders
