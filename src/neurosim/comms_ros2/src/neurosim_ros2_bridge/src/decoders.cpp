#include "neurosim_ros2_bridge/decoders.hpp"

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
// msgpack helpers — narrow on purpose. Only the shapes the neurosim simulator
// emits are accepted; anything else throws and the bridge logs+drops.
// --------------------------------------------------------------------------

double as_double(const msgpack::object & o)
{
  switch (o.type) {
    case msgpack::type::FLOAT32:
    case msgpack::type::FLOAT64: return o.via.f64;
    case msgpack::type::POSITIVE_INTEGER: return static_cast<double>(o.via.u64);
    case msgpack::type::NEGATIVE_INTEGER: return static_cast<double>(o.via.i64);
    default: throw WireDecodeError("expected float-like msgpack value");
  }
}

std::uint64_t as_uint(const msgpack::object & o)
{
  if (o.type == msgpack::type::POSITIVE_INTEGER) {return o.via.u64;}
  if (o.type == msgpack::type::NEGATIVE_INTEGER && o.via.i64 >= 0) {
    return static_cast<std::uint64_t>(o.via.i64);
  }
  throw WireDecodeError("expected non-negative integer msgpack value");
}

const msgpack::object * map_get(
  const msgpack::object & obj, std::string_view key)
{
  if (obj.type != msgpack::type::MAP) {return nullptr;}
  for (std::uint32_t i = 0; i < obj.via.map.size; ++i) {
    const auto & k = obj.via.map.ptr[i].key;
    if (k.type == msgpack::type::STR &&
      std::string_view(k.via.str.ptr, k.via.str.size) == key)
    {
      return &obj.via.map.ptr[i].val;
    }
  }
  return nullptr;
}

const msgpack::object & map_require(
  const msgpack::object & obj, std::string_view key, std::string_view ctx)
{
  const auto * v = map_get(obj, key);
  if (!v) {
    throw WireDecodeError(
            std::string(ctx) + ": missing key '" + std::string(key) + "'");
  }
  return *v;
}

template<std::size_t N>
void read_doubles(const msgpack::object & o, std::string_view ctx, double (&out)[N])
{
  if (o.type != msgpack::type::ARRAY || o.via.array.size != N) {
    throw WireDecodeError(
            std::string(ctx) + ": expected length-" + std::to_string(N) + " array");
  }
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = as_double(o.via.array.ptr[i]);
  }
}

// --------------------------------------------------------------------------
// OOB helpers — wrappers around cortex_wire::OobBuffer<T> that add the bridge's
// validation conventions (dtype, element count, frame-bounds).
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
// owns a shared_ptr to the frame so the bytes outlive any caller that holds
// the view — no raw-pointer lifetime games.
template<typename T>
OobBuffer<T> oob_view(
  const OobDescriptor & desc,
  const std::vector<ZmqFramePtr> & frames,
  std::string_view ctx,
  std::string_view expected_dtype = {})
{
  if (!expected_dtype.empty() && desc.dtype != expected_dtype) {
    throw WireDecodeError(
            std::string(ctx) + ": dtype mismatch (got '" + desc.dtype +
            "', expected '" + std::string(expected_dtype) + "')");
  }
  if (desc.buffer_index >= frames.size()) {
    throw WireDecodeError(
            std::string(ctx) + ": OOB buffer index " +
            std::to_string(desc.buffer_index) + " out of range");
  }
  const auto & frame = frames[desc.buffer_index];
  const std::size_t n = shape_elements(desc.shape);
  if (frame->size() < n * sizeof(T)) {
    throw WireDecodeError(
            std::string(ctx) + ": OOB frame too small (" +
            std::to_string(frame->size()) + " < " + std::to_string(n * sizeof(T)) + ")");
  }
  return OobBuffer<T>(frame, n);
}

// Look up an OOB descriptor under a map key (raises if absent or non-OOB),
// then materialise a typed view of it. Single call covers the events / IMU
// {key -> descriptor -> buffer} traversal.
template<typename T>
OobBuffer<T> map_oob_view(
  const msgpack::object & obj, std::string_view key,
  const std::vector<ZmqFramePtr> & frames,
  std::string_view ctx,
  std::string_view expected_dtype = {})
{
  const auto & v = map_require(obj, key, ctx);
  auto desc = DecodedMetadata::as_oob(v);
  if (!desc) {
    throw WireDecodeError(
            std::string(ctx) + ": key '" + std::string(key) +
            "' is not an OOB descriptor");
  }
  return oob_view<T>(*desc, frames, ctx, expected_dtype);
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
  const Inbound & in, std::string_view ctx, std::string_view expected_dtype)
{
  if (in.metadata.field_count() != 3) {
    throw WireDecodeError(std::string(ctx) + ": expected 3 metadata fields");
  }
  auto desc = DecodedMetadata::as_oob(in.metadata.field(0));
  if (!desc) {
    throw WireDecodeError(std::string(ctx) + ": field 0 is not an OOB descriptor");
  }
  auto shape = desc->shape;
  return ArrayView<T>{oob_view<T>(*desc, in.oob_frames, ctx, expected_dtype), std::move(shape)};
}

void stamp_header(
  std_msgs::msg::Header & h, const cortex_wire::MessageHeader & wire,
  const std::string & frame_id)
{
  h.stamp.sec = static_cast<std::int32_t>(wire.timestamp_ns / 1'000'000'000ULL);
  h.stamp.nanosec = static_cast<std::uint32_t>(wire.timestamp_ns % 1'000'000'000ULL);
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
  std::string_view ctx, geometry_msgs::msg::Vector3 & out)
{
  if (desc.dtype == "<f8") {
    auto v = oob_view<double>(desc, frames, ctx, "<f8");
    if (v.size() < 3) {
      throw WireDecodeError(std::string(ctx) + ": expected at least 3 elements");
    }
    out.x = v[0]; out.y = v[1]; out.z = v[2];
  } else if (desc.dtype == "<f4") {
    auto v = oob_view<float>(desc, frames, ctx, "<f4");
    if (v.size() < 3) {
      throw WireDecodeError(std::string(ctx) + ": expected at least 3 elements");
    }
    out.x = v[0]; out.y = v[1]; out.z = v[2];
  } else {
    throw WireDecodeError(
            std::string(ctx) + ": unsupported dtype '" + desc.dtype + "'");
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
  read_doubles(map_require(data, "x", "state"), "state.x", x);
  read_doubles(map_require(data, "q", "state"), "state.q", q);
  read_doubles(map_require(data, "v", "state"), "state.v", v);
  read_doubles(map_require(data, "w", "state"), "state.w", w);

  auto out = std::make_unique<msg::State>();
  stamp_header(out->header, in.header, in.frame_id);
  if (auto * t = map_get(data, "timestamp")) {out->timestamp = as_double(*t);}
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

  // We resolve the descriptors here, then let unpack_vec3_oob pick float vs
  // double based on the wire dtype.
  const auto & accel_field = map_require(data, "accel", "imu");
  const auto & gyro_field = map_require(data, "gyro", "imu");
  auto accel_desc = DecodedMetadata::as_oob(accel_field);
  auto gyro_desc = DecodedMetadata::as_oob(gyro_field);
  if (!accel_desc || !gyro_desc) {
    throw WireDecodeError("imu: accel/gyro fields are not OOB descriptors");
  }

  auto out = std::make_unique<msg::Imu>();
  stamp_header(out->header, in.header, in.frame_id);
  if (auto * t = map_get(data, "timestamp")) {out->timestamp = as_double(*t);}
  if (auto * s = map_get(data, "simsteps")) {out->simsteps = as_uint(*s);}
  if (auto * u = map_get(data, "uuid"); u && u->type == msgpack::type::STR) {
    out->uuid.assign(u->via.str.ptr, u->via.str.size);
  }
  unpack_vec3_oob(*accel_desc, in.oob_frames, "imu.accel", out->accel);
  unpack_vec3_oob(*gyro_desc, in.oob_frames, "imu.gyro", out->gyro);
  return out;
}

// ---- Events ---------------------------------------------------------------

std::unique_ptr<msg::Events> decode_events(const Inbound & in)
{
  if (in.metadata.field_count() != 2) {
    throw WireDecodeError("events: expected 2 metadata fields");
  }
  const auto & arrays = in.metadata.field(0);
  if (arrays.type != msgpack::type::MAP) {
    throw WireDecodeError("events: arrays field is not a map");
  }

  // dtype contract is fixed by the simulator's EventBuffer; we enforce it
  // here so a mismatch is loud instead of silently misinterpreted bytes.
  auto xv = map_oob_view<std::uint16_t>(arrays, "x", in.oob_frames, "events.x", "<u2");
  auto yv = map_oob_view<std::uint16_t>(arrays, "y", in.oob_frames, "events.y", "<u2");
  auto tv = map_oob_view<std::uint64_t>(arrays, "t", in.oob_frames, "events.t", "<u8");
  auto pv = map_oob_view<std::uint8_t>(arrays, "p", in.oob_frames, "events.p", "|u1");

  const std::size_t n = xv.size();
  if (yv.size() != n || tv.size() != n || pv.size() != n) {
    throw WireDecodeError("events: array lengths mismatch");
  }

  auto out = std::make_unique<msg::Events>();
  stamp_header(out->header, in.header, in.frame_id);
  copy_into_vector(xv, out->x);
  copy_into_vector(yv, out->y);
  copy_into_vector(tv, out->t);
  copy_into_vector(pv, out->p);
  return out;
}

// ---- Color / Depth Image --------------------------------------------------

std::unique_ptr<sensor_msgs::msg::Image> decode_color_image(const Inbound & in)
{
  auto av = array_message_view<std::uint8_t>(in, "color", "|u1");
  if (av.shape.size() != 3 || av.shape[2] != 3) {
    throw WireDecodeError("color: expected HxWx3 uint8 array");
  }
  const auto height = static_cast<std::uint32_t>(av.shape[0]);
  const auto width = static_cast<std::uint32_t>(av.shape[1]);

  auto out = std::make_unique<sensor_msgs::msg::Image>();
  stamp_header(out->header, in.header, in.frame_id);
  out->height = height;
  out->width = width;
  out->encoding = "rgb8";
  out->is_bigendian = 0;
  out->step = width * 3;
  copy_into_vector(av.data, out->data);  // single memcpy, size_bytes = H*W*3
  return out;
}

std::unique_ptr<sensor_msgs::msg::Image> decode_depth_image(const Inbound & in)
{
  auto av = array_message_view<float>(in, "depth", "<f4");
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
  stamp_header(out->header, in.header, in.frame_id);
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

// ---- Control (ROS 2 -> Cortex) --------------------------------------------

OutboundFrames encode_control(const std_msgs::msg::Float64MultiArray & msg)
{
  // Cortex DictMessage: 1 field, a msgpack MAP. The simulator's
  // receive_control expects {"cmd_motor_speeds": [...], ...}. We pack the
  // motor speeds as an inline msgpack array (they're tiny — 4 floats), and a
  // monotonic timestamp so the simulator's logging stays sensible.
  cortex_wire::MetadataBuilder b(1);

  auto & p = b.packer();
  const std::size_t map_size = 2;  // cmd_motor_speeds + timestamp
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

  auto frames = std::move(b).finish();
  return OutboundFrames{std::move(frames.metadata), std::move(frames.oob_buffers)};
}

}  // namespace neurosim_ros2_bridge::decoders
