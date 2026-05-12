// Copyright (c) 2026, Neurosim contributors. Apache-2.0.
#include "neurosim_ros2_bridge/decoders.hpp"

#include <cortex_wire/metadata.hpp>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace neurosim_ros2_bridge::decoders
{

namespace
{

using cortex_wire::DecodedMetadata;
using cortex_wire::OobDescriptor;
using cortex_wire::WireDecodeError;
using cortex_wire::ZmqFramePtr;

// ---- msgpack helpers ------------------------------------------------------
//
// These are intentionally narrow: only the shapes the neurosim simulator ever
// emits are accepted. Anything else throws and the bridge logs+drops the
// message.

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

// Look up `key` in a msgpack MAP `obj`. Returns nullptr if absent.
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

// Read an inline list-of-doubles (length-checked) into a fixed-size array.
template<std::size_t N>
void read_doubles(const msgpack::object & o, std::string_view ctx, double (&out)[N])
{
  if (o.type != msgpack::type::ARRAY || o.via.array.size != N) {
    throw WireDecodeError(
            std::string(ctx) + ": expected length-" + std::to_string(N) +
            " array");
  }
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = as_double(o.via.array.ptr[i]);
  }
}

// Look up an OOB descriptor under a map key and return it. Throws if absent
// or not actually an OOB descriptor.
OobDescriptor map_oob(
  const msgpack::object & obj, std::string_view key, std::string_view ctx)
{
  const auto & v = map_require(obj, key, ctx);
  auto desc = DecodedMetadata::as_oob(v);
  if (!desc) {
    throw WireDecodeError(
            std::string(ctx) + ": key '" + std::string(key) +
            "' is not an OOB descriptor");
  }
  return *desc;
}

// Resolve an OOB descriptor's frame and check sizes. Returns (ptr, size_bytes).
struct OobView
{
  const std::uint8_t * data = nullptr;
  std::size_t size = 0;
  OobDescriptor descriptor;
};

OobView oob_resolve(
  const OobDescriptor & desc,
  const std::vector<ZmqFramePtr> & frames,
  std::string_view ctx)
{
  if (desc.buffer_index >= frames.size()) {
    throw WireDecodeError(
            std::string(ctx) + ": OOB buffer index " +
            std::to_string(desc.buffer_index) + " out of range");
  }
  const auto & frame = frames[desc.buffer_index];
  OobView v;
  v.descriptor = desc;
  v.data = static_cast<const std::uint8_t *>(frame->data());
  v.size = frame->size();
  return v;
}

// Stamp a std_msgs/Header from the cortex wire header. The wire timestamp
// is nanoseconds since epoch.
void stamp_header(
  std_msgs::msg::Header & h, const cortex_wire::MessageHeader & wire,
  const std::string & frame_id)
{
  h.stamp.sec = static_cast<std::int32_t>(wire.timestamp_ns / 1'000'000'000ULL);
  h.stamp.nanosec = static_cast<std::uint32_t>(wire.timestamp_ns % 1'000'000'000ULL);
  h.frame_id = frame_id;
}

// Set a geometry_msgs/Vector3 from a length-3 double array.
void set_vec3(geometry_msgs::msg::Vector3 & out, const double (&v)[3])
{
  out.x = v[0];
  out.y = v[1];
  out.z = v[2];
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

  const auto accel_desc = map_oob(data, "accel", "imu");
  const auto gyro_desc = map_oob(data, "gyro", "imu");
  const auto accel = oob_resolve(accel_desc, in.oob_frames, "imu.accel");
  const auto gyro = oob_resolve(gyro_desc, in.oob_frames, "imu.gyro");

  // Both are length-3 float64 (the simulator does .cpu().numpy() on torch
  // accel/gyro tensors which are float32; the python ArrayMessage will keep
  // whatever dtype is given. Allow either f4/f8 and convert.)
  auto unpack_vec3 = [](const OobView & v, std::string_view ctx,
      geometry_msgs::msg::Vector3 & out) {
      if (v.descriptor.dtype == "<f8") {
        if (v.size < 3 * sizeof(double)) {
          throw WireDecodeError(std::string(ctx) + ": frame too small for f64[3]");
        }
        const auto * p = reinterpret_cast<const double *>(v.data);
        out.x = p[0]; out.y = p[1]; out.z = p[2];
      } else if (v.descriptor.dtype == "<f4") {
        if (v.size < 3 * sizeof(float)) {
          throw WireDecodeError(std::string(ctx) + ": frame too small for f32[3]");
        }
        const auto * p = reinterpret_cast<const float *>(v.data);
        out.x = p[0]; out.y = p[1]; out.z = p[2];
      } else {
        throw WireDecodeError(
                std::string(ctx) + ": unsupported dtype '" + v.descriptor.dtype + "'");
      }
    };

  auto out = std::make_unique<msg::Imu>();
  stamp_header(out->header, in.header, in.frame_id);
  if (auto * t = map_get(data, "timestamp")) {out->timestamp = as_double(*t);}
  if (auto * s = map_get(data, "simsteps")) {out->simsteps = as_uint(*s);}
  if (auto * u = map_get(data, "uuid"); u && u->type == msgpack::type::STR) {
    out->uuid.assign(u->via.str.ptr, u->via.str.size);
  }
  unpack_vec3(accel, "imu.accel", out->accel);
  unpack_vec3(gyro, "imu.gyro", out->gyro);
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

  const auto x_desc = map_oob(arrays, "x", "events");
  const auto y_desc = map_oob(arrays, "y", "events");
  const auto t_desc = map_oob(arrays, "t", "events");
  const auto p_desc = map_oob(arrays, "p", "events");

  if (x_desc.dtype != "<u2" || y_desc.dtype != "<u2" ||
    t_desc.dtype != "<u8" || p_desc.dtype != "|u1")
  {
    throw WireDecodeError(
            "events: expected dtypes x=<u2, y=<u2, t=<u8, p=|u1 (got " +
            x_desc.dtype + "/" + y_desc.dtype + "/" + t_desc.dtype + "/" +
            p_desc.dtype + ")");
  }

  const auto xv = oob_resolve(x_desc, in.oob_frames, "events.x");
  const auto yv = oob_resolve(y_desc, in.oob_frames, "events.y");
  const auto tv = oob_resolve(t_desc, in.oob_frames, "events.t");
  const auto pv = oob_resolve(p_desc, in.oob_frames, "events.p");

  const std::size_t n_events = xv.size / sizeof(std::uint16_t);
  if (yv.size / sizeof(std::uint16_t) != n_events ||
    tv.size / sizeof(std::uint64_t) != n_events ||
    pv.size / sizeof(std::uint8_t) != n_events)
  {
    throw WireDecodeError("events: array lengths mismatch");
  }

  auto out = std::make_unique<msg::Events>();
  stamp_header(out->header, in.header, in.frame_id);
  out->x.resize(n_events);
  out->y.resize(n_events);
  out->t.resize(n_events);
  out->p.resize(n_events);
  std::memcpy(out->x.data(), xv.data, n_events * sizeof(std::uint16_t));
  std::memcpy(out->y.data(), yv.data, n_events * sizeof(std::uint16_t));
  std::memcpy(out->t.data(), tv.data, n_events * sizeof(std::uint64_t));
  std::memcpy(out->p.data(), pv.data, n_events * sizeof(std::uint8_t));
  return out;
}

// ---- Color / Depth Image --------------------------------------------------

namespace
{

OobView decode_array_message(
  const Inbound & in, std::string_view ctx, std::string_view expected_dtype)
{
  if (in.metadata.field_count() != 3) {
    throw WireDecodeError(std::string(ctx) + ": expected 3 metadata fields");
  }
  const auto desc = DecodedMetadata::as_oob(in.metadata.field(0));
  if (!desc) {
    throw WireDecodeError(std::string(ctx) + ": field 0 is not an OOB descriptor");
  }
  if (desc->dtype != expected_dtype) {
    throw WireDecodeError(
            std::string(ctx) + ": dtype mismatch (got '" + desc->dtype +
            "', expected '" + std::string(expected_dtype) + "')");
  }
  return oob_resolve(*desc, in.oob_frames, ctx);
}

}  // namespace

std::unique_ptr<sensor_msgs::msg::Image> decode_color_image(const Inbound & in)
{
  const auto view = decode_array_message(in, "color", "|u1");
  const auto & shape = view.descriptor.shape;
  if (shape.size() != 3 || shape[2] != 3) {
    throw WireDecodeError("color: expected HxWx3 uint8 array");
  }
  const auto height = static_cast<std::uint32_t>(shape[0]);
  const auto width = static_cast<std::uint32_t>(shape[1]);
  const std::size_t nbytes = height * width * 3;
  if (view.size < nbytes) {
    throw WireDecodeError("color: OOB frame too small");
  }

  auto out = std::make_unique<sensor_msgs::msg::Image>();
  stamp_header(out->header, in.header, in.frame_id);
  out->height = height;
  out->width = width;
  out->encoding = "rgb8";
  out->is_bigendian = 0;
  out->step = width * 3;
  out->data.resize(nbytes);
  std::memcpy(out->data.data(), view.data, nbytes);
  return out;
}

std::unique_ptr<sensor_msgs::msg::Image> decode_depth_image(const Inbound & in)
{
  const auto view = decode_array_message(in, "depth", "<f4");
  const auto & shape = view.descriptor.shape;
  std::uint32_t height = 0, width = 0;
  if (shape.size() == 2) {
    height = static_cast<std::uint32_t>(shape[0]);
    width = static_cast<std::uint32_t>(shape[1]);
  } else if (shape.size() == 3 && shape[2] == 1) {
    height = static_cast<std::uint32_t>(shape[0]);
    width = static_cast<std::uint32_t>(shape[1]);
  } else {
    throw WireDecodeError("depth: expected HxW float32 array");
  }
  const std::size_t nbytes = static_cast<std::size_t>(height) * width * sizeof(float);
  if (view.size < nbytes) {
    throw WireDecodeError("depth: OOB frame too small");
  }

  auto out = std::make_unique<sensor_msgs::msg::Image>();
  stamp_header(out->header, in.header, in.frame_id);
  out->height = height;
  out->width = width;
  out->encoding = "32FC1";
  out->is_bigendian = 0;
  out->step = width * sizeof(float);
  out->data.resize(nbytes);
  std::memcpy(out->data.data(), view.data, nbytes);
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
