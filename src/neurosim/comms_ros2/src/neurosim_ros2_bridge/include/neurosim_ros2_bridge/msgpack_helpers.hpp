// Small free-function helpers for inspecting raw msgpack values pulled out of
// a cortex_wire::DecodedMetadata. These wrap the tagged-union dance into
// Python-ish accessors (`as_double`, `map_get`, `map_require`, `read_doubles`)
// so decoder code reads cleanly. Header-only; no class.
#ifndef NEUROSIM_ROS2_BRIDGE__MSGPACK_HELPERS_HPP_
#define NEUROSIM_ROS2_BRIDGE__MSGPACK_HELPERS_HPP_

#include <cortex_wire/header.hpp>       // WireDecodeError
#include <msgpack.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace neurosim_ros2_bridge
{

// FLOAT32/64 or any integer -> double.
inline double as_double(const msgpack::object & o)
{
  switch (o.type) {
    case msgpack::type::FLOAT32:
    case msgpack::type::FLOAT64: return o.via.f64;
    case msgpack::type::POSITIVE_INTEGER: return static_cast<double>(o.via.u64);
    case msgpack::type::NEGATIVE_INTEGER: return static_cast<double>(o.via.i64);
    default: throw cortex_wire::WireDecodeError("expected float-like msgpack value");
  }
}

// POS_INT (or non-negative NEG_INT) -> uint64.
inline std::uint64_t as_uint(const msgpack::object & o)
{
  if (o.type == msgpack::type::POSITIVE_INTEGER) {return o.via.u64;}
  if (o.type == msgpack::type::NEGATIVE_INTEGER && o.via.i64 >= 0) {
    return static_cast<std::uint64_t>(o.via.i64);
  }
  throw cortex_wire::WireDecodeError("expected non-negative integer");
}

// STR -> string_view. The bytes belong to the underlying DecodedMetadata.
inline std::string_view as_str(const msgpack::object & o)
{
  if (o.type != msgpack::type::STR) {
    throw cortex_wire::WireDecodeError("expected msgpack string");
  }
  return std::string_view(o.via.str.ptr, o.via.str.size);
}

// MAP[key] -> pointer, or nullptr if absent / not a map.
inline const msgpack::object * map_get(
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

// MAP[key] -> reference, throws if absent.
inline const msgpack::object & map_require(
  const msgpack::object & obj, std::string_view key)
{
  const auto * v = map_get(obj, key);
  if (!v) {
    throw cortex_wire::WireDecodeError("missing key '" + std::string(key) + "'");
  }
  return *v;
}

// ARRAY of exactly N FLOAT/int values into a fixed C array of doubles.
template<std::size_t N>
void read_doubles(const msgpack::object & o, double (&out)[N])
{
  if (o.type != msgpack::type::ARRAY || o.via.array.size != N) {
    throw cortex_wire::WireDecodeError(
            "expected length-" + std::to_string(N) + " array");
  }
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = as_double(o.via.array.ptr[i]);
  }
}

}  // namespace neurosim_ros2_bridge

#endif  // NEUROSIM_ROS2_BRIDGE__MSGPACK_HELPERS_HPP_
