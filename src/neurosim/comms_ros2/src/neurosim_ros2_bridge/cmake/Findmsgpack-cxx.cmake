# Findmsgpack-cxx.cmake
#
# Shim for distros where msgpack-cxx ships header-only with no CMake config
# (Ubuntu 22.04's `libmsgpack-dev`). cortex_wire_cpp's installed Config does
# `find_dependency(msgpack-cxx)`, which propagates REQUIRED from the consumer
# and would otherwise fail the build.
#
# This module:
#   - locates msgpack.hpp via the standard include paths,
#   - publishes a `msgpack-cxx` INTERFACE IMPORTED target that exports those
#     headers, which matches the target name the CMake config would create on
#     newer distros (Ubuntu 23.04+ / fedora / vcpkg).
#
# The bridge CMakeLists prepends the directory holding this file to
# CMAKE_MODULE_PATH before find_package(cortex_wire_cpp REQUIRED) so this
# module is reachable.

find_path(MSGPACK_INCLUDE_DIR
  NAMES msgpack.hpp
  HINTS /usr/include /usr/local/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(msgpack-cxx
  REQUIRED_VARS MSGPACK_INCLUDE_DIR
)

if(msgpack-cxx_FOUND AND NOT TARGET msgpack-cxx)
  add_library(msgpack-cxx INTERFACE IMPORTED)
  target_include_directories(msgpack-cxx INTERFACE "${MSGPACK_INCLUDE_DIR}")
endif()
