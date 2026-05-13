cmake_minimum_required(VERSION 3.16)
project(neurosim_ros2_bridge VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(cortex_wire_cpp REQUIRED)

# ---- ROS 2 message generation ---------------------------------------------

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/State.msg
  msg/Imu.msg
  msg/Events.msg
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

# ---- Composable bridge component ------------------------------------------

add_library(neurosim_ros2_bridge_component SHARED
  src/config.cpp
  src/decoders.cpp
  src/bridge_node.cpp
)
target_include_directories(neurosim_ros2_bridge_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(neurosim_ros2_bridge_component PUBLIC
  yaml-cpp
  cortex_wire::cortex_wire
)
ament_target_dependencies(neurosim_ros2_bridge_component PUBLIC
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  geometry_msgs
  builtin_interfaces
)

# Hook the generated rosidl typesupport target into our component so it can
# include msg/state.hpp etc.
rosidl_get_typesupport_target(_neurosim_msgs_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(neurosim_ros2_bridge_component PUBLIC ${_neurosim_msgs_target})

rclcpp_components_register_node(neurosim_ros2_bridge_component
  PLUGIN "neurosim_ros2_bridge::NeurosimRos2Bridge"
  EXECUTABLE neurosim_bridge
)

# ---- install --------------------------------------------------------------

install(DIRECTORY include/ DESTINATION include)

install(TARGETS neurosim_ros2_bridge_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_export_dependencies(
  rclcpp rclcpp_components std_msgs sensor_msgs geometry_msgs builtin_interfaces
  rosidl_default_runtime)

ament_package()
