cmake_minimum_required(VERSION 3.8)
project(zmq_ros2_bridge)

# Suppress runtime path warnings for ROS2 message generation
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(cppzmq REQUIRED)
find_package(nlohmann_json REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Event.msg"
  "msg/Imu.msg"
  "msg/State.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Include directories
include_directories(
  include
)

# Add executable
add_executable(zmq_ros2_bridge_node
  src/zmq_ros2_bridge.cpp
)

# Link dependencies
ament_target_dependencies(zmq_ros2_bridge_node
  rclcpp
  sensor_msgs
  std_msgs
)

target_link_libraries(zmq_ros2_bridge_node
  cppzmq
  nlohmann_json::nlohmann_json
)

# Link generated messages to the executable
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

target_link_libraries(zmq_ros2_bridge_node
  "${cpp_typesupport_target}"
)

# Install targets
install(TARGETS
  zmq_ros2_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Export dependencies for downstream packages
ament_export_dependencies(rosidl_default_runtime)

ament_package()
