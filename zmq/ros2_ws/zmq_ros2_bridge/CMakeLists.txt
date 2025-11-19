cmake_minimum_required(VERSION 3.8)
project(zmq_ros2_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(cppzmq REQUIRED)
find_package(nlohmann_json REQUIRED)

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
)

target_link_libraries(zmq_ros2_bridge_node
  cppzmq
  nlohmann_json::nlohmann_json
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

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
