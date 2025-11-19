from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_imu",
                default_value="true",
                description="Enable IMU subscriber",
            ),
            DeclareLaunchArgument(
                "enable_color",
                default_value="true",
                description="Enable Color image subscriber",
            ),
            DeclareLaunchArgument(
                "imu_zmq_address",
                default_value="ipc:///tmp/0",
                description="ZMQ address for IMU data",
            ),
            DeclareLaunchArgument(
                "imu_zmq_topic",
                default_value="imu",
                description="ZMQ topic for IMU data",
            ),
            DeclareLaunchArgument(
                "imu_ros2_topic",
                default_value="imu",
                description="ROS2 topic to publish IMU data",
            ),
            DeclareLaunchArgument(
                "color_zmq_address",
                default_value="ipc:///tmp/0",
                description="ZMQ address for Color image data",
            ),
            DeclareLaunchArgument(
                "color_zmq_topic",
                default_value="color",
                description="ZMQ topic for Color image data",
            ),
            DeclareLaunchArgument(
                "color_ros2_topic",
                default_value="color",
                description="ROS2 topic to publish Color image data",
            ),
            Node(
                package="zmq_ros2_bridge",
                executable="zmq_ros2_bridge_node",
                name="zmq_ros2_bridge",
                output="screen",
                parameters=[
                    {
                        "enable_imu": LaunchConfiguration("enable_imu"),
                        "enable_color": LaunchConfiguration("enable_color"),
                        "imu_zmq_address": LaunchConfiguration("imu_zmq_address"),
                        "imu_zmq_topic": LaunchConfiguration("imu_zmq_topic"),
                        "imu_ros2_topic": LaunchConfiguration("imu_ros2_topic"),
                        "color_zmq_address": LaunchConfiguration("color_zmq_address"),
                        "color_zmq_topic": LaunchConfiguration("color_zmq_topic"),
                        "color_ros2_topic": LaunchConfiguration("color_ros2_topic"),
                    }
                ],
            ),
        ]
    )
