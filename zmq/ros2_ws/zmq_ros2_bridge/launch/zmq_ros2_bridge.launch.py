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
                "enable_event",
                default_value="true",
                description="Enable Event data subscriber",
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
            DeclareLaunchArgument(
                "event_zmq_address",
                default_value="ipc:///tmp/0",
                description="ZMQ address for Event data",
            ),
            DeclareLaunchArgument(
                "event_zmq_topic",
                default_value="events",
                description="ZMQ topic for Event data",
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
                        "enable_event": LaunchConfiguration("enable_event"),
                        "imu_zmq_address": LaunchConfiguration("imu_zmq_address"),
                        "imu_zmq_topic": LaunchConfiguration("imu_zmq_topic"),
                        "imu_ros2_topic": LaunchConfiguration("imu_ros2_topic"),
                        "color_zmq_address": LaunchConfiguration("color_zmq_address"),
                        "color_zmq_topic": LaunchConfiguration("color_zmq_topic"),
                        "color_ros2_topic": LaunchConfiguration("color_ros2_topic"),
                        "event_zmq_address": LaunchConfiguration("event_zmq_address"),
                        "event_zmq_topic": LaunchConfiguration("event_zmq_topic"),
                    }
                ],
            ),
        ]
    )
