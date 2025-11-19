from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "zmq_address",
                default_value="tcp://localhost:5555",
                description="ZMQ publisher address",
            ),
            DeclareLaunchArgument(
                "zmq_topic", default_value="", description="ZMQ topic to subscribe to (empty subscribes to all)"
            ),
            Node(
                package="zmq_ros2_bridge",
                executable="zmq_ros2_bridge_node",
                name="zmq_ros2_bridge",
                output="screen",
                parameters=[
                    {
                        "zmq_address": LaunchConfiguration("zmq_address"),
                        "zmq_topic": LaunchConfiguration("zmq_topic"),
                    }
                ],
            ),
        ]
    )
