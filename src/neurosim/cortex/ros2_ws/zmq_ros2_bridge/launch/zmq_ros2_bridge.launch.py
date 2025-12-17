import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the path to the config file
    config_file = LaunchConfiguration("config_file")

    # Default config file path
    default_config = os.path.join(
        get_package_share_directory("zmq_ros2_bridge"), "config", "zmq_ros2_bridge.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to the configuration YAML file",
            ),
            Node(
                package="zmq_ros2_bridge",
                executable="zmq_ros2_bridge_node",
                name="zmq_ros2_bridge",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
