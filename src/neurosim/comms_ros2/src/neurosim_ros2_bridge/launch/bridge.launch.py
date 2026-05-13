"""Launch the neurosim ROS 2 bridge as a composable node.

Downstream consumers that want intra-process zero-copy delivery from the
bridge should compose into the same container — either extend this file or
launch their own ComposableNodeContainer with the same `container_name`.

Example:

    ros2 launch neurosim_ros2_bridge bridge.launch.py \
        config:=$(ros2 pkg prefix neurosim_ros2_bridge)/share/neurosim_ros2_bridge/config/apartment_1.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    config = LaunchConfiguration("config")
    container_name = LaunchConfiguration("container_name")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                description="Path to the neurosim_ros2_bridge YAML config",
            ),
            DeclareLaunchArgument(
                "container_name",
                default_value="neurosim_bridge_container",
            ),
            ComposableNodeContainer(
                name=container_name,
                namespace="",
                package="rclcpp_components",
                executable="component_container_mt",
                composable_node_descriptions=[
                    ComposableNode(
                        package="neurosim_ros2_bridge",
                        plugin="neurosim_ros2_bridge::NeurosimRos2Bridge",
                        name="neurosim_ros2_bridge",
                        parameters=[{"config_path": config}],
                        extra_arguments=[{"use_intra_process_comms": True}],
                    ),
                ],
                output="screen",
            ),
        ]
    )
