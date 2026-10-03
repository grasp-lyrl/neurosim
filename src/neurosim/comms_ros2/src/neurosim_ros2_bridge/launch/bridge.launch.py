"""Launch the neurosim ROS 2 bridge with rviz2, and optionally record a flight beside it.

ros2 launch neurosim_ros2_bridge bridge.launch.py record:=true
"""

import os
import time

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def record_flight(context):
    """Load the recorder into the bridge's container and write the calibration into its bag."""
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    bag = os.path.abspath(LaunchConfiguration("bag_prefix").perform(context) + stamp)
    recorder = ComposableNode(
        package="rosbag2_composable_recorder",
        plugin="rosbag2_composable_recorder::ComposableRecorder",
        name="recorder",
        parameters=[LaunchConfiguration("record_config"), {"bag_name": bag}],
        extra_arguments=[{"use_intra_process_comms": True}],
    )
    calibration = [
        "python",
        "-m",
        "neurosim.core.coord_trans.calibration",
        "--bag",
        bag,
        "--settings",
        LaunchConfiguration("settings"),
        "--config",
        LaunchConfiguration("config"),
    ]
    return [
        LogInfo(msg=f"recording flight bag to {bag}/"),
        LoadComposableNodes(
            target_container=LaunchConfiguration("container_name"),
            composable_node_descriptions=[recorder],
        ),
        ExecuteProcess(cmd=calibration, output="screen"),
    ]


def generate_launch_description():
    share = FindPackageShare("neurosim_ros2_bridge")
    bag_dir = EnvironmentVariable("NEUROSIM_BAG_DIR", default_value="outputs/bags")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [share, "config", "apartment_1.yaml"]
                ),
                description="Path to the neurosim_ros2_bridge YAML config",
            ),
            DeclareLaunchArgument(
                "container_name",
                default_value="neurosim_bridge_container",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Also open rviz2 with rviz/neurosim.rviz",
            ),
            DeclareLaunchArgument(
                "record",
                default_value="false",
                description="Record a flight bag, with the camera calibration beside it",
            ),
            DeclareLaunchArgument(
                "record_config",
                default_value=PathJoinSubstitution([share, "config", "recorder.yaml"]),
                description="Recorder parameters: topics, storage",
            ),
            DeclareLaunchArgument(
                "bag_prefix",
                default_value=[bag_dir, "/flight_"],
                description="Bag directory prefix, a date-time is appended; relative to "
                "the launch directory",
            ),
            DeclareLaunchArgument(
                "settings",
                default_value="configs/apartment_1-settings.yaml",
                description="Simulator settings of the flight, for the camera calibration",
            ),
            ComposableNodeContainer(
                name=LaunchConfiguration("container_name"),
                namespace="",
                package="rclcpp_components",
                executable="component_container_mt",
                composable_node_descriptions=[
                    ComposableNode(
                        package="neurosim_ros2_bridge",
                        plugin="neurosim_ros2_bridge::NeurosimRos2Bridge",
                        name="neurosim_ros2_bridge",
                        parameters=[{"config_path": LaunchConfiguration("config")}],
                        extra_arguments=[{"use_intra_process_comms": True}],
                    ),
                ],
                output="screen",
            ),
            OpaqueFunction(
                function=record_flight,
                condition=IfCondition(LaunchConfiguration("record")),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=[
                    "-d",
                    PathJoinSubstitution([share, "rviz", "neurosim.rviz"]),
                ],
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
