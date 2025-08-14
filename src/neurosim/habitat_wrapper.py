import yaml
import torch
import random
import numpy as np
import magnum as mn
from pathlib import Path
from typing import Any, Dict, Tuple

import habitat_sim as hsim
from habitat_sim.bindings import built_with_bullet

from neurosim.utils import color2intensity

try:
    from neurosim.utils import EventSimulatorCUDA as EventSimulator
except ImportError:
    from neurosim.utils import EventSimulatorTorch as EventSimulator

    print(
        "Warning: CUDA event simulator not available, using Torch version. Might be slower. Install CUDA version for better performance."
    )


BLACK = mn.Color4.from_linear_rgb_int(0)

default_sim_settings: Dict[str, Any] = {
    # path to .scene_dataset.json file
    "scene_dataset_config_file": "default",
    # name of an existing scene in the dataset, a scene, stage, or asset filepath, or "NONE" for an empty scene
    "scene": "habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-castle-rotated.glb",
    # camera sensor parameters
    "width": 640,
    "height": 480,
    # horizontal field of view in degrees
    "hfov": 90,
    # far clipping plane
    "zfar": 1000.0,
    # optional background color override for rgb sensors
    "clear_color": BLACK,
    # vertical offset of the camera from the agent's root position (e.g. height of eyes)
    "sensor_height": 0.05,
    # defaul agent ix
    "default_agent": 0,
    # radius of the agent cylinder approximation for navmesh
    "agent_radius": 0.1,
    # pick sensors to use
    "color_sensor": True,
    "event_camera": True,
    "semantic_sensor": False,
    "depth_sensor": False,
    "ortho_rgba_sensor": False,
    "ortho_depth_sensor": False,
    "ortho_semantic_sensor": False,
    "fisheye_rgba_sensor": False,
    "fisheye_depth_sensor": False,
    "fisheye_semantic_sensor": False,
    "equirect_rgba_sensor": False,
    "equirect_depth_sensor": False,
    "equirect_semantic_sensor": False,
    # random seed
    "seed": 1,
    # path to .physics_config.json file
    "physics_config_file": "data/default.physics_config.json",
    # use bullet physics for dyanimcs or not - make default value whether or not
    # Simulator was built with bullet enabled
    "enable_physics": built_with_bullet,
    # ensure or create compatible navmesh for agent paramters
    "default_agent_navmesh": True,
    # if configuring a navmesh, should STATIC MotionType objects be included
    "navmesh_include_static_objects": False,
    # Enable horizon-based ambient occlusion, which provides soft shadows in corners and crevices.
    "enable_hbao": False,
    # Frustum culling is a performance optimization that skips rendering objects outside the camera's view.
    "frustum_culling": True,
    # Position and orientation of agents:
    # "start_position": [-1.7926959 ,  1.50, 14.255245],
    "start_position": [-3, -14, 2.0],
}


class HabitatWrapper:
    def __init__(self, settings: Path):
        if settings is None:
            self.settings = default_sim_settings
        else:
            with open(settings, "r") as file:
                self.settings = yaml.safe_load(file)

        self._cfg, self._event_simulator = self.make_cfg(self.settings)
        self._sim = hsim.Simulator(self._cfg)
        self._scene_aabb = self._sim.scene_aabb

        # set seed
        random.seed(self.settings["seed"])
        self._sim.seed(self.settings["seed"])

        self.init_agent_state(self.settings["default_agent"])
        self.agent = self._sim.get_agent(self.settings["default_agent"])

    def init_agent_state(self, agent_id: int) -> hsim.AgentState:
        """Initialize the agent state."""
        agent = self._sim.initialize_agent(agent_id)

        agent_state = hsim.AgentState()
        agent_state.position = mn.Vector3(self.settings["start_position"])
        agent.set_state(agent_state)

        agent_state = agent.get_state()
        print(
            f"Agent {agent_id} initialized at position: {agent_state.position}, rotation: {agent_state.rotation}"
        )

        return agent_state

    @staticmethod
    def make_cfg(
        settings: Dict[str, Any],
    ) -> Tuple[hsim.Configuration, EventSimulator]:
        r"""Isolates the boilerplate code to create a habitat_sim.Configuration from a settings dictionary.
        :param settings: A dict with pre-defined keys, each a basic simulator initialization parameter.
        Allows configuration of dataset and scene, visual sensor parameters, and basic agent parameters.
        Optionally creates up to one of each of a variety of aligned visual sensors under Agent 0.
        The output can be passed directly into habitat_sim.simulator.Simulator constructor or reconfigure to initialize a Simulator instance.
        """
        sim_cfg = hsim.SimulatorConfiguration()
        if "scene_dataset_config_file" in settings:
            sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
        sim_cfg.frustum_culling = settings.get("frustum_culling", False)
        if "enable_physics" in settings:
            sim_cfg.enable_physics = settings["enable_physics"]
        if "physics_config_file" in settings:
            sim_cfg.physics_config_file = settings["physics_config_file"]
        if "scene_light_setup" in settings:
            sim_cfg.scene_light_setup = settings["scene_light_setup"]
        sim_cfg.enable_hbao = settings.get("enable_hbao", False)
        sim_cfg.gpu_device_id = 0

        sim_cfg.scene_id = settings["scene"]

        # define default sensor parameters (see src/esp/Sensor/Sensor.h)
        sensor_specs = []

        def create_camera_spec(**kw_args) -> hsim.CameraSensorSpec:
            """Create a camera sensor specification."""
            camera_sensor_spec = hsim.CameraSensorSpec()
            camera_sensor_spec.sensor_type = hsim.SensorType.COLOR
            camera_sensor_spec.resolution = mn.Vector2i([settings["height"], settings["width"]])
            camera_sensor_spec.position = mn.Vector3(0, 0, settings["sensor_height"])
            camera_sensor_spec.gpu2gpu_transfer = True  # Keep renderings in GPU
            for k in kw_args:
                setattr(camera_sensor_spec, k, kw_args[k])
            return camera_sensor_spec

        if settings["color_sensor"]:
            color_sensor_spec = create_camera_spec(
                uuid="color_sensor",
                hfov=settings["hfov"],
                far=settings["zfar"],
                sensor_type=hsim.SensorType.COLOR,
                sensor_subtype=hsim.SensorSubType.PINHOLE,
                clear_color=settings["clear_color"],
            )
            sensor_specs.append(color_sensor_spec)

        if settings["depth_sensor"]:
            depth_sensor_spec = create_camera_spec(
                uuid="depth_sensor",
                hfov=settings["hfov"],
                far=settings["zfar"],
                sensor_type=hsim.SensorType.DEPTH,
                channels=1,
                sensor_subtype=hsim.SensorSubType.PINHOLE,
            )
            sensor_specs.append(depth_sensor_spec)

        if settings["semantic_sensor"]:
            semantic_sensor_spec = create_camera_spec(
                uuid="semantic_sensor",
                hfov=settings["hfov"],
                far=settings["zfar"],
                sensor_type=hsim.SensorType.SEMANTIC,
                channels=1,
                sensor_subtype=hsim.SensorSubType.PINHOLE,
            )
            sensor_specs.append(semantic_sensor_spec)

        _event_simulator = None
        if settings["event_camera"]:
            _event_simulator = EventSimulator(settings["width"], settings["height"], 0)

        # create agent specifications
        agent_cfg = hsim.agent.AgentConfiguration()
        agent_cfg.height = settings["sensor_height"]
        agent_cfg.radius = settings["agent_radius"]
        agent_cfg.sensor_specifications = sensor_specs

        return hsim.Configuration(sim_cfg, [agent_cfg]), _event_simulator

    def update_agent_pose(
        self, agent_id: int, position: mn.Vector3, rotation: mn.Quaternion
    ) -> None:
        """Update the agent's pose."""
        agent = self._sim.get_agent(agent_id)
        agent_state = hsim.AgentState()
        agent_state.position = position
        agent_state.rotation = rotation
        agent.set_state(agent_state)

    def get_agent_state(self, agent_id: int) -> hsim.AgentState:
        """Get the agent's state."""
        agent = self._sim.get_agent(agent_id)
        return agent.get_state()

    def render_events(self, time: int, to_numpy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render events from the agent."""
        sensor = self._sim._sensors["color_sensor"]
        sensor.draw_observation()
        intensity_image = color2intensity(sensor.get_observation()[..., :3] / 255.0)
        events = self._event_simulator.image_callback(intensity_image, time)  # in us
        if to_numpy and events is not None:
            events = [
                events[0].cpu().numpy().astype(np.uint16),
                events[1].cpu().numpy().astype(np.uint16),
                events[2].cpu().numpy().astype(np.uint64),
                events[3].cpu().numpy().astype(np.uint8),
            ]
        return events

    def render_color_sensor(self) -> torch.Tensor:
        """Render the color sensor."""
        sensor = self._sim._sensors["color_sensor"]  # RGB only
        sensor.draw_observation()
        return sensor.get_observation()[..., :3]

    def render_depth_sensor(self) -> torch.Tensor:
        """Render the depth sensor."""
        sensor = self._sim._sensors["depth_sensor"]
        sensor.draw_observation()
        return sensor.get_observation()

    def close(self) -> None:
        """Close the simulator."""
        self._sim.close()

    def reset(self) -> None:
        """Reset the simulator."""
        self._sim.reset()
