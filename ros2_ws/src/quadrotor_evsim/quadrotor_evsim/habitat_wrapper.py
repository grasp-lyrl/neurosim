import yaml
import torch
import random
import magnum as mn
from pathlib import Path
from typing import Any, Dict, Tuple

import habitat_sim as hsim

from .utils import color2intensity, EventSimulatorTorch

class HabitatWrapper:
    def __init__(self, settings: Path):
        with open(settings, "r") as file:
            self.settings = yaml.safe_load(file)

        self._cfg, self._event_simulator = self.make_cfg(self.settings)
        self._sim = hsim.Simulator(self._cfg)

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
    def make_cfg(settings: Dict[str, Any]) -> Tuple[hsim.Configuration, EventSimulatorTorch]:
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
            _event_simulator = EventSimulatorTorch(settings["width"], settings["height"], 0)

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

    def render_observations(self, agent_id: int) -> Dict[str, Any]:
        """Render observations from the agent."""
        observations = self._sim.get_sensor_observations(agent_id)
        return observations

    def render_events(self, agent_id: int, time: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render events from the agent."""
        # observations = self._sim.get_sensor_observations(agent_id)
        # intensity_image = color2intensity(observations["color_sensor"][..., :3] / 255.0)
        sensor = self._sim._sensors["color_sensor"]
        sensor.draw_observation()
        intensity_image = color2intensity(sensor.get_observation()[..., :3] / 255.0)
        events = self._event_simulator.image_callback(intensity_image, time)  # in us
        return intensity_image, events

    def close(self) -> None:
        """Close the simulator."""
        self._sim.close()

    def reset(self) -> None:
        """Reset the simulator."""
        self._sim.reset()
