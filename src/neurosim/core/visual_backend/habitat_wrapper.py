"""
Habitat Wrapper for Neurosim.

This module provides a wrapper around the Habitat simulator with
event camera simulation support.
"""

import torch
import random
import logging
import numpy as np
import magnum as mn
from typing import Any, Optional

import habitat_sim as hsim

from neurosim.core.utils import color2intensity, RECOLOR_MAP, outline_border
from neurosim.core.event_sim import create_event_simulator, EventSimulatorProtocol

logger = logging.getLogger(__name__)


class HabitatWrapper:
    """Wrapper around Habitat simulator with event camera support.

    Args:
        settings: A dictionary containing simulator settings.
    """

    def __init__(self, settings: dict[str, Any]):
        self.settings = settings

        # Initialize event simulator container
        self._event_simulators: dict[str, EventSimulatorProtocol] = {}

        # Create Habitat configuration and fill in event simulators if any
        self._cfg = self._make_cfg()

        # Initialize simulator
        self._sim = hsim.Simulator(self._cfg)
        self._scene_bounds = self._sim.pathfinder.get_bounds()

        # Set seed
        self._set_seed(self.settings.get("seed", 324))

        # init the agent to the start position and orientation
        # self.agent = self._init_agent_state(self.settings["default_agent"])
        self.agent = self._sim.get_agent(self.settings["default_agent"])

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for the simulator and numpy.

        Args:
            seed: The seed value to set.
        """
        random.seed(seed)
        self._sim.seed(seed)
        np.random.seed(seed)

    def _create_camera_spec(
        self,
        uuid: str,
        sensor_type: hsim.SensorType | str,
        sensor_subtype: hsim.SensorSubType | str,
        resolution: mn.Vector2i | tuple[int, int] | list[int],
        hfov: float,
        far: float,
        position: mn.Vector3 | tuple[float, float, float] | list[float],
        orientation: mn.Vector3 | tuple[float, float, float] | list[float],
        clear_color: mn.Color4 | tuple[float, float, float, float] | list[float] = (
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    ) -> hsim.CameraSensorSpec:
        """Create a camera sensor specification."""
        camera_sensor_spec = hsim.CameraSensorSpec()
        camera_sensor_spec.uuid = uuid
        camera_sensor_spec.hfov = hfov
        camera_sensor_spec.far = far
        if not isinstance(sensor_type, hsim.SensorType):
            camera_sensor_spec.sensor_type = {
                "color": hsim.SensorType.COLOR,
                "depth": hsim.SensorType.DEPTH,
                "semantic": hsim.SensorType.SEMANTIC,
            }[sensor_type.lower()]
        else:
            camera_sensor_spec.sensor_type = sensor_type

        camera_sensor_spec.channels = {
            hsim.SensorType.COLOR: 3,
            hsim.SensorType.DEPTH: 1,
            hsim.SensorType.SEMANTIC: 1,
        }[camera_sensor_spec.sensor_type]

        if not isinstance(sensor_subtype, hsim.SensorSubType):
            camera_sensor_spec.sensor_subtype = {
                "pinhole": hsim.SensorSubType.PINHOLE,
                "fisheye": hsim.SensorSubType.FISHEYE,
            }[sensor_subtype.lower()]
        else:
            camera_sensor_spec.sensor_subtype = sensor_subtype

        if not isinstance(resolution, mn.Vector2i):
            camera_sensor_spec.resolution = mn.Vector2i(resolution)
        else:
            camera_sensor_spec.resolution = resolution

        if not isinstance(position, mn.Vector3):
            camera_sensor_spec.position = mn.Vector3(position)
        else:
            camera_sensor_spec.position = position

        if not isinstance(orientation, mn.Vector3):
            camera_sensor_spec.orientation = mn.Vector3(orientation)
        else:
            camera_sensor_spec.orientation = orientation

        if not isinstance(clear_color, mn.Color4):
            camera_sensor_spec.clear_color = mn.Color4(*clear_color)
        else:
            camera_sensor_spec.clear_color = clear_color

        camera_sensor_spec.gpu2gpu_transfer = True  # Keep renderings in GPU
        return camera_sensor_spec

    def _create_event_simulator(
        self, sensor_name: str, sensor_cfg: dict[str, Any]
    ) -> tuple[hsim.CameraSensorSpec, EventSimulatorProtocol]:
        """Create a color sensor to generate intensity images for the event camera.
        Adds the color sensor to the habitat sim agent and creates the event simulator
        in the current HabitatWrapper instance.

        Args:
            sensor_cfg: Configuration dictionary for the event sensor.

        Returns:
            An event simulator instance following EventSimulatorProtocol
        """
        # We need a color sensor to generate the intensity images for the event camera
        color_sensor_spec = self._create_camera_spec(
            uuid=sensor_name,
            sensor_type="color",
            sensor_subtype=sensor_cfg.get("subtype", "pinhole"),
            resolution=(sensor_cfg["height"], sensor_cfg["width"]),
            hfov=sensor_cfg["hfov"],
            far=sensor_cfg["zfar"],
            position=sensor_cfg["position"],
            orientation=sensor_cfg["orientation"],
        )

        backend = sensor_cfg.get("backend", "auto")

        # Get contrast thresholds from settings
        contrast_threshold_pos = sensor_cfg.get("contrast_threshold_pos", 0.35)
        contrast_threshold_neg = sensor_cfg.get("contrast_threshold_neg", 0.35)

        event_simulator = create_event_simulator(
            backend=backend,
            width=sensor_cfg["width"],
            height=sensor_cfg["height"],
            start_time=0,
            contrast_threshold_neg=contrast_threshold_neg,
            contrast_threshold_pos=contrast_threshold_pos,
        )

        return color_sensor_spec, event_simulator

    def _make_cfg(self) -> hsim.Configuration:
        """Create a Habitat configuration from a settings dictionary.

        Returns:
            A Habitat Configuration object.
        """
        sim_cfg = hsim.SimulatorConfiguration()

        sim_cfg.scene_dataset_config_file = self.settings.get(
            "scene_dataset_config_file", "default"
        )
        sim_cfg.enable_physics = self.settings.get("enable_physics", False)
        sim_cfg.physics_config_file = self.settings.get(
            "physics_config_file", "data/default.physics_config.json"
        )

        # TODO: Add scene_light_setup and other habitat settings

        sim_cfg.frustum_culling = self.settings.get("frustum_culling", False)
        sim_cfg.enable_hbao = self.settings.get("enable_hbao", False)
        sim_cfg.gpu_device_id = self.settings.get("gpu_id", 0)
        sim_cfg.scene_id = self.settings["scene"]

        # Create sensor specifications
        sensor_specifications = []
        for sensor_name, sensor_cfg in self.settings.get("sensors", {}).items():
            if sensor_cfg["type"] == "event":
                sensor_spec, self._event_simulators[sensor_name] = (
                    self._create_event_simulator(
                        sensor_name=sensor_name, sensor_cfg=sensor_cfg
                    )
                )
            else:
                sensor_spec = self._create_camera_spec(
                    uuid=sensor_name,
                    sensor_type=sensor_cfg["type"],
                    sensor_subtype=sensor_cfg.get("subtype", "pinhole"),
                    resolution=(sensor_cfg["height"], sensor_cfg["width"]),
                    hfov=sensor_cfg["hfov"],
                    far=sensor_cfg["zfar"],
                    position=sensor_cfg["position"],
                    orientation=sensor_cfg["orientation"],
                )
            sensor_specifications.append(sensor_spec)

        # Create agent specifications
        agent_cfg = hsim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specifications

        # TODO: Handle these two settings later
        # agent_cfg.height = settings.get("sensor_height", 0.05)
        # agent_cfg.radius = settings.get("agent_radius", 0.1)

        return hsim.Configuration(sim_cfg, [agent_cfg])

    def update_agent_state(self, position: np.ndarray, rotation: np.ndarray) -> None:
        """Update the agent's pose.

        Args:
            agent_id: The ID of the agent to update.
            position: The new position.
            rotation: The new rotation.
        """
        agent_state = hsim.AgentState(position=position, rotation=rotation)
        self.agent.set_state(agent_state, reset_sensors=False)

    def render_events(
        self, uuid: str, time: int, to_numpy: bool = False
    ) -> Optional[tuple[Any, ...]]:
        """Render events from the event camera.

        Args:
            time: The current timestamp in microseconds.
            to_numpy: Whether to convert events to numpy arrays.

        Returns:
            Tuple of (x, y, t, p) event arrays, or None if no events.
        """
        color_sensor = self._sim._sensors[uuid]
        color_sensor.draw_observation()
        color_observation = color_sensor.get_observation()[..., :3]  # RGB

        intensity_image = color2intensity(color_observation / 255.0)

        events = self._event_simulators[uuid].image_callback(
            intensity_image, time
        )  # in us

        if to_numpy and events is not None and isinstance(events[0], torch.Tensor):
            events = [
                events[0].cpu().numpy().astype(np.uint16),
                events[1].cpu().numpy().astype(np.uint16),
                events[2].cpu().numpy().astype(np.uint64),
                events[3].cpu().numpy().astype(np.uint8),
            ]
            # else we assume they are already numpy arrays
        return events

    def render_color(self, uuid: str) -> torch.Tensor:
        """Render the color sensor.

        Returns:
            RGB image tensor of shape (H, W, 3).
        """
        sensor = self._sim._sensors[uuid]
        sensor.draw_observation()
        return sensor.get_observation()[..., :3]

    def render_depth(self, uuid: str) -> torch.Tensor:
        """Render the depth sensor.

        Returns:
            Depth image tensor.
        """
        sensor = self._sim._sensors[uuid]
        sensor.draw_observation()
        return sensor.get_observation()

    def render_navmesh(
        self, meters_per_pixel: float = 0.1, height: Optional[float] = None
    ) -> np.ndarray:
        """Render the navmesh."""
        if height is None:
            height = self._scene_bounds[0][1] + 0.5
        sim_topdown_map = self._sim.pathfinder.get_topdown_view(
            meters_per_pixel, height
        ).astype(np.uint8)
        outline_border(sim_topdown_map)
        return RECOLOR_MAP[sim_topdown_map]

    def close(self) -> None:
        """Close the simulator."""
        self._sim.close()
