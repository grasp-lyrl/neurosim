"""
Habitat Wrapper for Neurosim.

This module provides a wrapper around the Habitat simulator with
event camera simulation support.
"""

import time as time_module
import yaml
import torch
import random
import numpy as np
import magnum as mn
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import habitat_sim as hsim

from neurosim.settings import default_sim_settings
from neurosim.utils import color2intensity
from neurosim.utils.evsim import (
    create_event_simulator,
    get_best_available_backend,
    EventSimulatorProtocol,
)


@dataclass
class RenderEventProfiler:
    """Profiler for tracking render_events performance.

    This class tracks timing statistics for render_events calls,
    useful for benchmarking different event simulator backends.
    """

    enabled: bool = False
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    times_ms: list = field(default_factory=list)

    # Breakdown timings
    total_render_time_ms: float = 0.0
    total_color2intensity_time_ms: float = 0.0
    total_evsim_time_ms: float = 0.0

    def reset(self) -> None:
        """Reset all profiler statistics."""
        self.call_count = 0
        self.total_time_ms = 0.0
        self.min_time_ms = float("inf")
        self.max_time_ms = 0.0
        self.times_ms = []
        self.total_render_time_ms = 0.0
        self.total_color2intensity_time_ms = 0.0
        self.total_evsim_time_ms = 0.0

    def record(
        self,
        total_ms: float,
        render_ms: float = 0.0,
        color2intensity_ms: float = 0.0,
        evsim_ms: float = 0.0,
    ) -> None:
        """Record a timing measurement."""
        self.call_count += 1
        self.total_time_ms += total_ms
        self.min_time_ms = min(self.min_time_ms, total_ms)
        self.max_time_ms = max(self.max_time_ms, total_ms)
        self.times_ms.append(total_ms)

        self.total_render_time_ms += render_ms
        self.total_color2intensity_time_ms += color2intensity_ms
        self.total_evsim_time_ms += evsim_ms

    @property
    def avg_time_ms(self) -> float:
        """Average time per call in milliseconds."""
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def std_time_ms(self) -> float:
        """Standard deviation of time per call in milliseconds."""
        if self.call_count < 2:
            return 0.0
        return float(np.std(self.times_ms))

    @property
    def avg_render_time_ms(self) -> float:
        """Average render time per call in milliseconds."""
        return self.total_render_time_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_color2intensity_time_ms(self) -> float:
        """Average color2intensity time per call in milliseconds."""
        return self.total_color2intensity_time_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_evsim_time_ms(self) -> float:
        """Average event simulation time per call in milliseconds."""
        return self.total_evsim_time_ms / self.call_count if self.call_count > 0 else 0.0

    def get_summary(self) -> dict:
        """Get a summary of profiling statistics."""
        return {
            "enabled": self.enabled,
            "call_count": self.call_count,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms if self.call_count > 0 else 0.0,
            "max_time_ms": self.max_time_ms,
            "breakdown": {
                "avg_render_ms": self.avg_render_time_ms,
                "avg_color2intensity_ms": self.avg_color2intensity_time_ms,
                "avg_evsim_ms": self.avg_evsim_time_ms,
            },
        }

    def print_summary(self) -> None:
        """Print a formatted summary of profiling statistics."""
        if not self.enabled or self.call_count == 0:
            print("[Profiler] No profiling data available.")
            return

        print("\n" + "=" * 60)
        print("RENDER_EVENTS PROFILING SUMMARY")
        print("=" * 60)
        print(f"Total calls: {self.call_count}")
        print(f"Total time: {self.total_time_ms:.2f} ms")
        print("-" * 60)
        print(f"Average time: {self.avg_time_ms:.3f} ms")
        print(f"Std dev: {self.std_time_ms:.3f} ms")
        print(f"Min time: {self.min_time_ms:.3f} ms")
        print(f"Max time: {self.max_time_ms:.3f} ms")
        print("-" * 60)
        print("Breakdown (average per call):")
        print(
            f"  Render:          {self.avg_render_time_ms:.3f} ms ({100*self.avg_render_time_ms/self.avg_time_ms:.1f}%)"
        )
        print(
            f"  Color2Intensity: {self.avg_color2intensity_time_ms:.3f} ms ({100*self.avg_color2intensity_time_ms/self.avg_time_ms:.1f}%)"
        )
        print(
            f"  Event Sim:       {self.avg_evsim_time_ms:.3f} ms ({100*self.avg_evsim_time_ms/self.avg_time_ms:.1f}%)"
        )
        print("=" * 60)


class HabitatWrapper:
    """Wrapper around Habitat simulator with event camera support.

    This class provides a convenient interface to the Habitat simulator
    with built-in event camera simulation. Users can choose between
    different event simulator backends (cuda, torch, airsim, vid2e).

    Args:
        settings: Path to a YAML settings file, or None to use defaults.
        event_camera_backend: Backend to use for event simulation.
            Options: "cuda" (recommended), "torch", "airsim", "vid2e", or "auto".
            If "auto", the best available backend will be selected.
        enable_profiling: Whether to enable profiling for render_events calls.
    """

    def __init__(
        self,
        settings: Optional[Path] = None,
        event_camera_backend: str = "auto",
        enable_profiling: bool = False,
    ):
        # Initialize profiler
        self.profiler = RenderEventProfiler(enabled=enable_profiling)

        # Load settings
        if settings is None:
            self.settings = default_sim_settings.copy()
        else:
            with open(settings, "r") as file:
                loaded_settings = yaml.safe_load(file)
                # Merge with defaults
                self.settings = default_sim_settings.copy()
                self.settings.update(loaded_settings)

        # Override event camera backend if specified
        if event_camera_backend != "auto":
            self.settings["event_camera_backend"] = event_camera_backend

        # Create Habitat configuration
        self._cfg = self._make_cfg(self.settings)

        # Create event simulator
        self._event_simulator = self._create_event_simulator(self.settings)

        # Initialize simulator
        self._sim = hsim.Simulator(self._cfg)
        self._scene_aabb = self._sim.scene_aabb

        # Set seed
        random.seed(self.settings["seed"])
        self._sim.seed(self.settings["seed"])

        self.init_agent_state(self.settings["default_agent"])
        self.agent = self._sim.get_agent(self.settings["default_agent"])

    def init_agent_state(self, agent_id: int) -> hsim.AgentState:
        """Initialize the agent state.

        Args:
            agent_id: The ID of the agent to initialize.

        Returns:
            The initialized agent state.
        """
        agent = self._sim.initialize_agent(agent_id)

        agent_state = hsim.AgentState()
        agent_state.position = mn.Vector3(self.settings["start_position"])
        agent.set_state(agent_state)

        agent_state = agent.get_state()
        print(
            f"Agent {agent_id} initialized at "
            f"position: {agent_state.position}, "
            f"rotation: {agent_state.rotation}"
        )

        return agent_state

    @staticmethod
    def _create_event_simulator(settings: dict[str, Any]) -> Optional[EventSimulatorProtocol]:
        """Create an event simulator based on settings.

        Args:
            settings: The simulation settings dictionary.

        Returns:
            An event simulator instance, or None if event camera is disabled.
        """
        if not settings.get("event_camera", False):
            return None

        # Get backend preference
        backend_str = settings.get("event_camera_backend", "auto")

        if backend_str == "auto":
            backend = get_best_available_backend()
            print(f"[HabitatWrapper] Auto-selected event simulator backend: {backend.value}")
        else:
            backend = backend_str

        # Get contrast thresholds from settings
        contrast_threshold_pos = settings.get("event_contrast_threshold_pos", 0.35)
        contrast_threshold_neg = settings.get("event_contrast_threshold_neg", 0.35)

        return create_event_simulator(
            backend=backend,
            width=settings["width"],
            height=settings["height"],
            start_time=0,
            contrast_threshold_neg=contrast_threshold_neg,
            contrast_threshold_pos=contrast_threshold_pos,
        )

    @staticmethod
    def _make_cfg(
        settings: dict[str, Any],
    ) -> hsim.Configuration:
        """Create a Habitat configuration from a settings dictionary.

        Args:
            settings: A dict with pre-defined keys for simulator initialization.

        Returns:
            A tuple of (Habitat Configuration, Event Simulator).
        """
        sim_cfg = hsim.SimulatorConfiguration()

        if "scene_dataset_config_file" in settings:
            sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]

        if "enable_physics" in settings:
            sim_cfg.enable_physics = settings["enable_physics"]

        if "physics_config_file" in settings:
            sim_cfg.physics_config_file = settings["physics_config_file"]

        if "scene_light_setup" in settings:
            sim_cfg.scene_light_setup = settings["scene_light_setup"]

        sim_cfg.frustum_culling = settings.get("frustum_culling", False)
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

        if settings.get("color_sensor", True):
            color_sensor_spec = create_camera_spec(
                uuid="color_sensor",
                hfov=settings["hfov"],
                far=settings["zfar"],
                sensor_type=hsim.SensorType.COLOR,
                sensor_subtype=hsim.SensorSubType.PINHOLE,
                clear_color=settings["clear_color"],
            )
            sensor_specs.append(color_sensor_spec)

        if settings.get("depth_sensor", False):
            depth_sensor_spec = create_camera_spec(
                uuid="depth_sensor",
                hfov=settings["hfov"],
                far=settings["zfar"],
                sensor_type=hsim.SensorType.DEPTH,
                channels=1,
                sensor_subtype=hsim.SensorSubType.PINHOLE,
            )
            sensor_specs.append(depth_sensor_spec)

        if settings.get("semantic_sensor", False):
            semantic_sensor_spec = create_camera_spec(
                uuid="semantic_sensor",
                hfov=settings["hfov"],
                far=settings["zfar"],
                sensor_type=hsim.SensorType.SEMANTIC,
                channels=1,
                sensor_subtype=hsim.SensorSubType.PINHOLE,
            )
            sensor_specs.append(semantic_sensor_spec)

        # Create agent specifications
        agent_cfg = hsim.agent.AgentConfiguration()
        agent_cfg.height = settings["sensor_height"]
        agent_cfg.radius = settings["agent_radius"]
        agent_cfg.sensor_specifications = sensor_specs

        return hsim.Configuration(sim_cfg, [agent_cfg])

    def update_agent_pose(
        self, agent_id: int, position: mn.Vector3, rotation: mn.Quaternion
    ) -> None:
        """Update the agent's pose.

        Args:
            agent_id: The ID of the agent to update.
            position: The new position.
            rotation: The new rotation.
        """
        agent = self._sim.get_agent(agent_id)
        agent_state = hsim.AgentState()
        agent_state.position = position
        agent_state.rotation = rotation
        agent.set_state(agent_state)

    def get_agent_state(self, agent_id: int) -> hsim.AgentState:
        """Get the agent's state.

        Args:
            agent_id: The ID of the agent.

        Returns:
            The agent's current state.
        """
        agent = self._sim.get_agent(agent_id)
        return agent.get_state()

    def render_events(self, time: int, to_numpy: bool = False) -> Optional[tuple[Any, ...]]:
        """Render events from the event camera.

        Args:
            time: The current timestamp in microseconds.
            to_numpy: Whether to convert events to numpy arrays.

        Returns:
            Tuple of (x, y, t, p) event arrays, or None if no events.
        """
        if self._event_simulator is None:
            raise RuntimeError("Event camera is not enabled. Set 'event_camera': True in settings.")

        # Start total timing
        if self.profiler.enabled:
            torch.cuda.synchronize()
            total_start = time_module.perf_counter()

        # Render color sensor
        if self.profiler.enabled:
            render_start = time_module.perf_counter()

        sensor = self._sim._sensors["color_sensor"]
        sensor.draw_observation()
        observation = sensor.get_observation()[..., :3]

        if self.profiler.enabled:
            torch.cuda.synchronize()
            render_end = time_module.perf_counter()

        # Convert to intensity
        if self.profiler.enabled:
            c2i_start = time_module.perf_counter()

        intensity_image = color2intensity(observation / 255.0)

        if self.profiler.enabled:
            torch.cuda.synchronize()
            c2i_end = time_module.perf_counter()

        # Run event simulator
        if self.profiler.enabled:
            evsim_start = time_module.perf_counter()

        events = self._event_simulator.image_callback(intensity_image, time)  # in us

        if self.profiler.enabled:
            torch.cuda.synchronize()
            evsim_end = time_module.perf_counter()
            total_end = time_module.perf_counter()

            # Record timings
            self.profiler.record(
                total_ms=(total_end - total_start) * 1000,
                render_ms=(render_end - render_start) * 1000,
                color2intensity_ms=(c2i_end - c2i_start) * 1000,
                evsim_ms=(evsim_end - evsim_start) * 1000,
            )

        if to_numpy and events is not None:
            events = [
                events[0].cpu().numpy().astype(np.uint16),
                events[1].cpu().numpy().astype(np.uint16),
                events[2].cpu().numpy().astype(np.uint64),
                events[3].cpu().numpy().astype(np.uint8),
            ]
        return events

    def render_color_sensor(self) -> torch.Tensor:
        """Render the color sensor.

        Returns:
            RGB image tensor of shape (H, W, 3).
        """
        sensor = self._sim._sensors["color_sensor"]
        sensor.draw_observation()
        return sensor.get_observation()[..., :3]

    def render_depth_sensor(self) -> torch.Tensor:
        """Render the depth sensor.

        Returns:
            Depth image tensor.
        """
        sensor = self._sim._sensors["depth_sensor"]
        sensor.draw_observation()
        return sensor.get_observation()

    def close(self) -> None:
        """Close the simulator."""
        self._sim.close()

    def reset(self) -> None:
        """Reset the simulator."""
        self._sim.reset()

    @property
    def event_simulator(self) -> Optional[EventSimulatorProtocol]:
        """Get the event simulator instance."""
        return self._event_simulator
