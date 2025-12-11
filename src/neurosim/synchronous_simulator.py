"""
Neurosim Simulator Module.

This module provides the main Simulator class for running synchronous
simulations with visual backends, dynamics, control, and sensors.
"""

import time
import yaml
import logging
import pprint
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from neurosim.core.visual_backend import HabitatWrapper
from neurosim.core.dynamics import create_dynamics
from neurosim.core.control import create_controller
from neurosim.core.trajectory import create_trajectory
from neurosim.core.imu_sim import create_imu_sensor
from neurosim.core.coord_trans import rotorpy_to_habitat
from neurosim.core.utils import RerunVisualizer


logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Container for simulation configuration."""

    world_rate: int
    control_rate: int
    sim_time: float
    sensor_rates: dict = field(default_factory=dict)
    viz_rates: dict = field(default_factory=dict)  # Optional visualization rates
    visual_sensors: dict = field(default_factory=dict)
    additional_sensors: dict = field(default_factory=dict)
    t_step: float = field(init=False)
    t_final: float = field(init=False)
    sensor_manager: Optional["SensorManager"] = field(init=False, default=None)

    def __post_init__(self):
        self.t_step = 1.0 / self.world_rate
        self.t_final = self.sim_time

        # Initialize sensor manager
        self.sensor_manager = SensorManager(
            world_rate=self.world_rate,
            sensor_rates=self.sensor_rates,
            viz_rates=self.viz_rates,
            visual_sensors=self.visual_sensors,
            additional_sensors=self.additional_sensors,
        )


@dataclass
class SensorConfig:
    """Container for a single sensor's configuration.

    Attributes:
        uuid: Unique identifier for the sensor
        sensor_type: Type of the sensor (e.g., event, color, depth, imu)
        sampling_rate: Sampling rate in Hz (how often to render/sample)
        sampling_steps: Number of simulation steps between samples
        viz_rate: Visualization rate in Hz (how often to display)
        viz_steps: Number of simulation steps between visualization
    """

    uuid: str
    sensor_type: str
    sampling_rate: float
    sampling_steps: int
    viz_rate: float
    viz_steps: int


class SensorManager:
    """Manages sensor configurations and sampling logic."""

    def __init__(
        self,
        world_rate: int,
        sensor_rates: dict[str, float],
        viz_rates: dict[str, float],
        visual_sensors: dict[str, dict],
        additional_sensors: dict[str, dict],
    ):
        """
        Initialize sensor manager.

        Args:
            world_rate: World simulation rate in Hz
            sensor_rates: Dictionary mapping sensor UUIDs to their sampling rates
            viz_rates: Dictionary mapping sensor UUIDs to their visualization rates (optional)
            visual_sensors: Visual sensors from visual_backend config
            additional_sensors: Additional sensors (IMU, etc.) from simulator config
        """
        self.world_rate = world_rate
        self.sensors: dict[str, SensorConfig] = {}

        # Parse all sensors (visual and additional)
        for uuid, sensor_cfg in {**visual_sensors, **additional_sensors}.items():
            sensor_type = sensor_cfg["type"]
            sampling_rate = sensor_rates[uuid]
            sampling_steps = int(world_rate / sampling_rate)

            # Visualization rate defaults to sampling rate if not specified
            viz_rate = viz_rates.get(uuid, sampling_rate)

            # Ensure viz_rate <= sampling_rate
            if viz_rate > sampling_rate:
                logger.warning(
                    f"Sensor {uuid}: viz_rate ({viz_rate}Hz) > sampling_rate ({sampling_rate}Hz). "
                    f"Setting viz_rate = sampling_rate"
                )
                viz_rate = sampling_rate

            viz_steps = int(world_rate / viz_rate)

            self.sensors[uuid] = SensorConfig(
                uuid=uuid,
                sensor_type=sensor_type,
                sampling_rate=sampling_rate,
                sampling_steps=sampling_steps,
                viz_rate=viz_rate,
                viz_steps=viz_steps,
            )

        logger.info(f"Initialized {len(self.sensors)} sensors")
        for uuid, cfg in self.sensors.items():
            logger.info(
                f"  • {uuid}: {cfg.sensor_type} @ {cfg.sampling_rate}Hz "
                f"(viz: {cfg.viz_rate}Hz)"
            )

    def should_sample(self, uuid: str, simstep: int) -> bool:
        """Check if a sensor should be sampled at this simulation step."""
        return simstep % self.sensors[uuid].sampling_steps == 0

    def should_visualize(self, uuid: str, simstep: int) -> bool:
        """Check if a sensor should be visualized at this simulation step."""
        return simstep % self.sensors[uuid].viz_steps == 0

    def get_sensor_config(self, uuid: str) -> Optional[SensorConfig]:
        """
        Get sensor configuration by UUID.

        Args:
            uuid: Sensor UUID
        Returns:
            SensorConfig object or None if not found
        """
        return self.sensors.get(uuid)

    def get_sensors_by_type(self, sensor_type: str) -> list[SensorConfig]:
        """Get all sensors of a specific type."""
        return [s for s in self.sensors.values() if s.sensor_type == sensor_type]


class SynchronousSimulator:
    """Main simulator class for neurosim."""

    def __init__(self, settings_path: Path | str):
        """
        Initialize the Simulator.

        Args:
            settings_path: Path to the settings YAML file.
        """
        self.settings_path = Path(settings_path)
        if not self.settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {self.settings_path}")

        # Load settings
        with open(self.settings_path, "r") as f:
            self.settings = yaml.safe_load(f)

        # Initialize simulation configuration (includes sensor manager)
        sim_cfg = self.settings.get("simulator", {})
        self.config = SimulationConfig(
            world_rate=sim_cfg["world_rate"],
            control_rate=sim_cfg["control_rate"],
            sim_time=sim_cfg["sim_time"],
            sensor_rates=sim_cfg.get("sensor_rates", {}),
            viz_rates=sim_cfg.get("viz_rates", {}),  # Optional visualization rates
            visual_sensors=self.settings.get("visual_backend", {}).get("sensors", {}),
            additional_sensors=sim_cfg.get("additional_sensors", {}),
        )

        # Simulation state
        self.time = 0.0
        self.simsteps = 0

        # Initialize backends and components
        self._init_visual_backend()
        self._init_dynamics()
        self._init_controller()
        self._init_trajectory()
        self._init_additional_sensors()  # Example: IMU

        # Initialize visualizer
        self.visualizer = RerunVisualizer(self.config)

        logger.info("🚀 Simulator initialized successfully")

    def _init_visual_backend(self) -> None:
        self.visual_backend = HabitatWrapper(self.settings["visual_backend"])
        logger.info("Visual backend initialized: HabitatWrapper")

    def _init_dynamics(self) -> None:
        self.dynamics = create_dynamics(
            **self.settings["dynamics"],
            initial_state={
                "x": np.array(self.settings["start_position"]),
                "q": np.array(self.settings["start_orientation"]),
            },
        )
        logger.info(
            f"Dynamics initialized at state: {pprint.pformat(self.dynamics.state, indent=4, width=60)}"
        )

    def _init_controller(self) -> None:
        self.controller = create_controller(**self.settings.get("controller", {}))
        logger.info(f"Controller initialized: {self.controller.__class__.__name__}")

    def _init_trajectory(self) -> None:
        self.trajectory = create_trajectory(**self.settings.get("trajectory", {}))
        logger.info(f"Trajectory initialized: {self.trajectory.__class__.__name__}")

    def _init_additional_sensors(self) -> None:
        """Initialize additional sensors like IMU."""
        self.additional_sensors = {}

        for uuid, sensor_cfg in self.config.additional_sensors.items():
            sensor_type = sensor_cfg.get("type")

            if sensor_type == "imu":
                self.additional_sensors[uuid] = create_imu_sensor(**sensor_cfg)
                logger.info(f"Initialized IMU sensor: {uuid}")
            else:
                raise ValueError(f"Unsupported additional sensor type: {sensor_type}")

    def step(self, control: dict) -> None:
        """
        Execute one simulation step.

        Args:
            control: Control input dictionary

        Returns:
            Updated state dictionary
        """
        self.time += self.config.t_step
        self.simsteps += 1

        # Update dynamics
        state = self.dynamics.step(control, self.config.t_step)

        # TODO: Choose the transform based on settings. Right now hardcoded from rotorpy to habitat
        # This is the coordinate transform step from the dynamics to the visual renderer
        position, rotation = rotorpy_to_habitat(state)
        self.visual_backend.update_agent_state(position, rotation)

    def _render_sensors(self) -> dict:
        """
        Render all sensors that should be sampled at this timestep.

        Keeps data on GPU to minimize expensive GPU->CPU transfers.
        Data is only moved to CPU during visualization.

        Returns:
            Dictionary mapping sensor UUIDs to their measurements (GPU tensors when possible)
        """
        measurements = {}

        for uuid, sensor_cfg in self.config.sensor_manager.sensors.items():
            if not self.config.sensor_manager.should_sample(uuid, self.simsteps):
                continue

            sensor_type = sensor_cfg.sensor_type

            if sensor_type == "event":
                events = self.visual_backend.render_events(
                    uuid=uuid,
                    time=int(self.time * 1e6),  # Convert to microseconds
                    to_numpy=False,  # Keep on GPU for fast accumulation
                )
                measurements[uuid] = events

            elif sensor_type == "color":
                color_img = self.visual_backend.render_color(uuid)
                measurements[uuid] = color_img  # Keep as GPU tensor

            elif sensor_type == "depth":
                depth_img = self.visual_backend.render_depth(uuid)
                measurements[uuid] = depth_img  # Keep as GPU tensor

            elif sensor_type == "imu" and uuid in self.additional_sensors:
                imu_data = self.additional_sensors[uuid].measurement(
                    self.dynamics.state, self.dynamics.statedot()
                )
                measurements[uuid] = imu_data

        return measurements

    def run(self, display: bool = False) -> dict:
        """
        Run the simulation.

        Args:
            display: Whether to display live visualization with Rerun

        Returns:
            Dictionary with simulation statistics
        """
        # Setup visualization
        if display:
            self.visualizer.initialize()

        # Run simulation loop
        latencies = self._run_simulation_loop(display)

        # Compute and display statistics
        stats = self._compute_statistics(latencies)
        logger.info("Simulation completed")
        logger.info("=" * 60)
        logger.info("\nSimulation Statistics:")
        logger.info(f"\n{pprint.pformat(stats, indent=4, width=80)}")

        return stats

    def _run_simulation_loop(self, display: bool) -> list[float]:
        """Run the main simulation loop."""
        # Initialize control
        flat = self.trajectory.update(self.time)
        control = self.controller.update(self.time, self.dynamics.state, flat)

        latencies = []
        steps_per_control = int(self.config.world_rate / self.config.control_rate)

        logger.info("=" * 60)
        logger.info(
            f"▶ Starting simulation: {self.config.sim_time}s @ {self.config.world_rate}Hz "
            f"(control @ {self.config.control_rate}Hz)"
        )
        logger.info("=" * 60)

        while self.time < self.config.t_final:
            # Inner loop: world rate steps between control updates
            for _ in range(steps_per_control):
                start_time = time.perf_counter()

                # Simulate one step
                self.step(control)

                # Render sensors
                measurements = self._render_sensors()

                # Record latency
                latencies.append(time.perf_counter() - start_time)

                # Display with Rerun
                if display:
                    self.visualizer.log_measurements(
                        measurements, self.time, self.simsteps
                    )
                    self.visualizer.log_state(self.dynamics.state)

            # Update control at control rate
            flat = self.trajectory.update(self.time)
            control = self.controller.update(self.time, self.dynamics.state, flat)

        return latencies

    def _compute_statistics(self, latencies: list[float]) -> dict:
        """Compute simulation statistics."""
        warmup = 20
        valid_latencies = latencies[warmup:] if len(latencies) > warmup else []

        return {
            "total_steps": self.simsteps,
            "sim_time": self.config.sim_time,
            "world_rate": self.config.world_rate,
            "control_rate": self.config.control_rate,
            "total_wall_time": sum(latencies),
            "avg_step_time": np.mean(valid_latencies),
            "median_step_time": np.median(valid_latencies),
            "max_step_time": np.max(latencies),
            "min_step_time": np.min(latencies),
            "FPS": len(valid_latencies) / sum(valid_latencies)
            if valid_latencies
            else 0,
        }

    def close(self) -> None:
        """Clean up simulator resources."""
        if hasattr(self, "visual_backend"):
            self.visual_backend.close()
        logger.info("Simulator closed gracefully")
