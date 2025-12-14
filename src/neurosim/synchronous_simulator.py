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
from typing import Callable, Any
from dataclasses import dataclass, field

from neurosim.core.visual_backend import HabitatWrapper
from neurosim.core.dynamics import create_dynamics
from neurosim.core.control import create_controller
from neurosim.core.trajectory import create_trajectory
from neurosim.core.imu_sim import create_imu_sensor
from neurosim.core.coord_trans import CoordinateTransform
from neurosim.core.utils import RerunVisualizer, H5Logger


logger = logging.getLogger(__name__)


@dataclass(slots=True)
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
    sensor_manager: "SensorManager" = field(init=False, default=None)

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


@dataclass(slots=True)
class SensorConfig:
    """Container for a single sensor's configuration.

    Attributes:
        uuid: Unique identifier for the sensor
        sensor_type: Type of the sensor (e.g., event, color, depth, imu)
        sampling_rate: Sampling rate in Hz (how often to render/sample)
        sampling_steps: Number of simulation steps between samples
        viz_rate: Visualization rate in Hz (how often to display)
        viz_steps: Number of simulation steps between visualization
        executor: Callable function to execute for obtaining sensor data
    """

    uuid: str
    config: dict
    sensor_type: str
    sampling_rate: float
    sampling_steps: int
    viz_rate: float
    viz_steps: int
    executor: Callable[[], Any] | None = field(default=None, repr=False)


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
                config=sensor_cfg,
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

    def add_executor(self, uuid: str, executor: Callable[[], Any]) -> None:
        """
        Add an executor function to a sensor.

        Args:
            uuid: Sensor UUID
            executor: Callable function to execute for obtaining sensor data
        """
        self.sensors[uuid].executor = executor

    def should_sample(self, uuid: str, simstep: int) -> bool:
        """Check if a sensor should be sampled at this simulation step."""
        return simstep % self.sensors[uuid].sampling_steps == 0

    def should_visualize(self, uuid: str, simstep: int) -> bool:
        """Check if a sensor should be visualized at this simulation step."""
        return simstep % self.sensors[uuid].viz_steps == 0

    def get_sensor_config(self, uuid: str) -> SensorConfig | None:
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

        ####################
        # Simulation state #
        ####################
        self.time = 0.0
        self.simsteps = 0

        ######################################
        # Initialize backends and components #
        ######################################

        # Initializes the visual backend and binds visual sensor executors
        self._init_visual_backend()

        # Initialize dynamics, controller, trajectory
        self._init_dynamics()
        self._init_controller()
        self._init_trajectory()

        # Initialize additional sensors like IMU and bind their executors
        self._init_additional_sensors()

        # Initialize visualizer
        self.visualizer = RerunVisualizer(self.config)

        logger.info("🚀 Simulator initialized successfully")

    @staticmethod
    def _create_sensor_executor(sensor_type: str, **kwargs) -> Callable[[], Any]:
        """Factory to create optimized sensor executor functions.

        Uses provider functions to access current values at execution time,
        not initialization time. Minimizes closure depth and overhead.

        Args:
            sensor_type: Type of the sensor (event, color, depth, imu, navmesh)
            **kwargs: Sensor-specific parameters:
                Visual sensors (event, color, depth):
                    - backend: Visual backend instance
                    - uuid: Sensor UUID
                    - time_provider: Callable that returns current simulation time
                IMU sensors:
                    - sensor: IMU sensor instance
                    - state_provider: Callable that returns current state
                    - statedot_provider: Callable that returns current state derivative
                Navmesh sensors:
                    - backend: Visual backend instance
                    - uuid: Sensor UUID
                    - meters_per_pixel: Scale factor for the navmesh (default: 0.1)
                    - height: Height at which to render the navmesh (optional)

        Returns:
            Callable executor function that samples the sensor
        """
        if sensor_type == "event":
            backend = kwargs["backend"]
            uuid = kwargs["uuid"]
            time_provider = kwargs["time_provider"]
            # Pre-bind methods to reduce attribute lookups
            render_events = backend.render_events

            def executor():
                # Minimize function call depth - call provider inline
                return render_events(
                    uuid=uuid, time=int(time_provider() * 1e6), to_numpy=False
                )

        elif sensor_type == "color":
            backend = kwargs["backend"]
            uuid = kwargs["uuid"]
            # Pre-bind method
            render_color = backend.render_color

            def executor():
                return render_color(uuid)

        elif sensor_type == "depth":
            backend = kwargs["backend"]
            uuid = kwargs["uuid"]
            # Pre-bind method
            render_depth = backend.render_depth

            def executor():
                return render_depth(uuid)

        elif sensor_type == "imu":
            sensor = kwargs["sensor"]
            state_provider = kwargs["state_provider"]
            statedot_provider = kwargs["statedot_provider"]
            # Pre-bind method
            measurement = sensor.measurement

            def executor():
                return measurement(state_provider(), statedot_provider())

        elif sensor_type == "navmesh":
            backend = kwargs["backend"]
            meters_per_pixel = kwargs.get("meters_per_pixel", 0.1)
            height = kwargs.get("height", None)
            # Pre-bind method
            render_navmesh = backend.render_navmesh

            def executor():
                return render_navmesh(meters_per_pixel=meters_per_pixel, height=height)

        else:
            raise ValueError(f"Unsupported sensor type: {sensor_type}")

        return executor

    def _init_visual_backend(self) -> None:
        self.visual_backend = HabitatWrapper(self.settings["visual_backend"])
        logger.info("Visual backend initialized: HabitatWrapper")

        # Bind executors for visual sensors
        for uuid, sensor_cfg in self.config.visual_sensors.items():
            sensor_type = sensor_cfg.get("type")

            # Build kwargs for the executor
            executor_kwargs = {
                "sensor_type": sensor_type,
                "backend": self.visual_backend,
                "uuid": uuid,
            }

            # Add type-specific parameters
            if sensor_type in ["event", "color", "depth"]:
                executor_kwargs["time_provider"] = lambda: self.time  # Lazy evaluation

            elif sensor_type == "navmesh":
                executor_kwargs["meters_per_pixel"] = sensor_cfg.get(
                    "meters_per_pixel", 0.1
                )
                executor_kwargs["height"] = sensor_cfg.get("height", None)

            executor = self._create_sensor_executor(**executor_kwargs)
            self.config.sensor_manager.add_executor(uuid, executor)

    def _init_dynamics(self) -> None:
        # Hardcoded coordinate transform from rotorpy to habitat,
        # can be made configurable later
        self.coord_trans = CoordinateTransform(
            pos_transform=np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
            quat_transform=np.array(
                [[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]]
            ),
        )
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

        for uuid, sensor_cfg in self.config.additional_sensors.items():
            sensor_type = sensor_cfg.get("type")

            if sensor_type == "imu":
                sensor = create_imu_sensor(**sensor_cfg)

                executor = self._create_sensor_executor(
                    sensor_type=sensor_type,
                    sensor=sensor,
                    state_provider=lambda: self.dynamics.state,  # Lazy evaluation
                    statedot_provider=lambda: self.dynamics.statedot(),  # Lazy evaluation
                )

                logger.info(f"Initialized IMU sensor: {uuid}")
            else:
                raise ValueError(f"Unsupported additional sensor type: {sensor_type}")

            self.config.sensor_manager.add_executor(uuid, executor)

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
        position, rotation = self.coord_trans.transform(state)
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
        sensor_manager = self.config.sensor_manager

        for uuid, sensor_cfg in sensor_manager.sensors.items():
            if sensor_manager.should_sample(uuid, self.simsteps):
                measurements[uuid] = sensor_cfg.executor()

        return measurements

    def run(self, display: bool = False, log_h5: str | None = None) -> dict:
        """
        Run the simulation.

        Args:
            display: Whether to display live visualization with Rerun
            log_h5: Path to HDF5 file for logging. If None, no logging.

        Returns:
            Dictionary with simulation statistics
        """
        # Setup visualization
        if display:
            self.visualizer.initialize()

        # Setup H5 logger
        h5_logger = None
        if log_h5:
            h5_logger = H5Logger(
                filename=log_h5,
                sensor_manager=self.config.sensor_manager,
                deepcopy_data=False,  # Use zero-copy for speed
                compression=None,  # Disable compression for speed (can enable 'lzf' for smaller files)
                verbose=True,
            )
            logger.info(f"H5 logging enabled: {log_h5}")

        # Run simulation loop
        latencies = self._run_simulation_loop(display, h5_logger)

        # Close H5 logger
        if h5_logger:
            h5_logger.close()

        # Compute and display statistics
        stats = self._compute_statistics(latencies)
        logger.info("Simulation completed")
        logger.info("=" * 60)
        logger.info("\nSimulation Statistics:")
        logger.info(f"\n{pprint.pformat(stats, indent=4, width=80)}")

        return stats

    def _run_simulation_loop(
        self, display: bool, h5_logger: H5Logger | None = None
    ) -> list[float]:
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

                # Log to H5
                if h5_logger:
                    h5_logger.log(
                        {
                            **measurements,
                            "state": self.dynamics.state,
                        },
                        self.time,
                        self.simsteps,
                    )

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
