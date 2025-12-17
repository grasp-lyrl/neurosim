"""
Neurosim Simulator Module.

This module provides the main Simulator class for running synchronous
simulations with visual backends, dynamics, control, and sensors.
"""

import time
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Callable, Any

from neurosim.core.visual_backend import HabitatWrapper
from neurosim.core.dynamics import create_dynamics
from neurosim.core.control import create_controller
from neurosim.core.trajectory import create_trajectory
from neurosim.core.imu_sim import create_imu_sensor
from neurosim.core.coord_trans import CoordinateTransform
from neurosim.core.utils import RerunVisualizer, H5Logger, SimulationConfig, format_dict


logger = logging.getLogger(__name__)


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

        # Initialize coordinate transform utility (hardcoded for now)
        self.coord_trans = CoordinateTransform("rotorpy_to_habitat")

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

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ Simulator initialized successfully")
        logger.info(f"   World rate: {self.config.world_rate} Hz")
        logger.info(f"   Control rate: {self.config.control_rate} Hz")
        logger.info(f"   Simulation time: {self.config.sim_time} s")
        logger.info("═══════════════════════════════════════════════════════════")

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
        self.dynamics = create_dynamics(**self.settings["dynamics"])

    def _init_controller(self) -> None:
        self.controller = create_controller(**self.settings.get("controller", {}))

    def _init_trajectory(self) -> None:
        # TODO: Right now this assumes the visual backend has a pathfinder
        # TODO: This is fine for HabitatWrapper, but may need to be more general later
        # TODO: Ideally, we want to move on from habitat pathfinder to a more general navmesh handler
        # TODO: and path planner, say GCOPTER with Recast/Detour or similar.
        traj_kwargs = self.settings.get("trajectory", {}).copy()
        traj_kwargs["pathfinder"] = self.visual_backend._sim.pathfinder
        traj_kwargs["coord_transform"] = self.coord_trans.inverse_transform_batch
        self.trajectory = create_trajectory(**traj_kwargs)

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
        position, quaternion = self.coord_trans.transform(state["x"], state["q"])
        self.visual_backend.update_agent_state(position, quaternion)

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

        # Run simulation loop
        latencies = self._run_simulation_loop(display, h5_logger)

        # Close H5 logger
        if h5_logger:
            h5_logger.close()

        # Compute and display statistics
        stats = self._compute_statistics(latencies)
        logger.info("Simulation completed")
        logger.info("════════════════════════════════════════════════════════════════")
        logger.info("\nSimulation Statistics:")
        logger.info(format_dict(stats))

        return stats

    def _run_simulation_loop(
        self, display: bool, h5_logger: H5Logger | None = None
    ) -> list[float]:
        """Run the main simulation loop."""
        # Initialize control and state from first trajectory point
        flat = self.trajectory.update(self.time)
        self.dynamics.state = {
            "x": flat["x"],
            "v": flat["x_dot"],
            "yaw": flat["yaw"],
            "yaw_dot": flat["yaw_dot"],
        }
        control = self.controller.update(self.time, self.dynamics.state, flat)

        latencies = []
        steps_per_control = int(self.config.world_rate / self.config.control_rate)

        logger.info("════════════════════════════════════════════════════════════════")
        logger.info(
            f"▶ Starting simulation: {self.config.sim_time}s @ {self.config.world_rate}Hz "
            f"(control @ {self.config.control_rate}Hz)"
        )
        logger.info("════════════════════════════════════════════════════════════════")

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
