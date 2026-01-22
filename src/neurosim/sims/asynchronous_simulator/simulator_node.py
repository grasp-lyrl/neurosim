"""
Simulator Node - Async simulator with dynamics, sensors and visual backend.

This node handles:
- Dynamics simulation (vehicle physics)
- Visual backend (Habitat rendering)
- Sensor simulation (events, color, depth, IMU)
- Publishing state and sensor data
- Receiving control commands
"""

import zmq
import yaml
import time
import logging
import asyncio
import numpy as np
from pathlib import Path
from collections import defaultdict

from neurosim.cortex.utils import ZMQNODE
from neurosim.core.visual_backend import HabitatWrapper
from neurosim.core.dynamics import create_dynamics
from neurosim.core.imu_sim import create_imu_sensor
from neurosim.core.coord_trans import CoordinateTransform
from neurosim.core.utils import SimulationConfig, SensorConfig, EventBuffer
from neurosim.sims.synchronous_simulator import SynchronousSimulator

logger = logging.getLogger(__name__)


class SimulatorNode(ZMQNODE):
    """
    Async Simulator Node.

    Handles dynamics simulation, visual backend rendering, and sensor data publishing.
    Receives control commands from the controller node.
    """

    MAX_EVENT_BUFFER_SIZE = 4000000

    def __init__(
        self,
        settings: Path | str | dict,
        ipc_pub_addr: str = "ipc:///tmp/neurosim_sim_pub",
        ipc_sub_addr: str = "ipc:///tmp/neurosim_ctrl_pub",
    ):
        """
        Initialize the Simulator Node.

        Args:
            settings_path: Path to the settings YAML file
            ipc_pub_addr: ZMQ address for publishing state/sensor data
            ipc_sub_addr: ZMQ address for subscribing to control commands
        """
        super().__init__()

        if isinstance(settings, (str, Path)):
            settings_path = Path(settings)
            if not settings_path.exists():
                raise FileNotFoundError(f"Settings file not found: {settings_path}")

            # Load settings
            with open(settings_path, "r") as f:
                self.settings = yaml.safe_load(f)
        elif isinstance(settings, dict):
            self.settings = settings
        else:
            raise TypeError("settings must be a dict or a path to a YAML file.")

        # Initialize simulation configuration (includes sensor manager)
        sim_cfg = self.settings.get("simulator", {})
        self.config = SimulationConfig(
            world_rate=sim_cfg["world_rate"],
            control_rate=sim_cfg["control_rate"],  # sends state at control rate
            sim_time=sim_cfg["sim_time"],
            sensor_rates=sim_cfg.get("sensor_rates", {}),
            viz_rates=sim_cfg.get("viz_rates", {}),  # Optional visualization rates
            visual_sensors=self.settings.get("visual_backend", {}).get("sensors", {}),
            additional_sensors=sim_cfg.get("additional_sensors", {}),
        )
        self.sensor_manager = self.config.sensor_manager

        #################
        # IPC addresses #
        #################
        self.ipc_pub_addr = ipc_pub_addr
        self.ipc_sub_addr = ipc_sub_addr

        ####################
        # Simulation state #
        ####################
        self.time = 0.0
        self.simsteps = 0
        self.control = None  # To be received from controller

        # Storage for sensor measurements
        self.measurements = {}  # Stores latest sensor measurements

        # Event camera buffers (pre-allocated GPU tensor buffers)
        self.event_buffers: dict[str, EventBuffer] = {}
        for sensor in self.config.sensor_manager.get_sensors_by_type("event"):
            self.event_buffers[sensor.uuid] = EventBuffer(
                max_size=self.MAX_EVENT_BUFFER_SIZE,
                use_gpu=True,  # Keep events on GPU until publish time
            )

        # Initialize coordinate transform (rotorpy to habitat)
        self.coord_trans = CoordinateTransform("rotorpy_to_habitat")

        ######################################
        # Initialize backends and components #
        ######################################

        # Initializes the visual backend and binds visual sensor executors
        self._init_visual_backend()

        # Initialize dynamics
        self._init_dynamics()

        # Initialize additional sensors like IMU and bind their executors
        self._init_additional_sensors()

        # Initialize components
        self._init_sockets()
        self._init_executors()

        # Stats tracking per sensor UUID
        self._stats = defaultdict(int)

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ SimulatorNode initialized successfully")
        logger.info(f"   World rate: {self.config.world_rate} Hz")
        logger.info(f"   Control rate: {self.config.control_rate} Hz")
        logger.info(f"   Publishing to: {self.ipc_pub_addr}")
        logger.info(f"   Subscribing to: {self.ipc_sub_addr}")
        logger.info("═══════════════════════════════════════════════════════════")

    def _init_visual_backend(self) -> None:
        """Initialize the visual backend (Habitat)."""
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
            if sensor_type == "event":
                executor_kwargs["time_provider"] = lambda: self.time  # Lazy evaluation

            elif sensor_type == "navmesh":
                executor_kwargs["meters_per_pixel"] = sensor_cfg.get(
                    "meters_per_pixel", 0.1
                )
                executor_kwargs["height"] = sensor_cfg.get("height", None)

            executor = SynchronousSimulator._create_sensor_executor(**executor_kwargs)
            self.config.sensor_manager.add_executor(uuid, executor)

    def _init_dynamics(self) -> None:
        """Initialize the dynamics model."""
        self.dynamics = create_dynamics(**self.settings["dynamics"])
        self.control = {
            # Initialize with hover command
            "cmd_motor_speeds": self.dynamics.state["rotor_speeds"]
        }

    def _init_additional_sensors(self) -> None:
        """Initialize additional sensors like IMU."""

        for uuid, sensor_cfg in self.config.additional_sensors.items():
            sensor_type = sensor_cfg.get("type")

            if sensor_type == "imu":
                sensor = create_imu_sensor(**sensor_cfg)

                executor = SynchronousSimulator._create_sensor_executor(
                    sensor_type=sensor_type,
                    sensor=sensor,
                    state_provider=lambda: self.dynamics.state,  # Lazy evaluation
                    statedot_provider=lambda: self.dynamics.statedot(),  # Lazy evaluation
                )
            else:
                raise ValueError(f"Unsupported additional sensor type: {sensor_type}")

            self.config.sensor_manager.add_executor(uuid, executor)

    def _init_sockets(self) -> None:
        """Initialize ZMQ sockets."""
        # Publisher socket for state and sensor data
        self.socket_pub = self.create_socket(
            zmq.PUB,
            self.ipc_pub_addr,
            setsockopt={
                zmq.SNDHWM: 1000,
                zmq.LINGER: 0,
                zmq.IMMEDIATE: 1,
            },
        )

        # Subscriber socket for control commands
        self.socket_control = self.create_socket(
            zmq.SUB,
            self.ipc_sub_addr,
            setsockopt={
                zmq.SUBSCRIBE: b"control",
            },
        )

    def _init_executors(self) -> None:
        """Initialize async executors."""
        # Main simulation loop at world rate
        self.create_constant_rate_executor(self.simulate_step, self.config.world_rate)

        # Publish state at control rate
        self.create_constant_rate_executor(self.publish_state, self.config.control_rate)

        # Create sensor publishing executors at viz rates
        for _, sensor in self.sensor_manager.sensors.items():
            sensor_type = sensor.sensor_type

            if sensor_type == "event":
                self.create_constant_rate_executor(
                    lambda s=sensor: self.publish_events(s),
                    sensor.viz_rate,
                )
            elif sensor_type == "imu":
                self.create_constant_rate_executor(
                    lambda s=sensor: self.publish_imu(s),
                    sensor.viz_rate,
                )
            elif sensor_type == "color":
                self.create_constant_rate_executor(
                    lambda s=sensor: self.publish_color(s),
                    sensor.viz_rate,
                )
            elif sensor_type == "depth":
                self.create_constant_rate_executor(
                    lambda s=sensor: self.publish_depth(s),
                    sensor.viz_rate,
                )

        # Subscriber for control
        self.create_async_executor(self.receive_control)

        # Stats printer
        self.create_constant_rate_executor(self.print_stats, 1)

    async def simulate_step(self) -> None:
        """Execute one simulation step: step dynamics AND render sensors at sampling rate."""
        self.time += self.config.t_step
        self.simsteps += 1

        # Update dynamics
        self.dynamics.step(self.control, self.config.t_step)

        # Transform state from dynamics coordinate system to visual backend
        position, quaternion = self.coord_trans.transform(
            self.dynamics.state["x"], self.dynamics.state["q"]
        )
        self.visual_backend.update_agent_state(position, quaternion)

        # Render sensors at their sampling rate (similar to synchronous simulator)
        self._render_sensors()

        self._stats["sim_steps"] += 1

    def _render_sensors(self) -> None:
        """Render sensors that should be sampled at this timestep and store results."""
        sensor_manager = self.sensor_manager

        for uuid, sensor_cfg in sensor_manager.sensors.items():
            if sensor_manager.should_sample(uuid, self.simsteps):
                sensor_type = sensor_cfg.sensor_type

                if sensor_type == "event":
                    # Events are accumulated in a buffer
                    events = sensor_cfg.executor()
                    if events is not None:
                        self.event_buffers[uuid].append(events)
                        self._stats[f"rendered_{uuid}"] += 1
                else:
                    # Other sensors: store the latest measurement
                    self.measurements[uuid] = sensor_cfg.executor()
                    self._stats[f"rendered_{uuid}"] += 1

    async def receive_control(self) -> None:
        """Receive control commands from controller."""
        _, msg = await self.recv_dict(self.socket_control, copy=False)

        if msg and "cmd_motor_speeds" in msg:
            self.control = {"cmd_motor_speeds": np.array(msg["cmd_motor_speeds"])}
            self._stats["received_controls"] += 1

    async def publish_state(self) -> None:
        """Publish current state."""
        if self.dynamics.state is None:
            return

        state = self.dynamics.state
        state_msg = {
            "x": state["x"].tolist(),
            "q": state["q"].tolist(),
            "v": state["v"].tolist(),
            "w": state["w"].tolist(),
            "timestamp": self.time,
            "simsteps": self.simsteps,
        }

        if self.send_dict(self.socket_pub, state_msg, topic="state", copy=False):
            self._stats["published_state"] += 1

    async def publish_imu(self, sensor: SensorConfig) -> None:
        """Publish IMU sensor data from stored measurements."""
        uuid = sensor.uuid
        if uuid not in self.measurements:
            return

        imu_data = self.measurements[uuid]

        imu_msg = {
            "uuid": uuid,
            "accel": imu_data["accel"].tolist(),
            "gyro": imu_data["gyro"].tolist(),
            "timestamp": self.time,
        }

        if self.send_dict(self.socket_pub, imu_msg, topic=f"imu/{uuid}", copy=False):
            self._stats[f"published_{uuid}"] += 1

    async def publish_color(self, sensor: SensorConfig) -> None:
        """Publish color camera data from stored measurements."""
        uuid = sensor.uuid
        if uuid not in self.measurements:
            return

        if self.send_array(
            self.socket_pub,
            self.measurements[uuid].cpu().numpy(),
            topic=f"color/{uuid}",
        ):
            self._stats[f"published_{uuid}"] += 1

    async def publish_depth(self, sensor: SensorConfig) -> None:
        """Publish depth camera data from stored measurements."""
        uuid = sensor.uuid
        if uuid not in self.measurements:
            return

        if self.send_array(
            self.socket_pub,
            self.measurements[uuid].cpu().numpy(),
            topic=f"depth/{uuid}",
        ):
            self._stats[f"published_{uuid}"] += 1

    async def publish_events(self, sensor: SensorConfig) -> None:
        """Publish event camera data from pre-allocated buffer."""
        uuid = sensor.uuid

        # Get events and clear buffer atomically
        events_dict = self.event_buffers[uuid].get_and_clear()
        if events_dict is None:
            return

        if self.send_dict_of_arrays(
            self.socket_pub, events_dict, topic=f"events/{uuid}", copy=True
        ):
            self._stats[f"published_{uuid}"] += 1

    async def print_stats(self) -> None:
        """Print statistics."""
        if self._cpu_clock_start_time is None:
            return

        elapsed = time.perf_counter() - self._cpu_clock_start_time
        logger.info("─" * 50)
        logger.info(
            f"[SimulatorNode] Elapsed: {elapsed:.2f}s | Sim time: {self.time:.2f}s"
        )
        logger.info(f"  Sim rate: {self._stats['sim_steps'] / elapsed:.1f} Hz")

        # Show per-UUID stats
        for key, value in sorted(self._stats.items()):
            if key != "sim_steps":
                logger.info(f"  {key}: {value / elapsed:.1f}/s")
        logger.info("─" * 50)


async def main():
    """Main entry point for simulator node."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the async simulator node.")
    parser.add_argument(
        "--settings",
        type=Path,
        required=True,
        help="Path to the settings YAML file.",
    )
    parser.add_argument(
        "--ipc-pub-addr",
        type=str,
        default="ipc:///tmp/neurosim_sim_pub",
        help="IPC address for publishing.",
    )
    parser.add_argument(
        "--ipc-sub-addr",
        type=str,
        default="ipc:///tmp/neurosim_ctrl_pub",
        help="IPC address for subscribing to control.",
    )
    args = parser.parse_args()

    node = SimulatorNode(
        settings=args.settings,
        ipc_pub_addr=args.ipc_pub_addr,
        ipc_sub_addr=args.ipc_sub_addr,
    )

    try:
        await node.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        await node.close()


if __name__ == "__main__":
    asyncio.run(main())
