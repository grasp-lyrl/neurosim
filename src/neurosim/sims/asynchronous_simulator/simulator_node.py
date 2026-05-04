"""
Simulator Node - Async simulator with dynamics, sensors and visual backend.

This node handles:
- Dynamics simulation (vehicle physics)
- Visual backend (Habitat rendering)
- Sensor simulation (events, color, depth, IMU)
- Publishing state and sensor data
- Receiving control commands
"""

import yaml
import time
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

import cortex
from cortex.core.node import Node
from cortex.discovery.daemon import DEFAULT_DISCOVERY_ADDRESS
from cortex.messages.standard import ArrayMessage, DictMessage, MultiArrayMessage

from neurosim.core.visual_backend import create_visual_backend
from neurosim.core.dynamics import create_dynamics
from neurosim.core.imu_sim import create_imu_sensor
from neurosim.core.coord_trans import CoordinateTransform
from neurosim.core.utils import SimulationConfig, SensorConfig, EventBuffer
from neurosim.sims.synchronous_simulator import SynchronousSimulator
from neurosim.sims.asynchronous_simulator.cortex_io import (
    CONTROL_TOPIC,
    SENSOR_TOPIC_TYPES,
    STATE_TOPIC,
    SUBSCRIBE_DEFAULTS,
    ensure_discovery_daemon,
    message_type_for_sensor,
    sensor_frame_id,
    sensor_topic,
)

logger = logging.getLogger(__name__)


class SimulatorNode(Node):
    """
    Async Simulator Node.

    Handles dynamics simulation, visual backend rendering, and sensor data publishing.
    Receives control commands from the controller node.
    """

    MAX_EVENT_BUFFER_SIZE = 4000000

    def __init__(
        self,
        settings: Path | str | dict,
        discovery_address: str = DEFAULT_DISCOVERY_ADDRESS,
    ):
        """
        Initialize the Simulator Node.

        Args:
            settings: Settings dictionary or path to the settings YAML file
            discovery_address: Cortex discovery daemon address
        """
        ensure_discovery_daemon(discovery_address)
        super().__init__("neurosim_simulator", discovery_address=discovery_address)
        self._cpu_clock_start_time = None

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
        self.config = SimulationConfig(
            **self.settings.get("simulator", {}),
            visual_sensors=self.settings.get("visual_backend", {}).get("sensors", {}),
        )
        self.sensor_manager = self.config.sensor_manager

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

        # Initialize coordinate transform from config
        self.coord_trans = CoordinateTransform(self.config.coord_transform)

        ######################################
        # Initialize backends and components #
        ######################################

        # Initializes the visual backend and binds visual sensor executors
        self._init_visual_backend()
        self._init_visual_sensors()

        # Initialize dynamics
        self._init_dynamics()

        # Initialize additional sensors like IMU and bind their executors
        self._init_additional_sensors()

        # Initialize components
        self._init_cortex_io()
        self._init_executors()

        # Stats tracking per sensor UUID
        self._stats = defaultdict(int)

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ SimulatorNode initialized successfully")
        logger.info(f"   World rate: {self.config.world_rate} Hz")
        logger.info(f"   Control rate: {self.config.control_rate} Hz")
        logger.info(f"   Discovery: {self.discovery_address}")
        logger.info("═══════════════════════════════════════════════════════════")

    def _init_visual_backend(self) -> None:
        """Initialize the visual backend (Habitat or CARLA)."""
        self.visual_backend = create_visual_backend(self.settings["visual_backend"])

    def _init_visual_sensors(self) -> None:
        """Bind visual sensor executors using the synchronous simulator factory."""
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
            if sensor_type in {
                "event",
                "color",
                "semantic",
                "depth",
                "optical_flow",
                "corner",
                "edge",
                "grayscale",
            }:
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
                    statedot_provider=lambda: (
                        self.dynamics.statedot()
                    ),  # Lazy evaluation
                )
            else:
                raise ValueError(f"Unsupported additional sensor type: {sensor_type}")

            self.config.sensor_manager.add_executor(uuid, executor)

    def _init_cortex_io(self) -> None:
        """Initialize Cortex publishers and subscribers."""
        self.state_pub = self.create_publisher(
            STATE_TOPIC, DictMessage, queue_size=1000
        )
        self.sensor_publishers = {}

        for _, sensor in self.sensor_manager.sensors.items():
            msg_type = message_type_for_sensor(sensor.sensor_type)
            topic_type = SENSOR_TOPIC_TYPES.get(sensor.sensor_type)
            if msg_type is None or topic_type is None:
                continue

            self.sensor_publishers[sensor.uuid] = self.create_publisher(
                sensor_topic(topic_type, sensor.uuid),
                msg_type,
                queue_size=1000,
            )

        publishers = [self.state_pub, *self.sensor_publishers.values()]
        unregistered = [pub.topic_name for pub in publishers if not pub.is_registered]
        if unregistered:
            raise RuntimeError(
                "Failed to register Cortex simulator topic(s): "
                + ", ".join(unregistered)
            )

        self.create_subscriber(
            CONTROL_TOPIC,
            DictMessage,
            callback=self.receive_control,
            **SUBSCRIBE_DEFAULTS,
        )

    def _init_executors(self) -> None:
        """Initialize async executors."""
        # Main simulation loop at world rate
        self.create_timer(1.0 / self.config.world_rate, self.simulate_step)

        # Publish state at control rate
        self.create_timer(1.0 / self.config.control_rate, self.publish_state)

        # Create sensor publishing executors at viz rates
        for _, sensor in self.sensor_manager.sensors.items():
            if sensor.uuid in self.sensor_publishers:
                self.create_timer(
                    1.0 / sensor.viz_rate,
                    lambda s=sensor: self.publish_sensor(s),
                )

        # Stats printer
        self.create_timer(1.0, self.print_stats)

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
        self.visual_backend.update_dynamic_obstacles(
            sim_time=self.time, dt=self.config.t_step
        )

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

    @staticmethod
    def _to_numpy(measurement) -> np.ndarray:
        """Convert sensor measurements to numpy before Cortex serialization."""
        if hasattr(measurement, "detach"):
            measurement = measurement.detach()
        if hasattr(measurement, "cpu"):
            measurement = measurement.cpu()
        if hasattr(measurement, "numpy"):
            return measurement.numpy()
        return np.asarray(measurement)

    def _sensor_frame_id(self, uuid: str) -> str:
        return sensor_frame_id(uuid, self.time, self.simsteps)

    def _corner_to_dict(self, uuid: str, measurement) -> dict:
        """Serialize FeatureDetectionResult-style corner measurements."""
        return {
            "uuid": uuid,
            "timestamp": self.time,
            "simsteps": self.simsteps,
            "keypoints": self._to_numpy(measurement.keypoints),
            "scores": self._to_numpy(measurement.scores),
            "descriptors": None
            if measurement.descriptors is None
            else self._to_numpy(measurement.descriptors),
            "sizes": self._to_numpy(measurement.sizes),
            "angles": self._to_numpy(measurement.angles),
            "octaves": self._to_numpy(measurement.octaves),
            "num_keypoints": int(measurement.num_keypoints),
            "detector_name": measurement.detector_name,
            "descriptor_name": measurement.descriptor_name,
        }

    async def receive_control(self, msg: DictMessage, _header) -> None:
        """Receive control commands from controller."""
        data = msg.data
        if data and "cmd_motor_speeds" in data:
            self.control = {"cmd_motor_speeds": np.array(data["cmd_motor_speeds"])}
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

        if self.state_pub.publish(DictMessage(data=state_msg)):
            self._stats["published_state"] += 1

    async def publish_sensor(self, sensor: SensorConfig) -> None:
        """Publish the latest measurement for any supported sensor type."""
        uuid = sensor.uuid
        sensor_type = sensor.sensor_type

        if sensor_type == "event":
            events_dict = self.event_buffers[uuid].get_and_clear()
            if events_dict is None:
                return
            message = MultiArrayMessage(
                arrays=events_dict,
                frame_id=self._sensor_frame_id(uuid),
            )
        else:
            if uuid not in self.measurements:
                return
            measurement = self.measurements[uuid]
            if sensor_type == "imu":
                message = DictMessage(
                    data={
                        "uuid": uuid,
                        "accel": self._to_numpy(measurement["accel"]),
                        "gyro": self._to_numpy(measurement["gyro"]),
                        "timestamp": self.time,
                        "simsteps": self.simsteps,
                    }
                )
            elif sensor_type == "corner":
                message = DictMessage(data=self._corner_to_dict(uuid, measurement))
            else:
                message = ArrayMessage(
                    data=self._to_numpy(measurement),
                    name=uuid,
                    frame_id=self._sensor_frame_id(uuid),
                )

        if self.sensor_publishers[uuid].publish(message):
            self._stats[f"published_{uuid}"] += 1

    async def publish_imu(self, sensor: SensorConfig) -> None:
        """Publish IMU sensor data from stored measurements."""
        await self.publish_sensor(sensor)

    async def publish_color(self, sensor: SensorConfig) -> None:
        """Publish color camera data from stored measurements."""
        await self.publish_sensor(sensor)

    async def publish_depth(self, sensor: SensorConfig) -> None:
        """Publish depth camera data from stored measurements."""
        await self.publish_sensor(sensor)

    async def publish_events(self, sensor: SensorConfig) -> None:
        """Publish event camera data from pre-allocated buffer."""
        await self.publish_sensor(sensor)

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

    async def run(self) -> None:
        """Run the Cortex simulator node."""
        self._cpu_clock_start_time = time.perf_counter()
        await super().run()

    async def close(self) -> None:
        """Close Cortex resources and simulator backends."""
        await super().close()
        if hasattr(self, "visual_backend"):
            self.visual_backend.close()


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
        "--discovery-address",
        type=str,
        default=DEFAULT_DISCOVERY_ADDRESS,
        help=(
            "Cortex discovery daemon address. Start discovery first with "
            "`cortex-discovery`."
        ),
    )
    args = parser.parse_args()

    async with SimulatorNode(
        settings=args.settings,
        discovery_address=args.discovery_address,
    ) as node:
        try:
            await node.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")


if __name__ == "__main__":
    cortex.run(main())
