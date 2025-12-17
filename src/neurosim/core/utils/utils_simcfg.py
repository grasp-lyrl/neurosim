"""Utility classes and functions for simulation configuration.

Used in all simulator types: asynchronous, synchronous, and batched."""

import logging
from typing import Any, Callable
from dataclasses import dataclass, field

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
