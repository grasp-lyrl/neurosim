import logging
from dataclasses import dataclass, field

import numpy as np

try:
    import rerun as rr

    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False

logger = logging.getLogger(__name__)


@dataclass
class EventVisualizationState:
    """Visualization state for an event sensor."""

    uuid: str
    width: int
    height: int
    buffer: np.ndarray = field(init=False)

    def __post_init__(self):
        self.buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def accumulate(self, events: tuple) -> None:
        """Accumulate events into the visualization buffer."""
        if events is not None and len(events[0]) > 0:
            x, y, t, p = events
            self.buffer[y, x, p * 2] = 255

    def reset(self) -> None:
        """Reset the visualization buffer."""
        self.buffer.fill(0)

    def get_image(self) -> np.ndarray:
        """Get the current visualization buffer."""
        return self.buffer


class RerunVisualizer:
    """Handles Rerun visualization for the simulator."""

    def __init__(self, config):
        """
        Initialize the Rerun visualizer.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self.enabled = False

        # Create event visualization states for each event sensor
        self.event_viz_states: dict[str, EventVisualizationState] = {}
        for sensor in config.sensor_manager.get_sensors_by_type("event"):
            sensor_cfg = config.visual_sensors[sensor.uuid]
            self.event_viz_states[sensor.uuid] = EventVisualizationState(
                uuid=sensor.uuid, width=sensor_cfg["width"], height=sensor_cfg["height"]
            )

        logger.info(
            f"RerunVisualizer initialized with {len(self.event_viz_states)} event sensors and friends"
        )

    def initialize(self) -> None:
        """Initialize Rerun recording."""
        if not HAS_RERUN:
            raise ImportError("Rerun package is not installed.")

        rr.init("neurosim", spawn=True)
        self.enabled = True
        logger.info("🎬 Rerun visualization started")

    def log_measurements(self, measurements: dict, time: float, simsteps: int) -> None:
        """
        Log sensor measurements to Rerun.

        Args:
            measurements: Dictionary of sensor measurements by UUID
            time: Current simulation time
            simsteps: Current simulation step
        """
        if not self.enabled:
            return

        rr.set_time("sim_time", timestamp=time)

        for uuid, measurement in measurements.items():
            sensor_cfg = self.config.sensor_manager.get_sensor_config(uuid)
            sensor_type = sensor_cfg.sensor_type

            # For event sensors, always accumulate
            if sensor_type == "event":
                self.event_viz_states[uuid].accumulate(measurement)

            # Check if we should visualize this sensor at this step
            if not self.config.sensor_manager.should_visualize(uuid, simsteps):
                continue

            if sensor_type == "event":
                # Accumulate events and log the visualization
                rr.log(
                    f"sensors/{uuid}/events",
                    rr.Image(self.event_viz_states[uuid].get_image()),
                )
                # Reset buffer after visualization
                self.event_viz_states[uuid].reset()

            elif sensor_type == "color":
                rr.log(f"sensors/{uuid}/color", rr.Image(measurement))

            elif sensor_type == "depth":
                rr.log(f"sensors/{uuid}/depth", rr.DepthImage(measurement))

            elif sensor_type == "imu":
                rr.log(f"sensors/{uuid}/accel", rr.Scalars(measurement["accel"]))
                rr.log(f"sensors/{uuid}/gyro", rr.Scalars(measurement["gyro"]))

    def log_state(self, state: dict) -> None:
        """Log vehicle state to Rerun."""
        if not self.enabled:
            return

        rr.log(
            "navigation/pose",
            rr.Transform3D(
                translation=state["x"],
                rotation=rr.Quaternion(xyzw=state["q"]),
                axis_length=1.0,
                relation=rr.TransformRelation.ParentFromChild,
            ),
        )
        rr.log("navigation/trajectory", rr.Points3D(positions=state["x"][None, :]))
