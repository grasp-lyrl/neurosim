import torch
import logging
import numpy as np
from dataclasses import dataclass, field


try:
    import rerun as rr

    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False

logger = logging.getLogger(__name__)


@dataclass
class EventBuffer:
    """Pre-allocated buffer for fast event camera data accumulation."""

    max_size: int = 2000000
    use_gpu: bool = True

    def __post_init__(self):
        """Initialize pre-allocated buffers (GPU tensors or numpy arrays)."""
        if self.use_gpu:
            # Keep buffers on GPU to avoid unnecessary CPU transfers
            self.x = torch.empty((self.max_size,), dtype=torch.uint16, device="cuda")
            self.y = torch.empty((self.max_size,), dtype=torch.uint16, device="cuda")
            self.t = torch.empty((self.max_size,), dtype=torch.uint64, device="cuda")
            self.p = torch.empty((self.max_size,), dtype=torch.uint8, device="cuda")
        else:
            # Use numpy arrays for CPU-only operation
            self.x = np.empty((self.max_size,), dtype=np.uint16)
            self.y = np.empty((self.max_size,), dtype=np.uint16)
            self.t = np.empty((self.max_size,), dtype=np.uint64)
            self.p = np.empty((self.max_size,), dtype=np.uint8)
        self.size = 0

    def append(self, events):
        """Append events to buffer. events is (x, y, t, p) tuple of tensors."""
        x, y, t, p = events
        n_events = x.shape[0]

        new_size = self.size + n_events
        if new_size > self.max_size:
            logger.warning(
                f"Event buffer overflow: {new_size}/{self.max_size}. Dropping events."
            )
            return False

        # Assume events are already on correct device (GPU or CPU)
        self.x[self.size : new_size] = x
        self.y[self.size : new_size] = y
        self.t[self.size : new_size] = t
        self.p[self.size : new_size] = p

        self.size = new_size
        return True

    def get_and_clear(self):
        """Get current events as dict of numpy arrays and clear buffer."""
        if self.size == 0:
            return None

        _size = self.size  # Local copy for thread safety

        if self.use_gpu:
            # Convert GPU tensors to numpy only when publishing
            events_dict = {
                "x": self.x[:_size].cpu().numpy(),
                "y": self.y[:_size].cpu().numpy(),
                "t": self.t[:_size].cpu().numpy(),
                "p": self.p[:_size].cpu().numpy(),
            }
        else:
            # Already numpy arrays
            events_dict = {
                "x": self.x[:_size],
                "y": self.y[:_size],
                "t": self.t[:_size],
                "p": self.p[:_size],
            }

        self.size = 0
        return events_dict


@dataclass
class EventVisualizationState:
    """Visualization state for an event sensor.

    Supports both CPU (numpy) and GPU (torch) buffers for efficient event accumulation.
    """

    uuid: str
    width: int
    height: int
    device: str = "cuda:0"
    use_gpu: bool = True  # Whether to use GPU buffer
    buffer: torch.Tensor | np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        if self.use_gpu:
            try:
                # Use GPU buffer for faster accumulation
                self.buffer = torch.zeros(
                    (self.height, self.width, 3), dtype=torch.uint8, device=self.device
                )
                logger.debug(
                    f"Event sensor {self.uuid} using GPU buffer on {self.device}"
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ Could not initialize GPU buffer for sensor {self.uuid}: {e}"
                )
                # Fallback to CPU buffer
                self.buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.use_gpu = False
                logger.debug(f"Event sensor {self.uuid} using CPU buffer")
        else:
            # Use CPU buffer
            self.buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            logger.debug(f"Event sensor {self.uuid} using CPU buffer")

    def accumulate(self, events: tuple[torch.Tensor | np.ndarray, ...]) -> None:
        """Accumulate events into the visualization buffer.

        Args:
            events: Tuple of (x, y, t, p) event arrays. Can be GPU or CPU tensors.
        """
        if events is None or len(events[0]) == 0:
            return

        x, y, _, p = events

        if self.use_gpu:
            # Ensure indices are integer tensors for GPU buffer
            x = x.to(torch.int32)
            y = y.to(torch.int32)
            p = p.to(torch.int32)
        else:
            # CPU-based accumulation
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
                y = y.cpu().numpy()
                p = p.cpu().numpy()

        self.buffer[y, x, p * 2] = 255

    def reset(self) -> None:
        """Reset the visualization buffer."""
        if self.use_gpu:
            self.buffer.zero_()
        else:
            self.buffer.fill(0)

    def get_image(self) -> np.ndarray:
        """Get the current visualization buffer as numpy array (for Rerun)."""
        if self.use_gpu:
            # Transfer from GPU only when needed for visualization
            return self.buffer.cpu().numpy()
        else:
            return self.buffer


class RerunVisualizer:
    """Handles Rerun visualization for the simulator.

    Optimized for GPU-based sensors to minimize CPU-GPU transfers.
    Only transfers data to CPU when visualization is actually performed.
    """

    def __init__(self, config, use_gpu: bool = True, device: str = "cuda:0"):
        """
        Initialize the Rerun visualizer.

        Args:
            config: Simulation configuration
            use_gpu: Use GPU buffers for event accumulation (default: True)
        """
        self.config = config
        self.enabled = False
        self.use_gpu = use_gpu
        self.device = device

        if use_gpu:
            assert self.device.startswith("cuda"), "Device must be a CUDA device string"
            if not torch.cuda.is_available():
                logger.warning(
                    "⚠️ GPU requested for visualization but not available. Falling back to CPU."
                )
                self.use_gpu = False
                self.device = "cpu"

        # Create event visualization states for each event sensor
        self.event_viz_states: dict[str, EventVisualizationState] = {}
        for sensor in config.sensor_manager.get_sensors_by_type("event"):
            sensor_cfg = config.visual_sensors[sensor.uuid]
            self.event_viz_states[sensor.uuid] = EventVisualizationState(
                uuid=sensor.uuid,
                width=sensor_cfg["width"],
                height=sensor_cfg["height"],
                device=self.device,
                use_gpu=self.use_gpu,
            )

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info(
            f"🎬 RerunVisualizer ready: {len(self.event_viz_states)} event sensor(s) "
            f"configured ({'GPU' if self.use_gpu else 'CPU'} mode)"
        )
        logger.info("═══════════════════════════════════════════════════════════")

    def initialize(self, memory_limit: str = "10%") -> None:
        """Initialize Rerun recording."""
        if not HAS_RERUN:
            raise ImportError("Rerun package is not installed.")

        rr.init("neurosim")
        rr.spawn(memory_limit=memory_limit)
        self.enabled = True
        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("🎬 RerunVisualizer started")
        logger.info("═══════════════════════════════════════════════════════════")

    def log_measurements(self, measurements: dict, time: float, simsteps: int) -> None:
        """
        Log sensor measurements to Rerun.

        Optimized to minimize GPU->CPU transfers. Data is only transferred
        when visualization is actually needed.

        Args:
            measurements: Dictionary of sensor measurements by UUID (can be GPU tensors)
            time: Current simulation time
            simsteps: Current simulation step
        """
        if not self.enabled:
            return

        rr.set_time("sim_time", timestamp=time)

        for uuid, measurement in measurements.items():
            sensor_cfg = self.config.sensor_manager.get_sensor_config(uuid)
            sensor_type = sensor_cfg.sensor_type

            # For event sensors, always accumulate (stays on GPU if possible)
            if sensor_type == "event":
                self.event_viz_states[uuid].accumulate(measurement)

            # Check if we should visualize this sensor at this step
            if not self.config.sensor_manager.should_visualize(uuid, simsteps):
                continue

            # Only transfer to CPU when we actually need to visualize
            if sensor_type == "event":
                # Transfer from GPU and log visualization
                rr.log(
                    f"sensors/{uuid}/events",
                    rr.Image(self.event_viz_states[uuid].get_image()),
                )
                # Reset buffer after visualization
                self.event_viz_states[uuid].reset()

            elif sensor_type == "color":
                # Transfer from GPU only for visualization
                if hasattr(measurement, "cpu"):
                    measurement = measurement.cpu().numpy()
                rr.log(f"sensors/{uuid}/color", rr.Image(measurement))

            elif sensor_type == "depth":
                # Transfer from GPU only for visualization
                if hasattr(measurement, "cpu"):
                    measurement = measurement.cpu().numpy()
                rr.log(f"sensors/{uuid}/depth", rr.DepthImage(measurement))

            elif sensor_type == "imu":
                # IMU data is typically small, already on CPU
                rr.log(f"sensors/{uuid}/accel", rr.Scalars(measurement["accel"]))
                rr.log(f"sensors/{uuid}/gyro", rr.Scalars(measurement["gyro"]))

            elif sensor_type == "navmesh":
                # Navmesh is already a numpy array on CPU
                rr.log(f"sensors/{uuid}/navmesh", rr.Image(measurement))

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
