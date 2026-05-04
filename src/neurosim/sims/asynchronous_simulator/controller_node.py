"""
Controller Node - Async controller with trajectory generation and control.

This node handles:
- Trajectory generation (minsnap, polynomial, etc.)
- Control computation (SE3, etc.)
- Receiving state from simulator
- Publishing control commands
"""

import yaml
import time
import logging
from pathlib import Path
from collections import defaultdict

import cortex
from cortex.core.node import Node
from cortex.discovery.daemon import DEFAULT_DISCOVERY_ADDRESS
from cortex.messages.standard import DictMessage, MultiArrayMessage

from neurosim.core.visual_backend import HabitatWrapper
from neurosim.core.control import create_controller
from neurosim.core.trajectory import create_trajectory
from neurosim.core.coord_trans import CoordinateTransform
from neurosim.sims.asynchronous_simulator.cortex_io import (
    CONTROL_TOPIC,
    STATE_TOPIC,
    SUBSCRIBE_DEFAULTS,
    ensure_discovery_daemon,
    sensor_topics_from_settings,
    state_from_message,
)

logger = logging.getLogger(__name__)


class ControllerNode(Node):
    """
    Async Controller Node.

    Handles trajectory generation and control computation.
    Receives state from simulator and publishes control commands.
    """

    def __init__(
        self,
        settings_path: Path | str,
        discovery_address: str = DEFAULT_DISCOVERY_ADDRESS,
    ):
        """
        Initialize the Controller Node.

        Args:
            settings_path: Path to the settings YAML file
            discovery_address: Cortex discovery daemon address
        """
        ensure_discovery_daemon(discovery_address)
        super().__init__("neurosim_controller", discovery_address=discovery_address)
        self._cpu_clock_start_time = None

        self.settings_path = Path(settings_path)
        if not self.settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {self.settings_path}")

        # Load settings
        with open(self.settings_path, "r") as f:
            self.settings = yaml.safe_load(f)

        # Parse configuration
        sim_cfg = self.settings.get("simulator", {})
        self.control_rate = sim_cfg["control_rate"]
        coord_transform = sim_cfg.get("coord_transform", "rotorpy_to_hm3d")

        # State tracking
        self.state = None
        self.time = 0.0

        # Initialize coordinate transform for trajectory
        self.coord_trans = CoordinateTransform(coord_transform)

        # TODO: Visual backend instantiation shouldn't be necessary here in the future.
        # TODO: Ideally, we want to move to a more general navmesh handler and path planner
        # TODO: (e.g., GCOPTER with Recast/Detour) that doesn't require the full visual backend.
        # For now, we need it to extract the pathfinder for trajectory generation.
        logger.info("Initializing visual backend to extract pathfinder...")
        visual_backend = HabitatWrapper(self.settings["visual_backend"])
        # Note: We keep the visual backend alive as it owns the pathfinder
        self._visual_backend = visual_backend

        # Initialize components
        self._init_controller()
        self._init_trajectory()
        self._init_cortex_io()
        self._init_executors()

        # Stats tracking
        self._stats = defaultdict(int)

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ ControllerNode initialized successfully")
        logger.info(f"   Control rate: {self.control_rate} Hz")
        logger.info(f"   Discovery: {self.discovery_address}")
        logger.info("═══════════════════════════════════════════════════════════")

    def _init_controller(self) -> None:
        """Initialize the controller."""
        self.controller = create_controller(**self.settings.get("controller", {}))
        logger.info(f"Controller initialized: {self.controller.__class__.__name__}")

    def _init_trajectory(self) -> None:
        """Initialize the trajectory generator."""
        traj_kwargs = self.settings.get("trajectory", {}).copy()

        # Add pathfinder and coordinate transform for habitat-based trajectories
        traj_kwargs["pathfinder"] = self._visual_backend._sim.pathfinder
        traj_kwargs["coord_transform"] = self.coord_trans.inverse_transform_batch

        self.trajectory = create_trajectory(**traj_kwargs)
        logger.info(f"Trajectory initialized: {self.trajectory.__class__.__name__}")

    def _init_cortex_io(self) -> None:
        """Initialize Cortex publishers and subscribers."""
        self.control_pub = self.create_publisher(
            CONTROL_TOPIC,
            DictMessage,
            queue_size=1000,
        )
        if not self.control_pub.is_registered:
            raise RuntimeError(f"Failed to register Cortex topic: {CONTROL_TOPIC}")

        self.create_subscriber(
            STATE_TOPIC,
            DictMessage,
            callback=self.receive_state,
            **SUBSCRIBE_DEFAULTS,
        )

        for topic, _uuid in sensor_topics_from_settings(self.settings)["events"]:
            self.create_subscriber(
                topic,
                MultiArrayMessage,
                callback=self.receive_events,
                **SUBSCRIBE_DEFAULTS,
            )

    def _init_executors(self) -> None:
        """Initialize async executors."""
        # Control loop at control rate
        self.create_timer(1.0 / self.control_rate, self.compute_and_publish_control)

        # Stats printer
        self.create_timer(1.0, self.print_stats)

    async def receive_state(self, msg: DictMessage, _header) -> None:
        """Receive state from simulator and update internal state."""
        if msg.data is not None:
            self.state = state_from_message(msg.data)
            self._stats["received_state"] += 1

    async def receive_events(self, msg: MultiArrayMessage, _header) -> None:
        """Receive event packets (placeholder for event-driven control)."""
        if msg.arrays is not None:
            self._stats["received_event_packets"] += 1

    async def compute_and_publish_control(self) -> None:
        """Compute control command and publish to simulator."""
        if self.state is None:
            logger.debug("State not yet received, skipping control computation.")
            return

        self.time = time.perf_counter() - self._cpu_clock_start_time
        # OR self.time += 1.0 / self.control_rate.
        # The uncommented version shows the real time behavior of the simulator.

        # Update trajectory reference
        flat = self.trajectory.update(self.time)

        # Compute control command
        control = self.controller.update(self.time, self.state, flat)

        # Publish control command
        control_msg = {
            "cmd_motor_speeds": control["cmd_motor_speeds"].tolist(),
            "timestamp": self.time,
        }

        if self.control_pub.publish(DictMessage(data=control_msg)):
            self._stats["published_control"] += 1

    async def print_stats(self) -> None:
        """Print statistics."""
        if self._cpu_clock_start_time is None:
            return

        elapsed = time.perf_counter() - self._cpu_clock_start_time
        logger.info("─" * 50)
        logger.info(
            f"[ControllerNode] Elapsed: {elapsed:.2f}s | Sim time: {self.time:.2f}s"
        )
        for key, value in sorted(self._stats.items()):
            logger.info(f"  {key}: {value / elapsed:.1f}/s")
        logger.info("─" * 50)

    async def run(self) -> None:
        """Run the Cortex controller node."""
        self._cpu_clock_start_time = time.perf_counter()
        await super().run()

    async def close(self) -> None:
        """Close Cortex resources and the controller visual backend."""
        await super().close()
        if hasattr(self, "_visual_backend"):
            self._visual_backend.close()


async def main():
    """Main entry point for controller node."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the async controller node.")
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

    async with ControllerNode(
        settings_path=args.settings,
        discovery_address=args.discovery_address,
    ) as node:
        try:
            await node.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")


if __name__ == "__main__":
    cortex.run(main())
