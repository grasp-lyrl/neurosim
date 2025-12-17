"""
Controller Node - Async controller with trajectory generation and control.

This node handles:
- Trajectory generation (minsnap, polynomial, etc.)
- Control computation (SE3, etc.)
- Receiving state from simulator
- Publishing control commands
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
from neurosim.core.control import create_controller
from neurosim.core.trajectory import create_trajectory
from neurosim.core.coord_trans import CoordinateTransform

logger = logging.getLogger(__name__)


class ControllerNode(ZMQNODE):
    """
    Async Controller Node.

    Handles trajectory generation and control computation.
    Receives state from simulator and publishes control commands.
    """

    def __init__(
        self,
        settings_path: Path | str,
        ipc_pub_addr: str = "ipc:///tmp/neurosim_ctrl_pub",
        ipc_sub_addr: str = "ipc:///tmp/neurosim_sim_pub",
    ):
        """
        Initialize the Controller Node.

        Args:
            settings_path: Path to the settings YAML file
            ipc_pub_addr: ZMQ address for publishing control commands
            ipc_sub_addr: ZMQ address for subscribing to state
        """
        super().__init__()

        self.settings_path = Path(settings_path)
        if not self.settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {self.settings_path}")

        # Load settings
        with open(self.settings_path, "r") as f:
            self.settings = yaml.safe_load(f)

        # Parse configuration
        sim_cfg = self.settings.get("simulator", {})
        self.control_rate = sim_cfg["control_rate"]

        # IPC addresses
        self.ipc_pub_addr = ipc_pub_addr
        self.ipc_sub_addr = ipc_sub_addr

        # State tracking
        self.state = None
        self.time = 0.0

        # Initialize coordinate transform for trajectory
        self.coord_trans = CoordinateTransform("rotorpy_to_habitat")

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
        self._init_sockets()
        self._init_executors()

        # Stats tracking
        self._stats = defaultdict(int)

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ ControllerNode initialized successfully")
        logger.info(f"   Control rate: {self.control_rate} Hz")
        logger.info(f"   Publishing to: {self.ipc_pub_addr}")
        logger.info(f"   Subscribing to: {self.ipc_sub_addr}")
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

    def _init_sockets(self) -> None:
        """Initialize ZMQ sockets."""
        # Publisher socket for control commands
        self.socket_pub = self.create_socket(
            zmq.PUB,
            self.ipc_pub_addr,
            setsockopt={
                zmq.SNDHWM: 1000,
                zmq.LINGER: 0,
                zmq.IMMEDIATE: 1,
            },
        )

        # Subscriber socket for state
        self.socket_sub_state = self.create_socket(
            zmq.SUB,
            self.ipc_sub_addr,
            setsockopt={
                zmq.SUBSCRIBE: b"state",
            },
        )

        # Optional: subscribe to events for event-based control
        self.socket_sub_events = self.create_socket(
            zmq.SUB,
            self.ipc_sub_addr,
            setsockopt={
                zmq.SUBSCRIBE: b"events",
            },
        )

    def _init_executors(self) -> None:
        """Initialize async executors."""
        # Control loop at control rate
        self.create_constant_rate_executor(
            self.compute_and_publish_control, self.control_rate
        )

        # State receiver
        self.create_async_executor(self.receive_state)

        # Events receiver (optional, for event-based control)
        self.create_async_executor(self.receive_events)

        # Stats printer
        self.create_constant_rate_executor(self.print_stats, 1)

    async def receive_state(self) -> None:
        """Receive state from simulator and update internal state."""
        _, msg = await self.recv_dict(self.socket_sub_state, copy=False)

        if msg is not None:
            self.state = {
                "x": np.array(msg["x"]),
                "q": np.array(msg["q"]),
                "v": np.array(msg["v"]),
                "w": np.array(msg["w"]),
            }
            self._stats["received_state"] += 1

    async def receive_events(self) -> None:
        """Receive events from simulator.

        This serves as a proxy to demonstrate event-based control capabilities.
        Events can be processed here for advanced control algorithms (e.g., event-based
        visual servoing, direct event feedback, etc.), though currently unused.
        """
        topic, events_dict = await self.recv_dict_of_arrays(
            self.socket_sub_events, copy=False
        )

        if events_dict is not None:
            # Events received successfully - can be processed here for event-based control
            # For example:
            #   x, y, t, p = events_dict["x"], events_dict["y"], events_dict["t"], events_dict["p"]
            #   ... process events for control ...
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

        if self.send_dict(self.socket_pub, control_msg, topic="control", copy=False):
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
        "--ipc-pub-addr",
        type=str,
        default="ipc:///tmp/neurosim_ctrl_pub",
        help="IPC address for publishing control.",
    )
    parser.add_argument(
        "--ipc-sub-addr",
        type=str,
        default="ipc:///tmp/neurosim_sim_pub",
        help="IPC address for subscribing to state.",
    )
    args = parser.parse_args()

    node = ControllerNode(
        settings_path=args.settings,
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
