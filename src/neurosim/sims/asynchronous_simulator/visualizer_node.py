"""
Visualizer Node - Async visualization with Rerun.

This node handles:
- Receiving sensor data from simulator
- Receiving state data from simulator
- Visualizing data with Rerun
"""

import zmq
import time
import logging
import asyncio
import numpy as np
import rerun as rr
from collections import defaultdict

from neurosim.cortex.utils import ZMQNODE
from neurosim.core.utils.utils_viz import EventVisualizationState

logger = logging.getLogger(__name__)


class VisualizerNode(ZMQNODE):
    """
    Async Visualizer Node.

    Receives data from simulator and visualizes with Rerun.
    """

    def __init__(
        self,
        ipc_sub_addr: str = "ipc:///tmp/neurosim_sim_pub",
        spawn: bool = True,
    ):
        """
        Initialize the Visualizer Node.

        Args:
            settings_path: Path to the settings YAML file
            ipc_sub_addr: ZMQ address for subscribing to simulator data
            spawn: Whether to spawn Rerun viewer
        """
        super().__init__()

        # IPC address
        self.ipc_sub_addr = ipc_sub_addr

        # Initialize Rerun
        rr.init("neurosim_async", spawn=spawn)
        rr.set_time("sim_time", timestamp=0)

        # Initialize sockets and executors
        self._init_sockets()
        self._init_executors()

        # Stats tracking
        self._stats = defaultdict(int)

        # Event visualization states - one per event sensor
        # Using CPU-based buffers (use_gpu=False) since we're working with numpy arrays
        self.event_viz_states: dict[str, EventVisualizationState] = {}

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ VisualizerNode initialized successfully")
        logger.info(f"   Subscribing to: {self.ipc_sub_addr}")
        logger.info("═══════════════════════════════════════════════════════════")

    def _init_sockets(self) -> None:
        """Initialize ZMQ sockets."""
        # Subscribe to state
        self.socket_sub_state = self.create_socket(
            zmq.SUB,
            self.ipc_sub_addr,
            setsockopt={zmq.SUBSCRIBE: b"state"},
        )

        # Subscribe to events (using prefix to catch all event sensors)
        self.socket_sub_events = self.create_socket(
            zmq.SUB,
            self.ipc_sub_addr,
            setsockopt={zmq.SUBSCRIBE: b"events"},
        )

        # Subscribe to IMU
        self.socket_sub_imu = self.create_socket(
            zmq.SUB,
            self.ipc_sub_addr,
            setsockopt={zmq.SUBSCRIBE: b"imu"},
        )

        # Subscribe to color cameras (using prefix subscription)
        self.socket_sub_color = self.create_socket(
            zmq.SUB,
            self.ipc_sub_addr,
            setsockopt={zmq.SUBSCRIBE: b"color"},
        )

        # Subscribe to depth cameras
        self.socket_sub_depth = self.create_socket(
            zmq.SUB,
            self.ipc_sub_addr,
            setsockopt={zmq.SUBSCRIBE: b"depth"},
        )

    def _init_executors(self) -> None:
        """Initialize async executors."""
        # Receivers for each data type
        self.create_async_executor(self.receive_state)
        self.create_async_executor(self.receive_events)
        self.create_async_executor(self.receive_imu)
        self.create_async_executor(self.receive_color)
        self.create_async_executor(self.receive_depth)

        # Stats printer
        self.create_constant_rate_executor(self.print_stats, 1)

    async def receive_state(self) -> None:
        """Receive and visualize state."""
        _, msg = await self.recv_dict(self.socket_sub_state, copy=False)

        if msg is not None:
            state = {
                "x": np.array(msg["x"]),
                "q": np.array(msg["q"]),
                "v": np.array(msg["v"]),
                "w": np.array(msg["w"]),
            }
            timestamp = msg.get("timestamp", 0)
            self._stats["received_state"] += 1

            # Visualize with Rerun
            rr.set_time("sim_time", timestamp=timestamp)
            rr.log(
                "navigation/pose",
                rr.Transform3D(
                    translation=state["x"],
                    rotation=rr.Quaternion(xyzw=state["q"]),
                    axis_length=1.0,
                    relation=rr.TransformRelation.ParentFromChild,
                ),
            )
            rr.log(
                "navigation/trajectory",
                rr.Points3D(positions=state["x"][None, :]),
            )

    async def receive_events(self) -> None:
        """Receive and visualize events as dict of arrays."""
        topic, events_dict = await self.recv_dict_of_arrays(
            self.socket_sub_events, copy=False
        )

        if events_dict is not None and topic is not None:
            self._stats["received_" + topic] += 1

            # Initialize visualization state for this sensor if needed
            if topic not in self.event_viz_states:
                self.event_viz_states[topic] = EventVisualizationState(
                    uuid=topic,
                    width=640,
                    height=480,
                    device="cpu",
                    use_gpu=False,  # Use CPU since we're receiving numpy arrays
                )

            # Convert dict format to tuple format expected by accumulate
            # accumulate expects: (x, y, t, p)
            # We receive: {'x': array, 'y': array, 'p': array}
            # Accumulate events in the visualization state
            self.event_viz_states[topic].accumulate(
                (events_dict["x"], events_dict["y"], None, events_dict["p"])
            )

            # Visualize: get the accumulated image and reset
            img = self.event_viz_states[topic].get_image()
            rr.log(topic, rr.Image(img))
            self.event_viz_states[topic].reset()

    async def receive_imu(self) -> None:
        """Receive and visualize IMU data."""
        topic, msg = await self.recv_dict(self.socket_sub_imu, copy=False)

        if msg is not None:
            uuid = msg["uuid"]
            self._stats["received_" + topic] += 1
            timestamp = msg.get("timestamp", 0)

            rr.set_time("sim_time", timestamp=timestamp)
            rr.log(f"sensors/{uuid}/accel", rr.Scalars(msg["accel"]))
            rr.log(f"sensors/{uuid}/gyro", rr.Scalars(msg["gyro"]))

    async def receive_color(self) -> None:
        """Receive and visualize color images."""
        topic, color_img = await self.recv_array(self.socket_sub_color, copy=False)

        if color_img is not None and topic is not None:
            self._stats["received_" + topic] += 1
            rr.log(topic, rr.Image(color_img))

    async def receive_depth(self) -> None:
        """Receive and visualize depth images."""
        topic, depth_img = await self.recv_array(self.socket_sub_depth, copy=False)

        if depth_img is not None and topic is not None:
            self._stats["received_" + topic] += 1
            rr.log(topic, rr.DepthImage(depth_img))

    async def print_stats(self) -> None:
        """Print statistics."""
        if self._cpu_clock_start_time is None:
            return

        elapsed = time.perf_counter() - self._cpu_clock_start_time
        logger.info("─" * 50)
        logger.info(f"[VisualizerNode] Elapsed: {elapsed:.2f}s")
        for key, value in self._stats.items():
            logger.info(f"  {key}: {value / elapsed:.1f}/s")
        logger.info("─" * 50)


async def main():
    """Main entry point for visualizer node."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the async visualizer node.")
    parser.add_argument(
        "--ipc-sub-addr",
        type=str,
        default="ipc:///tmp/neurosim_sim_pub",
        help="IPC address for subscribing to simulator.",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Don't spawn Rerun viewer (connect to existing).",
    )
    args = parser.parse_args()

    node = VisualizerNode(
        # settings_path=args.settings,
        ipc_sub_addr=args.ipc_sub_addr,
        spawn=not args.no_spawn,
    )

    try:
        await node.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        await node.close()


if __name__ == "__main__":
    asyncio.run(main())
