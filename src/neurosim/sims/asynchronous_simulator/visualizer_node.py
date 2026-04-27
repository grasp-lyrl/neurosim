"""
Visualizer Node - Async visualization with Rerun.

This node handles:
- Receiving sensor data from simulator
- Receiving state data from simulator
- Visualizing data with Rerun
"""

import yaml
import time
import logging
import rerun as rr
from pathlib import Path
from collections import defaultdict

import cortex
from cortex.core.node import Node
from cortex.discovery.daemon import DEFAULT_DISCOVERY_ADDRESS
from cortex.messages.standard import ArrayMessage, DictMessage, MultiArrayMessage

from neurosim.core.utils import EventVisualizationState
from neurosim.sims.asynchronous_simulator.cortex_io import (
    STATE_TOPIC,
    SUBSCRIBE_DEFAULTS,
    ensure_discovery_daemon,
    sensor_topics_from_settings,
    state_from_message,
)

logger = logging.getLogger(__name__)


class VisualizerNode(Node):
    """
    Async Visualizer Node.

    Receives data from simulator and visualizes with Rerun.
    """

    def __init__(
        self,
        settings: Path | str | dict,
        discovery_address: str = DEFAULT_DISCOVERY_ADDRESS,
        memory_limit: str = "10%",
        keep_latest: bool = True,
        mode: str = "spawn",
    ):
        """
        Initialize the Visualizer Node.

        - Can use this to visualize data from the SimulatorNode in real-time.

        - Can also use this to visualize data from the OnlineDataPublisher -- so you can
          see what data is being sent to the training process. Very cool.

        Args:
            settings: Settings dictionary or path used to derive explicit sensor topics
            discovery_address: Cortex discovery daemon address
            memory_limit: Memory limit for Rerun process (e.g., "10%", "2GB")
            keep_latest: If True, always keep latest timestamp in Rerun. Saves memory. Disregards sim time.
                         Helpful for simulations which keep on loading different scenes and resetting time.
            mode: Rerun operating mode. Options:
                  - "spawn": Spawn local viewer (default)
                  - "serve": Start gRPC server for remote viewing (run on server)
        """
        ensure_discovery_daemon(discovery_address)
        super().__init__("neurosim_visualizer", discovery_address=discovery_address)
        self._cpu_clock_start_time = None

        self.settings = self._load_settings(settings)
        self.visual_sensor_settings = self.settings.get("visual_backend", {}).get(
            "sensors", {}
        )

        # Initialize Rerun with appropriate mode
        rr.init("neurosim_async")

        if mode == "spawn":
            rr.spawn(memory_limit=memory_limit)
            logger.info("Rerun viewer spawned locally")
        elif mode == "serve":
            server_uri = rr.serve_grpc(server_memory_limit=memory_limit)
            logger.info("═══════════════════════════════════════════════════════════")
            logger.info("🌐 Rerun gRPC server started for remote viewing")
            logger.info(f"   Server URI: {server_uri}")
            logger.info("   To connect from your personal machine, run:")
            logger.info(
                f"   rerun --connect {server_uri.replace('127.0.0.1', '<SERVER_IP_ADDRESS>')}"
            )
            logger.info("═══════════════════════════════════════════════════════════")
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'spawn' or 'serve'")

        rr.set_time("sim_time", timestamp=0)
        self.keep_latest = keep_latest

        # Initialize Cortex subscriptions and timers
        self._init_cortex_io()
        self._init_executors()

        # Stats tracking
        self._stats = defaultdict(int)

        # Event visualization states - one per event sensor
        # Using CPU-based buffers (use_gpu=False) since we're working with numpy arrays
        self.event_viz_states: dict[str, EventVisualizationState] = {}

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ VisualizerNode initialized successfully")
        logger.info(f"   Discovery: {self.discovery_address}")
        logger.info("═══════════════════════════════════════════════════════════")

    @staticmethod
    def _load_settings(settings: Path | str | dict) -> dict:
        """Load settings from a dict or YAML file path."""
        if isinstance(settings, dict):
            return settings

        settings_path = Path(settings)
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")

        with open(settings_path, "r") as f:
            return yaml.safe_load(f)

    def _init_cortex_io(self) -> None:
        """Initialize explicit Cortex subscriptions."""
        self.create_subscriber(
            STATE_TOPIC,
            DictMessage,
            callback=self.receive_state,
            **SUBSCRIBE_DEFAULTS,
        )
        topics = sensor_topics_from_settings(self.settings)
        for kind, msg_type, factory in (
            ("events", MultiArrayMessage, self._event_callback),
            ("imu", DictMessage, self._imu_callback),
            ("color", ArrayMessage, self._color_callback),
            ("depth", ArrayMessage, self._depth_callback),
        ):
            for topic, uuid in topics[kind]:
                self.create_subscriber(
                    topic,
                    msg_type,
                    callback=factory(topic, uuid),
                    **SUBSCRIBE_DEFAULTS,
                )

    def _init_executors(self) -> None:
        """Initialize async executors."""
        # Stats printer
        self.create_timer(1.0, self.print_stats)

    def _event_callback(self, topic: str, uuid: str):
        async def callback(msg: MultiArrayMessage, header) -> None:
            await self.receive_events(topic, uuid, msg, header)

        return callback

    def _imu_callback(self, topic: str, uuid: str):
        async def callback(msg: DictMessage, header) -> None:
            await self.receive_imu(topic, uuid, msg, header)

        return callback

    def _color_callback(self, topic: str, uuid: str):
        async def callback(msg: ArrayMessage, header) -> None:
            await self.receive_color(topic, uuid, msg, header)

        return callback

    def _depth_callback(self, topic: str, uuid: str):
        async def callback(msg: ArrayMessage, header) -> None:
            await self.receive_depth(topic, uuid, msg, header)

        return callback

    async def receive_state(self, msg: DictMessage, _header) -> None:
        """Receive and visualize state."""
        if msg.data is not None:
            state = state_from_message(msg.data)
            timestamp = msg.data.get("timestamp", 0)
            self._stats["received_state"] += 1

            # Visualize with Rerun
            if not self.keep_latest:
                rr.set_time("sim_time", timestamp=timestamp)
            rr.log(
                "navigation/pose",
                rr.Transform3D(
                    translation=state["x"],
                    rotation=rr.Quaternion(xyzw=state["q"]),
                    relation=rr.TransformRelation.ParentFromChild,
                ),
                rr.TransformAxes3D(
                    1.0
                ),  # Separate archetype for axis visualization in 0.28.1+
            )
            rr.log(
                "navigation/trajectory",
                rr.Points3D(positions=state["x"][None, :]),
            )

    async def receive_events(
        self, topic: str, uuid: str, msg: MultiArrayMessage, _header
    ) -> None:
        """Receive and visualize events as a Cortex multi-array message."""
        events_dict = msg.arrays
        if events_dict is not None:
            self._stats["received_" + topic] += 1

            # Initialize visualization state for this sensor if needed
            if topic not in self.event_viz_states:
                sensor_cfg = self.visual_sensor_settings.get(uuid, {})
                self.event_viz_states[topic] = EventVisualizationState(
                    uuid=topic,
                    width=sensor_cfg.get("width", 640),
                    height=sensor_cfg.get("height", 480),
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
            rr.log(
                topic,
                rr.Image(self.event_viz_states[topic].get_image()),
            )
            self.event_viz_states[topic].reset()

    async def receive_imu(
        self, topic: str, _uuid: str, msg: DictMessage, _header
    ) -> None:
        """Receive and visualize IMU data."""
        if msg.data is not None:
            uuid = msg.data["uuid"]
            self._stats["received_" + topic] += 1
            timestamp = msg.data.get("timestamp", 0)

            if not self.keep_latest:
                rr.set_time("sim_time", timestamp=timestamp)
            rr.log(f"sensors/{uuid}/accel", rr.Scalars(msg.data["accel"]))
            rr.log(f"sensors/{uuid}/gyro", rr.Scalars(msg.data["gyro"]))

    async def receive_color(
        self, topic: str, _uuid: str, msg: ArrayMessage, _header
    ) -> None:
        """Receive and visualize color images."""
        color_img = msg.data
        if color_img is not None:
            self._stats["received_" + topic] += 1
            rr.log(topic, rr.Image(color_img))

    async def receive_depth(
        self, topic: str, _uuid: str, msg: ArrayMessage, _header
    ) -> None:
        """Receive and visualize depth images."""
        depth_img = msg.data
        if depth_img is not None:
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

    async def run(self) -> None:
        """Run the Cortex visualizer node."""
        self._cpu_clock_start_time = time.perf_counter()
        await super().run()


async def main():
    """Main entry point for visualizer node."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the async visualizer node.")
    parser.add_argument(
        "--settings",
        type=Path,
        required=True,
        help="Path to the settings YAML file used to derive sensor topics.",
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
    parser.add_argument(
        "--mode",
        type=str,
        default="spawn",
        choices=["spawn", "serve"],
        help="Rerun operating mode: 'spawn' (local viewer), 'serve' (gRPC server for remote)",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="10%",
        help="Memory limit for Rerun process (e.g., '10%%', '2GB')",
    )
    args = parser.parse_args()

    node = VisualizerNode(
        settings=args.settings,
        discovery_address=args.discovery_address,
        mode=args.mode,
        memory_limit=args.memory_limit,
    )

    try:
        await node.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        await node.close()


if __name__ == "__main__":
    cortex.run(main())
