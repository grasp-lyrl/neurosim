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

from neurosim.core.utils import RerunVisualizer, SimulationConfig
from neurosim.core.visual_backend.corner_detector import FeatureDetectionResult
from neurosim.sims.asynchronous_simulator.cortex_io import (
    STATE_TOPIC,
    SUBSCRIBE_DEFAULTS,
    ensure_discovery_daemon,
    message_type_for_sensor,
    sensor_metadata_from_frame_id,
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
        stream_only: bool = False,
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
            stream_only: If True, log everything as static data so each entity
                         overwrites in place. The viewer never accumulates a
                         time-series, so it never hits the memory limit and
                         goes black. No scrubbing, no history — just live.
                         Implies ``keep_latest`` semantics.
        """
        ensure_discovery_daemon(discovery_address)
        super().__init__("neurosim_visualizer", discovery_address=discovery_address)
        self._cpu_clock_start_time = None

        self.settings = self._load_settings(settings)
        self.config = SimulationConfig(
            **self.settings.get("simulator", {}),
            visual_sensors=self.settings.get("visual_backend", {}).get("sensors", {}),
        )
        self.sensor_manager = self.config.sensor_manager

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

        self.stream_only = stream_only
        # In stream-only mode there's no timeline — everything is static.
        self.keep_latest = keep_latest or stream_only
        if not stream_only:
            rr.set_time("sim_time", timestamp=0)
        self.visualizer = RerunVisualizer(
            self.config, use_gpu=False, device="cpu", stream_only=stream_only
        )
        self.visualizer.enabled = True

        # Initialize Cortex subscriptions and timers
        self._init_cortex_io()
        self._init_executors()

        # Stats tracking
        self._stats = defaultdict(int)

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
        for topic_pairs in topics.values():
            for topic, uuid in topic_pairs:
                sensor_type = self.sensor_manager.sensors[uuid].sensor_type
                self.create_subscriber(
                    topic,
                    message_type_for_sensor(sensor_type),
                    callback=self._sensor_callback(sensor_type, topic, uuid),
                    **SUBSCRIBE_DEFAULTS,
                )

    def _init_executors(self) -> None:
        """Initialize async executors."""
        # Stats printer
        self.create_timer(1.0, self.print_stats)

    def _sensor_callback(self, sensor_type: str, topic: str, uuid: str):
        handlers = {
            "event": self.receive_events,
            "imu": self.receive_imu,
            "color": self.receive_array,
            "semantic": self.receive_array,
            "depth": self.receive_array,
            "navmesh": self.receive_array,
            "optical_flow": self.receive_array,
            "corner": self.receive_corner,
            "edge": self.receive_array,
            "grayscale": self.receive_array,
        }
        handler = handlers[sensor_type]

        async def callback(msg, header) -> None:
            await handler(topic, uuid, msg, header)

        return callback

    def _timestamp_from_frame_id(self, frame_id: str) -> float:
        _uuid, timestamp, _simsteps = sensor_metadata_from_frame_id(frame_id)
        return 0.0 if self.keep_latest else timestamp

    def _log_measurement(self, uuid: str, measurement, timestamp: float) -> None:
        self.visualizer.log_measurements(
            {uuid: measurement},
            0.0 if self.keep_latest else timestamp,
            0,  # Messages are already published at viz rate; force logging.
        )

    async def receive_state(self, msg: DictMessage, _header) -> None:
        """Receive and visualize state."""
        if msg.data is not None:
            state = state_from_message(msg.data)
            timestamp = msg.data.get("timestamp", 0)
            self._stats["received_state"] += 1

            # Visualize with Rerun
            if not self.stream_only:
                rr.set_time(
                    "sim_time", timestamp=0.0 if self.keep_latest else timestamp
                )
            self.visualizer.log_state(state)

    async def receive_events(
        self, topic: str, uuid: str, msg: MultiArrayMessage, _header
    ) -> None:
        """Receive and visualize events as a Cortex multi-array message."""
        events_dict = msg.arrays
        if events_dict is not None:
            self._stats["received_" + topic] += 1
            self._log_measurement(
                uuid,
                (events_dict["x"], events_dict["y"], None, events_dict["p"]),
                self._timestamp_from_frame_id(msg.frame_id),
            )

    async def receive_imu(
        self, topic: str, _uuid: str, msg: DictMessage, _header
    ) -> None:
        """Receive and visualize IMU data."""
        if msg.data is not None:
            uuid = msg.data["uuid"]
            self._stats["received_" + topic] += 1
            timestamp = msg.data.get("timestamp", 0)
            self._log_measurement(
                uuid,
                {"accel": msg.data["accel"], "gyro": msg.data["gyro"]},
                timestamp,
            )

    async def receive_array(
        self, topic: str, uuid: str, msg: ArrayMessage, _header
    ) -> None:
        """Receive and visualize array-backed sensors via RerunVisualizer."""
        if msg.data is not None:
            self._stats["received_" + topic] += 1
            self._log_measurement(
                uuid,
                msg.data,
                self._timestamp_from_frame_id(msg.frame_id),
            )

    async def receive_corner(
        self, topic: str, uuid: str, msg: DictMessage, _header
    ) -> None:
        """Receive and visualize corner detector results."""
        if msg.data is None:
            return

        self._stats["received_" + topic] += 1
        data = msg.data
        measurement = FeatureDetectionResult(
            keypoints=data["keypoints"],
            scores=data["scores"],
            descriptors=data["descriptors"],
            sizes=data["sizes"],
            angles=data["angles"],
            octaves=data["octaves"],
            num_keypoints=data["num_keypoints"],
            detector_name=data["detector_name"],
            descriptor_name=data["descriptor_name"],
        )
        self._log_measurement(uuid, measurement, data.get("timestamp", 0.0))

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
    parser.add_argument(
        "--stream-only",
        action="store_true",
        help=(
            "Log everything as static data so the viewer never accumulates a "
            "time-series. Avoids the black-screen issue when memory_limit is hit."
        ),
    )
    args = parser.parse_args()

    async with VisualizerNode(
        settings=args.settings,
        discovery_address=args.discovery_address,
        mode=args.mode,
        memory_limit=args.memory_limit,
        stream_only=args.stream_only,
    ) as node:
        try:
            await node.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")


if __name__ == "__main__":
    cortex.run(main())
