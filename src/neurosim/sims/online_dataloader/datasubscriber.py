"""
This module provides the DataSubscriber class which receives simulation
data via ZMQ and buffers it in shared storage for the OnlineDataLoader to
access and create PyTorch batches.
"""

import zmq
import time
import asyncio
import logging
import multiprocessing as mp
from collections import defaultdict

from neurosim.cortex.utils import ZMQNODE
from neurosim.sims.online_dataloader.config import DatasetConfig

logger = logging.getLogger(__name__)


class DataSubscriber(ZMQNODE):
    """
    Async ZMQ subscriber for receiving simulation data.

    Similar to VisualizerNode - runs asynchronously with separate async
    executors for each sensor type. Buffers data in shared storage that
    OnlineDataLoader can access for batching.

    Data is not copied - uses zero-copy receiving where possible.
    """

    def __init__(
        self,
        config: DatasetConfig,
        data_queue: mp.Queue,
        sensors: list[str] | set[str] | None = None,
        ipc_sub_addr: str | None = None,
    ):
        """
        Initialize async data subscriber.

        Args:
            config: Dataset configuration
            data_queue: Multiprocessing queue for sending data to dataloader
            sensors: List of sensor UUIDs to include (None = all)
            ipc_sub_addr: ZMQ address to subscribe to (default: from config)
        """
        super().__init__()

        self.config = config
        self.data_queue = data_queue
        self.sensors = set(sensors) if sensors else None
        self.ipc_sub_addr = ipc_sub_addr or config.ipc_pub_addr

        # Statistics
        self._stats = defaultdict(int)

        # Initialize sockets and executors
        self._init_sockets()
        self._init_executors()

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ DataSubscriber initialized")
        logger.info(f"   Subscribing to: {self.ipc_sub_addr}")
        if self.sensors:
            logger.info(f"   Filtering sensors: {list(self.sensors)}")
        logger.info("═══════════════════════════════════════════════════════════")

    def _init_sockets(self) -> None:
        """Initialize ZMQ subscriber sockets."""
        # Subscribe to color cameras
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

        # Subscribe to events
        self.socket_sub_events = self.create_socket(
            zmq.SUB,
            self.ipc_sub_addr,
            setsockopt={zmq.SUBSCRIBE: b"events"},
        )

    def _init_executors(self) -> None:
        """Initialize async executors for each data type."""
        # Async receivers for each sensor type
        self.create_async_executor(self.receive_color)
        self.create_async_executor(self.receive_depth)
        self.create_async_executor(self.receive_events)

        # Stats printer
        self.create_constant_rate_executor(self.print_stats, 1)

    def _should_include_sensor(self, uuid: str) -> bool:
        """Check if sensor should be included based on filter."""
        return self.sensors is None or uuid in self.sensors

    async def receive_color(self) -> None:
        """Receive color images (async)."""
        topic, img_array = await self.recv_array(self.socket_sub_color, copy=False)

        if img_array is not None and topic is not None:
            # Extract UUID from topic (e.g., "color/camera_1" -> "camera_1")
            uuid = topic.split("/")[1]

            if self._should_include_sensor(uuid):
                # Put data into queue as (sensor_type, uuid, data)
                self.data_queue.put(("color", uuid, img_array))
                self._stats[f"received_{uuid}"] += 1

    async def receive_depth(self) -> None:
        """Receive depth images (async)."""
        topic, depth_array = await self.recv_array(self.socket_sub_depth, copy=False)

        if depth_array is not None and topic is not None:
            uuid = topic.split("/")[1]

            if self._should_include_sensor(uuid):
                # Put data into queue as (sensor_type, uuid, data)
                self.data_queue.put(("depth", uuid, depth_array))
                self._stats[f"received_{uuid}"] += 1

    async def receive_events(self) -> None:
        """Receive event camera data (async)."""
        topic, events_dict = await self.recv_dict_of_arrays(
            self.socket_sub_events, copy=False
        )

        if events_dict is not None and topic is not None:
            # Extract UUID from topic (e.g., "events/camera_1" -> "camera_1")
            uuid = topic.split("/")[1]

            if self._should_include_sensor(uuid):
                # Events are already bucketed at fixed rate (e.g., 20Hz = 50ms)
                # Put data into queue as (sensor_type, uuid, events_dict)
                self.data_queue.put(("events", uuid, events_dict))
                self._stats[f"received_{uuid}"] += 1

    async def print_stats(self) -> None:
        """Print statistics."""
        if self._cpu_clock_start_time is None:
            return

        elapsed = time.perf_counter() - self._cpu_clock_start_time
        logger.info("─" * 50)
        logger.info(f"[DataSubscriber] Elapsed: {elapsed:.2f}s")
        for key, value in sorted(self._stats.items()):
            logger.info(f"  {key}: {value / elapsed:.1f}/s")
        logger.info("─" * 50)


def run_subscriber_process(
    config: DatasetConfig,
    data_queue: mp.Queue,
    sensors: list[str] | None,
    ipc_sub_addr: str | None,
):
    """
    Entry point for running DataSubscriber in a separate process.

    This function creates a new event loop and runs the subscriber asynchronously.
    """
    subscriber = DataSubscriber(
        config=config,
        data_queue=data_queue,
        sensors=sensors,
        ipc_sub_addr=ipc_sub_addr,
    )

    try:
        asyncio.run(subscriber.run())
    except KeyboardInterrupt:
        logger.info("Subscriber process interrupted")
