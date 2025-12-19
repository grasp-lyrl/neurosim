"""
Online DataLoader for PyTorch training integration.

This module provides the OnlineDataLoader class which receives simulation
data via ZMQ and packages it into PyTorch batches for training.

Data is buffered in shared storage and batched into preallocated torch tensors.

TODO: Since it knows what sensor uuids and types are being used, it can
TODO: do the buffer initialization in the init.
"""

import torch
from torch.utils.data import IterableDataset

import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Iterator
import multiprocessing as mp
from dataclasses import dataclass, field
from collections import defaultdict

from neurosim.sims.online_dataloader.config import DatasetConfig
from neurosim.sims.online_dataloader.datasubscriber import run_subscriber_process

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameBuffer:
    batch_size: int
    sample_shape: tuple
    dtype: np.dtype
    data: np.ndarray = field(init=False)
    idx: int = 0

    def __post_init__(self):
        """Allocate tensor after initialization."""
        shape = (self.batch_size,) + self.sample_shape
        self.data = np.zeros(shape, dtype=self.dtype)

    def reset(self):
        """Reset buffer index."""
        self.idx = 0


@dataclass(slots=True)
class EventBuffer:
    batch_size: int
    max_events: int
    events: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)
    sample_idx: int = 0
    event_idx: int = 0

    def __post_init__(self):
        """Allocate tensors after initialization."""
        self.events = np.zeros((self.max_events, 4), dtype=np.float32)
        self.counts = np.zeros(self.batch_size, dtype=np.int32)

    def reset(self):
        """Reset buffer indices."""
        self.sample_idx = 0
        self.event_idx = 0


class OnlineDataLoader:
    """
    PyTorch-compatible dataloader that receives data from DataSubscriber via mp.Queue.

    The subscriber runs in a separate process and puts data into a multiprocessing Queue.
    The dataloader reads from the queue and creates batched torch tensors.

    Features:
    - Separate process subscriber with mp.Queue for data sharing
    - Preallocated torch tensor batches (no repeated allocations)
    - Config-based sensor filtering
    - Event formatting: (B,) list of event counts, (N_total, 4) tensor of events
                        N_total = N_1 + N_2 + ... + N_B
    """

    def __init__(
        self,
        config: DatasetConfig,
        sensor_uuids: list[str],
        sensor_batch_sizes: dict[str, int],
        ipc_sub_addr: str | None = None,
        queue_maxsize: int = 1000,
        max_events: int = 4_000_000,
    ):
        """
        Initialize the OnlineDataLoader.

        Args:
            config: Dataset configuration
            sensor_uuids: List of sensor UUIDs to load
            batch_size: Default samples per batch (used if sensor_batch_sizes not provided)
            max_events: Maximum events to buffer per batch
            ipc_sub_addr: ZMQ address to subscribe to (default: from config)
            queue_maxsize: Max size of multiprocessing queue
            sensor_batch_sizes: Optional per-UUID batch sizes
        """
        self.config = config
        self.sensor_uuids = sensor_uuids
        self.batch_sizes = sensor_batch_sizes
        self.ipc_sub_addr = ipc_sub_addr
        self.max_events = max_events

        # Per-sensor buffers (initialized on first sample)
        self.buffers: dict[str, FrameBuffer | EventBuffer] = {}

        # Track which buffers are ready for a batch
        self._ready: set[str] = set()

        # Statistics
        self._stats = defaultdict(int)

        # Create multiprocessing queue for data transfer
        self.data_queue = mp.Queue(maxsize=queue_maxsize)

        # Start subscriber in separate process
        self.subscriber_process = mp.Process(
            target=run_subscriber_process,
            args=(config, self.data_queue, sensor_uuids, ipc_sub_addr),
            daemon=True,
        )
        self.subscriber_process.start()

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ OnlineDataLoader initialized")
        logger.info(f"   Batch sizes: {self.batch_sizes}")
        logger.info(f"   Sensors: {sensor_uuids}")
        logger.info(
            f"🚀 Started data subscriber in process {self.subscriber_process.pid}"
        )
        logger.info("═══════════════════════════════════════════════════════════")

    def _initialize_buffer(
        self, uuid: str, sensor_type: str, sample_data: np.ndarray
    ) -> None:
        """Initialize preallocated np/torch buffer based on first sample."""
        target = self.batch_sizes[uuid]
        if sensor_type == "color":
            self.buffers[uuid] = FrameBuffer(target, sample_data.shape, np.uint8)
        elif sensor_type == "depth":
            self.buffers[uuid] = FrameBuffer(target, sample_data.shape, np.float32)
        elif sensor_type == "events":
            self.buffers[uuid] = EventBuffer(target, self.max_events)
        else:
            logger.warning(
                f"Unknown sensor type for buffer initialization: {sensor_type}"
            )

    def __iter__(self) -> Iterator[dict[str, np.ndarray | tuple]]:
        """
        Iterate over batches from the subscriber queue.

        Yields per-sensor batches as they fill up.
        """
        logger.info("🔄 Reading data from queue...")

        try:
            while True:
                # If all desired buffers are ready, emit a batch
                if len(self._ready) == len(self.batch_sizes):
                    batch: dict[str, np.ndarray] = {}
                    for uuid, buf in self.buffers.items():
                        target = self.batch_sizes[uuid]
                        if isinstance(buf, FrameBuffer):
                            batch[uuid] = buf.data.copy()
                        elif isinstance(buf, EventBuffer):
                            batch[uuid] = (
                                buf.counts.copy(),
                                buf.events[: buf.event_idx].copy(),
                            )
                        self.buffers[uuid].reset()

                    self._ready.clear()
                    self._stats["batches_yielded"] += 1
                    yield batch

                # Get data from queue (sensor_type, uuid, data)
                sensor_type, uuid, data = self.data_queue.get()

                # Initialize buffer on first sample
                if uuid not in self.buffers:
                    self._initialize_buffer(uuid, sensor_type, data)

                # Fill buffer
                buf = self.buffers[uuid]
                target = self.batch_sizes[uuid]

                if sensor_type in ["color", "depth"]:
                    if buf.idx < target:
                        buf.data[buf.idx] = data
                        buf.idx = buf.idx + 1
                        if buf.idx == target:
                            self._ready.add(uuid)

                elif sensor_type == "events":
                    events_dict = data
                    n_events = len(events_dict["x"])

                    start = buf.event_idx
                    end = start + n_events

                    if buf.sample_idx < target:
                        if end < buf.max_events:
                            buf.events[start:end, 0] = events_dict["x"]
                            buf.events[start:end, 1] = events_dict["y"]
                            buf.events[start:end, 2] = events_dict["t"]
                            buf.events[start:end, 3] = events_dict["p"]
                            buf.counts[buf.sample_idx] = n_events

                            # TODO: Time needs to be normalized or relative to batch start
                            # TODO: Otherwise it will definitely overflow for float32

                            buf.event_idx = end
                            buf.sample_idx = buf.sample_idx + 1
                            if buf.sample_idx == target:
                                self._ready.add(uuid)
                        else:
                            logger.warning(
                                f"Event buffer overflow for sensor {uuid}: "
                                f"max_events={buf.max_events}, "
                                f"needed={end}. Dropping events."
                            )
                            self._ready.add(uuid)

                else:
                    logger.warning(f"Unknown sensor type received: {sensor_type}")

        except KeyboardInterrupt:
            logger.info("DataLoader interrupted by user")

        logger.info(f"📊 DataLoader stats: {dict(self._stats)}")

    def as_torch_dataset(self) -> "OnlineIterableDataset":
        """Return a PyTorch IterableDataset wrapper."""
        return OnlineIterableDataset(self)

    def get_stats(self) -> dict:
        """Get dataloader statistics."""
        return dict(self._stats)

    def close(self) -> None:
        """Clean up resources."""
        if self.subscriber_process:
            self.subscriber_process.terminate()
            self.subscriber_process.join(timeout=2.0)
            if self.subscriber_process.is_alive():
                self.subscriber_process.kill()
            logger.info("🛑 Subscriber process terminated")

        # Clean up queue
        try:
            self.data_queue.close()
            self.data_queue.join_thread()
        except Exception:
            pass


class OnlineIterableDataset(IterableDataset):
    """PyTorch IterableDataset wrapper for OnlineDataLoader."""

    def __init__(self, dataloader: OnlineDataLoader):
        self.dataloader = dataloader

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        return iter(self.dataloader)


def main():
    """Main entry point for testing the dataloader standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the online dataloader with subscriber for testing. In practice, "
        "the dataloader would be used imported and used within a training script."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to dataset configuration YAML",
    )
    parser.add_argument(
        "--ipc-addr",
        type=str,
        default=None,
        help="Override ZMQ IPC address (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--sensors",
        type=str,
        nargs="+",
        required=True,
        help="Sensor UUIDs to load",
    )

    args = parser.parse_args()

    config = DatasetConfig.from_yaml(args.config)

    # Create dataloader (starts subscriber automatically in separate process)
    dataloader = OnlineDataLoader(
        config=config,
        sensor_uuids=args.sensors,
        sensor_batch_sizes={uuid: args.batch_size for uuid in args.sensors},
        ipc_sub_addr=args.ipc_addr,
    )

    # Give subscriber time to initialize
    time.sleep(3.0)

    batch_count = 0
    total_samples = 0

    try:
        for batch in dataloader:
            batch_count += 1

            # Count actual samples in batch (from first tensor, skip lists like events)
            batch_size = 0
            for v in batch.values():
                if hasattr(v, "shape"):
                    batch_size = v.shape[0]
                    break
            total_samples += batch_size

            # Print batch info
            keys = list(batch.keys())
            shapes = {}
            for k, v in batch.items():
                if isinstance(v, tuple) and len(v) == 2:
                    # Events: (batch_sizes, events_tensor)
                    shapes[k] = f"(batch_sizes={v[0]}, events={tuple(v[1].shape)})"
                elif hasattr(v, "shape"):
                    shapes[k] = tuple(v.shape)
                else:
                    shapes[k] = "unknown"

            logger.info(
                f"Batch {batch_count}: {batch_size} samples\n"
                f"   Keys: {keys[:3]}{'...' if len(keys) > 3 else ''}\n"
                f"   Shapes: {list(shapes.items())[:3]}"
            )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        dataloader.close()

    logger.info(f"Total: {batch_count} batches, {total_samples} samples")


if __name__ == "__main__":
    main()
