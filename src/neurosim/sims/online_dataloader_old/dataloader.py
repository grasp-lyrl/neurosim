"""
Online DataLoader for PyTorch training integration.

This module provides the OnlineDataLoader class which receives simulation
data via ZMQ and packages it into PyTorch batches for training.

Data is buffered in shared storage and batched into preallocated torch tensors.
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
from collections import defaultdict
from dataclasses import dataclass, field

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
        self.data = np.zeros((self.batch_size,) + self.sample_shape, dtype=self.dtype)

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
        self.counts.fill(0)


@dataclass(slots=True, frozen=True)
class BatchBuildingConfig:
    """Configuration for batch building operations.

    Currently handles event normalization, with potential for additional
    batch processing options in the future.
    """

    # Event normalization settings
    normalize_events: bool = True
    event_width: int = 640
    event_height: int = 480
    event_time_window: int = 50000


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
        max_events: int = 8_000_000,
        prefetch_factor: int = 2,
        verbose: bool = False,
        batch_building_config: BatchBuildingConfig | dict = BatchBuildingConfig(),
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
            prefetch_factor: Number of batches to prefetch (0 = no prefetch)
            batch_building_config: Configuration for batch building (default: event normalization enabled with 640x480, 50000 time window)
        """
        self.config = config
        self.sensor_uuids = sensor_uuids
        self.batch_sizes = sensor_batch_sizes
        self.ipc_sub_addr = ipc_sub_addr
        self.max_events = max_events
        self.prefetch_factor = prefetch_factor

        if isinstance(batch_building_config, dict):
            self.batch_building_config = BatchBuildingConfig(**batch_building_config)
        elif isinstance(batch_building_config, BatchBuildingConfig):
            self.batch_building_config = batch_building_config
        else:
            raise ValueError(
                "batch_building_config must be a BatchBuildingConfig or dict"
            )

        # Statistics (shared across processes)
        self._stats = defaultdict(int)

        # Create multiprocessing queue for data transfer
        self.data_queue = mp.Queue(maxsize=queue_maxsize)

        # Create batch queue for prefetching
        self.batch_queue = (
            mp.Queue(maxsize=max(1, prefetch_factor)) if prefetch_factor > 0 else None
        )

        # Start subscriber in separate process
        self.subscriber_process = mp.Process(
            target=run_subscriber_process,
            args=(self.data_queue, ipc_sub_addr, sensor_uuids, verbose),
            daemon=True,
        )
        self.subscriber_process.start()

        # Start batch builder in separate process if prefetching
        self.builder_process = None
        if prefetch_factor > 0:
            self.builder_process = mp.Process(
                target=self._batch_builder_worker,
                args=(self.batch_building_config,),
                daemon=True,
            )
            self.builder_process.start()

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ OnlineDataLoader initialized")
        logger.info(f"   Batch sizes: {self.batch_sizes}")
        logger.info(f"   Sensors: {sensor_uuids}")
        logger.info(f"   Prefetch factor: {prefetch_factor}")
        logger.info(
            f"🚀 Started data subscriber in process {self.subscriber_process.pid}"
        )
        if self.builder_process:
            logger.info(
                f"🚀 Started batch builder in process {self.builder_process.pid}"
            )
        logger.info("═══════════════════════════════════════════════════════════")

    def _initialize_all_buffers(self) -> dict[str, FrameBuffer | EventBuffer]:
        """Initialize all buffers eagerly from config at startup."""
        buffers: dict[str, FrameBuffer | EventBuffer] = {}

        for uuid in self.sensor_uuids:
            sensor_cfg = self.config.visual_backend["sensors"][uuid]
            target = self.batch_sizes[uuid]
            sensor_type = sensor_cfg["type"]

            if sensor_type == "color":
                shape = (sensor_cfg["height"], sensor_cfg["width"], 3)
                buffers[uuid] = FrameBuffer(target, shape, np.uint8)
                logger.info(f"   Initialized color buffer {uuid}: {shape}")
            elif sensor_type == "depth":
                shape = (sensor_cfg["height"], sensor_cfg["width"])
                buffers[uuid] = FrameBuffer(target, shape, np.float32)
                logger.info(f"   Initialized depth buffer {uuid}: {shape}")
            elif sensor_type == "event":
                buffers[uuid] = EventBuffer(target, self.max_events)
                logger.info(
                    f"   Initialized event buffer {uuid}: {target} samples, {self.max_events} events"
                )
            else:
                logger.warning(f"Unknown sensor type '{sensor_type}' for {uuid}")

        return buffers

    @staticmethod
    def _build_batch_from_buffers(
        buffers: dict[str, FrameBuffer | EventBuffer],
    ) -> dict[str, np.ndarray | tuple]:
        """Build batch from buffers and reset them (shared logic)."""
        batch: dict[str, np.ndarray | tuple] = {}
        for uuid, buf in buffers.items():
            if isinstance(buf, FrameBuffer):
                batch[uuid] = buf.data.copy()
            elif isinstance(buf, EventBuffer):
                batch[uuid] = (buf.counts.copy(), buf.events[: buf.event_idx].copy())
            buf.reset()
        return batch

    @staticmethod
    def _process_sample(
        buffers: dict[str, FrameBuffer | EventBuffer],
        ready: set[str],
        uuid: str,
        sensor_type: str,
        data: np.ndarray | dict,
        batch_sizes: dict[str, int],
        batch_building_config: BatchBuildingConfig,
    ) -> None:
        """Process a single sample into buffer (shared logic)."""
        buf = buffers[uuid]
        target = batch_sizes[uuid]

        if sensor_type in ["color", "depth"]:
            if buf.idx < target:
                buf.data[buf.idx] = data
                buf.idx += 1
                if buf.idx == target:
                    ready.add(uuid)

        elif sensor_type == "events":
            events_dict = data

            # Filter events based on time window (events are sorted in time)
            if len(events_dict["t"]) > 0:
                t_max = events_dict["t"][-1]
                t_threshold = t_max - batch_building_config.event_time_window
                start_idx = np.searchsorted(events_dict["t"], t_threshold, side="right")

                # Filter all event arrays to keep only events within time window
                events_dict["x"] = events_dict["x"][start_idx:]
                events_dict["y"] = events_dict["y"][start_idx:]
                events_dict["t"] = events_dict["t"][start_idx:]
                events_dict["p"] = events_dict["p"][start_idx:]

            n_events = len(events_dict["x"])
            start = buf.event_idx
            end = start + n_events

            if buf.sample_idx < target:
                if end <= buf.max_events:
                    if batch_building_config.normalize_events:
                        events_stacked = np.column_stack(
                            [
                                events_dict["x"] / batch_building_config.event_width,
                                events_dict["y"] / batch_building_config.event_height,
                                (events_dict["t"][-1] - events_dict["t"])
                                / batch_building_config.event_time_window,
                                events_dict["p"],
                            ]
                        )
                    else:
                        events_stacked = np.column_stack(
                            [
                                events_dict["x"],
                                events_dict["y"],
                                events_dict["t"],
                                events_dict["p"],
                            ]
                        )
                    buf.events[start:end] = events_stacked
                    buf.counts[buf.sample_idx] = n_events
                    buf.event_idx = end
                    buf.sample_idx += 1

                    if buf.sample_idx == target:
                        ready.add(uuid)
                else:
                    logger.warning(
                        f"Event buffer overflow for sensor {uuid}: "
                        f"max_events={buf.max_events}, needed={end}. Dropping events."
                    )
                    ready.add(uuid)
        else:
            logger.warning(f"Unknown sensor type: {sensor_type}")

    def _batch_builder_worker(
        self,
        batch_building_config: BatchBuildingConfig,
    ) -> None:
        """
        Worker process that builds batches and puts them in the batch queue.
        Runs continuously in background.
        """
        buffers = self._initialize_all_buffers()
        ready: set[str] = set()

        logger.info("🔄 Batch builder started, reading data from queue...")

        try:
            while True:
                # If all desired buffers are ready, emit a batch
                if len(ready) == len(self.batch_sizes):
                    batch = self._build_batch_from_buffers(buffers)
                    ready.clear()
                    self.batch_queue.put(batch)

                sensor_type, uuid, data = self.data_queue.get()

                # Process sample
                self._process_sample(
                    buffers,
                    ready,
                    uuid,
                    sensor_type,
                    data,
                    self.batch_sizes,
                    batch_building_config,
                )

        except KeyboardInterrupt:
            logger.info("Batch builder interrupted")
        except Exception as e:
            logger.error(f"Batch builder error: {e}", exc_info=True)

    def __iter__(self) -> Iterator[dict[str, np.ndarray | tuple]]:
        """
        Iterate over pre-built batches from the batch queue.
        """
        if self.prefetch_factor > 0:
            # Prefetch mode: just pull from batch queue
            logger.info("🔄 Yielding pre-built batches...")
            try:
                while True:
                    batch = self.batch_queue.get()
                    self._stats["batches_yielded"] += 1
                    yield batch
            except KeyboardInterrupt:
                logger.info("DataLoader interrupted by user")
        else:
            # No prefetch: build batches inline
            logger.info("🔄 Building batches inline (no prefetch)...")
            buffers = self._initialize_all_buffers()
            ready: set[str] = set()

            try:
                while True:
                    if len(ready) == len(self.batch_sizes):
                        batch = self._build_batch_from_buffers(buffers)
                        ready.clear()
                        self._stats["batches_yielded"] += 1
                        yield batch
                        continue

                    sensor_type, uuid, data = self.data_queue.get()

                    self._process_sample(
                        buffers,
                        ready,
                        uuid,
                        sensor_type,
                        data,
                        self.batch_sizes,
                        self.batch_building_config,
                    )

            except KeyboardInterrupt:
                logger.info("DataLoader interrupted by user")

    def as_torch_dataset(self) -> "OnlineIterableDataset":
        """Return a PyTorch IterableDataset wrapper."""
        return OnlineIterableDataset(self)

    def get_stats(self) -> dict:
        """Get dataloader statistics."""
        return dict(self._stats)

    def close(self) -> None:
        """Clean up resources."""
        if self.builder_process:
            self.builder_process.terminate()
            self.builder_process.join(timeout=2.0)
            if self.builder_process.is_alive():
                self.builder_process.kill()
            logger.info("🛑 Batch builder (Prefetching) process terminated")

        if self.subscriber_process:
            self.subscriber_process.terminate()
            self.subscriber_process.join(timeout=2.0)
            if self.subscriber_process.is_alive():
                self.subscriber_process.kill()
            logger.info("🛑 Subscriber process terminated")

        # Clean up queues
        try:
            self.data_queue.close()
            self.data_queue.join_thread()
        except Exception:
            pass

        if self.batch_queue:
            try:
                self.batch_queue.close()
                self.batch_queue.join_thread()
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
        ipc_sub_addr=config.ipc_pub_addr,
        prefetch_factor=8,
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
