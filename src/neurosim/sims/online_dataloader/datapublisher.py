"""
Data Publisher for online dataset generation.

This module provides the DataPublisher class which runs simulations
sequentially across different scenes and seeds, publishing sensor data
via ZMQ for consumption by a training dataloader.

Inherits from ZMQNODE and follows the same patterns as SimulatorNode
for sensor rendering and publishing.

Note: In this code, we can afford to zero-copy data sending since the
data is copied once from GPU to CPU before being sent over ZMQ. So it
is always a new memory. This might not be the case in other scenarios.
Be careful.

Usage:
    python -m neurosim.sims.online_dataloader.publisher --config configs/dataset.yaml
"""

import zmq
import time
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from neurosim.cortex.utils import ZMQNODE
from neurosim.core.utils import EventBuffer, SensorManager
from neurosim.sims.online_dataloader.config import DatasetConfig
from neurosim.sims.synchronous_simulator import SynchronousSimulator

logger = logging.getLogger(__name__)


class DataPublisher(ZMQNODE):
    """
    Publishes simulation data for online training.

    Runs simulations sequentially through configured scenes and seeds,
    publishing sensor data and state via ZMQ.

    This publisher runs synchronously and publishes immediately after rendering.
    """

    MAX_EVENT_BUFFER_SIZE = 2000000

    def __init__(
        self,
        dataset_config: DatasetConfig,
        ipc_pub_addr: str | None = None,
    ):
        """
        Initialize the DataPublisher.

        Args:
            dataset_config: Configuration specifying scenes, seeds, and parameters
            ipc_pub_addr: ZMQ IPC address for publishing (overrides config if set)
        """
        super().__init__()

        # Override the async context with sync context for synchronous publishing
        # TODO: Add better support for this using Cortex.
        self._context = zmq.Context(self._context)

        self.config = dataset_config
        self.ipc_pub_addr = ipc_pub_addr or dataset_config.ipc_pub_addr

        # Create publisher socket
        self._init_socket()

        # Statistics
        self._stats = defaultdict(int)

        # Current simulation metadata (set during run)
        self._current_metadata = None

        # Current simulator sensor manager (set during simulation)
        self._sensor_manager: SensorManager | None = None

        # Event camera buffers (initialized per simulation)
        self.event_buffers: dict[str, EventBuffer] = {}

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ DataPublisher initialized")
        logger.info(f"   Publishing to: {self.ipc_pub_addr}")
        logger.info(f"   Total runs: {self.config.get_total_runs()}")
        logger.info(f"   Scenes: {[s.name for s in self.config.scenes]}")
        logger.info("═══════════════════════════════════════════════════════════")

    def _init_socket(self) -> None:
        """Initialize ZMQ publisher socket."""
        self.socket_pub = self.create_socket(
            zmq.PUB,
            self.ipc_pub_addr,
            setsockopt={
                zmq.SNDHWM: 100,
                zmq.LINGER: 0,
                zmq.IMMEDIATE: 1,
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Publishing Methods (following simulator_node pattern)
    # ─────────────────────────────────────────────────────────────────────────

    def publish_state(
        self,
        state: dict[str, np.ndarray],
        timestamp: float,
        simsteps: int,
        metadata: dict,
    ) -> bool:
        """Publish current state."""
        if state is None:
            return False

        state_msg = {
            "x": state["x"].tolist(),
            "q": state["q"].tolist(),
            "v": state["v"].tolist(),
            "w": state["w"].tolist(),
            "timestamp": timestamp,
            "simsteps": simsteps,
            "metadata": metadata,  # What metadata
        }

        if self.send_dict(self.socket_pub, state_msg, topic="state", copy=False):
            self._stats["published_state"] += 1
            return True
        return False

    def publish_imu(
        self,
        uuid: str,
        imu_data: dict[str, np.ndarray],
        timestamp: float,
    ) -> bool:
        """Publish IMU sensor data from stored measurements."""
        imu_msg = {
            "uuid": uuid,
            "accel": imu_data["accel"].tolist(),
            "gyro": imu_data["gyro"].tolist(),
            "timestamp": timestamp,
        }

        if self.send_dict(self.socket_pub, imu_msg, topic=f"imu/{uuid}", copy=False):
            self._stats[f"published_{uuid}"] += 1
            return True
        return False

    def publish_color(
        self,
        uuid: str,
        data: torch.Tensor | np.ndarray,
    ) -> bool:
        """Publish color camera data from stored measurements."""
        # Handle GPU tensors
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()

        if self.send_array(self.socket_pub, data, topic=f"color/{uuid}", copy=False):
            self._stats[f"published_{uuid}"] += 1
            return True
        return False

    def publish_depth(
        self,
        uuid: str,
        data: torch.Tensor | np.ndarray,
    ) -> bool:
        """Publish depth camera data from stored measurements."""
        # Handle GPU tensors
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()

        if self.send_array(self.socket_pub, data, topic=f"depth/{uuid}", copy=False):
            self._stats[f"published_{uuid}"] += 1
            return True
        return False

    def publish_events(
        self,
        uuid: str,
        events_dict: dict[str, np.ndarray],
    ) -> bool:
        """Publish event camera data."""
        if events_dict is None:
            return False

        if self.send_dict_of_arrays(
            self.socket_pub, events_dict, topic=f"events/{uuid}", copy=False
        ):
            self._stats[f"published_{uuid}"] += 1
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation Execution
    # ─────────────────────────────────────────────────────────────────────────

    def step_callback(
        self, measurements: dict, state: dict, sim_time: float, simsteps: int
    ) -> None:
        """
        Callback invoked by SynchronousSimulator at each step to publish data.

        Args:
            measurements: Sensor measurements from this step
            state: Current dynamics state
            sim_time: Current simulation time
            simsteps: Current simulation step count
        """
        # Publish state if enabled
        if self.config.publish_state:
            self.publish_state(state, sim_time, simsteps, self._current_metadata)

        _sensor_dict = self._sensor_manager.sensors

        # Process sensor measurements
        for uuid, data in measurements.items():
            # Query sensor type from sensor manager
            sensor = _sensor_dict[uuid]
            sensor_type = sensor.sensor_type

            if sensor_type == "event":
                # Accumulate events in buffer (like simulator_node)
                if data is not None:
                    self.event_buffers[uuid].append(data)

                # Only publish when should_visualize returns True
                if self._sensor_manager.should_visualize_sensor(sensor, simsteps):
                    events_dict = self.event_buffers[uuid].get_and_clear()
                    if events_dict is not None:
                        self.publish_events(uuid, events_dict)
                        self._stats["published_samples"] += 1

            elif sensor_type == "imu":
                # Publish IMU at visualization rate
                if self._sensor_manager.should_visualize_sensor(sensor, simsteps):
                    self.publish_imu(uuid, data, sim_time)
                    self._stats["published_samples"] += 1

            elif sensor_type == "color":
                # Publish color at visualization rate
                if self._sensor_manager.should_visualize_sensor(sensor, simsteps):
                    self.publish_color(uuid, data)
                    self._stats["published_samples"] += 1

            elif sensor_type == "depth":
                # Publish depth at visualization rate
                if self._sensor_manager.should_visualize_sensor(sensor, simsteps):
                    self.publish_depth(uuid, data)
                    self._stats["published_samples"] += 1

    def _run_single_simulation(self, settings: dict, metadata: dict) -> int:
        """
        Run a single simulation and publish data.

        Args:
            settings: Complete settings dict for the simulation
            metadata: Metadata about this run (scene, seed, etc.)

        Returns:
            Number of samples published
        """
        # Store metadata for use in callback
        self._current_metadata = metadata

        # Create simulator with settings dict
        simulator = SynchronousSimulator(settings=settings, visualizer_disabled=True)

        # Store simulator and sensor manager references for callback
        self._sensor_manager = simulator.config.sensor_manager

        # Initialize event buffers for this simulation
        # TODO: Modify SynchronousSimulator to handle this internally
        self.event_buffers.clear()
        for sensor in self._sensor_manager.get_sensors_by_type("event"):
            self.event_buffers[sensor.uuid] = EventBuffer(
                max_size=self.MAX_EVENT_BUFFER_SIZE,
                use_gpu=True,
            )

        # Run simulation with step callback for publishing
        samples_before = self._stats["published_samples"]
        try:
            simulator.run(display=False, log_h5=None, callback_hook_=self.step_callback)
        finally:
            simulator.close()
            self._sensor_manager = None

        num_samples = self._stats["published_samples"] - samples_before

        return num_samples

    def run(self, verbose: bool = True) -> defaultdict:
        """
        Run all configured simulations and publish data.

        Args:
            verbose: Whether to log progress

        Returns:
            Statistics dictionary
        """
        start_time = time.perf_counter()

        for settings, metadata in self.config.generate_run_configs():
            if verbose:
                logger.info(
                    f"▶ Starting run {metadata['run_idx'] + 1}/{self.config.get_total_runs()}: "
                    f"{metadata['scene_name']} (seed={metadata['seed']})"
                )

            run_start = time.perf_counter()
            num_samples = self._run_single_simulation(settings, metadata)
            run_time = time.perf_counter() - run_start

            self._stats["total_samples"] += num_samples
            self._stats["total_runs"] += 1
            self._stats["total_time"] += run_time

            if verbose:
                logger.info(
                    f"   ✓ Completed: {num_samples} samples in {run_time:.2f}s "
                    f"({num_samples / run_time:.1f} samples/s)"
                )

        total_time = time.perf_counter() - start_time
        self._stats["wall_time"] = total_time

        logger.info("═══════════════════════════════════════════════════════════")
        logger.info("✅ Dataset generation complete")
        logger.info(f"   Total runs: {self._stats['total_runs']}")
        logger.info(f"   Total samples: {self._stats['total_samples']}")
        logger.info(f"   Wall time: {total_time:.2f}s")
        if total_time > 0:
            logger.info(
                f"   Throughput: {self._stats['total_samples'] / total_time:.1f} samples/s"
            )
        logger.info("═══════════════════════════════════════════════════════════")

        return self._stats

    def close(self) -> None:
        """Close publisher resources."""
        super().close()
        logger.info("✅ DataPublisher closed")


def main():
    """Main entry point for running the publisher standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the data publisher for online training",
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
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity",
    )

    args = parser.parse_args()

    config = DatasetConfig.from_yaml(args.config)
    publisher = DataPublisher(config, ipc_pub_addr=args.ipc_addr)

    try:
        publisher.run(verbose=not args.quiet)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
