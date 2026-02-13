"""
HDF5 Logger for Neurosim - Asynchronous batched writes.

Features:
- Batched writes: accumulates data, writes in large batches
- GPU-to-CPU conversion in writer process
- Sensor-aware batching (events: 50, images: 50, IMU/state: 100)
- Optimized chunking and optional compression

Usage:
    logger = H5Logger("output.h5", sensor_manager=sim.config.sensor_manager)
    logger.log(measurements, time=0.1, step=10)
    logger.close()
"""

import h5py
import torch
import logging
import numpy as np
from typing import Any
import multiprocessing as mp
from copy import deepcopy as dcopy

logger = logging.getLogger(__name__)


class H5Logger:
    """Asynchronous HDF5 Logger with batched writes."""

    # Batch sizes: how many samples to accumulate before writing
    BATCH_SIZES = {
        "event": 50,
        "color": 50,
        "depth": 50,
        "semantic": 50,
        "navmesh": 50,
        "imu": 100,
        "state": 100,
    }

    # Chunk sizes: HDF5 chunking for efficient I/O
    CHUNK_SIZES = {
        "event": 40000,  # Large chunks for event streams
        "color": 100,  # Medium chunks for images
        "depth": 100,
        "semantic": 100,
        "navmesh": 100,
        "imu": 1000,  # Small chunks for IMU data
        "state": 1000,  # Small chunks for state data
        "metadata": 1000,  # Small chunks for metadata (sim_time, sim_step)
    }

    IGNORED_SENSOR_TYPES = set()

    def __init__(
        self,
        filename: str,
        sensor_manager: Any = None,
        deepcopy_data: bool = False,
        compression: str = "lzf",
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.filename = filename
        self.deepcopy_data = deepcopy_data
        self.sensor_types = {"state": "state"}

        # Initialize HDF5 file and write sensor metadata
        with h5py.File(self.filename, "w", libver="latest") as f:
            if sensor_manager is not None:
                for uuid, cfg in sensor_manager.sensors.items():
                    self.sensor_types[uuid] = cfg.sensor_type

                    if cfg.sensor_type not in self.IGNORED_SENSOR_TYPES:
                        grp = f.create_group(uuid)
                        grp.attrs.update(cfg.config)

        ctx = mp.get_context("spawn")
        self.queue = ctx.Queue()
        self.process = ctx.Process(
            target=self._writer_process,
            args=(
                self.queue,
                self.filename,
                self.sensor_types,
                compression,
                self.verbose,
            ),
        )
        self.process.start()
        logger.info("═══════════════════════════════════════════════════════")
        logger.info("📝 H5Logger STARTED")
        logger.info(f"    File: {self.filename}")
        logger.info(f"    Available Sensors: {list(self.sensor_types.keys())}")
        logger.info(
            f"    Logged Sensor Types: {set(self.sensor_types.values()) - self.IGNORED_SENSOR_TYPES}"
        )
        logger.info("═══════════════════════════════════════════════════════")

    def log(self, measurements: dict[str, Any], time: float, step: int) -> None:
        """Log measurements to HDF5."""
        data = {"_sim_time": time, "_sim_step": step}
        for uuid, val in measurements.items():
            data[uuid] = dcopy(val) if self.deepcopy_data else val
        self.queue.put(data)

    def close(self) -> None:
        """Stop logger and flush remaining data."""
        self.queue.put(None)
        self.process.join()
        logger.info("═══════════════════════════════════════════════════════")
        logger.info("📝 H5Logger CLOSED")
        logger.info("═══════════════════════════════════════════════════════")

    @staticmethod
    def _torch_to_numpy(data):
        """Convert PyTorch tensors to NumPy."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, dict):
            return {k: H5Logger._torch_to_numpy(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(H5Logger._torch_to_numpy(x) for x in data)
        return data

    @staticmethod
    def _writer_process(queue, filename, sensor_types, compression, verbose=False):
        """Writer process: batches data and writes to HDF5."""
        buffers = {}  # uuid -> {data: [], sim_time: [], sim_step: []}

        def flush(f, uuid, buf, stype):
            if not buf or not buf["data"]:
                return
            grp = f[uuid] if uuid in f else f.create_group(uuid)
            times, steps = np.array(buf["sim_time"]), np.array(buf["sim_step"])

            if stype == "event":
                H5Logger._write_events(
                    grp,
                    buf["data"],
                    times,
                    steps,
                    H5Logger.CHUNK_SIZES["event"],
                    compression,
                )
            elif stype in ["color", "depth", "semantic", "navmesh"]:
                H5Logger._write_images(
                    grp,
                    buf["data"],
                    times,
                    steps,
                    H5Logger.CHUNK_SIZES[stype],
                    compression,
                )
            elif stype in ["imu", "state"]:
                H5Logger._write_dicts(
                    grp,
                    buf["data"],
                    times,
                    steps,
                    H5Logger.CHUNK_SIZES[stype],
                    compression,
                )
            else:
                H5Logger._write_generic(grp, buf["data"], times, steps, compression)

            buf["data"].clear()
            buf["sim_time"].clear()
            buf["sim_step"].clear()

        with h5py.File(filename, "a", libver="latest") as f:
            while True:
                batch = queue.get()
                if batch is None:
                    for uuid, buf in buffers.items():
                        flush(f, uuid, buf, sensor_types[uuid])
                    break

                sim_time, sim_step = batch.pop("_sim_time"), batch.pop("_sim_step")
                if verbose and sim_step % 1000 == 0:
                    print("═══════════════════════════════════════════════════════")
                    print(f"📝 H5 Writer: step={sim_step}, queue_size={queue.qsize()}")
                    print("═══════════════════════════════════════════════════════")

                for uuid, data in batch.items():
                    stype = sensor_types[uuid]

                    if stype in H5Logger.IGNORED_SENSOR_TYPES:
                        continue

                    if uuid not in buffers:
                        buffers[uuid] = {"data": [], "sim_time": [], "sim_step": []}

                    buffers[uuid]["data"].append(H5Logger._torch_to_numpy(data))
                    buffers[uuid]["sim_time"].append(sim_time)
                    buffers[uuid]["sim_step"].append(sim_step)

                    if len(buffers[uuid]["data"]) >= H5Logger.BATCH_SIZES[stype]:
                        flush(f, uuid, buffers[uuid], stype)

    @staticmethod
    def _write_events(grp, data_list, times, steps, chunk_size, compression):
        """Write batched event data by concatenating x,y,t,p arrays."""
        if not data_list:
            return
        names = ["x", "y", "t", "p"]
        for i, name in enumerate(names):
            arrays = [
                d[i]
                for d in data_list
                if isinstance(d, (tuple, list))
                and len(d) == 4
                and isinstance(d[i], np.ndarray)
                and len(d[i]) > 0
            ]
            if arrays:
                # Concatenate variable-length event streams
                H5Logger._append_concat(
                    grp, name, np.concatenate(arrays), chunk_size, compression
                )

        # Metadata uses stacking (fixed-size per sample)
        H5Logger._append_stack(
            grp, "sim_time", times, H5Logger.CHUNK_SIZES["metadata"], None
        )
        H5Logger._append_stack(
            grp, "sim_step", steps, H5Logger.CHUNK_SIZES["metadata"], None
        )

    @staticmethod
    def _write_images(grp, data_list, times, steps, chunk_size, compression):
        """Write batched image data."""
        if data_list:
            H5Logger._append_stack(
                grp, "data", np.stack(data_list), chunk_size, compression
            )
            H5Logger._append_stack(
                grp, "sim_time", times, H5Logger.CHUNK_SIZES["metadata"], None
            )
            H5Logger._append_stack(
                grp, "sim_step", steps, H5Logger.CHUNK_SIZES["metadata"], None
            )

    @staticmethod
    def _write_dicts(grp, data_list, times, steps, chunk_size, compression):
        """Write batched dictionary data (IMU/state)."""
        if not data_list:
            return
        for key in data_list[0].keys():
            values = [np.asarray(d[key]) for d in data_list]
            stacked = np.stack(values, axis=0)
            H5Logger._append_stack(grp, key, stacked, chunk_size, compression)
        H5Logger._append_stack(
            grp, "sim_time", times, H5Logger.CHUNK_SIZES["metadata"], None
        )
        H5Logger._append_stack(
            grp, "sim_step", steps, H5Logger.CHUNK_SIZES["metadata"], None
        )

    @staticmethod
    def _write_generic(grp, data_list, times, steps, compression):
        """Write batched generic data."""
        if not data_list:
            return
        # Default chunk size for unknown types
        chunk_size = 100
        if all(isinstance(d, np.ndarray) for d in data_list):
            H5Logger._append_stack(
                grp, "data", np.stack(data_list), chunk_size, compression
            )
        elif all(isinstance(d, dict) for d in data_list):
            for key in data_list[0].keys():
                values = [np.asarray(d[key]) for d in data_list]
                H5Logger._append_stack(
                    grp, key, np.stack(values), chunk_size, compression
                )
        H5Logger._append_stack(
            grp, "sim_time", times, H5Logger.CHUNK_SIZES["metadata"], None
        )
        H5Logger._append_stack(
            grp, "sim_step", steps, H5Logger.CHUNK_SIZES["metadata"], None
        )

    @staticmethod
    def _append_stack(grp, name, data, chunk_size, compression):
        """
        Append fixed-size data by stacking (e.g., images, IMU, state).
        Each sample has the same shape, stacked along axis 0.

        Examples:
        - Images: (H, W, C) -> dataset shape (N, H, W, C)
        - IMU accel: (3,) -> dataset shape (N, 3)
        - State position: (3,) -> dataset shape (N, 3)
        - Metadata scalars: () -> dataset shape (N,)
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim == 0:
            data = data.reshape(1)

        if name not in grp:
            # Create dataset with shape (0, ...) where ... matches data.shape[1:]
            if data.ndim == 1:
                # 1D data: scalars or metadata
                shape, maxshape = (0,), (None,)
                chunks = (chunk_size,)
            else:
                # Multi-dimensional: (N, d1, d2, ...)
                shape = (0,) + data.shape[1:]
                maxshape = (None,) + data.shape[1:]
                chunks = (chunk_size,) + data.shape[1:]

            grp.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                dtype=data.dtype,
                chunks=chunks,
                compression=compression,
            )

        dset = grp[name]
        old_size = dset.shape[0]
        new_size = old_size + data.shape[0]
        dset.resize(new_size, axis=0)
        dset[old_size:new_size] = data

    @staticmethod
    def _append_concat(grp, name, data, chunk_size, compression):
        """
        Append variable-length data by concatenation (e.g., event streams).
        Data is concatenated along axis 0, typically 1D arrays.

        Examples:
        - Event x: (N_events,) -> dataset shape (Total_events,)
        - Event y: (N_events,) -> dataset shape (Total_events,)
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim == 0:
            data = data.reshape(1)

        if name not in grp:
            # Create 1D dataset for concatenation
            shape = (0,)
            maxshape = (None,)

            grp.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                dtype=data.dtype,
                chunks=(chunk_size,),
                compression=compression,
            )

        dset = grp[name]
        old_size = dset.shape[0]
        new_size = old_size + data.shape[0]
        dset.resize(new_size, axis=0)
        dset[old_size:new_size] = data
