import zmq
import zmq.asyncio

import json
import time
import struct
import asyncio
import numpy as np
from abc import ABC
from typing import Callable


class Executor:
    """
    Utility class for executing async functions at a constant rate.
    """

    def __init__(self, func: Callable, rate_hz: float = None):
        """
        Initialize constant rate executor.

        Args:
            rate_hz: Target execution rate in Hz
            func: Async function to execute
        """
        self.func = func
        self._running = False

        if rate_hz is None:
            self.run = self._async_run
        else:
            self.interval = 1.0 / rate_hz
            self.run = self._constant_rate_run

    @property
    def running(self) -> bool:
        """Check if the executor is running."""
        return self._running

    def start(self):
        """Start the executor."""
        self._running = True

    def stop(self):
        """Stop the executor."""
        self._running = False

    async def _async_run(self, *args, **kwargs) -> None:
        """Run the async function."""
        while self._running:
            try:
                await self.func(*args, **kwargs)
                await asyncio.sleep(0)  ##############################
            except Exception as e:
                print(f"Error in _async_run: {e}")
                await asyncio.sleep(0.001)  ###########################

    async def _constant_rate_run(self, *args, **kwargs) -> None:
        """
        Run a function at constant rate with precise timing.
        Executions happen at exact intervals regardless of execution time.

        Args:
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
        """
        next_exec_time = time.perf_counter()

        while self._running:
            try:
                if time.perf_counter() >= next_exec_time:
                    await self.func(*args, **kwargs)
                    next_exec_time += self.interval

                await asyncio.sleep(0)  ##############################
            except Exception as e:
                print(f"Error in _constant_rate_run: {e}")
                await asyncio.sleep(0.001)  ###########################

            await asyncio.sleep(
                min(
                    max(
                        0,
                        next_exec_time - time.perf_counter(),
                    ),
                    0.001,  ############################################
                )
            )


class ZMQNODE(ABC):
    def __init__(self, *args, **kwargs):
        # General settings
        self._running = False
        self._cpu_clock_start_time = None

        # Tasks and executors
        self._tasks = []
        self._async_executors = []
        self._constant_rate_executors = []

        # ZMQ context and sockets
        self._context = zmq.asyncio.Context()
        self._sockets = []

        # Sending Numpy Arrays support
        self.header_struct = struct.Struct(kwargs.get("header_format", "!IIII"))

        # Pre-compiled struct formats for common array dimensions.
        # Only support up to 6D arrays for efficiency.
        self._ndim_struct = struct.Struct("!B")  # 8-bit unsigned int (0-255)
        self._shape_structs = {
            1: struct.Struct("!I"),
            2: struct.Struct("!II"),
            3: struct.Struct("!III"),
            4: struct.Struct("!IIII"),
            5: struct.Struct("!IIIII"),
            6: struct.Struct("!IIIIII"),
        }

        # Cache for dtype strings to avoid repeated encoding
        self._dtype_cache = {}
        for dtype in [
            np.float64,
            np.float32,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint64,
            np.uint32,
            np.uint16,
            np.uint8,
            np.bool_,
            np.complex128,
            np.complex64,
        ]:
            dt = np.dtype(dtype)
            self._dtype_cache[dt.str] = dt.str.encode("ascii")

    def create_socket(self, socket_type: int, addr: str, setsockopt: dict = {}) -> zmq.Socket:
        """
        Create a ZMQ socket and connect/bind it to the specified address.

        Args:
            socket_type: Type of the ZMQ socket (zmq.PUB | zmq.SUB)
            addr: Address to connect/bind the socket
            **kwargs: Additional setsockopt options
        """
        socket = self._context.socket(socket_type)
        for option, value in setsockopt.items():
            socket.setsockopt(option, value)
        if socket_type == zmq.PUB:
            socket.bind(addr)
        else:
            socket.connect(addr)
        self._sockets.append(socket)
        return socket

    def create_constant_rate_executor(self, func: Callable, rate_hz: float):
        """
        Create a constant rate executor for the given function.

        Args:
            rate_hz: Target execution rate in Hz
            func: Async function to execute

        Returns:
            ConstantRateExecutor instance
        """
        self._constant_rate_executors.append(Executor(func, rate_hz))

    def create_async_executor(self, func: Callable):
        """
        Create an async executor for the given function.

        Args:
            func: Async function to execute

        Returns:
            AsyncExecutor instance
        """
        self._async_executors.append(Executor(func))

    def send_array(
        self,
        socket: zmq.Socket,
        array: np.ndarray,
        topic: str,
        copy: bool = True,
        flags: int = zmq.NOBLOCK,
    ) -> bool:
        """
        Send a Numpy array over ZMQ with a header.

        [topic, ndim, dtype, shape, data]

        Args:
            socket: ZMQ socket to send the array
            array: Numpy array to send
            topic: Topic to publish the array under

        Returns:
            bool: True if sent successfully, False if would block
        """
        try:
            ndim_data = self._ndim_struct.pack(array.ndim)
            # Use cached dtype
            dtype_code = self._dtype_cache[array.dtype.str]
            # Use precompiled struct for shape
            shape_data = self._shape_structs[array.ndim].pack(*array.shape)

            # Send multipart message: topic, header, dtype, shape, data
            socket.send_multipart(
                [topic.encode("utf-8"), ndim_data, dtype_code, shape_data, array],
                flags=flags,
                copy=copy,  # sometimes setting copy=True, might be necessary. Check your use-case
            )

            return True

        except zmq.Again:
            return False

    async def recv_array(
        self,
        socket: zmq.Socket,
        copy: bool = False,
    ) -> tuple[str, np.ndarray]:
        """
        Receive a Numpy array over ZMQ with a header.

        [topic, ndim, dtype, shape, data]

        Args:
            socket: ZMQ socket to receive the array
            copy: Whether to copy the data when receiving

        Returns:
            tuple: (topic, array) if received successfully, (None, None) if would block
        """
        try:
            messages = await socket.recv_multipart(copy=copy)

            if len(messages) != 5:
                print(f"Warning: Expected 5 parts, got {len(messages)}")
                return None, None

            topic_msg, ndim_msg, dtype_msg, shape_msg, array_msg = messages
            if not copy:
                topic_msg = topic_msg.buffer.tobytes()
                dtype_msg = dtype_msg.buffer.tobytes()
                ndim_msg = ndim_msg.buffer.tobytes()
                shape_msg = shape_msg.buffer
                array_msg = array_msg.buffer

            topic = topic_msg.decode("utf-8")
            dtype = np.dtype(dtype_msg.decode("ascii"))
            ndim = self._ndim_struct.unpack(ndim_msg)[0]

            # Use precompiled struct for shape
            shape = self._shape_structs[ndim].unpack(shape_msg)

            array = np.frombuffer(array_msg, dtype=dtype).reshape(shape)

            return topic, array
        except zmq.Again:
            return None, None
        except Exception as e:
            print(f"Error recv_array: {e}")
            return None, None

    def send_dict(
        self,
        socket: zmq.Socket,
        data: dict,
        topic: str,
        copy: bool = True,
        flags: int = zmq.NOBLOCK,
    ) -> bool:
        """
        Send a dictionary over ZMQ with a topic.

        Args:
            socket: ZMQ socket to send the data
            data: Dictionary to send
            topic: Topic to publish the data under

        Returns:
            bool: True if sent successfully, False if would block
        """
        try:
            socket.send_multipart(
                [topic.encode("utf-8"), json.dumps(data).encode("utf-8")],
                flags=flags,
                copy=copy,
            )
            return True
        except zmq.Again:
            return False

    async def recv_dict(
        self,
        socket: zmq.Socket,
        copy: bool = True,
    ) -> bool:
        """
        Receive a dictionary over ZMQ with a topic.

        Args:
            socket: ZMQ socket to receive the data
        Returns:
            tuple: (topic, data) if received successfully, None if would block
        """
        try:
            topic, msg = await socket.recv_multipart(copy=copy)
            if not copy:
                msg = msg.buffer.tobytes()
            msg2dict = json.loads(msg.decode("utf-8"))
            return topic, msg2dict
        except zmq.Again:
            return None, None
        except Exception as e:
            print(f"Error recv_dict: {e}")
            return None, None

    def send_dict_of_arrays(
        self,
        socket: zmq.Socket,
        data: dict[str, np.ndarray],
        topic: str,
        copy: bool = True,
        flags: int = zmq.NOBLOCK,
    ) -> bool:
        """
        Send a dictionary of numpy arrays over ZMQ with a topic.

        Message format: [topic, key1, ndim1, dtype1, shape1, array1, key2, ndim2, dtype2, shape2, array2, ...]

        Args:
            socket: ZMQ socket to send the data
            data: Dictionary of numpy arrays to send
            topic: Topic to publish the data under
            copy: Whether to copy the data when sending
            flags: ZMQ flags for sending

        Returns:
            bool: True if sent successfully, False if would block
        """
        try:
            # Pre-allocate: 1 topic + 5 parts per array
            message_parts = [None] * (1 + len(data) * 5)
            message_parts[0] = topic.encode("utf-8")

            idx = 1
            for key, array in data.items():
                if not isinstance(array, np.ndarray):
                    raise ValueError(f"Value for key '{key}' is not a numpy array")

                key_bytes = key.encode("ascii")

                # Use cached dtype
                dtype_code = self._dtype_cache[array.dtype.str]
                # Use precompiled struct for shape
                shape_data = self._shape_structs[array.ndim].pack(*array.shape)
                ndim_data = self._ndim_struct.pack(array.ndim)

                message_parts[idx] = key_bytes
                message_parts[idx + 1] = ndim_data
                message_parts[idx + 2] = dtype_code
                message_parts[idx + 3] = shape_data
                message_parts[idx + 4] = array
                idx += 5

            socket.send_multipart(message_parts, flags=flags, copy=copy)
            return True

        except zmq.Again:
            return False
        except Exception as e:
            print(f"Error in send_dict_of_arrays: {e}")
            return False

    async def recv_dict_of_arrays(
        self,
        socket: zmq.Socket,
        copy: bool = False,
    ) -> tuple[str, dict[str, np.ndarray]]:
        """
        Receive a dictionary of numpy arrays over ZMQ with a topic.

        Message format: [topic, key1, ndim1, dtype1, shape1, array1, key2, ndim2, dtype2, shape2, array2, ...]

        Args:
            socket: ZMQ socket to receive the data
            copy: Whether to copy the data when receiving

        Returns:
            tuple: (topic, dict_of_arrays) if received successfully, (None, None) if would block
        """
        try:
            messages = await socket.recv_multipart(copy=copy)

            if len(messages) < 6:
                print(f"Warning: Expected at least 6 parts, got {len(messages)}")
                return None, None

            if not copy:
                topic = messages[0].buffer.tobytes().decode("utf-8")
                result_dict = {}

                for msg_idx in range(1, len(messages), 5):
                    key = messages[msg_idx].buffer.tobytes().decode("ascii")
                    ndim = self._ndim_struct.unpack(messages[msg_idx + 1].buffer)[0]
                    dtype = np.dtype(messages[msg_idx + 2].buffer.tobytes().decode("ascii"))

                    # Use precompiled struct for shape
                    shape = self._shape_structs[ndim].unpack(messages[msg_idx + 3].buffer)

                    array = np.frombuffer(messages[msg_idx + 4].buffer, dtype=dtype).reshape(shape)
                    result_dict[key] = array
            else:
                topic = messages[0].decode("utf-8")
                result_dict = {}

                for msg_idx in range(1, len(messages), 5):
                    key = messages[msg_idx].decode("ascii")
                    ndim = self._ndim_struct.unpack(messages[msg_idx + 1])[0]
                    dtype = np.dtype(messages[msg_idx + 2].decode("ascii"))

                    # Use precompiled struct for shape
                    shape = self._shape_structs[ndim].unpack(messages[msg_idx + 3])

                    array = np.frombuffer(messages[msg_idx + 4], dtype=dtype).reshape(shape)
                    result_dict[key] = array

            return topic, result_dict

        except zmq.Again:
            return None, None
        except Exception as e:
            print(f"Error in recv_dict_of_arrays: {e}")
            return None, None

    async def run(self):
        """Run the ZMQ node."""
        self._running = True
        for executor in self._constant_rate_executors + self._async_executors:
            executor.start()

        self._tasks = [
            asyncio.create_task(executor.run())
            for executor in self._constant_rate_executors + self._async_executors
        ]

        try:
            print(f"{self.__class__.__name__} is running. Press Ctrl+C to stop.")
            self._cpu_clock_start_time = time.perf_counter()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.CancelledError:
            print(f"{self.__class__.__name__} stopped.")
        finally:
            self._running = False
            for executor in self._constant_rate_executors + self._async_executors:
                executor.stop()

            for task in self._tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def close(self):
        """Close the ZMQ node."""
        self._running = False
        for executor in self._constant_rate_executors + self._async_executors:
            executor.stop()
        print(f"Closing {self.__class__.__name__}...")
        for socket in self._sockets:
            socket.close()
        self._context.term()
        print(f"{self.__class__.__name__} closed.")
