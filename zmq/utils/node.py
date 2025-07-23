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

        Args:
            socket: ZMQ socket to send the array
            array: Numpy array to send
            topic: Topic to publish the array under

        Returns:
            bool: True if sent successfully, False if would block
        """
        try:
            dtype_code = array.dtype.str.encode("ascii")
            shape_data = struct.pack("!" + "I" * len(array.shape), *array.shape)

            header = self.header_struct.pack(
                array.ndim, len(dtype_code), len(shape_data), array.nbytes
            )

            # Send multipart message: topic, header, dtype, shape, data
            socket.send_multipart(
                [topic.encode("utf-8"), header, dtype_code, shape_data, array],
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
        try:
            messages = await socket.recv_multipart(copy=copy)

            if len(messages) != 5:
                print(f"Warning: Expected 5 parts, got {len(messages)}")
                return None, None

            topic_msg, header_msg, dtype_msg, shape_msg, array_msg = messages
            if not copy:
                topic_msg = topic_msg.buffer.tobytes()
                dtype_msg = dtype_msg.buffer.tobytes()
                header_msg = header_msg.buffer
                shape_msg = shape_msg.buffer
                array_msg = array_msg.buffer

            topic = topic_msg.decode("utf-8")
            dtype = np.dtype(dtype_msg.decode("ascii"))

            ndim, dtype_len, shape_len, nbytes = self.header_struct.unpack(header_msg)
            shape = struct.unpack("!" + "I" * ndim, shape_msg)

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
