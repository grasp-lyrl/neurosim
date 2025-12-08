import zmq
import time
import yaml
import asyncio
import numpy as np
from pathlib import Path
from collections import defaultdict

from utils import ZMQNODE

from rotorpy.vehicles.crazyflie_params import quad_params

# Controller
from rotorpy.controllers.quadrotor_control import SE3Control

# Trajectories
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.polynomial_traj import Polynomial


class ControllerNode(ZMQNODE):
    def __init__(
        self,
        settings: Path,
        control_rate: int = 100,
        ipc_pub_path: str = "/tmp/1",
        ipc_sub_path: str = "/tmp/0",
    ):
        """
        settings: Path to the settings file for the habitat scene and simulator.
        if None, loads default settings.
        """
        super(ControllerNode, self).__init__()

        self.state = None
        self.control_rate = control_rate
        self.ipc_pub_path = ipc_pub_path
        self.ipc_sub_path = ipc_sub_path

        # Load trajectory settings
        if settings is not None:
            with open(settings, "r") as file:
                settings = yaml.safe_load(file)
        trajectory = settings["trajectory"]
        start_position = settings["start_position"]

        #! Use Rotorpy controller for simulation for now
        self._controller = SE3Control(quad_params)
        if trajectory["type"] == "constant_speed":
            self._trajectory = ConstantSpeed(
                init_pos=start_position,
                dist=trajectory["dist"],
                v_avg=trajectory["v_avg"],
                axis=trajectory["axis"],
                repeat=True,
            )
        elif trajectory["type"] == "polynomial":
            self._trajectory = Polynomial(
                points=np.array(trajectory["points"]),
                v_avg=trajectory["v_avg"],
            )
        else:
            raise ValueError(
                "Invalid trajectory type. Use 'constant_speed' or 'polynomial'."
            )

        self._init_sockets()
        self._init_executors()

    def _init_sockets(self):
        """
        Initialize the ZMQ sockets for the controller.
        """
        self.socket_pub = self.create_socket(
            zmq.PUB,
            f"ipc://{self.ipc_pub_path}",
            setsockopt={
                zmq.SNDHWM: 1000,
                zmq.LINGER: 0,
                zmq.IMMEDIATE: 1,
            },
        )
        self.socket_sub_state = self.create_socket(
            zmq.SUB,
            f"ipc://{self.ipc_sub_path}",
            setsockopt={
                zmq.SUBSCRIBE: b"state",
            },
        )
        self.socket_sub_events = self.create_socket(
            zmq.SUB,
            f"ipc://{self.ipc_sub_path}",
            setsockopt={
                zmq.SUBSCRIBE: b"events",
            },
        )

    def _init_executors(self):
        """
        Initialize the executors for the controller.
        """
        # Publishers
        self.create_constant_rate_executor(self.publish_control, self.control_rate)
        self.create_constant_rate_executor(
            self.print_stats, 1
        )  # Print stats every second

        # Subscribers
        self.create_async_executor(self.receive_state)
        self.create_async_executor(self.receive_events)

        self._pub_sub_stats = defaultdict(int)

    async def publish_control(
        self,
    ):  ##################################### maybe need to separate the update controls and publish controls
        if self.state is None:
            print("State is not initialized. Cannot publish control.")
            return

        self.time = time.perf_counter() - self._cpu_clock_start_time
        flat = self._trajectory.update(self.time)
        control = self._controller.update(self.time, self.state, flat)

        control_msg = {
            "cmd_motor_speeds": control["cmd_motor_speeds"].tolist(),
        }

        if self.send_dict(self.socket_pub, control_msg, topic="control"):
            self._pub_sub_stats["sent_controls"] += 1

    async def receive_state(self):
        """
        Callback for the state subscriber.
        """
        _, msg = await self.recv_dict(self.socket_sub_state, copy=True)

        self.state = {
            "x": np.array(msg["x"]),
            "q": np.array(msg["q"]),
            "v": np.array(msg["v"]),
            "w": np.array(msg["w"]),
        }
        self._pub_sub_stats["received_states"] += 1

    async def receive_events(self):
        """
        Receive numpy array with zero-copy.

        Returns:
            Tuple of (array, message_counter, zmq_message) or None
            Note: Keep zmq_message alive as long as you need the array!
        """
        topic, arrays_dict = await self.recv_dict_of_arrays(self.socket_sub_events)

        if arrays_dict is not None:
            x = arrays_dict["x"]  # noqa
            y = arrays_dict["y"]  # noqa
            p = arrays_dict["p"]  # noqa
            t = arrays_dict["t"]  # noqa

            self._pub_sub_stats["received_event_packets"] += 1

            # process here or later

            return topic, arrays_dict

    async def print_stats(self):
        """
        Print statistics about the controller.
        """
        elapsed_time = time.perf_counter() - self._cpu_clock_start_time
        print("-" * 40)
        print(f"Elapsed controller time: {elapsed_time:.2f} seconds")
        for key, value in self._pub_sub_stats.items():
            rate = value / elapsed_time
            print(f"{key.replace('_', ' ').title()} Rate: {rate:.2f} Hz")
        print("-" * 40)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the controller node.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Path to the settings file for the habitat scene and simulator.",
    )
    parser.add_argument(
        "--ipc-pub-path",
        "-ipp",
        type=str,
        default="/tmp/1",
        help="IPC path for the publisher socket.",
    )
    parser.add_argument(
        "--ipc-sub-path",
        "-isp",
        type=str,
        default="/tmp/0",
        help="IPC path for the subscriber socket.",
    )
    parser.add_argument(
        "--control_rate",
        type=int,
        default=100,
        help="Control rate in Hz.",
    )
    args = parser.parse_args()

    controller_node = ControllerNode(
        settings=args.settings,
        control_rate=args.control_rate,
        ipc_pub_path=args.ipc_pub_path,
        ipc_sub_path=args.ipc_sub_path,
    )

    try:
        await controller_node.run()
    except KeyboardInterrupt:
        print("Controller node interrupted. Closing...")
    finally:
        await controller_node.close()


if __name__ == "__main__":
    asyncio.run(main())
