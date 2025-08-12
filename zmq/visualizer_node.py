import zmq
import asyncio
import rerun as rr
import numpy as np
from collections import defaultdict

from utils import ZMQNODE


class RRViz(ZMQNODE):
    def __init__(self, ipc_sub_path: str):
        """
        RRViz is a subclass of ZMQNODE that provides visualization capabilities.
        It inherits from ZMQNODE to utilize its ZeroMQ communication features.
        """
        super(RRViz, self).__init__()

        self.ipc_sub_path = ipc_sub_path

        self._init_sockets()
        self._init_executors()
        self._init_rr()

    def _init_sockets(self):
        """
        Initialize the ZMQ sockets for the visualizer.
        """
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
        self.socket_sub_color = self.create_socket(
            zmq.SUB,
            f"ipc://{self.ipc_sub_path}",
            setsockopt={
                zmq.SUBSCRIBE: b"color",
            },
        )
        self.socket_sub_depth = self.create_socket(
            zmq.SUB,
            f"ipc://{self.ipc_sub_path}",
            setsockopt={
                zmq.SUBSCRIBE: b"depth",
            },
        )
        self.socket_sub_imu = self.create_socket(
            zmq.SUB,
            f"ipc://{self.ipc_sub_path}",
            setsockopt={
                zmq.SUBSCRIBE: b"imu",
            },
        )

    def _init_executors(self):
        """
        Initialize the executors for the visualizer.
        """
        # Subscribers
        self.create_async_executor(self.receive_state)
        self.create_async_executor(self.receive_events)
        self.create_async_executor(self.receive_color)
        self.create_async_executor(self.receive_depth)
        self.create_async_executor(self.receive_imu)

        self._pub_sub_stats = defaultdict(int)

    def _init_rr(self):
        """
        Initialize Rerun for visualization.
        """
        rr.init("neurosim_viz", spawn=True)
        rr.set_time("stable_time", duration=0)

    async def receive_state(self):
        """
        Callback for the state subscriber.
        """
        _, msg = await self.recv_dict(self.socket_sub_state, copy=True)

        state = {
            "x": np.array(msg["x"]),
            "q": np.array(msg["q"]),
            "v": np.array(msg["v"]),
            "w": np.array(msg["w"]),
        }
        time = msg["timestamp"]
        self._pub_sub_stats["received_states"] += 1

        # Visualize the state using Rerun
        rr.set_time("stable_time", duration=time)
        rr.log(
            "navigation/pose",
            rr.Transform3D(
                translation=state["x"],
                rotation=rr.Quaternion(xyzw=state["q"]),
                axis_length=5.0,
                relation=rr.TransformRelation.ParentFromChild,
            ),
        )
        rr.log("navigation/trajectory", rr.Points3D(positions=state["x"][None, :]))

    async def receive_events(self):
        """
        Callback for the events subscriber.
        """
        _, events = await self.recv_dict_of_arrays(self.socket_sub_events, copy=True)

        if events is not None:
            self._pub_sub_stats["received_event_packets"] += 1
            if not hasattr(self, "event_img"):
                self.event_img = np.zeros(
                    (480, 640, 3), dtype=np.uint8
                )  #! replace with actual dimensions
            self.event_img[events["y"], events["x"], events["p"] * 2] = 255
            rr.log("events", rr.Image(self.event_img))
            self.event_img.fill(0)

    async def receive_color(self):
        """
        Callback for receiving color data.
        """
        _, color_img = await self.recv_array(self.socket_sub_color)
        if color_img is not None:
            self._pub_sub_stats["received_color_images"] += 1
            rr.log("color", rr.Image(color_img))

    async def receive_depth(self):
        """
        Callback for receiving depth data.
        """
        _, depth_img = await self.recv_array(self.socket_sub_depth)
        if depth_img is not None:
            self._pub_sub_stats["received_depth_images"] += 1
            rr.log("depth", rr.DepthImage(depth_img))

    async def receive_imu(self):
        """
        Receive IMU data and log it using Rerun.
        """
        _, imu_data = await self.recv_dict(self.socket_sub_imu, copy=True)
        if imu_data is not None:
            self._pub_sub_stats["received_imu_data"] += 1
            rr.set_time("stable_time", duration=imu_data["timestamp"])
            rr.log("imu/accel", rr.Scalars(imu_data["accel"]))
            rr.log("imu/gyro", rr.Scalars(imu_data["gyro"]))


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the controller node.")
    parser.add_argument(
        "--ipc-sub-path",
        "-isp",
        type=str,
        default="/tmp/0",
        help="IPC path for the subscriber socket.",
    )
    args = parser.parse_args()

    visualizer_node = RRViz(args.ipc_sub_path)

    try:
        await visualizer_node.run()
    except KeyboardInterrupt:
        print("Controller node interrupted. Closing...")
    finally:
        await visualizer_node.close()


if __name__ == "__main__":
    asyncio.run(main())
