import zmq
import asyncio
import rerun as rr
import numpy as np

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
        self.socket_state = self.create_socket(
            zmq.SUB,
            f"ipc://{self.ipc_sub_path}",
            setsockopt={
                zmq.SUBSCRIBE: b"state",
            },
        )

    def _init_executors(self):
        """
        Initialize the executors for the visualizer.
        """
        # Subscribers
        self.create_async_executor(self.receive_state)

        self._pub_sub_stats = {
            "received_states": 0,
        }

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
        _, msg = await self.recv_dict(self.socket_state, copy=True)

        print(f"Received state: {msg}")

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
