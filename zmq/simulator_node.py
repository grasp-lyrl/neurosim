import zmq
import copy
import time
import asyncio
import numpy as np
from pathlib import Path
from collections import defaultdict

from neurosim.habitat_wrapper import HabitatWrapper
from utils import ZMQNODE

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

# Sensors
from rotorpy.sensors.imu import Imu


class SimulatorNode(ZMQNODE):

    MAX_EVENT_BUFFER_SIZE = 4000000

    def __init__(
        self,
        settings: Path,
        world_rate: int = 1000,
        publish_rate: int = 100,
        ipc_pub_path: str = "/tmp/0",
        ipc_sub_path: str = "/tmp/1",
    ):
        """
        settings: Path to the settings file for the habitat scene and simulator.
        if None, loads default settings.
        """
        super(SimulatorNode, self).__init__()

        # Visual Simulator instance
        self._hwrapper = HabitatWrapper(settings)

        # Dynamics Simulator instance
        self._quadsim = Multirotor(quad_params, aero=False, integration_method="euler")
        quad_state = self._hwrapper.agent.get_state()
        self._quadsim.initial_state["x"] = quad_state.position
        self._quadsim.initial_state["q"] = np.array(
            [
                quad_state.rotation.x,
                quad_state.rotation.y,
                quad_state.rotation.z,
                quad_state.rotation.w,
            ]
        )

        # Quad INIT
        self.state = copy.deepcopy(self._quadsim.initial_state)
        self.control = {"cmd_motor_speeds": self._quadsim.initial_state["rotor_speeds"]}

        # Load IMU sensor

        self.time = 0  # time in seconds
        self.simsteps = 0  # number of simulation steps
        self.t_step = 1 / world_rate  # seconds
        self.world_rate = world_rate  # Hz
        self.publish_rate = publish_rate  # Hz
        self.ipc_pub_path = ipc_pub_path
        self.ipc_sub_path = ipc_sub_path

        self._init_sensors_and_buffers()
        self._init_sockets()
        self._init_executors()

    def _init_sensors_and_buffers(self):
        """
        Initialize sensors and buffers for the simulator.
        """
        # Sensors
        self.has_imu_sensor = self._hwrapper.settings["imu_sensor"]
        self.imu_sampling_rate = self._hwrapper.settings.get("imu_sampling_rate", 100)
        self.imu = Imu(p_BS=np.zeros(3), R_BS=np.eye(3), sampling_rate=self.imu_sampling_rate)

        self.has_color_sensor = self._hwrapper.settings["color_sensor"]
        self.color_sensor_rate = self._hwrapper.settings.get("color_sensor_rate", 25)

        self.has_depth_sensor = self._hwrapper.settings["depth_sensor"]
        self.depth_sensor_rate = self._hwrapper.settings.get("depth_sensor_rate", 10)

        # Buffers
        self._event_size = 0
        self._event_buffer = [
            np.empty((self.MAX_EVENT_BUFFER_SIZE,), dtype=np.uint16),
            np.empty((self.MAX_EVENT_BUFFER_SIZE,), dtype=np.uint16),
            np.empty((self.MAX_EVENT_BUFFER_SIZE,), dtype=np.uint8),
            np.empty((self.MAX_EVENT_BUFFER_SIZE,), dtype=np.uint64),
        ]
        self._imu_data = None
        self._color_img = None
        self._depth_img = None

    def _init_sockets(self):
        """
        Initialize the ZMQ sockets for the simulator.
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
        self.socket_control = self.create_socket(
            zmq.SUB,
            f"ipc://{self.ipc_sub_path}",
            setsockopt={
                zmq.SUBSCRIBE: b"control",
            },
        )

    def _init_executors(self):
        """
        Initialize the executors for the simulator.
        """
        # Simulators and Publishers
        self.create_constant_rate_executor(self.print_stats, 1)  # Print stats every second
        self.create_constant_rate_executor(self.simulate_control_and_events, self.world_rate)
        self.create_constant_rate_executor(self.publish_state, self.publish_rate)
        self.create_constant_rate_executor(self.publish_events, self.publish_rate)

        # Sensors
        if self.has_imu_sensor:
            self.create_constant_rate_executor(
                self.simulate_and_publish_imu_sensor, self.imu_sampling_rate
            )
        if self.has_color_sensor:
            self.create_constant_rate_executor(
                self.simulate_and_publish_color_sensor, self.color_sensor_rate
            )
        if self.has_depth_sensor:
            self.create_constant_rate_executor(
                self.simulate_and_publish_depth_sensor, self.depth_sensor_rate
            )

        # Subscribers
        self.create_async_executor(self.receive_control)
        self._pub_sub_stats = defaultdict(int)

    async def receive_control(self):
        """
        Callback for the control subscriber.
        """
        _, msg = await self.recv_dict(self.socket_control, copy=True)

        if "cmd_motor_speeds" in msg:
            self.control["cmd_motor_speeds"] = np.array(msg["cmd_motor_speeds"])
            self._pub_sub_stats["received_controls"] += 1
        else:
            print("Received control message without 'cmd_motor_speeds' key.")

    def simulate_control(self):
        """
        Simulate one step of the environment.
        """
        self.time += self.t_step
        self.simsteps += 1
        self.state = self._quadsim.step(self.state, self.control, self.t_step)

        position = np.array([self.state["x"][0], self.state["x"][2], -self.state["x"][1]])
        rotation = np.quaternion(
            self.state["q"][3],
            self.state["q"][0],
            self.state["q"][2],
            -self.state["q"][1],
        )
        self._hwrapper.update_agent_pose(0, position, rotation)

    def simulate_events(self):
        events = self._hwrapper.render_events(self.time * 1e6, to_numpy=True)  # in us
        if events is not None:
            _old_event_size = self._event_size
            _new_event_size = events[0].shape[0] + _old_event_size
            if _new_event_size > self.MAX_EVENT_BUFFER_SIZE:
                print(f"Event buffer overflow: {self._event_size}/{self.MAX_EVENT_BUFFER_SIZE}")
            else:
                self._event_buffer[0][_old_event_size:_new_event_size] = events[0]
                self._event_buffer[1][_old_event_size:_new_event_size] = events[1]
                self._event_buffer[2][_old_event_size:_new_event_size] = events[2]
                self._event_buffer[3][_old_event_size:_new_event_size] = events[3]
                self._event_size = _new_event_size

    async def simulate_control_and_events(self):
        self.simulate_control()
        self.simulate_events()

    async def simulate_and_publish_imu_sensor(self):
        statedot = self._quadsim.statedot(self.state, self.control, 0)
        self.imu_data = self.imu.measurement(self.state, statedot, with_noise=False)
        imu_msg = {
            "accel": self.imu_data["accel"].tolist(),
            "gyro": self.imu_data["gyro"].tolist(),
            "timestamp": self.time,
        }
        if self.send_dict(self.socket_pub, imu_msg, topic="imu"):
            self._pub_sub_stats["sent_imu_data"] += 1

    async def simulate_and_publish_color_sensor(self):
        self._color_img = self._hwrapper.render_color_sensor().cpu().numpy()
        if self.send_array(self.socket_pub, self._color_img, topic="color"):
            self._pub_sub_stats["sent_color_images"] += 1

    async def simulate_and_publish_depth_sensor(self):
        self._depth_img = self._hwrapper.render_depth_sensor().cpu().numpy()
        if self.send_array(self.socket_pub, self._depth_img, topic="depth"):
            self._pub_sub_stats["sent_depth_images"] += 1

    async def publish_state(self):
        """
        Publish the current state of the simulator.
        """
        state_msg = {
            "x": self.state["x"].tolist(),  # [x, y, z]
            "q": self.state["q"].tolist(),  # [qx, qy, qz, qw]
            "v": self.state["v"].tolist(),  # [vx, vy, vz]
            "w": self.state["w"].tolist(),  # [wx, wy, wz]
            "timestamp": self.time,
            "simsteps": self.simsteps,
        }

        if self.send_dict(self.socket_pub, state_msg, topic="state"):
            self._pub_sub_stats["sent_states"] += 1

    async def publish_events(self):
        """Async method to publish event arrays."""
        if self._event_size == 0:
            return

        sent_success = True
        _size = self._event_size
        for _buf, _key in zip(self._event_buffer, ["x", "y", "p", "t"]):
            sent_success &= self.send_array(self.socket_pub, _buf[:_size], topic=f"events/{_key}")

        if sent_success:
            self._pub_sub_stats["sent_event_packets"] += 4
            self._event_size = 0

    async def print_stats(self):
        """Print statistics about the simulator."""
        elapsed = time.perf_counter() - self._cpu_clock_start_time
        print("-" * 40)
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Current simulator time: {self.time:.2f} seconds")
        print(f"Simulator running at {self.simsteps / elapsed:.2f} Hz")
        print(f"Event buffer size: {self._event_size}/{self.MAX_EVENT_BUFFER_SIZE}")
        print(f"State position: {self.state['x']}")
        for key, value in self._pub_sub_stats.items():
            rate = value / elapsed
            print(f"{key.replace('_', ' ').title()} Rate: {rate:.2f} Hz")
        print("-" * 40)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the simulator node.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("settings.json"),
        help="Path to the settings file for the habitat scene and simulator.",
    )
    parser.add_argument(
        "--ipc-pub-path",
        "-ipp",
        type=str,
        default="/tmp/0",
        help="IPC path for the publisher socket.",
    )
    parser.add_argument(
        "--ipc-sub-path",
        "-isp",
        type=str,
        default="/tmp/1",
        help="IPC path for the subscriber socket.",
    )
    parser.add_argument(
        "--world_rate",
        type=int,
        default=1000,
        help="World simulation rate in Hz.",
    )
    parser.add_argument(
        "--publish_rate",
        type=int,
        default=100,
        help="Rate to publish sensor data in Hz.",
    )
    args = parser.parse_args()

    simulator_node = SimulatorNode(
        settings=args.settings,
        world_rate=args.world_rate,
        publish_rate=args.publish_rate,
    )

    try:
        await simulator_node.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping the simulator node.")
    finally:
        await simulator_node.close()


if __name__ == "__main__":
    asyncio.run(main())
