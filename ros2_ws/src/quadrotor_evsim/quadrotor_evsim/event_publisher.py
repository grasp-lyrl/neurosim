import copy
import torch
import numpy as np
from pathlib import Path

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params

import rclpy
from rclpy.node import Node

from quadrotor_interfaces.msg import Control, State, Events
from quadrotor_evsim.habitat_wrapper import HabitatWrapper

MAX_EVENT_BUFFER_SIZE = 400000


class HabitatROS2SimulatorNode(Node):
    def __init__(
        self,
        settings: Path,
        world_rate: int = 1000,  # Hz
        publish_rate: int = 100,  # Hz
    ):
        """
        settings: Path to the settings file for the habitat scene and simulator.
        if None, loads default settings.
        """
        super().__init__("habitatros2_simulator_node")
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

        # TODO: have no notion of logical time. Move to clock time or ROS time.
        self.time = 0
        self.frames = 0
        self.t_step = 1 / world_rate  # seconds

        self.publisher_state_ = self.create_publisher(State, "state", 10)
        self.publisher_events_ = self.create_publisher(Events, "events", 10)
        self.timer = self.create_timer(self.t_step, self.simulate_callback)
        self.subscriber_ = self.create_subscription(Control, "control", self.control_callback, 10)

        self.state = copy.deepcopy(self._quadsim.initial_state)
        self.control = {
            "cmd_motor_speeds": self._quadsim.initial_state["rotor_speeds"]
        }  # hover control

        self.publish_ctr = 0
        self.publish_steps = world_rate // publish_rate  # publish every few world steps
        self._event_size = 0
        self._event_buffer_x = torch.empty((MAX_EVENT_BUFFER_SIZE,), dtype=torch.uint16)
        self._event_buffer_y = torch.empty((MAX_EVENT_BUFFER_SIZE,), dtype=torch.uint16)
        self._event_buffer_p = torch.empty((MAX_EVENT_BUFFER_SIZE,), dtype=torch.uint8)
        self._event_buffer_t = torch.empty((MAX_EVENT_BUFFER_SIZE,), dtype=torch.uint64)

    def control_callback(self, msg):
        """
        Callback for the control subscriber.
        """
        # self.get_logger().info(f"Received control: {msg}")
        self.control = {"cmd_motor_speeds": [msg.rotor1, msg.rotor2, msg.rotor3, msg.rotor4]}

    def simulate_step(self, control):
        """
        Simulate one step of the environment.
        """
        self.time += self.t_step
        self.state = self._quadsim.step(self.state, control, self.t_step)

        position = np.array([self.state["x"][0], self.state["x"][2], -self.state["x"][1]])
        rotation = np.quaternion(
            self.state["q"][3], self.state["q"][0], self.state["q"][2], -self.state["q"][1]
        )
        self._hwrapper.update_agent_pose(0, position, rotation)
        intensity_img, events = self._hwrapper.render_events(0, self.time * 1e6)  # in us

        return intensity_img, events

    def simulate_callback(self):
        _, events = self.simulate_step(self.control)
        if events is not None:
            _new_event_size = events[0].shape[0] + self._event_size
            self._event_buffer_x[self._event_size : _new_event_size] = events[0].cpu()
            self._event_buffer_y[self._event_size : _new_event_size] = events[1].cpu()
            self._event_buffer_t[self._event_size : _new_event_size] = events[2].cpu()
            self._event_buffer_p[self._event_size : _new_event_size] = events[3].cpu()
            self._event_size = _new_event_size

            if (self.publish_ctr + 1) % self.publish_steps == 0:
                msg = State()
                msg.time = self.time
                msg.px, msg.py, msg.pz = self.state["x"]
                msg.qx, msg.qy, msg.qz, msg.qw = self.state["q"]
                msg.vx, msg.vy, msg.vz = self.state["v"]
                msg.wx, msg.wy, msg.wz = self.state["w"]
                self.publisher_state_.publish(msg)
                self.get_logger().info(f"Publishing {self._event_size} events")
                self._event_size = 0
        self.frames += 1
        self.publish_ctr += 1


def main(args=None):
    rclpy.init(args=args)

    cfg = "/workspace/configs/00014-nYYcLpSzihC-settings.yaml"
    minimal_publisher = HabitatROS2SimulatorNode(cfg)

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
