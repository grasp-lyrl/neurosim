import yaml
import numpy as np
from pathlib import Path

from rotorpy.vehicles.crazyflie_params import quad_params

from rotorpy.controllers.quadrotor_control import SE3Control

# Trajectories
from rotorpy.trajectories.speed_traj import ConstantSpeed
from rotorpy.trajectories.polynomial_traj import Polynomial

import rclpy
from rclpy.node import Node

from quadrotor_interfaces.msg import Control, State


class RotorpySE3ControllerNode(Node):
    def __init__(
        self,
        settings: Path,
        control_rate: int = 100,
    ):
        """
        settings: Path to the settings file for the habitat scene and simulator.
        if None, loads default settings.
        """
        super().__init__("rotorpy_se3controller_node")

        # Load trajectory settings
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
            raise ValueError("Invalid trajectory type. Use 'constant_speed' or 'polynomial'.")

        timer_period = 1 / control_rate
        self.timer = self.create_timer(timer_period, self.control_callback)
        self.publisher_ = self.create_publisher(Control, "control", 10)
        self.subscriber_ = self.create_subscription(State, "state", self.state_callback, 10)

        self.time = 0
        self.state = None
        self.old_time = 0

    def control_callback(self):
        """
        Simulate the trajectory.
        """
        if self.state is None:
            self.get_logger().warn("State is not initialized. Waiting for state update.")
            return
        flat = self._trajectory.update(self.time)
        control = self._controller.update(self.time, self.state, flat)
        msg = Control()
        msg.rotor1, msg.rotor2, msg.rotor3, msg.rotor4 = control["cmd_motor_speeds"]
        self.publisher_.publish(msg)

    def state_callback(self, msg):
        """
        Callback for the state subscriber.
        """
        self.time = msg.time
        self.state = {
            "x": np.array([msg.px, msg.py, msg.pz]),
            "q": np.array([msg.qx, msg.qy, msg.qz, msg.qw]),
            "v": np.array([msg.vx, msg.vy, msg.vz]),
            "w": np.array([msg.wx, msg.wy, msg.wz]),
        }
        # get the ros time
        rostime = self.get_clock().now().nanoseconds / 1e6  # in ms
        self.get_logger().info(f"Event time: {rostime - self.old_time} ms")
        self.old_time = rostime


def main(args=None):
    rclpy.init(args=args)

    cfg = "/workspace/configs/00014-nYYcLpSzihC-settings.yaml"
    minimal_publisher = RotorpySE3ControllerNode(cfg)

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
