from typing import Dict, Any
from rotorpy.controllers.quadrotor_control import SE3Control
from .types import ControllerProtocol, ControllerType


def get_vehicle_params(vehicle_name: str = "crazyflie") -> Dict[str, Any]:
    if vehicle_name == "crazyflie":
        from rotorpy.vehicles.crazyflie_params import quad_params

        return quad_params
    elif vehicle_name == "hummingbird":
        from rotorpy.vehicles.hummingbird_params import quad_params

        return quad_params
    else:
        raise ValueError(f"Unknown vehicle name: {vehicle_name}")


class RotorpySE3Controller(ControllerProtocol):
    def __init__(
        self,
        vehicle: str = "crazyflie",
        controller_type: ControllerType = ControllerType.ROTORPY_SE3,
    ):
        vehicle_params = get_vehicle_params(vehicle)

        if controller_type == ControllerType.ROTORPY_SE3:
            self._controller = SE3Control(vehicle_params)
        else:
            raise ValueError(f"Unsupported controller type: {controller_type}")

    def update(
        self, time: float, state: Dict[str, Any], flat_out: Dict[str, Any]
    ) -> Any:
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_motor_thrusts, N
                cmd_thrust, N
                cmd_moment, N*m
                cmd_q, quaternion [i,j,k,w]
                cmd_w, angular rates in the body frame, rad/s
                cmd_v, velocity in the world frame, m/s
                cmd_acc, mass normalized thrust vector in the world frame, m/s/s.

                Not all keys are used, it depends on the control_abstraction selected when initializing the Multirotor object.
        """
        return self._controller.update(time, state, flat_out)
