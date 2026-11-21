import logging
import numpy as np
from typing import Any
from rotorpy.vehicles.multirotor import Multirotor

from .types import DynamicsProtocol, DynamicsType
from neurosim.core.coord_trans import CoordinateTransform

logger = logging.getLogger(__name__)


def get_vehicle_params(vehicle_name: str = "crazyflie") -> dict[str, Any]:
    if vehicle_name == "crazyflie":
        from rotorpy.vehicles.crazyflie_params import quad_params

        return quad_params
    elif vehicle_name == "hummingbird":
        from rotorpy.vehicles.hummingbird_params import quad_params

        return quad_params
    elif vehicle_name == "px4_sihsim_quadx":
        from rotorpy.vehicles.px4_sihsim_quadx_params import quad_params

        # RotorPy ships no attitude gains for it, and its defaults (3000/360 on every axis) ask
        # ~180x the yaw torque the vehicle has; these fit its authority, with yaw softer as in
        # PX4 (roll/pitch: no saturation on 10 deg steps; yaw: 0.8 rad/s ramps). PX4's
        # allocation (MultirotorEuler) gives up yaw, not thrust, when the rotors saturate.
        return {
            **quad_params,
            "kp_att": np.array([150.0, 150.0, 30.0]),
            "kd_att": np.array([25.0, 25.0, 12.0]),
            "px4_allocation": True,
        }
    else:
        raise ValueError(f"Unknown vehicle name: {vehicle_name}")


def get_multirotor_model(
    dynamics_type: DynamicsType,
    vehicle_params: dict[str, Any],
    control_abstraction: str = "cmd_motor_speeds",
    enable_ground: bool = False,
) -> Multirotor:
    if dynamics_type == DynamicsType.ROTORPY_MULTIROTOR:
        quadsim = Multirotor(
            vehicle_params,
            control_abstraction=control_abstraction,
            aero=False,
            enable_ground=enable_ground,
            integrator_kwargs={"method": "RK45"},
        )
    elif dynamics_type == DynamicsType.ROTORPY_MULTIROTOR_EULER:
        from .multirotor_euler import MultirotorEuler

        quadsim = MultirotorEuler(
            vehicle_params,
            control_abstraction=control_abstraction,
            aero=False,
            enable_ground=enable_ground,
        )
    else:
        raise ValueError(f"Unsupported multirotor model: {dynamics_type}")

    return quadsim


class RotorpyDynamics(DynamicsProtocol):
    def __init__(
        self,
        vehicle: str = "crazyflie",
        dynamics_type: DynamicsType = DynamicsType.ROTORPY_MULTIROTOR,
        control_abstraction: str = "cmd_motor_speeds",
        initial_state: dict[str, np.ndarray] = {},
        enable_ground: bool = False,
    ):
        vehicle_params = get_vehicle_params(vehicle)
        self._multirotor = get_multirotor_model(
            dynamics_type,
            vehicle_params,
            control_abstraction=control_abstraction,
            enable_ground=enable_ground,
        )

        # a new dict: RotorPy shares its default initial_state between every vehicle
        self._multirotor.initial_state = {
            **self._multirotor.initial_state,
            **{k: np.asarray(v, dtype=np.float64) for k, v in initial_state.items()},
        }
        self._last_control = self.default_control()
        self._state = self._multirotor.initial_state

    def default_control(self) -> dict[str, np.ndarray | float]:
        """Hover, or motors idle when the vehicle starts on the ground."""
        control = self._default_control()
        if self._multirotor._enable_ground:
            for key in ("cmd_thrust", "cmd_motor_speeds", "cmd_motor_thrusts"):
                if key in control:
                    control[key] = control[key] * 0.0
        return control

    def _default_control(self) -> dict[str, np.ndarray | float]:
        abstraction = getattr(
            self._multirotor, "control_abstraction", "cmd_motor_speeds"
        )
        if abstraction == "cmd_ctbr":
            return {
                "cmd_thrust": float(self._multirotor.mass * self._multirotor.g),
                "cmd_w": np.zeros(3, dtype=np.float64),
            }
        if abstraction == "cmd_ctbm":
            return {
                "cmd_thrust": float(self._multirotor.mass * self._multirotor.g),
                "cmd_moment": np.zeros(3, dtype=np.float64),
            }
        if abstraction == "cmd_motor_thrusts":
            return {
                "cmd_motor_thrusts": np.zeros(
                    self._multirotor.num_rotors, dtype=np.float64
                )
            }
        if abstraction == "cmd_vel":
            return {"cmd_v": np.zeros(3, dtype=np.float64)}
        if abstraction == "cmd_ctatt":
            return {
                "cmd_thrust": float(self._multirotor.mass * self._multirotor.g),
                "cmd_q": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            }
        if abstraction == "cmd_acc":
            return {
                "cmd_acc": np.array([0.0, 0.0, self._multirotor.g], dtype=np.float64)
            }
        return {
            "cmd_motor_speeds": np.asarray(
                self._multirotor.initial_state["rotor_speeds"], dtype=np.float64
            )
        }

    @property
    def state(self) -> dict[str, np.ndarray]:
        return self._state

    @state.setter
    def state(self, value: dict[str, np.ndarray]) -> None:
        """Partially/fully update the state dictionary.

        Supports setting yaw/yaw_dot which are automatically converted to q/w.
        """
        # Handle yaw -> quaternion conversion if yaw is provided
        if "yaw" in value or "yaw_dot" in value:
            yaw = value.pop("yaw", 0.0)
            yaw_dot = value.pop("yaw_dot", 0.0)
            q, w = CoordinateTransform.euler_to_quat_and_body_rates(
                0.0, 0.0, yaw, 0.0, 0.0, yaw_dot
            )
            value["q"] = q
            value["w"] = w

        for k, v in value.items():
            if k in self._state:
                self._state[k] = v

    def step(self, control: Any, dt: float) -> dict[str, np.ndarray]:
        """Advance dynamics by one timestep and store the control."""
        self._last_control = control
        self._state = self._multirotor.step(self._state, control, dt)
        return self._state

    def statedot(self) -> dict[str, np.ndarray]:
        """Compute state derivatives using current state and last control."""
        return self._multirotor.statedot(self._state, self._last_control, 0)
