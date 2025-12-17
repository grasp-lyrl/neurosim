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
    else:
        raise ValueError(f"Unknown vehicle name: {vehicle_name}")


def get_multirotor_model(
    dynamics_type: DynamicsType, vehicle_params: dict[str, Any]
) -> Multirotor:
    if dynamics_type == DynamicsType.ROTORPY_MULTIROTOR:
        quadsim = Multirotor(
            vehicle_params, aero=False, integrator_kwargs={"method": "RK45"}
        )
    elif dynamics_type == DynamicsType.ROTORPY_MULTIROTOR_EULER:
        from .multirotor_euler import MultirotorEuler

        quadsim = MultirotorEuler(vehicle_params, aero=False)
    else:
        raise ValueError(f"Unsupported multirotor model: {dynamics_type}")

    return quadsim


class RotorpyDynamics(DynamicsProtocol):
    def __init__(
        self,
        vehicle: str = "crazyflie",
        dynamics_type: DynamicsType = DynamicsType.ROTORPY_MULTIROTOR,
        initial_state: dict[str, np.ndarray] = {},
    ):
        vehicle_params = get_vehicle_params(vehicle)
        self._multirotor = get_multirotor_model(dynamics_type, vehicle_params)

        if initial_state:
            # Update keys, keeping rest of the initial state intact
            for k, v in initial_state.items():
                self._multirotor.initial_state[k] = v
        self._last_control = {}
        self._state = self._multirotor.initial_state

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
