from typing import Any
from rotorpy.vehicles.multirotor import Multirotor

from .types import DynamicsProtocol, DynamicsType


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
        initial_state: dict[str, Any] = {},
    ):
        vehicle_params = get_vehicle_params(vehicle)
        self._multirotor = get_multirotor_model(dynamics_type, vehicle_params)

        if initial_state:
            # Update keys, keeping rest of the initial state intact
            for k, v in initial_state.items():
                self._multirotor.initial_state[k] = v
        self._state = self._multirotor.initial_state

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]):
        self._state = value

    def step(self, control: Any, dt: float) -> dict[str, Any]:
        self._state = self._multirotor.step(self._state, control, dt)
        return self._state

    def statedot(self, state: dict[str, Any], control: Any, t: float) -> dict[str, Any]:
        return self._multirotor.statedot(state, control, t)
