from enum import Enum
from typing import Protocol, Dict, Any


class DynamicsType(Enum):
    ROTORPY_MULTIROTOR = "rotorpy_multirotor"
    ROTORPY_PX4_MULTIROTOR = "rotorpy_px4_multirotor"
    ROTORPY_ARDUPILOT_MULTIROTOR = "rotorpy_ardupilot_multirotor"
    ROTORPY_MULTIROTOR_EULER = "rotorpy_multirotor_euler"


class DynamicsProtocol(Protocol):
    @property
    def state(self) -> Dict[str, Any]: ...

    @state.setter
    def state(self, value: Dict[str, Any]): ...

    def step(self, control: Any, dt: float) -> Dict[str, Any]: ...

    def statedot(
        self, state: Dict[str, Any], control: Any, t: float
    ) -> Dict[str, Any]: ...
