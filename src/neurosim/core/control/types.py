from enum import Enum
from typing import Protocol, Dict, Any


class ControllerProtocol(Protocol):
    def update(
        self, time: float, state: Dict[str, Any], flat_out: Dict[str, Any]
    ) -> Any: ...


class ControllerType(Enum):
    ROTORPY_SE3 = "rotorpy_se3"
