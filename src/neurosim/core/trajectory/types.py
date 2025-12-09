from enum import Enum
from typing import Protocol, Dict, Any


class TrajectoryType(Enum):
    CONSTANT_SPEED = "constant_speed"
    MINSNAP = "minsnap"
    POLYNOMIAL = "polynomial"


class TrajectoryProtocol(Protocol):
    def update(self, time: float) -> Dict[str, Any]: ...
