from enum import Enum
from typing import Protocol, Any


class TrajectoryType(Enum):
    CONSTANT_SPEED = "constant_speed"
    MINSNAP = "minsnap"
    POLYNOMIAL = "polynomial"
    HABITAT_RANDOM_MINSNAP = "habitat_random_minsnap"


class TrajectoryProtocol(Protocol):
    def update(self, time: float) -> dict[str, Any]: ...
