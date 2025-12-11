from enum import Enum
from typing import Protocol, Dict, Any


class IMUSensorType(Enum):
    ROTORPY = "rotorpy"


class IMUSensorProtocol(Protocol):
    def measurement(
        self, state: Dict[str, Any], statedot: Dict[str, Any]
    ) -> Dict[str, Any]: ...
