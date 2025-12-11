from enum import Enum
from typing import Protocol, Dict, Any


class DynamicsType(Enum):
    ROTORPY_MULTIROTOR = "rotorpy_multirotor"
    ROTORPY_PX4_MULTIROTOR = "rotorpy_px4_multirotor"
    ROTORPY_ARDUPILOT_MULTIROTOR = "rotorpy_ardupilot_multirotor"
    ROTORPY_MULTIROTOR_EULER = "rotorpy_multirotor_euler"


class DynamicsProtocol(Protocol):
    """Protocol for dynamics models.

    The dynamics object maintains internal state and the last control input.
    The step() method updates both the state and last control.
    The statedot() method computes derivatives using the current state and last control.
    """

    @property
    def state(self) -> Dict[str, Any]:
        """Current state of the system."""
        ...

    @state.setter
    def state(self, value: Dict[str, Any]):
        """Set the current state of the system."""
        ...

    def step(self, control: Any, dt: float) -> Dict[str, Any]:
        """
        Advance the dynamics by one timestep.

        Args:
            control: Control input
            dt: Time step

        Returns:
            Updated state dictionary
        """
        ...

    def statedot(self) -> Dict[str, Any]:
        """
        Compute state derivatives using current state and last control.

        Returns:
            Dictionary of state derivatives
        """
        ...
