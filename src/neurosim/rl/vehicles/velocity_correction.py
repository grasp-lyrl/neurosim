"""Velocity-correction vehicle for trajectory-dodge RL tasks.

The policy outputs a 3-D velocity correction that is added to the desired
velocity from the trajectory before being passed to the SE3 controller.  This
vehicle does **not** produce motor or body-rate commands directly.
"""

from typing import Any

import numpy as np
from gymnasium import spaces

from .base import RLVehicle


class VelocityCorrectionVehicle(RLVehicle):
    """Maps normalized [-1, 1]^3 actions to a velocity correction vector.

    Parameters
    ----------
    max_correction_mps : float
        Maximum per-axis velocity correction magnitude (m/s).
    """

    def __init__(self, *, max_correction_mps: float = 2.0, **_kwargs: Any):
        self._max_correction = float(max_correction_mps)
        self._action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    def randomize(self, episode_count: int, rng: np.random.Generator) -> None:
        """No vehicle-side randomization for velocity correction."""

    def action_to_correction(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action to a velocity correction in m/s."""
        action = np.asarray(action, dtype=np.float32)
        return np.clip(action, -1.0, 1.0) * self._max_correction

    def action_to_control(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
        """Not used directly — the env composes correction + SE3 controller.

        Returns a dummy dict to satisfy the interface.
        """
        raise NotImplementedError(
            "VelocityCorrectionVehicle does not produce control commands directly. "
            "Use action_to_correction() and compose with the SE3 controller."
        )
