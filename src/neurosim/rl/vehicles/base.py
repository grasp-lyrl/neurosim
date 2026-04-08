"""Vehicle interfaces for RL environments."""

from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces


class RLVehicle(ABC):
    """Maps normalized policy actions to simulator control commands."""

    @property
    @abstractmethod
    def action_space(self) -> spaces.Box:
        """Normalized policy action space."""

    @abstractmethod
    def on_reset(self, rng: np.random.Generator) -> None:
        """Apply optional episode-level updates (e.g., randomization)."""

    @abstractmethod
    def action_to_control(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
        """Convert normalized action to simulator control dictionary."""
