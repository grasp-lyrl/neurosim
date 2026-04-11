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
    def randomize(self, episode_count: int, rng: np.random.Generator) -> None:
        """Optionally resample vehicle-side randomization (see dynamics config)."""

    @abstractmethod
    def action_to_control(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
        """Convert normalized action to simulator control dictionary."""
