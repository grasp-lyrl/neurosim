"""Task interfaces for Neurosim RL environments."""

from abc import ABC, abstractmethod
import numpy as np


class RLTask(ABC):
    """Interface for task-specific reward, success, and termination logic."""

    @property
    @abstractmethod
    def crash_penalty(self) -> float:
        """Penalty applied by the environment when the episode terminates unsafely."""

    @abstractmethod
    def on_reset(self) -> None:
        """Reset task-specific episode state."""

    @abstractmethod
    def compute_reward(
        self,
        *,
        state: dict[str, np.ndarray],
        action: np.ndarray,
        prev_action: np.ndarray | None,
        event_count: int,
        event_shape: tuple[int, int],
        obs_mode: str,
    ) -> tuple[float, dict[str, float]]:
        """Compute reward and structured reward terms."""

    @abstractmethod
    def check_success(self, *, state: dict[str, np.ndarray]) -> bool:
        """Return whether the current state satisfies task success criteria."""

    def check_terminated(self, *, state: dict[str, np.ndarray]) -> tuple[bool, str]:
        """Optional task-specific termination checks."""
        _ = state
        return False, ""
