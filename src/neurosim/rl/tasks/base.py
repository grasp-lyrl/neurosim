"""Task interfaces for Neurosim RL environments."""

from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from neurosim.rl.representations import EventRepresentationManager


@dataclass(slots=True)
class TaskStep:
    """Inputs handed to :meth:`RLTask.compute_reward` once per env step.

    The env packs this struct from its own state and per-step bookkeeping,
    so the task signature does not need to grow each time we add an input.
    """

    state: dict[str, np.ndarray]
    base_state: np.ndarray
    action: np.ndarray
    prev_action: np.ndarray | None
    sim_time: float
    dt: float
    event_manager: EventRepresentationManager
    obs_mode: str


@dataclass(slots=True)
class RewardOutcome:
    """Result of :meth:`RLTask.compute_reward`."""

    reward: float
    terms: dict[str, float] = field(default_factory=dict)


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
    def compute_reward(self, step: TaskStep) -> RewardOutcome:
        """Compute reward, term breakdown, and the post-step observation features."""

    @abstractmethod
    def check_success(self, *, state: dict[str, np.ndarray]) -> bool:
        """Return whether the current state satisfies task success criteria."""

    def check_terminated(self, *, state: dict[str, np.ndarray]) -> tuple[bool, str]:
        """Optional task-specific termination checks."""
        return False, ""

    @property
    def state_observation_dim(self) -> int:
        """Dimension of the vector observation for this task."""
        return 13

    @property
    def action_dim(self) -> int | None:
        """Override the vehicle action dimension when the task owns action semantics."""
        return None

    @property
    def uses_nominal_controller(self) -> bool:
        """Whether the environment should build trajectory/controller context."""
        return False

    def set_context(self, context: dict[str, Any]) -> None:
        """Receive environment-computed context before reward/observation calls."""

    def make_state_observation(
        self,
        *,
        state: dict[str, np.ndarray],
        base_state: np.ndarray,
    ) -> np.ndarray:
        """Build the vector observation for this task (used at reset)."""
        return np.asarray(base_state, dtype=np.float32)
