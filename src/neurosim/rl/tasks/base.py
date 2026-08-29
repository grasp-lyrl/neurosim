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

    def set_previous_action(self, action: np.ndarray | None) -> None:
        """Refresh only the previous-action field of the current context.

        The env populates context *before* computing reward, at which point
        "previous action" still means the action before the one being
        applied. The observation returned from that same step is consumed
        by the *next* decision, for which the previous action is the one
        just applied -- so it is refreshed here rather than leaving the
        policy to act on a correction that is a step stale.
        """

    def make_state_observation(
        self,
        *,
        state: dict[str, np.ndarray],
        base_state: np.ndarray,
    ) -> np.ndarray:
        """Build the vector observation for this task (used at reset)."""
        return np.asarray(base_state, dtype=np.float32)

    @property
    def privileged_observation_dim(self) -> int:
        """Width of the critic-only observation; 0 disables the channel.

        Asymmetric actor-critic: the value function may see ground-truth
        quantities the actor cannot perceive, which cuts value-estimation
        variance in exactly the states that drive the advantage signal.

        Privilege the critic only with information that is *in principle*
        recoverable from the actor's observation (e.g. obstacle geometry
        that is visible in the event stream). Under partial observability
        the variance-minimising baseline is ``E[V | actor observation]``,
        not ``V(true state)`` -- feeding the critic genuinely unknowable
        quantities makes it explain away variance the actor cannot act on,
        which *raises* advantage variance instead of lowering it.
        """
        return 0

    def make_privileged_observation(
        self,
        *,
        state: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Build the critic-only observation vector.

        Only called when :attr:`privileged_observation_dim` is positive.
        """
        return np.zeros(0, dtype=np.float32)
