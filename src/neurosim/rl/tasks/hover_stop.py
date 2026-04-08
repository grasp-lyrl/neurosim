"""Hover-stop task implementation."""

import numpy as np

from .base import RLTask


class HoverStopTask(RLTask):
    """Reward and success logic for decelerating into a stable hover."""

    def __init__(
        self,
        w_velocity: float,
        w_events: float,
        w_angular: float,
        w_action: float,
        w_survival: float,
        crash_penalty: float,
        success_velocity_threshold: float,
        success_steps_required: int,
    ):
        self.w_velocity = float(w_velocity)
        self.w_events = float(w_events)
        self.w_angular = float(w_angular)
        self.w_action = float(w_action)
        self.w_survival = float(w_survival)
        self._crash_penalty = float(crash_penalty)
        self.success_velocity_threshold = float(success_velocity_threshold)
        self.success_steps_required = int(success_steps_required)
        self._consecutive_success_steps = 0

    @property
    def crash_penalty(self) -> float:
        return self._crash_penalty

    def on_reset(self) -> None:
        self._consecutive_success_steps = 0

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
        v = np.asarray(state["v"], dtype=np.float32)
        w = np.asarray(state["w"], dtype=np.float32)

        vel_norm = float(np.linalg.norm(v))
        ang_rate_norm = float(np.linalg.norm(w))
        if obs_mode == "state":
            event_activity = 0.0
            event_activity_density = 0.0
        else:
            event_activity = float(event_count)
            event_activity_density = event_activity / (event_shape[0] * event_shape[1])

        if prev_action is None:
            action_smoothness = 0.0
        else:
            action_smoothness = float(np.linalg.norm(action - prev_action))

        r_velocity = -self.w_velocity * vel_norm
        r_events = -self.w_events * event_activity_density
        r_angular = -self.w_angular * ang_rate_norm
        r_action = -self.w_action * action_smoothness
        r_survival = self.w_survival

        reward = r_velocity + r_events + r_angular + r_action + r_survival

        return reward, {
            "vel_norm": vel_norm,
            "ang_rate_norm": ang_rate_norm,
            "event_activity": event_activity,
            "event_activity_density": event_activity_density,
            "action_smoothness": action_smoothness,
            "r_velocity": r_velocity,
            "r_events": r_events,
            "r_angular": r_angular,
            "r_action": r_action,
            "r_survival": r_survival,
        }

    def check_success(self, *, state: dict[str, np.ndarray]) -> bool:
        vel_norm = float(np.linalg.norm(np.asarray(state["v"])))
        if vel_norm < self.success_velocity_threshold:
            self._consecutive_success_steps += 1
        else:
            self._consecutive_success_steps = 0
        if self.success_steps_required > 0:
            return self._consecutive_success_steps >= self.success_steps_required
        return False
