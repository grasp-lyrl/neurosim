"""Constrained PPO wrapper around a frozen event-BC commitment policy.

PPO can make a small correction to the BC trigger logit and proposed y/z
direction only before a pulse commits.  The selected smooth, non-reversing
pulse remains fixed after commitment, and no correction is accepted without
a real event-active tracker detection.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EventPulseResidualWrapper(gym.Wrapper):
    """Expose frozen event-policy features and execute pulse-level residuals."""

    FEATURE_NAMES = (
        "base_gate_probability",
        "base_direction_y",
        "base_direction_z",
        "tracker_probability",
        "inbound_probability",
        "event_activity",
        "adjustment_enabled",
        "confirm_progress",
        "pulse_active",
        "pulse_progress",
        "refractory",
        "absence_progress",
        "latched_direction_y",
        "latched_direction_z",
        "previous_residual_gate",
        "previous_residual_y",
        "previous_residual_z",
    )

    def __init__(
        self,
        env: gym.Env,
        *,
        action_checkpoint: str,
        device: str = "cuda:0",
        gate_logit_scale: float = 1.0,
        direction_scale: float = 0.25,
        tracker_threshold: float = 0.7,
        min_event_activity: float = 1e-8,
        intervention: str = "real",
        pulse_kwargs: dict[str, Any] | None = None,
        policy: Any | None = None,
    ):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError("event pulse PPO requires a Dict observation")
        base_spaces = env.observation_space.spaces
        if "events" not in base_spaces or "state" not in base_spaces:
            raise ValueError("event pulse PPO requires events and state")
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("event pulse PPO requires a Box action space")

        if policy is None:
            from train_event_tracker_action_head import EventActionPolicy

            policy = EventActionPolicy(
                action_checkpoint,
                device=device,
                intervention=intervention,
                structured_pulse=True,
                pulse_kwargs=pulse_kwargs,
            )
        if policy.pulse_controller is None:
            raise ValueError("event pulse PPO requires a structured pulse policy")

        self.policy = policy
        self.gate_logit_scale = float(gate_logit_scale)
        self.direction_scale = float(direction_scale)
        self.tracker_threshold = float(tracker_threshold)
        self.min_event_activity = float(min_event_activity)
        if self.gate_logit_scale < 0.0 or self.direction_scale < 0.0:
            raise ValueError("residual scales must be non-negative")

        input_dim = int(np.asarray(self.policy.mean).size)
        compact_dim = input_dim + len(self.FEATURE_NAMES)
        output_spaces: dict[str, spaces.Space] = {
            "state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(compact_dim,), dtype=np.float32
            )
        }
        if "privileged" in base_spaces:
            output_spaces["privileged"] = base_spaces["privileged"]
        self.observation_space = spaces.Dict(output_spaces)
        # [trigger-logit correction, direction-y correction, direction-z correction]
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

        self.last_residual = np.zeros(3, dtype=np.float32)
        self.last_event_activity = 0.0

    @property
    def controller(self):
        return self.policy.pulse_controller

    def _adjustment_is_enabled(self) -> bool:
        controller = self.controller
        return bool(
            self.last_event_activity > self.min_event_activity
            and self.policy.last_prediction.probability >= self.tracker_threshold
            and controller.pulse_step is None
            and not controller.refractory
        )

    def _infer(self, observation: dict[str, np.ndarray]) -> None:
        events = np.asarray(observation["events"])
        self.last_event_activity = float(np.mean(np.abs(events) > 1e-6))
        self.policy.infer(observation)

    def _compact_observation(self, observation):
        controller = self.controller
        pulse_active = controller.pulse_step is not None
        pulse_progress = (
            float(controller.pulse_step)
            / max(controller.pulse_end_step - 1, 1)
            if pulse_active
            else 0.0
        )
        features = np.asarray(
            [
                self.policy.last_gate_probability,
                *self.policy.last_direction,
                self.policy.last_prediction.probability,
                self.policy.last_prediction.inbound_probability,
                min(100.0 * self.last_event_activity, 2.0),
                float(self._adjustment_is_enabled()),
                controller.confirmed / max(controller.confirm_steps, 1),
                float(pulse_active),
                pulse_progress,
                float(controller.refractory),
                controller.absent_steps
                / max(controller.refractory_absence_steps, 1),
                *controller.direction,
                *self.last_residual,
            ],
            dtype=np.float32,
        )
        normalized_input = (
            np.asarray(self.policy.last_input, dtype=np.float32) - self.policy.mean
        ) / self.policy.std
        compact = {
            "state": np.concatenate([normalized_input, features]).astype(
                np.float32, copy=False
            )
        }
        if "privileged" in observation:
            compact["privileged"] = np.asarray(
                observation["privileged"], dtype=np.float32
            )
        return compact

    @staticmethod
    def _logit(probability: float) -> float:
        probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        return float(np.log(probability) - np.log1p(-probability))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.policy.reset()
        self.last_residual.fill(0.0)
        self._infer(observation)
        return self._compact_observation(observation), info

    def step(self, action):
        residual = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        enabled = self._adjustment_is_enabled()
        base_gate = float(self.policy.last_gate_probability)
        base_direction = self.policy.last_direction.copy()

        adjusted_gate = base_gate
        adjusted_direction = base_direction
        if enabled:
            adjusted_logit = self._logit(base_gate) + self.gate_logit_scale * float(
                residual[0]
            )
            adjusted_gate = float(1.0 / (1.0 + np.exp(-adjusted_logit)))
            adjusted_direction = np.clip(
                base_direction + self.direction_scale * residual[1:], -1.0, 1.0
            )

        # Advance a copy to record the exact no-residual baseline without
        # perturbing the real pulse state.
        baseline_controller = deepcopy(self.controller)
        track_present = (
            self.policy.last_event_active
            and self.policy.last_prediction.probability >= self.tracker_threshold
        )
        release_threshold = baseline_controller.release_inbound_threshold
        baseline = baseline_controller.step(
            base_gate,
            base_direction,
            track_present=track_present,
            threat_passed=(
                track_present
                and release_threshold is not None
                and self.policy.last_prediction.inbound_probability
                <= release_threshold
            ),
        )
        executed, _ = self.policy.action_from_inference(
            adjusted_gate, adjusted_direction
        )
        executed = np.clip(
            executed, self.env.action_space.low, self.env.action_space.high
        ).astype(np.float32)
        committed = bool(self.policy.last_committed)
        prediction_used = self.policy.last_prediction
        event_activity_used = self.last_event_activity
        self.last_residual = residual.copy()

        observation, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info)
        info["event_pulse_residual"] = {
            "tracker_probability": float(prediction_used.probability),
            "event_activity": float(event_activity_used),
            "inbound_probability": float(prediction_used.inbound_probability),
            "gate": float(enabled),
            "baseline_action": np.asarray(baseline, dtype=np.float32),
            "residual_action": residual.copy(),
            "executed_action": executed.copy(),
            "base_gate_probability": base_gate,
            "adjusted_gate_probability": adjusted_gate,
            "committed": committed,
            "renewed": bool(self.policy.pulse_controller.last_renewed),
        }
        self._infer(observation)
        return (
            self._compact_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )
