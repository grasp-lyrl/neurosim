"""Guarded residual-control wrapper for the frozen event geometry policy.

The event networks remain supervised perception modules.  PPO receives their
compact causal outputs and learns only a bounded correction around the
latched geometric controller.  Exact obstacle geometry, when enabled by the
task, remains under the ``privileged`` key for the asymmetric critic.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EventGeometryResidualWrapper(gym.Wrapper):
    """Expose object-centric observations and execute guarded residual actions.

    A residual is permitted only while the frozen tracker sees an object or
    while the baseline controller has an active/pending manoeuvre.  Therefore
    blank events cannot produce the standing-offset/random-motion shortcut
    that invalidated earlier PPO experiments.
    """

    FEATURE_NAMES = (
        "tracker_probability",
        "event_activity",
        "centre_x",
        "centre_y",
        "radius_norm",
        "depth_norm",
        "inbound_probability",
        "centre_motion_x",
        "centre_motion_y",
        "track_age",
        "hold_remaining",
        "pending",
        "refractory",
        "direction_x",
        "direction_y",
        "direction_z",
        "baseline_x",
        "baseline_y",
        "baseline_z",
        "residual_gate",
        "previous_residual_x",
        "previous_residual_y",
        "previous_residual_z",
    )

    def __init__(
        self,
        env: gym.Env,
        *,
        spatial_checkpoint: str,
        temporal_checkpoint: str,
        device: str = "cuda:0",
        residual_scale: float = 0.35,
        gate_threshold: float = 0.7,
        min_event_activity: float = 1e-8,
        controller: Any | None = None,
        tracker: Any | None = None,
        controller_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError("event residual PPO requires a Dict observation")
        base_spaces = env.observation_space.spaces
        if "events" not in base_spaces or "state" not in base_spaces:
            raise ValueError("event residual PPO requires events and state")
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("event residual PPO requires a Box action space")

        if tracker is None or controller is None:
            # This application module is intentionally imported lazily by the
            # generic trainer so ordinary PPO runs do not load perception code.
            from evaluate_event_geometry_controller import (
                EventObstacleTracker,
                LatchedGeometryController,
            )

            if tracker is None:
                tracker = EventObstacleTracker(
                    spatial_checkpoint, temporal_checkpoint, device=device
                )
            if controller is None:
                controller = LatchedGeometryController(**(controller_kwargs or {}))

        self.tracker = tracker
        self.controller = controller
        self.residual_scale = float(residual_scale)
        self.gate_threshold = float(gate_threshold)
        self.min_event_activity = float(min_event_activity)
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")

        state_space = base_spaces["state"]
        compact_dim = int(np.prod(state_space.shape)) + len(self.FEATURE_NAMES)
        output_spaces: dict[str, spaces.Space] = {
            "state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(compact_dim,), dtype=np.float32
            )
        }
        if "privileged" in base_spaces:
            output_spaces["privileged"] = base_spaces["privileged"]
        self.observation_space = spaces.Dict(output_spaces)
        # PPO's action denotes a normalized residual; the wrapped environment
        # retains the same normalized [-1, 1] command dimensionality.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

        self.last_prediction = None
        self.last_event_activity = 0.0
        self.last_baseline = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_residual = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_gate = 0.0
        self.last_committed = False

    def _residual_is_enabled(self, baseline: np.ndarray) -> bool:
        controller_active = bool(
            np.linalg.norm(baseline) > 1e-6
            or self.controller.pending_direction is not None
            or self.controller.hold_remaining > 0
        )
        return bool(
            self.last_event_activity > self.min_event_activity
            and (
                self.last_prediction.probability >= self.gate_threshold
                or controller_active
            )
        )

    def _advance_controller(self, observation: dict[str, np.ndarray]) -> None:
        events = np.asarray(observation["events"])
        self.last_event_activity = float(np.mean(np.abs(events) > 1e-6))
        self.last_prediction = self.tracker.step(events)
        baseline, self.last_committed = self.controller.step(self.last_prediction)
        self.last_baseline = np.asarray(baseline, dtype=np.float32)
        self.last_gate = float(self._residual_is_enabled(self.last_baseline))

    def _compact_observation(
        self, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        prediction = self.last_prediction
        controller = self.controller
        geometry = np.asarray(prediction.geometry, dtype=np.float32).reshape(2)
        centre = 2.0 * (np.asarray(prediction.centre, dtype=np.float32) - 0.5)
        # Centre motion is in normalized image coordinates per 50 ms policy
        # step.  Scale it into an order-one range and clip rare tracker jumps.
        motion = np.clip(
            20.0 * np.asarray(controller.motion, dtype=np.float32), -2.0, 2.0
        )
        tracker_features = np.asarray(
            [
                prediction.probability,
                min(100.0 * self.last_event_activity, 2.0),
                centre[0],
                centre[1],
                2.0 * geometry[0] - 1.0,
                2.0 * geometry[1] - 1.0,
                prediction.inbound_probability,
                motion[0],
                motion[1],
                min(controller.track_age / 20.0, 2.0),
                controller.hold_remaining / max(controller.hold_steps, 1),
                float(controller.pending_direction is not None),
                float(controller.refractory),
                *np.asarray(controller.direction, dtype=np.float32),
                *self.last_baseline,
                self.last_gate,
                *self.last_residual,
            ],
            dtype=np.float32,
        )
        compact = {
            "state": np.concatenate(
                [np.asarray(observation["state"], dtype=np.float32), tracker_features]
            ).astype(np.float32, copy=False)
        }
        if "privileged" in observation:
            compact["privileged"] = np.asarray(
                observation["privileged"], dtype=np.float32
            )
        return compact

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.tracker.reset()
        self.controller.reset()
        self.last_residual.fill(0.0)
        self._advance_controller(observation)
        return self._compact_observation(observation), info

    def step(self, action):
        residual = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        executed = np.clip(
            self.last_baseline + self.last_gate * self.residual_scale * residual,
            self.env.action_space.low,
            self.env.action_space.high,
        ).astype(np.float32)

        baseline_used = self.last_baseline.copy()
        gate_used = float(self.last_gate)
        prediction_used = self.last_prediction
        self.last_residual = residual.copy()
        observation, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info)
        info["event_geometry_residual"] = {
            "tracker_probability": float(prediction_used.probability),
            "event_activity": float(self.last_event_activity),
            "inbound_probability": float(prediction_used.inbound_probability),
            "gate": gate_used,
            "baseline_action": baseline_used,
            "residual_action": residual.copy(),
            "executed_action": executed.copy(),
            "committed": bool(self.last_committed),
        }
        self._advance_controller(observation)
        return (
            self._compact_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )
