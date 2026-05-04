"""Reactive-dodge task for residual control over nominal trajectory tracking."""

from typing import Any
from dataclasses import dataclass

import numpy as np

from .base import RewardOutcome, RLTask, TaskStep


@dataclass(slots=True)
class ReactiveDodgeContext:
    """Environment-side quantities used by the task but not all policy-visible."""

    flat: dict[str, Any] | None = None
    sim_time: float = 0.0
    lookahead_errors: list[np.ndarray] | None = None
    nominal_control_normalized: np.ndarray | None = None
    previous_action: np.ndarray | None = None
    min_obstacle_distance: float = np.inf
    predicted_closest_distance: float = np.inf
    time_to_closest_approach: float = np.inf
    obstacle_threat: bool = False
    threat_ids: tuple[int, ...] = ()
    near_miss_ids: tuple[int, ...] = ()
    nominal_counterfactual_collision: bool = False
    correction_energy: float = 0.0
    gate_value: float = 1.0


def _yaw_from_quat_xyzw(q: np.ndarray) -> float:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _angle_wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class ReactiveDodgeTask(RLTask):
    """Track a nominal trajectory while learning sparse evasive corrections."""

    def __init__(
        self,
        *,
        w_survival: float = 0.2,
        w_pos: float = 1.5,
        w_vel: float = 0.5,
        w_yaw: float = 0.1,
        w_correction: float = 0.02,
        w_no_threat_correction: float = 0.2,
        w_smooth: float = 0.02,
        w_angular: float = 0.02,
        w_tilt: float = 0.05,
        w_gate: float = 0.05,
        w_dodge_success: float = 1.0,
        crash_penalty_scene: float = 100.0,
        crash_penalty_obstacle: float = 120.0,
        tracking_failure_penalty: float = 25.0,
        success_pos_rmse_threshold: float = 0.5,
        success_vel_rmse_threshold: float = 0.75,
        success_max_correction_energy: float = 0.5,
        near_miss_radius_m: float = 1.0,
        threat_distance_m: float = 1.5,
        threat_time_horizon_s: float = 1.0,
        recovery_pos_error_m: float = 0.75,
        recovery_window_s: float = 1.0,
        tracking_failure_pos_error_m: float = 3.0,
        tracking_failure_steps: int = 20,
        require_obstacle_encounter: bool = True,
        action_dim: int | None = None,
        lookahead_seconds: list[float] | tuple[float, ...] = (0.25, 0.5),
        controller: dict[str, Any] | None = None,
        trajectory: dict[str, Any] | None = None,
        residual_control: dict[str, Any] | None = None,
    ):
        self.w_survival = float(w_survival)
        self.w_pos = float(w_pos)
        self.w_vel = float(w_vel)
        self.w_yaw = float(w_yaw)
        self.w_correction = float(w_correction)
        self.w_no_threat_correction = float(w_no_threat_correction)
        self.w_smooth = float(w_smooth)
        self.w_angular = float(w_angular)
        self.w_tilt = float(w_tilt)
        self.w_gate = float(w_gate)
        self.w_dodge_success = float(w_dodge_success)

        self.crash_penalty_scene = float(crash_penalty_scene)
        self.crash_penalty_obstacle = float(crash_penalty_obstacle)
        self.tracking_failure_penalty = float(tracking_failure_penalty)

        self.success_pos_rmse_threshold = float(success_pos_rmse_threshold)
        self.success_vel_rmse_threshold = float(success_vel_rmse_threshold)
        self.success_max_correction_energy = float(success_max_correction_energy)
        self.near_miss_radius_m = float(near_miss_radius_m)
        self.threat_distance_m = float(threat_distance_m)
        self.threat_time_horizon_s = float(threat_time_horizon_s)
        self.recovery_pos_error_m = float(recovery_pos_error_m)
        self.recovery_window_s = float(recovery_window_s)
        self.tracking_failure_pos_error_m = float(tracking_failure_pos_error_m)
        self.tracking_failure_steps = int(tracking_failure_steps)
        self.require_obstacle_encounter = bool(require_obstacle_encounter)

        self.lookahead_seconds = tuple(float(x) for x in lookahead_seconds)
        self.controller_config = dict(controller or {})
        self.trajectory_config = dict(trajectory or {})
        self.residual_control_config = dict(residual_control or {})
        rc = dict(self.residual_control_config)
        mode = str(rc.pop("mode", "ctbr_delta"))
        if mode not in {"ctbr_delta", "gated_ctbr_delta"}:
            raise ValueError(
                "reactive_dodge residual_control.mode must be "
                "'ctbr_delta' or 'gated_ctbr_delta'"
            )
        self.residual_control_mode = mode
        self.delta_thrust_fraction = float(rc.pop("delta_thrust_fraction", 0.2))
        self.delta_rate_limits = np.asarray(
            rc.pop("delta_rate_limits", [2.0, 2.0, 1.0]),
            dtype=np.float64,
        ).reshape(3)
        if rc:
            unknown = ", ".join(sorted(rc))
            raise ValueError(f"Unknown residual_control keys: {unknown}")
        default_dim = 5 if mode == "gated_ctbr_delta" else 4
        self._action_dim = default_dim if action_dim is None else int(action_dim)
        self._context = ReactiveDodgeContext()
        self._last_termination_reason = ""
        self.on_reset()

    @property
    def crash_penalty(self) -> float:
        if self._last_termination_reason == "obstacle_collision":
            return self.crash_penalty_obstacle
        if self._last_termination_reason == "tracking_failure":
            return self.tracking_failure_penalty
        return self.crash_penalty_scene

    @property
    def uses_nominal_controller(self) -> bool:
        return True

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_observation_dim(self) -> int:
        # base state + position error + velocity error + yaw error +
        # lookahead position errors + normalized nominal CTBR + previous action
        return 13 + 3 + 3 + 1 + 3 * len(self.lookahead_seconds) + 4 + self._action_dim

    def on_reset(self) -> None:
        self._context = ReactiveDodgeContext()
        self._last_termination_reason = ""
        self._pos_sq_sum = 0.0
        self._vel_sq_sum = 0.0
        self._sample_count = 0
        self._correction_energy_sum = 0.0
        self._near_miss_ids: set[int] = set()
        self._meaningful_encounter_ids: set[int] = set()
        self._nominal_counterfactual_collision_count = 0
        self._dodge_success_count = 0
        self._last_threat_time: float | None = None
        self._awaiting_recovery = False
        self._last_recovery_time = np.inf
        self._tracking_failure_count = 0

    def set_context(self, context: dict[str, Any]) -> None:
        self._context = ReactiveDodgeContext(
            flat=context.get("flat"),
            sim_time=float(context.get("sim_time", 0.0)),
            lookahead_errors=context.get("lookahead_errors"),
            nominal_control_normalized=context.get("nominal_control_normalized"),
            previous_action=context.get("previous_action"),
            min_obstacle_distance=float(context.get("min_obstacle_distance", np.inf)),
            predicted_closest_distance=float(
                context.get("predicted_closest_distance", np.inf)
            ),
            time_to_closest_approach=float(
                context.get("time_to_closest_approach", np.inf)
            ),
            obstacle_threat=bool(context.get("obstacle_threat", False)),
            threat_ids=tuple(int(x) for x in context.get("threat_ids", ())),
            near_miss_ids=tuple(int(x) for x in context.get("near_miss_ids", ())),
            nominal_counterfactual_collision=bool(
                context.get("nominal_counterfactual_collision", False)
            ),
            correction_energy=float(context.get("correction_energy", 0.0)),
            gate_value=float(context.get("gate_value", 1.0)),
        )

    def set_termination_reason(self, reason: str) -> None:
        self._last_termination_reason = str(reason)

    def _tracking_errors(
        self, state: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        flat = self._context.flat or {}
        x_ref = np.asarray(flat.get("x", state["x"]), dtype=np.float32)
        v_ref = np.asarray(flat.get("x_dot", state["v"]), dtype=np.float32)
        yaw_ref = float(flat.get("yaw", _yaw_from_quat_xyzw(state["q"])))

        pos_err = np.asarray(state["x"], dtype=np.float32) - x_ref
        vel_err = np.asarray(state["v"], dtype=np.float32) - v_ref
        yaw_err = _angle_wrap(_yaw_from_quat_xyzw(state["q"]) - yaw_ref)
        return pos_err, vel_err, yaw_err

    def make_state_observation(
        self,
        *,
        state: dict[str, np.ndarray],
        base_state: np.ndarray,
    ) -> np.ndarray:
        pos_err, vel_err, yaw_err = self._tracking_errors(state)
        lookahead = self._context.lookahead_errors or []
        lookahead_vecs = [
            np.asarray(err, dtype=np.float32).reshape(3) for err in lookahead
        ]

        while len(lookahead_vecs) < len(self.lookahead_seconds):
            lookahead_vecs.append(np.zeros(3, dtype=np.float32))

        nominal = self._context.nominal_control_normalized
        if nominal is None:
            nominal = np.zeros(4, dtype=np.float32)
        prev_action = self._context.previous_action
        if prev_action is None:
            prev_action = np.zeros(self._action_dim, dtype=np.float32)

        obs_parts = [
            np.asarray(base_state, dtype=np.float32),
            pos_err.astype(np.float32, copy=False),
            vel_err.astype(np.float32, copy=False),
            np.asarray([yaw_err], dtype=np.float32),
            *lookahead_vecs[: len(self.lookahead_seconds)],
            np.asarray(nominal, dtype=np.float32).reshape(4),
            np.asarray(prev_action, dtype=np.float32).reshape(self._action_dim),
        ]
        return np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)

    def split_action(self, action: np.ndarray) -> tuple[float, np.ndarray]:
        """Split a raw policy action into ``(gate, residual_delta)``.

        For ``ctbr_delta`` mode: gate is fixed to 1.0 and the delta is the
        first 4 components of ``action``. For ``gated_ctbr_delta`` mode:
        ``action[0]`` is clipped to ``[0, 1]`` as the gate and scales
        ``action[1:5]`` to produce the delta.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.residual_control_mode == "gated_ctbr_delta":
            gate = float(np.clip(action[0], 0.0, 1.0))
            delta = (gate * action[1:5]).astype(np.float32, copy=False)
            return gate, delta
        return 1.0, action[:4].astype(np.float32, copy=False)

    def _update_encounter_metrics(self, pos_norm: float) -> tuple[bool, float]:
        for oid in self._context.threat_ids:
            self._meaningful_encounter_ids.add(oid)
        for oid in self._context.near_miss_ids:
            self._near_miss_ids.add(oid)

        if self._context.nominal_counterfactual_collision:
            self._nominal_counterfactual_collision_count += 1

        if self._context.obstacle_threat:
            self._last_threat_time = self._context.sim_time
            self._awaiting_recovery = True
            return False, np.inf

        if self._awaiting_recovery and self._last_threat_time is not None:
            elapsed = self._context.sim_time - self._last_threat_time
            if (
                0.0 <= elapsed <= self.recovery_window_s
                and pos_norm <= self.recovery_pos_error_m
            ):
                self._awaiting_recovery = False
                self._dodge_success_count += 1
                self._last_recovery_time = elapsed
                return True, elapsed
            if elapsed > self.recovery_window_s:
                self._awaiting_recovery = False

        return False, self._last_recovery_time

    def compute_reward(self, step: TaskStep) -> RewardOutcome:
        state = step.state
        pos_err, vel_err, yaw_err = self._tracking_errors(state)
        pos_norm = float(np.linalg.norm(pos_err))
        vel_norm = float(np.linalg.norm(vel_err))
        yaw_abs = abs(float(yaw_err))
        ang_rate_norm = float(np.linalg.norm(np.asarray(state["w"], dtype=np.float32)))

        q = np.asarray(state["q"], dtype=np.float32)
        tilt_penalty = float(np.linalg.norm(q[:2]))

        action = np.asarray(step.action, dtype=np.float32)
        gate_value, applied_delta = self.split_action(action)
        correction_energy = float(np.mean(np.square(applied_delta)))
        if step.prev_action is None:
            action_smoothness = 0.0
        else:
            action_smoothness = float(np.linalg.norm(action - step.prev_action))

        no_threat_weight = (
            0.0 if self._context.obstacle_threat else self.w_no_threat_correction
        )
        no_threat_gate_weight = (
            0.0
            if self._context.obstacle_threat
            or self.residual_control_mode != "gated_ctbr_delta"
            else self.w_gate
        )
        recovered, recovery_time = self._update_encounter_metrics(pos_norm)

        r_survival = self.w_survival
        r_track = -self.w_pos * pos_norm - self.w_vel * vel_norm - self.w_yaw * yaw_abs
        r_correction = -(self.w_correction + no_threat_weight) * correction_energy
        r_gate = -no_threat_gate_weight * gate_value
        r_smooth = -self.w_smooth * action_smoothness
        r_attitude = -self.w_angular * ang_rate_norm - self.w_tilt * tilt_penalty
        r_dodge_success = self.w_dodge_success if recovered else 0.0
        reward = (
            r_survival
            + r_track
            + r_correction
            + r_gate
            + r_smooth
            + r_attitude
            + r_dodge_success
        )

        self._sample_count += 1
        self._pos_sq_sum += pos_norm * pos_norm
        self._vel_sq_sum += vel_norm * vel_norm
        self._correction_energy_sum += correction_energy

        if pos_norm > self.tracking_failure_pos_error_m:
            self._tracking_failure_count += 1
        else:
            self._tracking_failure_count = 0

        terms = {
            "pos_error": pos_norm,
            "vel_error": vel_norm,
            "yaw_error": yaw_abs,
            "ang_rate_norm": ang_rate_norm,
            "tilt_penalty": tilt_penalty,
            "action_smoothness": action_smoothness,
            "correction_energy": correction_energy,
            "gate_value": gate_value,
            "min_obstacle_distance": self._context.min_obstacle_distance,
            "predicted_closest_distance": self._context.predicted_closest_distance,
            "time_to_closest_approach": self._context.time_to_closest_approach,
            "obstacle_threat": float(self._context.obstacle_threat),
            "meaningful_encounter_count": float(len(self._meaningful_encounter_ids)),
            "near_miss_count": float(len(self._near_miss_ids)),
            "nominal_counterfactual_collision_count": float(
                self._nominal_counterfactual_collision_count
            ),
            "dodge_success_count": float(self._dodge_success_count),
            "post_dodge_recovery_time": float(recovery_time),
            "r_survival": r_survival,
            "r_track": r_track,
            "r_correction": r_correction,
            "r_gate": r_gate,
            "r_smooth": r_smooth,
            "r_attitude": r_attitude,
            "r_dodge_success": r_dodge_success,
        }
        return RewardOutcome(reward=reward, terms=terms)

    def check_success(self, *, state: dict[str, np.ndarray]) -> bool:
        if self._sample_count == 0:
            return False
        pos_rmse = float(np.sqrt(self._pos_sq_sum / self._sample_count))
        vel_rmse = float(np.sqrt(self._vel_sq_sum / self._sample_count))
        mean_correction_energy = self._correction_energy_sum / self._sample_count
        has_encounter = (
            len(self._meaningful_encounter_ids) > 0
            or len(self._near_miss_ids) > 0
            or not self.require_obstacle_encounter
        )
        recovered = self._dodge_success_count > 0 or not self.require_obstacle_encounter
        return (
            has_encounter
            and recovered
            and pos_rmse < self.success_pos_rmse_threshold
            and vel_rmse < self.success_vel_rmse_threshold
            and mean_correction_energy < self.success_max_correction_energy
        )

    def check_terminated(self, *, state: dict[str, np.ndarray]) -> tuple[bool, str]:
        if (
            self.tracking_failure_steps > 0
            and self._tracking_failure_count >= self.tracking_failure_steps
        ):
            self._last_termination_reason = "tracking_failure"
            return True, "tracking_failure"
        return False, ""
