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
    offset_cmd: np.ndarray | None = None
    delta_velocity: np.ndarray | None = None
    active_obstacle_count: int = 0
    obstacle_relative_states: list[dict[str, Any]] | None = None
    bounds_margin_m: float = np.inf
    # Policy steps left before truncation. Lets a terminal penalty price
    # the remaining episode a crash forfeits (see
    # VelocityDodgeTask.crash_penalty).
    remaining_steps: float = 0.0


def _yaw_from_quat_xyzw(q: np.ndarray) -> float:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _angle_wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class ReactiveDodgeTask(RLTask):
    """Track a nominal trajectory while learning sparse evasive corrections."""

    # residual_control.mode -> default action width
    RESIDUAL_MODES = {
        "ctbr_delta": 4,
        "gated_ctbr_delta": 5,
        "velocity_delta_integrated": 3,
        # Same integrator and displacement clip as the ungated mode; the
        # only difference is that action[0] scales the correction, so a
        # closed gate drives the delta to zero and the leaky integrator
        # decays back to the nominal path on its own.
        "gated_velocity_delta_integrated": 4,
        "desired_body_offset": 3,
        "cascaded_velocity": 3,
        "gated_cascaded_velocity": 4,
        "velocity_command": 3,
    }

    def __init__(
        self,
        *,
        threat_gated_actions: bool = False,
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
        if mode not in self.RESIDUAL_MODES:
            allowed = ", ".join(sorted(self.RESIDUAL_MODES))
            raise ValueError(
                f"reactive_dodge residual_control.mode must be one of: {allowed}"
            )
        self.residual_control_mode = mode
        self.delta_thrust_fraction = float(rc.pop("delta_thrust_fraction", 0.2))
        self.delta_rate_limits = np.asarray(
            rc.pop("delta_rate_limits", [2.0, 2.0, 1.0]),
            dtype=np.float64,
        ).reshape(3)

        # velocity_delta_integrated: the policy emits a body-frame velocity
        # correction which is integrated (with leak) into an offset applied
        # to the SE3 *reference position*. Adding a correction to the
        # reference velocity alone cannot work: SE3's position-error term
        # pulls back toward the un-shifted path, capping the achievable
        # deviation at (Kd/Kp) * dv -- about 0.6 * dv with stock gains,
        # regardless of how large the correction is made.
        self.delta_velocity_limits_mps = np.asarray(
            rc.pop("delta_velocity_limits_mps", [0.6, 0.6, 0.4]),
            dtype=np.float64,
        ).reshape(3)
        # Leak time constant: a held correction settles at tau * dv, and the
        # offset decays back to the nominal path with this time constant once
        # the policy stops correcting -- so "return to the trajectory" is
        # structural rather than something the reward has to enforce.
        self.offset_tau_s = float(rc.pop("offset_tau_s", 1.5))
        # First-order low-pass on the action before it is integrated. The
        # offset itself averages zero-mean action noise away, but its
        # derivative -- which is what is handed to SE3 as reference velocity
        # -- reproduces that noise directly, and Kd amplifies it into
        # acceleration. Measured: unfiltered random actions put 7/8 episodes
        # out of bounds; filtering at 0.15 s cut that to 3/8 and max tracking
        # error from 3.18 m to 0.99 m.
        self.action_filter_tau_s = float(rc.pop("action_filter_tau_s", 0.15))
        # desired_body_offset directly selects a bounded reference-position
        # offset. Filtering and rate limiting retain a smooth SE3 reference.
        self.offset_command_tau_s = float(rc.pop("offset_command_tau_s", 0.20))
        # Bounds the offset command's acceleration. Zero keeps the original
        # first-order lag, whose velocity steps to ~amp/tau immediately and
        # flips the vehicle past 109 deg on a 0.45 m sidestep (4/4 seeds, no
        # obstacles). See _control_from_desired_body_offset.
        self.offset_accel_limit_mps2 = float(
            rc.pop("offset_accel_limit_mps2", 0.0) or 0.0
        )
        self.offset_rate_limits_mps = np.asarray(
            rc.pop("offset_rate_limits_mps", [0.8, 0.8, 0.5]),
            dtype=np.float64,
        ).reshape(3)
        # velocity_command: v_cmd = v_ref + dv + k_return * (x_ref - x).
        # A soft, bounded pull back toward the path replaces SE3's stiff
        # position term -- enough to stop unbounded drift after a dodge,
        # far too weak to rotate the thrust axis toward horizontal.
        self.return_gain_hz = float(rc.pop("return_gain_hz", 1.0))
        self.max_return_speed_mps = float(rc.pop("max_return_speed_mps", 1.0))
        # cascaded_velocity: an explicit position outer loop produces a
        # restoring velocity, then the policy adds a body-frame avoidance
        # velocity.  The SE3 position reference is set to the measured
        # position so its inner position term is exactly zero.
        self.outer_position_gain_hz = np.asarray(
            rc.pop("outer_position_gain_hz", [1.5, 1.5, 2.0]),
            dtype=np.float64,
        ).reshape(3)
        self.max_outer_return_velocity_mps = np.asarray(
            rc.pop("max_outer_return_velocity_mps", [0.8, 0.8, 0.5]),
            dtype=np.float64,
        ).reshape(3)
        self.offset_max_m = np.asarray(
            rc.pop("offset_max_m", [1.2, 1.2, 0.8]),
            dtype=np.float64,
        ).reshape(3)
        if self.offset_tau_s <= 0.0:
            raise ValueError("residual_control.offset_tau_s must be positive")
        if self.offset_command_tau_s <= 0.0:
            raise ValueError(
                "residual_control.offset_command_tau_s must be positive"
            )
        if np.any(self.offset_rate_limits_mps <= 0.0):
            raise ValueError(
                "residual_control.offset_rate_limits_mps must be positive"
            )
        if np.any(self.outer_position_gain_hz <= 0.0):
            raise ValueError(
                "residual_control.outer_position_gain_hz must be positive"
            )
        if np.any(self.max_outer_return_velocity_mps <= 0.0):
            raise ValueError(
                "residual_control.max_outer_return_velocity_mps must be positive"
            )

        if rc:
            unknown = ", ".join(sorted(rc))
            raise ValueError(f"Unknown residual_control keys: {unknown}")
        self.threat_gated_actions = bool(threat_gated_actions)
        default_dim = self.RESIDUAL_MODES[mode]
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
            offset_cmd=context.get("offset_cmd"),
            delta_velocity=context.get("delta_velocity"),
            active_obstacle_count=int(context.get("active_obstacle_count", 0)),
            obstacle_relative_states=context.get("obstacle_relative_states"),
            bounds_margin_m=float(context.get("bounds_margin_m", np.inf)),
            remaining_steps=float(context.get("remaining_steps", 0.0)),
        )

    def set_previous_action(self, action: np.ndarray | None) -> None:
        self._context.previous_action = action

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
        if self.threat_gated_actions and not self._context.obstacle_threat:
            # Hard gate: with nothing inbound the policy commands nothing, so
            # the vehicle returns to the nominal path mechanically rather than
            # being taxed back onto it by a reward term.
            #
            # Reward shaping could not do this. Measured on v24, no-threat
            # steps ran 1.633 m off the path while the commanded offset was
            # only 0.280 m -- most of the excursion was SE3 chasing a
            # reference that sampled actions were shaking, which no penalty on
            # the *command* can reach. And the drift paid: obstacles are aimed
            # once at spawn with no re-solve, so wandering ~1.6 m over their
            # 1-2 s flight made them miss outright (threat fraction 51.5%
            # deterministic vs 20.8% stochastic). The policy was collecting
            # survival reward for not being where the throw was aimed.
            #
            # This uses the privileged threat flag, so a policy trained this
            # way is not deployable as-is. It buys a clean separation of "when
            # to act" from "which way to go": if dodging still fails with the
            # timing given for free, the failure is direction inference from
            # events, not detection.
            # Zeroing the action and falling through keeps the returned shapes
            # correct for every mode (the delta is 3 or 4 wide depending on
            # mode, and gated modes derive their gate from action[0], which
            # clips to 0 here).
            action = np.zeros_like(action)
        if self.residual_control_mode == "gated_ctbr_delta":
            gate = float(np.clip(action[0], 0.0, 1.0))
            delta = (gate * action[1:5]).astype(np.float32, copy=False)
            return gate, delta
        if self.residual_control_mode == "gated_cascaded_velocity":
            gate = float(np.clip(action[0], 0.0, 1.0))
            delta = (gate * action[1:4]).astype(np.float32, copy=False)
            return gate, delta
        if self.residual_control_mode == "gated_velocity_delta_integrated":
            gate = float(np.clip(action[0], 0.0, 1.0))
            delta = (gate * action[1:4]).astype(np.float32, copy=False)
            return gate, delta
        if self.residual_control_mode == "velocity_delta_integrated":
            delta = action[:3].astype(np.float32, copy=True)
            if getattr(self, "lateral_axis_only", False):
                # Left/right only: body x is forward, body z is up.
                delta[0] = 0.0
                delta[2] = 0.0
            elif getattr(self, "lateral_only", False):
                # Body x is forward; zeroing it removes braking/accelerating
                # and leaves only lateral + vertical evasion.
                delta[0] = 0.0
            return 1.0, delta
        if self.residual_control_mode == "desired_body_offset":
            delta = action[:3].astype(np.float32, copy=True)
            if getattr(self, "lateral_axis_only", False):
                delta[0] = 0.0
                delta[2] = 0.0
            elif getattr(self, "lateral_only", False):
                delta[0] = 0.0
            return 1.0, delta
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
