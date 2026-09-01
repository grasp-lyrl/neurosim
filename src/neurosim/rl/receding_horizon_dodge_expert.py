"""Sampling-MPC expert for privileged dynamic-obstacle avoidance.

The historical trajectory expert searches a finite menu of one-shot quintic
bumps.  This module instead repeatedly optimises the commands that the
``velocity_command`` task actually consumes.  It deliberately has no Habitat
dependency: moving-obstacle scoring is vectorised NumPy, while callers may
provide a (more expensive) callback for checking a short list of trajectories
against static scene geometry and world bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class MovingSpherePrediction:
    """World-frame constant-acceleration prediction for one obstacle."""

    object_id: int
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    combined_radius: float


@dataclass(frozen=True)
class RecedingHorizonConfig:
    horizon_s: float = 1.50
    step_dt_s: float = 0.05
    control_segments: int = 6
    samples_per_mode: int = 40
    iterations: int = 3
    elite_fraction: float = 0.15
    initial_std: float = 0.42
    minimum_std: float = 0.08
    replan_interval_s: float = 0.10
    # Hard bound on the rate of the residual velocity command. The task's
    # ``offset_accel_limit_mps2`` does not apply in velocity-command mode, so
    # without this constraint a replan may step the target by the full action
    # range and ask the inner velocity controller for an implausible impulse.
    max_residual_command_acceleration_mps2: float = 5.0
    # The first-order rollout is slightly optimistic once command acceleration
    # is bounded. Request 0.12 m of headroom above the task's 0.10 m success
    # clearance to reduce marginal full-vehicle misses.
    safety_margin_m: float = 0.22
    velocity_response_tau_s: float = 0.10
    # Optional bound on the predicted vehicle velocity change. Zero preserves
    # the historical first-order response. Low-acceleration pipelines set it
    # to the controller-layer desired-acceleration cap so MPC does not plan
    # trajectories the execution shield will subsequently slow down.
    max_vehicle_acceleration_mps2: float = 0.0
    static_shortlist: int = 24
    # The first two terms make collision slack lexicographically more
    # important than nominal tracking.  Unlike a hard feasibility filter,
    # finite slack leaves a least-dangerous command in genuinely infeasible
    # states.
    max_slack_cost: float = 1.0e6
    integrated_slack_cost: float = 1.0e5
    clearance_risk_cost: float = 2.0
    offset_cost: float = 0.8
    along_track_cost: float = 5.0
    vertical_cost: float = 0.35
    relative_velocity_cost: float = 0.05
    control_cost: float = 0.025
    control_smoothness_cost: float = 0.04
    # Optional temporal-continuity terms.  The historical smoothness term
    # only compares segments *within* a candidate.  It does not price a new
    # plan's first command jumping away from the command currently in flight,
    # or a replan abandoning the previously selected dodge mode.  Both default
    # to zero so historical oracle configs remain reproducible.
    initial_control_smoothness_cost: float = 0.0
    plan_change_cost: float = 0.0
    terminal_offset_cost: float = 3.0
    terminal_velocity_cost: float = 0.5
    seed: int = 20260831


@dataclass(frozen=True)
class RecedingHorizonPlan:
    """The best shooting trajectory found at one MPC update."""

    start_time: float
    step_dt_s: float
    actions: np.ndarray
    offsets: np.ndarray
    relative_velocities: np.ndarray
    predicted_min_clearance: float
    safety_slack: float
    objective: float
    static_clear: bool
    object_ids: tuple[int, ...]

    @property
    def end_time(self) -> float:
        return self.start_time + self.step_dt_s * len(self.actions)

    def action_at(self, time: float) -> np.ndarray:
        if len(self.actions) == 0:
            return np.zeros(3, dtype=np.float64)
        index = int(np.floor((float(time) - self.start_time) / self.step_dt_s))
        return np.asarray(
            self.actions[int(np.clip(index, 0, len(self.actions) - 1))],
            dtype=np.float64,
        ).copy()


class RecedingHorizonDodgeExpert:
    """Cross-entropy shooting MPC in the task's normalised action space."""

    def __init__(self, config: RecedingHorizonConfig | None = None):
        self.config = config or RecedingHorizonConfig()
        self.plan: RecedingHorizonPlan | None = None
        self._rng = np.random.default_rng(self.config.seed)
        self.replans = 0
        self.slack_replans = 0
        self.static_fallbacks = 0

    def reset(self) -> None:
        self.plan = None
        self._rng = np.random.default_rng(self.config.seed)
        self.replans = 0
        self.slack_replans = 0
        self.static_fallbacks = 0

    def should_replan(self, now: float) -> bool:
        return self.plan is None or (
            float(now) - self.plan.start_time >= self.config.replan_interval_s - 1e-9
        )

    @staticmethod
    def _normalised_directions(
        relative_velocity_body: np.ndarray,
        action_mask: np.ndarray,
    ) -> list[np.ndarray]:
        """Supply multimodal seeds so symmetric left/right solutions survive."""
        mask = np.asarray(action_mask, dtype=np.float64)
        directions: list[np.ndarray] = []
        for axis in range(3):
            if mask[axis] > 0.0:
                unit = np.zeros(3)
                unit[axis] = 1.0
                directions.extend((unit, -unit))

        velocity = np.asarray(relative_velocity_body, dtype=np.float64) * mask
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-8:
            incoming = velocity / speed
            basis_a = np.cross(incoming, np.array([0.0, 0.0, 1.0]))
            basis_a *= mask
            if np.linalg.norm(basis_a) < 1e-8:
                basis_a = np.cross(incoming, np.array([0.0, 1.0, 0.0])) * mask
            if np.linalg.norm(basis_a) > 1e-8:
                basis_a /= np.linalg.norm(basis_a)
                basis_b = np.cross(incoming, basis_a) * mask
                if np.linalg.norm(basis_b) > 1e-8:
                    basis_b /= np.linalg.norm(basis_b)
                    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                        direction = np.cos(angle) * basis_a + np.sin(angle) * basis_b
                        direction *= mask
                        norm = float(np.linalg.norm(direction))
                        if norm > 1e-8:
                            directions.append(direction / norm)

        unique: list[np.ndarray] = []
        for direction in directions:
            if not any(np.linalg.norm(direction - old) < 1e-6 for old in unique):
                unique.append(direction)
        return unique

    def _warm_segments(self, now: float) -> np.ndarray | None:
        if self.plan is None:
            return None
        cfg = self.config
        segment_dt = cfg.horizon_s / cfg.control_segments
        return np.asarray(
            [
                self.plan.action_at(now + (index + 0.5) * segment_dt)
                for index in range(cfg.control_segments)
            ],
            dtype=np.float64,
        )

    def _initial_means(
        self,
        *,
        now: float,
        time_to_closest_approach: float | None,
        relative_velocity_body: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        mask = np.asarray(action_mask, dtype=np.float64)
        means = [np.zeros((cfg.control_segments, 3), dtype=np.float64)]
        warm = self._warm_segments(now)
        if warm is not None:
            means.append(np.clip(warm, -1.0, 1.0) * mask)

        segment_dt = cfg.horizon_s / cfg.control_segments
        tca = cfg.horizon_s * 0.5 if time_to_closest_approach is None else max(
            float(time_to_closest_approach), segment_dt
        )
        outward_segments = int(np.clip(np.ceil(tca / segment_dt), 1, cfg.control_segments))
        for direction in self._normalised_directions(relative_velocity_body, mask):
            mean = np.zeros((cfg.control_segments, 3), dtype=np.float64)
            mean[:outward_segments] = 0.90 * direction
            # Begin cancelling velocity after closest approach.  The task's
            # own return controller then finishes convergence to the path.
            if outward_segments < cfg.control_segments:
                return_segments = min(2, cfg.control_segments - outward_segments)
                mean[outward_segments : outward_segments + return_segments] = (
                    -0.55 * direction
                )
            means.append(mean * mask)
        return np.asarray(means, dtype=np.float64)

    def _expand_controls(self, controls: np.ndarray) -> np.ndarray:
        cfg = self.config
        steps = int(round(cfg.horizon_s / cfg.step_dt_s))
        indices = np.minimum(
            np.arange(steps) * cfg.control_segments // steps,
            cfg.control_segments - 1,
        )
        return controls[:, indices]

    def _rollout(
        self,
        controls: np.ndarray,
        *,
        initial_offset: np.ndarray,
        initial_relative_velocity: np.ndarray,
        initial_filtered_action: np.ndarray,
        body_to_world: np.ndarray,
        delta_velocity_limits: np.ndarray,
        action_filter_alpha: float,
        max_target_delta_velocity_mps: float,
        return_gain_hz: float,
        max_return_speed_mps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        desired_actions = self._expand_controls(
            np.asarray(controls, dtype=np.float64)
        )
        count, steps, _ = desired_actions.shape
        offset = np.broadcast_to(np.asarray(initial_offset), (count, 3)).copy()
        velocity = np.broadcast_to(
            np.asarray(initial_relative_velocity), (count, 3)
        ).copy()
        filtered = np.broadcast_to(
            np.asarray(initial_filtered_action), (count, 3)
        ).copy()
        actions = np.empty_like(desired_actions)
        offsets = np.empty((count, steps, 3), dtype=np.float64)
        velocities = np.empty_like(offsets)
        response_alpha = 1.0 - np.exp(
            -cfg.step_dt_s / max(cfg.velocity_response_tau_s, 1e-6)
        )
        limits = np.asarray(delta_velocity_limits, dtype=np.float64)
        rotation = np.asarray(body_to_world, dtype=np.float64)
        max_target_gap = float(max_target_delta_velocity_mps)

        for step in range(steps):
            command = desired_actions[:, step].copy()
            if np.isfinite(max_target_gap):
                physical_gap = (command - filtered) * limits
                gap_norm = np.linalg.norm(physical_gap, axis=1)
                scale = np.minimum(
                    1.0,
                    max_target_gap / np.maximum(gap_norm, 1e-12),
                )
                command = filtered + (command - filtered) * scale[:, None]
            actions[:, step] = command
            filtered += float(action_filter_alpha) * (command - filtered)
            body_delta = filtered * limits
            world_delta = body_delta @ rotation.T
            return_velocity = -float(return_gain_hz) * offset
            return_speed = np.linalg.norm(return_velocity, axis=1)
            scale = np.minimum(
                1.0,
                float(max_return_speed_mps) / np.maximum(return_speed, 1e-12),
            )
            return_velocity *= scale[:, None]
            target_velocity = world_delta + return_velocity
            velocity_change = response_alpha * (target_velocity - velocity)
            max_acceleration = float(cfg.max_vehicle_acceleration_mps2)
            if max_acceleration > 0.0:
                change_norm = np.linalg.norm(velocity_change, axis=1)
                maximum_change = max_acceleration * cfg.step_dt_s
                scale = np.minimum(
                    1.0,
                    maximum_change / np.maximum(change_norm, 1e-12),
                )
                velocity_change *= scale[:, None]
            velocity += velocity_change
            offset += cfg.step_dt_s * velocity
            offsets[:, step] = offset
            velocities[:, step] = velocity
        return actions, offsets, velocities

    def _score(
        self,
        controls: np.ndarray,
        *,
        nominal_positions: np.ndarray,
        nominal_velocity: np.ndarray,
        obstacles: tuple[MovingSpherePrediction, ...],
        rollout_kwargs: dict,
        previous_controls: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[np.ndarray, ...]]:
        cfg = self.config
        actions, offsets, velocities = self._rollout(controls, **rollout_kwargs)
        reference = np.asarray(nominal_positions)[None, :, :] + offsets
        count = len(controls)
        min_clearance = np.full(count, np.inf, dtype=np.float64)
        integrated_violation = np.zeros(count, dtype=np.float64)
        risk = np.zeros(count, dtype=np.float64)
        times = cfg.step_dt_s * (np.arange(reference.shape[1]) + 1.0)
        clearance_series: list[np.ndarray] = []
        for obstacle in obstacles:
            swept = (
                np.asarray(obstacle.position, dtype=np.float64)[None, :]
                + times[:, None] * np.asarray(obstacle.velocity, dtype=np.float64)[None, :]
                + 0.5
                * times[:, None] ** 2
                * np.asarray(obstacle.acceleration, dtype=np.float64)[None, :]
            )
            clearance = (
                np.linalg.norm(reference - swept[None, :, :], axis=2)
                - float(obstacle.combined_radius)
            )
            clearance_series.append(clearance)
            min_clearance = np.minimum(min_clearance, np.min(clearance, axis=1))
            violation = np.maximum(cfg.safety_margin_m - clearance, 0.0)
            integrated_violation += np.mean(violation**2, axis=1)
            risk += np.mean(
                np.exp(
                    np.clip(
                        -(clearance - cfg.safety_margin_m) / 0.15,
                        -12.0,
                        12.0,
                    )
                ),
                axis=1,
            )

        slack = np.maximum(cfg.safety_margin_m - min_clearance, 0.0)
        if not obstacles:
            slack.fill(0.0)
            min_clearance.fill(np.inf)

        direction = np.asarray(nominal_velocity, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-9:
            direction /= direction_norm
            along = offsets @ direction
        else:
            along = np.zeros(offsets.shape[:2], dtype=np.float64)
        tracking_cost = cfg.step_dt_s * (
            cfg.offset_cost * np.sum(offsets**2, axis=(1, 2))
            + cfg.along_track_cost * np.sum(along**2, axis=1)
            + cfg.vertical_cost * np.sum(offsets[:, :, 2] ** 2, axis=1)
            + cfg.relative_velocity_cost * np.sum(velocities**2, axis=(1, 2))
        )
        effort = cfg.step_dt_s * cfg.control_cost * np.sum(
            actions**2, axis=(1, 2)
        )
        smooth = cfg.control_smoothness_cost * np.sum(
            np.diff(controls, axis=1) ** 2, axis=(1, 2)
        )
        initial_control = np.asarray(
            rollout_kwargs["initial_filtered_action"], dtype=np.float64
        )
        initial_smooth = cfg.initial_control_smoothness_cost * np.sum(
            (controls[:, 0] - initial_control[None, :]) ** 2,
            axis=1,
        )
        if previous_controls is None:
            plan_change = np.zeros(count, dtype=np.float64)
        else:
            previous = np.asarray(previous_controls, dtype=np.float64)
            plan_change = cfg.plan_change_cost * np.sum(
                (controls - previous[None, :, :]) ** 2,
                axis=(1, 2),
            )
        terminal = (
            cfg.terminal_offset_cost * np.sum(offsets[:, -1] ** 2, axis=1)
            + cfg.terminal_velocity_cost * np.sum(velocities[:, -1] ** 2, axis=1)
        )
        objective = (
            cfg.max_slack_cost * slack**2
            + cfg.integrated_slack_cost * integrated_violation
            + cfg.clearance_risk_cost * risk
            + tracking_cost
            + effort
            + smooth
            + initial_smooth
            + plan_change
            + terminal
        )
        details = {
            "actions": actions,
            "offsets": offsets,
            "velocities": velocities,
            "min_clearance": min_clearance,
            "slack": slack,
        }
        return objective, details, tuple(clearance_series)

    def make_plan(
        self,
        *,
        now: float,
        time_to_closest_approach: float | None,
        initial_offset: np.ndarray,
        initial_relative_velocity: np.ndarray,
        initial_filtered_action: np.ndarray,
        body_to_world: np.ndarray,
        delta_velocity_limits: np.ndarray,
        action_filter_alpha: float,
        max_target_delta_velocity_mps: float = np.inf,
        return_gain_hz: float,
        max_return_speed_mps: float,
        nominal_positions: np.ndarray,
        nominal_velocity: np.ndarray,
        obstacles: Iterable[MovingSpherePrediction] = (),
        action_mask: np.ndarray | None = None,
        static_path_is_clear: Callable[[np.ndarray], bool] | None = None,
    ) -> RecedingHorizonPlan:
        """Optimise and return a plan; finite safety slack prevents failure."""
        cfg = self.config
        obstacles = tuple(obstacles)
        mask = (
            np.ones(3, dtype=np.float64)
            if action_mask is None
            else np.asarray(action_mask, dtype=np.float64)
        )
        primary_velocity_body = np.zeros(3, dtype=np.float64)
        if obstacles:
            primary_velocity_body = (
                np.asarray(body_to_world, dtype=np.float64).T
                @ (np.asarray(obstacles[0].velocity) - np.asarray(nominal_velocity))
            )
        means = self._initial_means(
            now=now,
            time_to_closest_approach=time_to_closest_approach,
            relative_velocity_body=primary_velocity_body,
            action_mask=mask,
        )
        stds = np.full_like(means, cfg.initial_std)
        stds *= mask[None, None, :]
        rollout_kwargs = {
            "initial_offset": np.asarray(initial_offset, dtype=np.float64),
            "initial_relative_velocity": np.asarray(
                initial_relative_velocity, dtype=np.float64
            ),
            "initial_filtered_action": np.asarray(
                initial_filtered_action, dtype=np.float64
            ),
            "body_to_world": np.asarray(body_to_world, dtype=np.float64),
            "delta_velocity_limits": np.asarray(
                delta_velocity_limits, dtype=np.float64
            ),
            "action_filter_alpha": float(action_filter_alpha),
            "max_target_delta_velocity_mps": float(
                max_target_delta_velocity_mps
            ),
            "return_gain_hz": float(return_gain_hz),
            "max_return_speed_mps": float(max_return_speed_mps),
        }
        previous_controls = self._warm_segments(now)

        for _ in range(cfg.iterations):
            noise = self._rng.normal(
                size=(len(means), cfg.samples_per_mode, cfg.control_segments, 3)
            )
            population = np.clip(
                means[:, None] + stds[:, None] * noise, -1.0, 1.0
            )
            population *= mask[None, None, None, :]
            flat_population = population.reshape(-1, cfg.control_segments, 3)
            objective, _, _ = self._score(
                flat_population,
                nominal_positions=nominal_positions,
                nominal_velocity=nominal_velocity,
                obstacles=obstacles,
                rollout_kwargs=rollout_kwargs,
                previous_controls=previous_controls,
            )
            objective = objective.reshape(len(means), cfg.samples_per_mode)
            elite_count = max(2, int(round(cfg.samples_per_mode * cfg.elite_fraction)))
            elite_indices = np.argpartition(
                objective, elite_count - 1, axis=1
            )[:, :elite_count]
            elites = np.take_along_axis(
                population, elite_indices[:, :, None, None], axis=1
            )
            means = 0.25 * means + 0.75 * np.mean(elites, axis=1)
            elite_std = np.std(elites, axis=1)
            stds = np.maximum(
                cfg.minimum_std * mask[None, None, :],
                0.25 * stds + 0.75 * elite_std,
            )

        # Compare the refined modal centres as well as a final stochastic
        # population.  Keeping distinct modes prevents a symmetric left/right
        # solution from averaging back into a collision course.
        noise = self._rng.normal(
            size=(len(means), cfg.samples_per_mode, cfg.control_segments, 3)
        )
        final_population = np.clip(
            means[:, None] + stds[:, None] * noise, -1.0, 1.0
        )
        final_population *= mask[None, None, None, :]
        candidates = np.concatenate(
            [
                final_population.reshape(-1, cfg.control_segments, 3),
                means,
                np.zeros((1, cfg.control_segments, 3), dtype=np.float64),
            ],
            axis=0,
        )
        objective, details, _ = self._score(
            candidates,
            nominal_positions=nominal_positions,
            nominal_velocity=nominal_velocity,
            obstacles=obstacles,
            rollout_kwargs=rollout_kwargs,
            previous_controls=previous_controls,
        )
        order = np.argsort(objective)
        chosen = int(order[0])
        static_clear = static_path_is_clear is None
        if static_path_is_clear is not None:
            static_clear = False
            for index in order[: cfg.static_shortlist]:
                reference = np.asarray(nominal_positions) + details["offsets"][index]
                if static_path_is_clear(reference):
                    chosen = int(index)
                    static_clear = True
                    break
            if not static_clear:
                # The zero-control rollout follows the task's own return law
                # and is the most conservative static fallback.  It is always
                # present even when every evasive sample is rejected.
                zero_index = len(candidates) - 1
                reference = (
                    np.asarray(nominal_positions) + details["offsets"][zero_index]
                )
                if static_path_is_clear(reference):
                    chosen = zero_index
                    static_clear = True
                self.static_fallbacks += 1

        plan = RecedingHorizonPlan(
            start_time=float(now),
            step_dt_s=cfg.step_dt_s,
            actions=np.asarray(details["actions"][chosen], dtype=np.float64),
            offsets=np.asarray(details["offsets"][chosen], dtype=np.float64),
            relative_velocities=np.asarray(
                details["velocities"][chosen], dtype=np.float64
            ),
            predicted_min_clearance=float(details["min_clearance"][chosen]),
            safety_slack=float(details["slack"][chosen]),
            objective=float(objective[chosen]),
            static_clear=bool(static_clear),
            object_ids=tuple(int(obstacle.object_id) for obstacle in obstacles),
        )
        self.plan = plan
        self.replans += 1
        if plan.safety_slack > 1e-6:
            self.slack_replans += 1
        return plan
