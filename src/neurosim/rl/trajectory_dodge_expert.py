"""Smooth local trajectory expert for dynamic-obstacle avoidance.

The policy action in ``gated_cascaded_velocity`` is a body-frame velocity
residual.  This module first plans a *position trajectory* around a predicted
constant-velocity obstacle and only then converts that trajectory into the
controller's action coordinates.  It is deliberately independent of Habitat
so the optimizer and its collision checks can be tested without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np


def _smoothstep5(u: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quintic smoothstep and its first two derivatives with respect to ``u``."""
    u = np.asarray(u, dtype=np.float64)
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    dds = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
    return s, ds, dds


def _quintic_boundary(
    u: float,
    duration: float,
    start_position: np.ndarray,
    start_velocity: np.ndarray,
    start_acceleration: np.ndarray,
    end_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quintic segment ending at zero velocity/acceleration."""
    p0 = np.asarray(start_position, dtype=np.float64)
    a0 = p0
    a1 = np.asarray(start_velocity, dtype=np.float64) * duration
    a2 = 0.5 * np.asarray(start_acceleration, dtype=np.float64) * duration**2
    c = np.asarray(end_position, dtype=np.float64) - (a0 + a1 + a2)
    d = -a1 - 2.0 * a2
    e = -2.0 * a2
    a3 = 10.0 * c - 4.0 * d + 0.5 * e
    a4 = -15.0 * c + 7.0 * d - e
    a5 = 6.0 * c - 3.0 * d + 0.5 * e
    u = float(np.clip(u, 0.0, 1.0))
    position = a0 + a1 * u + a2 * u**2 + a3 * u**3 + a4 * u**4 + a5 * u**5
    du = a1 + 2 * a2 * u + 3 * a3 * u**2 + 4 * a4 * u**3 + 5 * a5 * u**4
    ddu = 2 * a2 + 6 * a3 * u + 12 * a4 * u**2 + 20 * a5 * u**3
    return position, du / duration, ddu / duration**2


@dataclass(frozen=True)
class QuinticAvoidancePlan:
    """Zero-velocity/acceleration departure, apex, and return bump."""

    object_id: int
    start_time: float
    peak_time: float
    end_time: float
    peak_offset: np.ndarray
    predicted_min_clearance: float
    start_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    start_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    start_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    validated_object_ids: tuple[int, ...] = ()

    def evaluate(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return world-frame displacement, velocity and acceleration."""
        t = float(time)
        offset = np.asarray(self.peak_offset, dtype=np.float64)
        if t <= self.start_time:
            return (
                np.asarray(self.start_offset, dtype=np.float64).copy(),
                np.asarray(self.start_velocity, dtype=np.float64).copy(),
                np.asarray(self.start_acceleration, dtype=np.float64).copy(),
            )
        if t >= self.end_time:
            z = np.zeros(3, dtype=np.float64)
            return z, z.copy(), z.copy()

        if t <= self.peak_time:
            duration = self.peak_time - self.start_time
            u = np.clip((t - self.start_time) / duration, 0.0, 1.0)
            return _quintic_boundary(
                u,
                duration,
                self.start_offset,
                self.start_velocity,
                self.start_acceleration,
                offset,
            )

        duration = self.end_time - self.peak_time
        u = np.clip((t - self.peak_time) / duration, 0.0, 1.0)
        s, ds, dds = _smoothstep5(u)
        return (
            offset * (1.0 - s),
            -offset * ds / duration,
            -offset * dds / duration**2,
        )


@dataclass(frozen=True)
class TrajectoryExpertConfig:
    # Narrow-corridor validation found 0.20--0.25 m buffers can leave no path
    # that is both static-clear and reachable within the 2 s visual horizon.
    # 0.15 m is the largest tested buffer that retained those solutions.
    safety_margin_m: float = 0.15
    # MEASURED 2026-08-29 (oracle_magnitude_probe.py, v20, 20 seeds, 16
    # threatening encounters, 64 directions x magnitudes to 2.00 m):
    #
    #   smallest displacement that clears by >= 0.10 m
    #     median 0.28 m   p75 0.30 m   p90 0.30 m   MAX 0.35 m
    #   encounters needing more than 1.00 m:  0 (0%)
    #   encounters unclearable within 2.00 m: 0 (0%)
    #
    # So the whole menu below is oversized: the largest displacement any
    # encounter has ever required is 0.35 m, which is the SMALLEST entry.
    # Do not widen this menu to "give the expert more authority" -- that was
    # proposed on the strength of `peak_cross_track` 0.295 m sitting at 23%
    # of the 1.275 m reachable, but 0.28 m is all the geometry asks for, so
    # 0.295 m is the expert correctly taking a minimum-sufficient dodge, not
    # under-committing. The expert's collisions are not magnitude-limited.
    #
    # Note escape_set_probe.py reports 2/16 (12%) "impossible" encounters.
    # That is an artifact of its single cross-track axis: sweeping the full
    # sphere clears all 16, so those two need an out-of-plane (vertical)
    # component rather than being unwinnable.
    candidate_offsets_m: tuple[float, ...] = (0.35, 0.50, 0.65, 0.80, 1.00)
    # MEASURED 2026-08-29 (oracle_prediction_probe.py, v20, 40 episodes,
    # seeds 9001+, reproducing the handoff's oracle at 14/40 collisions):
    #
    #   plans made 120, FAILED 666        -> 84.7% plan failure rate
    #   peak_offset_m                      median 0.35 (the smallest
    #                                      candidate), 1/68 near the 1.00 cap
    #   rise_time_s                        median 1.07
    #   predicted min clearance            median +0.298 m
    #   ACHIEVED min clearance             median +0.202 m
    #   gap (predicted - achieved)         median +0.114 m, p90 +0.556 m
    #   predicted clear but made contact   7/68 (10%)
    #   collisions: 7 on obstacles WITH a plan, 7 on obstacles with NONE
    #
    # Two distinct failures, roughly equal in weight:
    #
    # 1. FEASIBILITY. 85% of planning attempts find no admissible curve.
    #    Note this is not contradicted by oracle_magnitude_probe.py finding
    #    every encounter clearable by a 0.28 m constant displacement: that is
    #    pure geometry on the nominal path, whereas the planner must fit a
    #    dynamically feasible quintic FROM ITS CURRENT OFFSET AND VELOCITY
    #    that clears every obstacle by safety_margin_m within the speed
    #    envelope. Most of the time no such curve exists.
    #
    # 2. TIMING. rise_time_s median 1.07 s against a commit lead of 0.85 s
    #    (oracle_commit_probe.py), so the bump PEAKS AFTER CLOSEST APPROACH.
    #    The vehicle is only partway up when the obstacle arrives -- which is
    #    the 0.35 m commanded / 0.20 m achieved shortfall, and why 10% of
    #    encounters the planner predicted clear end in contact.
    #    minimum_rise_time_s is a floor of 0.55; the realised 1.07 comes from
    #    the magnitude/speed-limit schedule, not from this constant, so
    #    lowering the floor alone will not fix it.
    #
    # Before changing any constant here, note both failures are structural to
    # ONE-SHOT OPEN-LOOP PLANNING: a single curve committed per encounter,
    # predicted against an idealised model, with no feedback on realised
    # displacement. A closed-loop policy trained on the actual dynamics has
    # neither problem by construction. See HANDOFF section 7a.
    minimum_rise_time_s: float = 0.55
    return_time_s: float = 1.0
    sample_dt_s: float = 0.04
    # Matched to offset_rate_limits_mps (1.5 / 1.0), which is what the
    # controller can actually deliver. At 0.70/0.27 the planner rejected
    # 96.0% of its own candidates on this envelope alone -- against 0.7% for
    # moving obstacles and 0.0% for static geometry -- so the "expert" was
    # throttled by a hand-set constant rather than by the avoidance problem,
    # and every oracle ceiling quoted from it understated what is achievable.
    #
    # It also shortens exposure: return_duration is 1.875 * magnitude /
    # speed_limit, so a 1.0 m dodge took 2.68 s at 0.70 m/s and has to stay
    # clear of every obstacle across that whole window. At 1.5 m/s the same
    # dodge is over in 1.25 s.
    max_horizontal_speed_mps: float = 1.5
    max_vertical_speed_mps: float = 1.0
    vertical_cost: float = 1.35
    acceleration_cost: float = 0.015


class LocalTrajectoryExpert:
    """Search smooth local avoidance bumps and retain one per encounter."""

    def __init__(self, config: TrajectoryExpertConfig | None = None):
        self.config = config or TrajectoryExpertConfig()
        self.plan: QuinticAvoidancePlan | None = None

    def reset(self) -> None:
        self.plan = None

    @staticmethod
    def candidate_directions(
        nominal_velocity: np.ndarray,
        obstacle_velocity: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        velocity = np.asarray(nominal_velocity, dtype=np.float64)
        horizontal = velocity.copy()
        horizontal[2] = 0.0
        if np.linalg.norm(horizontal) < 1e-6:
            horizontal = np.array([1.0, 0.0, 0.0])
        horizontal /= np.linalg.norm(horizontal)
        lateral = np.cross(np.array([0.0, 0.0, 1.0]), horizontal)
        lateral /= np.linalg.norm(lateral)
        directions = [
            lateral,
            -lateral,
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]
        if obstacle_velocity is None:
            return directions

        relative_velocity = np.asarray(obstacle_velocity, dtype=np.float64) - velocity
        relative_norm = float(np.linalg.norm(relative_velocity))
        if relative_norm < 1e-6:
            return directions
        incoming = relative_velocity / relative_norm
        basis_a = np.cross(incoming, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(basis_a) < 1e-6:
            basis_a = np.cross(incoming, np.array([1.0, 0.0, 0.0]))
        basis_a /= np.linalg.norm(basis_a)
        basis_b = np.cross(incoming, basis_a)
        basis_b /= np.linalg.norm(basis_b)
        # Eight evenly spaced directions in the plane that most efficiently
        # increases miss distance for this particular incoming trajectory.
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            directions.append(np.cos(angle) * basis_a + np.sin(angle) * basis_b)
        return directions

    def plan_clears(
        self,
        plan,
        now: float,
        nominal_position,
        obstacles,
        static_path_is_clear=None,
    ) -> bool:
        """Does an existing plan still clear the current obstacle set?

        Used instead of discarding a plan whenever a new obstacle becomes
        actionable. With static obstacles the world does not change -- only
        which obstacles happen to trip a proximity threshold -- so replanning
        on that signal made the vehicle reverse direction mid-dodge for no
        physical reason. Keeping a plan that remains safe removes the
        thrashing; a plan that genuinely no longer clears is still replaced.
        """
        cfg = self.config
        end = float(plan.end_time)
        if end <= now:
            return False
        times = np.arange(float(now), end, cfg.sample_dt_s)
        if len(times) == 0:
            return False
        times = np.append(times, end)
        offsets = np.asarray([plan.evaluate(float(t))[0] for t in times])
        reference = (
            np.asarray([nominal_position(float(t)) for t in times]) + offsets
        )
        for entry in obstacles:
            # (pos, vel, r) or (pos, vel, acc, r). Same constant-acceleration
            # propagation as make_plan -- a plan kept alive by this check has
            # to be validated against the motion the obstacles actually
            # follow, or hysteresis just preserves plans that were certified
            # against phantoms.
            if len(entry) == 4:
                position, velocity, acceleration, radius = entry
            else:
                position, velocity, radius = entry[-3], entry[-2], entry[-1]
                acceleration = np.zeros(3)
            horizon = (times - float(now))[:, None]
            swept = (
                np.asarray(position, dtype=np.float64)
                + horizon * np.asarray(velocity, dtype=np.float64)
                + 0.5 * horizon**2 * np.asarray(acceleration, dtype=np.float64)
            )
            clearance = float(
                np.min(np.linalg.norm(reference - swept, axis=1) - float(radius))
            )
            if clearance < cfg.safety_margin_m:
                return False
        if static_path_is_clear is not None and not static_path_is_clear(reference):
            return False
        return True

    def make_plan(
        self,
        *,
        object_id: int,
        now: float,
        time_to_closest_approach: float,
        obstacle_position: np.ndarray,
        obstacle_velocity: np.ndarray,
        combined_radius: float,
        obstacle_acceleration: np.ndarray | None = None,
        nominal_position: Callable[[float], np.ndarray],
        nominal_velocity: np.ndarray,
        static_path_is_clear: Callable[[np.ndarray], bool] | None = None,
        directions: Iterable[np.ndarray] | None = None,
        additional_obstacles: Iterable[
            tuple[np.ndarray, np.ndarray, float]
            | tuple[int, np.ndarray, np.ndarray, float]
        ] = (),
        start_offset: np.ndarray | None = None,
        start_velocity: np.ndarray | None = None,
        start_acceleration: np.ndarray | None = None,
    ) -> QuinticAvoidancePlan | None:
        """Choose the lowest-cost candidate satisfying static and swept clearance.

        ``additional_obstacles`` contains ``(position, velocity,
        combined_radius)`` tuples in the same world frame.  A candidate is
        accepted only if its complete departure/return curve clears every
        active moving sphere, not merely the threat that triggered replanning.
        """
        cfg = self.config
        rise = max(float(time_to_closest_approach), cfg.minimum_rise_time_s)
        peak_time = float(now) + rise
        obstacle_position = np.asarray(obstacle_position, dtype=np.float64)
        obstacle_velocity = np.asarray(obstacle_velocity, dtype=np.float64)
        start_offset = np.zeros(3) if start_offset is None else np.asarray(start_offset)
        start_velocity = (
            np.zeros(3) if start_velocity is None else np.asarray(start_velocity)
        )
        start_acceleration = (
            np.zeros(3) if start_acceleration is None else np.asarray(start_acceleration)
        )
        obstacle_acceleration = (
            np.zeros(3)
            if obstacle_acceleration is None
            else np.asarray(obstacle_acceleration, dtype=np.float64)
        )
        moving_obstacles = [
            (
                obstacle_position,
                obstacle_velocity,
                obstacle_acceleration,
                float(combined_radius),
            )
        ]
        validated_ids = {int(object_id)}
        for obstacle in additional_obstacles:
            # (pos, vel, r) | (id, pos, vel, r) | (id, pos, vel, acc, r).
            # Acceleration is optional so existing three- and four-element
            # callers keep working; omitting it means "straight line".
            acceleration = np.zeros(3)
            if len(obstacle) == 5:
                other_id, position, velocity, acceleration, radius = obstacle
                validated_ids.add(int(other_id))
            elif len(obstacle) == 4:
                other_id, position, velocity, radius = obstacle
                validated_ids.add(int(other_id))
            else:
                position, velocity, radius = obstacle
            moving_obstacles.append(
                (
                    np.asarray(position, dtype=np.float64),
                    np.asarray(velocity, dtype=np.float64),
                    np.asarray(acceleration, dtype=np.float64),
                    float(radius),
                )
            )
        best: tuple[float, QuinticAvoidancePlan] | None = None
        # Why candidates get rejected. The expert fails to find any plan on
        # 89% of attempts even at the sparsest density, so which constraint
        # actually binds decides what is worth widening.
        rejects = {"speed": 0, "moving": 0, "static": 0, "considered": 0}

        candidate_dirs = list(
            directions
            or self.candidate_directions(nominal_velocity, obstacle_velocity)
        )
        for direction_index, direction in enumerate(candidate_dirs):
            direction = np.asarray(direction, dtype=np.float64)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                continue
            direction /= norm
            for magnitude in cfg.candidate_offsets_m:
                rejects["considered"] += 1
                direction_limit = (
                    cfg.max_vertical_speed_mps
                    if abs(float(direction[2])) > 0.5
                    else cfg.max_horizontal_speed_mps
                )
                # Quintic smoothstep has max(ds/du)=1.875.  Choose the
                # shortest return that stays inside the command envelope;
                # unnecessarily long offsets can collide with later walls
                # on a curving nominal path.
                return_duration = max(
                    cfg.return_time_s,
                    1.875 * float(magnitude) / max(direction_limit, 1e-9),
                )
                end_time = peak_time + return_duration
                times = np.arange(float(now), end_time, cfg.sample_dt_s)
                times = np.append(times, end_time)
                proto = QuinticAvoidancePlan(
                    object_id=int(object_id),
                    start_time=float(now),
                    peak_time=peak_time,
                    end_time=end_time,
                    peak_offset=direction * float(magnitude),
                    predicted_min_clearance=-np.inf,
                    start_offset=start_offset,
                    start_velocity=start_velocity,
                    start_acceleration=start_acceleration,
                    validated_object_ids=tuple(sorted(validated_ids)),
                )
                offsets = np.asarray([proto.evaluate(t)[0] for t in times])
                velocities = np.asarray([proto.evaluate(t)[1] for t in times])
                horizontal_speed = np.linalg.norm(velocities[:, :2], axis=1)
                if (
                    float(np.max(horizontal_speed)) > cfg.max_horizontal_speed_mps
                    or float(np.max(np.abs(velocities[:, 2]))) > cfg.max_vertical_speed_mps
                ):
                    rejects["speed"] += 1
                    continue
                reference = np.asarray([nominal_position(float(t)) for t in times]) + offsets
                clearances = []
                for (
                    moving_position,
                    moving_velocity,
                    moving_acceleration,
                    moving_radius,
                ) in moving_obstacles:
                    # Constant-acceleration propagation. A straight line here
                    # mispredicted every kinematic_parabola obstacle by
                    # g*t*T + 0.5*g*T^2 -- about 7 m at 1 s of flight over a
                    # 1.4 s horizon, ~24x the 0.30 m combined hit radius. The
                    # planner certified dodges at a median 0.216 m predicted
                    # clearance that actually grazed at 0.007 m, and rejected
                    # 87% of candidates against obstacles that were not where
                    # it thought they were.
                    horizon = (times - float(now))[:, None]
                    obstacle = (
                        moving_position
                        + horizon * moving_velocity
                        + 0.5 * horizon**2 * moving_acceleration
                    )
                    clearances.append(
                        float(
                            np.min(
                                np.linalg.norm(reference - obstacle, axis=1)
                                - moving_radius
                            )
                        )
                    )
                min_clearance = min(clearances)
                if any(clearance < cfg.safety_margin_m for clearance in clearances):
                    rejects["moving"] += 1
                    continue
                if static_path_is_clear is not None and not static_path_is_clear(reference):
                    rejects["static"] += 1
                    continue
                accelerations = np.asarray([proto.evaluate(t)[2] for t in times])
                vertical = abs(float(direction[2]))
                cost = (
                    float(magnitude) ** 2 * (1.0 + cfg.vertical_cost * vertical)
                    + cfg.acceleration_cost * float(np.mean(np.sum(accelerations**2, axis=1)))
                    + 1e-6 * direction_index
                )
                plan = QuinticAvoidancePlan(
                    object_id=proto.object_id,
                    start_time=proto.start_time,
                    peak_time=proto.peak_time,
                    end_time=proto.end_time,
                    peak_offset=proto.peak_offset,
                    predicted_min_clearance=min_clearance,
                    start_offset=proto.start_offset,
                    start_velocity=proto.start_velocity,
                    start_acceleration=proto.start_acceleration,
                    validated_object_ids=proto.validated_object_ids,
                )
                if best is None or cost < best[0]:
                    best = (cost, plan)

        self.last_rejects = rejects
        self.plan = None if best is None else best[1]
        return self.plan

    def world_velocity_residual(
        self,
        time: float,
        outer_position_gain_hz: np.ndarray,
        gate: float = 1.0,
        actual_displacement: np.ndarray | None = None,
    ) -> np.ndarray:
        """Residual that makes the cascaded controller track the planned bump.

        With an open gate the environment intentionally removes restoration
        toward the *nominal* path.  When the actual displacement is supplied,
        restore toward the local avoidance plan instead.  This makes oracle
        labels corrective on policy-visited DAgger states rather than merely
        replaying a feed-forward velocity profile that assumes perfect prior
        expert execution.
        """
        if self.plan is None or float(time) >= self.plan.end_time:
            self.plan = None
            return np.zeros(3, dtype=np.float64)
        displacement, velocity, _ = self.plan.evaluate(time)
        gain = np.asarray(outer_position_gain_hz, dtype=np.float64)
        if actual_displacement is not None:
            tracking_error = displacement - np.asarray(
                actual_displacement, dtype=np.float64
            )
            return velocity + gain * tracking_error
        return velocity + (1.0 - float(gate)) * gain * displacement
