"""Reactive-dodge env: residual CTBR control on top of an SE3 + minsnap nominal.

This subclass of :class:`BaseNeurosimRLEnv` owns everything that is
specific to the reactive_dodge task:

- a per-episode SE3 controller + random minsnap reference trajectory,
- residual CTBR control: the policy outputs a (gated) delta added to the
  nominal command and clipped to vehicle bounds,
- threat metrics: per-step closest-approach prediction and a nominal
  counterfactual collision check used as reward / observation context,
- a richer observation built from tracking errors + lookahead errors +
  the normalized nominal command.

Hover-stop and any other simpler task lives in ``env.py``; this file is
intentionally separate so the residual / nominal / threat machinery does
not leak into the shared base.
"""

import logging
from typing import Any

import numpy as np
from gymnasium import spaces

from neurosim.core.control import create_controller
from neurosim.core.coord_trans import rotate_vector_by_quat
from neurosim.core.trajectory import create_trajectory
from neurosim.core.trajectory.habitat_trajs import (
    sample_random_navigable_point_with_height,
)

from .env import BaseNeurosimRLEnv

logger = logging.getLogger(__name__)

_THREAT_INFO_KEYS = (
    "min_obstacle_distance",
    "predicted_closest_distance",
    "time_to_closest_approach",
    "obstacle_threat",
    "threat_ids",
    "near_miss_ids",
    "nominal_counterfactual_collision",
)
_TRACKING_INFO_KEYS = (
    "flat",
    "lookahead_errors",
    "nominal_control_normalized",
)
_TASK_METRIC_KEYS = (
    "min_obstacle_distance",
    "predicted_closest_distance",
    "time_to_closest_approach",
    "obstacle_threat",
    "meaningful_encounter_count",
    "near_miss_count",
    "nominal_counterfactual_collision_count",
    "dodge_success_count",
    "post_dodge_recovery_time",
    "correction_energy",
    "gate_value",
)


def constant_velocity_closest_approach(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    horizon_s: float,
) -> tuple[float, float]:
    """Closest approach (clearance, time) under constant relative velocity.

    Time is clipped to ``[0, horizon_s]`` so receding-only encounters
    report ``time = 0`` and a clearance equal to the current distance.
    """
    rel_pos = np.asarray(rel_pos, dtype=np.float64)
    rel_vel = np.asarray(rel_vel, dtype=np.float64)
    speed_sq = float(np.dot(rel_vel, rel_vel))
    if speed_sq < 1e-9:
        return float(np.linalg.norm(rel_pos)), np.inf
    tca = -float(np.dot(rel_pos, rel_vel)) / speed_sq
    tca = float(np.clip(tca, 0.0, horizon_s))
    closest = rel_pos + rel_vel * tca
    return float(np.linalg.norm(closest)), tca


_KINEMATIC_MODES = {"kinematic_line", "kinematic_parabola"}


def obstacle_acceleration(item: Any) -> np.ndarray:
    """Habitat-frame acceleration of a dynamic-obstacle item.

    Only ``kinematic_parabola`` accelerates, and it does so purely along
    Habitat -Y, matching the ``displacement[1] -= 0.5 * g * t**2`` term in
    ``DynamicObstacleManager._update_kinematic``. Everything else is a
    straight line.
    """
    if getattr(item, "motion_mode", None) == "kinematic_parabola":
        return np.array([0.0, -float(item.gravity_mps2), 0.0])
    return np.zeros(3)


def obstacle_velocity(item: Any, sim_time: float | None = None) -> np.ndarray:
    """Habitat-frame linear velocity of a dynamic-obstacle item.

    Kinematic obstacles are moved by directly writing ``translation``, and
    their rigid-body ``linear_velocity`` is deliberately set to zero -- so
    reading it would report a stationary obstacle and silently zero out
    every closest-approach / time-to-contact estimate. The integrated
    velocity on the item is the real one.

    ``item.velocity`` is the *launch* velocity. For a parabola that is only
    correct at t = 0: the true velocity is ``v0 - g*t`` along Habitat Y, so
    reporting v0 understates the fall by ``g*t``. Passing ``sim_time``
    returns the current velocity; omitting it preserves the launch value for
    callers that have no clock. Measured against a 0.30 m combined hit
    radius, the stale value contributed roughly 4.2 m of predicted-position
    error at 1 s of flight over a 1.4 s planning horizon, which is why the
    privileged oracle scored *below* a do-nothing control (7.5% vs 12.5%).
    """
    mode = getattr(item, "motion_mode", None)
    if mode in _KINEMATIC_MODES:
        velocity = np.asarray(item.velocity, dtype=np.float64)
        if sim_time is not None and mode == "kinematic_parabola":
            elapsed = max(0.0, float(sim_time) - float(item.born_time))
            velocity = velocity + obstacle_acceleration(item) * elapsed
        return velocity
    obj_vel = getattr(item.obj, "linear_velocity", None)
    if obj_vel is not None:
        return np.asarray(obj_vel, dtype=np.float64)
    return np.asarray(item.velocity, dtype=np.float64)


def obstacle_threat_priority(row: dict[str, Any]) -> tuple[float, float, float]:
    """Order imminent collision threats ahead of merely nearby obstacles."""
    return (
        float(row["predicted_clearance"]),
        float(row["time_to_closest_approach"]),
        float(row["clearance"]),
    )


class ReactiveDodgeEnv(BaseNeurosimRLEnv):
    """Residual-control env for the reactive_dodge task."""

    def __init__(self, env_config: dict[str, Any], *, train: bool = False):
        # Per-episode state, must exist before the first reset (which is
        # triggered indirectly by ``super().__init__`` through
        # ``_sync_from_simulator`` → ``_build_action_space``).
        self._nominal_controller = None
        self._nominal_trajectory = None
        self._current_flat: dict[str, Any] | None = None
        self._cached_nominal_control: dict[str, np.ndarray | float] | None = None
        self._last_threats: dict[str, Any] = {}
        self._last_tracking: dict[str, Any] = {}
        self._last_gate: float = 1.0
        self._last_correction_energy: float = 0.0
        # Reference-position offset accumulated by velocity_delta_integrated.
        # This is genuine hidden state, so it is also fed to the task for the
        # observation -- without it the policy cannot know how far off the
        # nominal path its own integrator has already carried it.
        self._offset_cmd = np.zeros(3, dtype=np.float64)
        self._offset_vel = np.zeros(3, dtype=np.float64)
        self._last_delta_velocity = np.zeros(3, dtype=np.float64)
        self._filtered_action = np.zeros(3, dtype=np.float64)
        self._intercept_config = dict(
            (env_config.get("visual_backend", {}) or {})
            .get("dynamic_obstacles", {})
            .get("intercept_schedule", {})
            or {}
        )
        # Eval-only: aim obstacle spawns/throws at the seed-determined
        # nominal trajectory instead of the live (policy-dependent) drone
        # position, so "the same seed" is a genuinely reproducible scenario
        # across different checkpoints. Off by default -- training keeps
        # the organic spawner chasing the actual drone.
        self._deterministic_obstacle_aim = bool(
            env_config.get("deterministic_obstacle_aim", False)
        )
        super().__init__(env_config, train=train)

    # ---- Subclass hooks ---------------------------------------------------

    def _build_action_space(self) -> spaces.Space:
        # Action shape comes from the task (4 for ctbr_delta, 5 for gated).
        action_dim = int(self._task.action_dim or 4)
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

    def _place_static_obstacles(self, rng: np.random.Generator) -> None:
        """Put pre-placed obstacles directly on the nominal path.

        Sampled from the trajectory itself rather than extrapolated from the
        drone's heading, so a passive drone flies into them by construction.
        Enabled by ``static_obstacle_count`` on the obstacle config; zero
        leaves the normal spawner untouched.
        """
        manager = getattr(self.sim.visual_backend, "_dynamic_obstacles", None)
        if manager is None or not getattr(manager, "cfg", None):
            return
        count = int(getattr(manager.cfg, "static_obstacle_count", 0) or 0)
        if count <= 0:
            return

        # Span the episode, not the trajectory. The MinSnap path runs ~40 s
        # while an episode covers only its first 10 -- placing across the full
        # path put four of five obstacles beyond anywhere the drone ever
        # reached (measured 1.1 encounters/episode for five obstacles).
        end = min(
            float(self._nominal_trajectory.t_keyframes[-1]),
            float(getattr(self, "episode_seconds", 0.0)) or float(
                self._nominal_trajectory.t_keyframes[-1]
            ),
        )
        # Keep clear of the very start and end so the drone is established on
        # the path before the first one and is not blocked at the finish.
        # Spaced by arc length, not time. MinSnap starts from rest, so the
        # first second covers almost no distance -- evenly spaced *times* put
        # an obstacle within the hit radius of the start position and the
        # episode ended on step 1.
        probe = np.linspace(0.0, end, 400)
        samples = np.asarray(
            [self._nominal_trajectory.update(float(t))["x"] for t in probe],
            dtype=np.float64,
        )
        travelled = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1)))
        )
        total = float(travelled[-1])
        if total < 2.0:
            return
        # Keep the first obstacle a safe distance ahead of the start.
        wanted = np.linspace(max(2.5, 0.12 * total), 0.88 * total, count)
        times = np.interp(wanted, travelled, probe)
        scatter = float(getattr(manager.cfg, "aim_scatter_m", 0.0))
        # Slalom offsets. Placing every obstacle near the path makes the field
        # degenerate: an obstacle must sit within the hit radius (~0.45 m) of
        # the path to block a passive drone at all, so a single constant
        # lateral offset of ~0.9 m clears all of them at once -- no perception
        # required, and measurably what a cloned policy converges to (0.919 m
        # while dodging, 0.457 m standing).
        #
        # Interleaving obstacles at the offset the drone would escape *to*
        # removes that solution: centre obstacles force |x| >= 0.7, while one
        # at +0.9 m forces x < 0.45 or x > 1.35, past the 1.2 m authority. No
        # constant offset satisfies both, so the drone has to weave.
        # Randomised per episode. The fixed pattern [centre, +s, centre, -s]
        # was identical in every episode, and obstacles sit at fixed
        # arc-length fractions, so the whole layout was a deterministic
        # function of the nominal path -- which the state already observes
        # through its lookahead errors. A policy could memorise "a third of
        # the way along, go left" and never look at anything. Measured: a
        # clone with its event frames ZEROED scored 28.3% against a 0.0%
        # passive control, and *beat* the same clone with vision (21.7%),
        # despite its encoder reaching aux distance R2 0.308.
        #
        # Structure is preserved -- centre obstacles block the path so a
        # passive drone always fails, side obstacles block the escape lane so
        # no constant offset works -- but the side and magnitude are drawn
        # per episode, so which way to dodge can only come from perception.
        slalom = float(getattr(manager.cfg, "slalom_offset_m", 0.0))
        lane_pattern = []
        if slalom > 0.0:
            for i in range(count):
                if i % 2 == 0:
                    # Blocking obstacle: near the path, slightly jittered.
                    lane_pattern.append(float(rng.uniform(-0.2, 0.2)))
                else:
                    # Trap obstacle: random side, random magnitude.
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    lane_pattern.append(sign * float(rng.uniform(0.78, 1.22)) * slalom)
        points = []
        for i, t in enumerate(times):
            flat = self._nominal_trajectory.update(float(t))
            position = np.asarray(flat["x"], dtype=np.float64)
            velocity = np.asarray(flat["x_dot"], dtype=np.float64)
            speed = float(np.linalg.norm(velocity))
            offset = np.zeros(3)
            if speed > 1e-6:
                heading = velocity / speed
                lateral_axis = np.cross(np.array([0.0, 0.0, 1.0]), heading)
                axis_norm = float(np.linalg.norm(lateral_axis))
                lateral_axis = (
                    lateral_axis / axis_norm
                    if axis_norm > 1e-9
                    else np.array([0.0, 1.0, 0.0])
                )
                if lane_pattern:
                    offset = lateral_axis * lane_pattern[i]
                if scatter > 0.0:
                    jitter = rng.normal(size=3)
                    jitter -= heading * float(np.dot(jitter, heading))
                    norm = float(np.linalg.norm(jitter))
                    if norm > 1e-9:
                        offset = offset + jitter / norm * float(
                            rng.uniform(0.0, scatter)
                        )
            points.append(
                self.sim.coord_trans.transform_batch(
                    (position + offset)[None, :]
                )[0]
            )
        manager.place_static_obstacles(points, sim_time=float(self.sim.time))

    def _on_episode_reset(
        self,
        *,
        rng: np.random.Generator,
        hab_start: np.ndarray,
    ) -> None:
        # Build a fresh tracking controller and reference trajectory for the
        # episode; seed the dynamics state from the trajectory's first
        # waypoint (overriding the base's random-velocity init).
        self._build_nominal_controller()
        self._build_nominal_trajectory(hab_start=hab_start, rng=rng)
        self._current_flat = self._nominal_trajectory.update(self.sim.time)

        yaw = float(self._current_flat.get("yaw", 0.0))
        yaw_dot = float(self._current_flat.get("yaw_dot", 0.0))
        self.sim.dynamics.state = {
            "x": np.asarray(self._current_flat["x"], dtype=np.float32),
            "v": np.asarray(self._current_flat["x_dot"], dtype=np.float32),
            "q": self._quat_from_yaw(yaw),
            "w": np.zeros(3, dtype=np.float32),
            "yaw": yaw,
            "yaw_dot": yaw_dot,
        }

        # Cache the initial nominal command so the first observation has a
        # meaningful nominal_control_normalized field instead of zeros.
        self._cached_nominal_control = self._nominal_controller.update(
            self.sim.time, self.sim.dynamics.state, self._current_flat
        )
        self._place_static_obstacles(rng)
        # Solved fresh each episode by the privileged oracle, if used.
        self._global_static_plan = "unset"
        self._last_threats = {}
        self._last_tracking = {}
        self._last_gate = 1.0
        self._last_correction_energy = 0.0
        self._offset_cmd = np.zeros(3, dtype=np.float64)
        self._offset_vel = np.zeros(3, dtype=np.float64)
        self._last_delta_velocity = np.zeros(3, dtype=np.float64)
        self._filtered_action = np.zeros(3, dtype=np.float64)

        # Solved against the trajectory that was just built, so it must come
        # after _build_nominal_trajectory and after the manager's own reset.
        manager = getattr(self.sim.visual_backend, "_dynamic_obstacles", None)
        if manager is not None and getattr(manager, "enabled", False):
            manager.set_intercept_schedule(self._build_intercept_schedule(rng))

    def _update_obstacle_aim_override(self) -> None:
        manager = getattr(self.sim.visual_backend, "_dynamic_obstacles", None)
        if manager is None or self.sim.safety is None:
            return
        nominal_x = np.asarray(self._nominal_flat()["x"], dtype=np.float64)
        manager.set_aim_override(self.sim.safety.dynamics_to_habitat(nominal_x))

    def _control_from_action(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
        if self._deterministic_obstacle_aim:
            self._update_obstacle_aim_override()
        if self._task.residual_control_mode == "velocity_command":
            return self._control_from_velocity_command(action)
        if self._task.residual_control_mode in {
            "velocity_delta_integrated",
            "gated_velocity_delta_integrated",
        }:
            # Identical control path: split_action already folds the gate
            # into the delta, so the gated variant differs only in that the
            # policy can drive its own correction to exactly zero.
            return self._control_from_velocity_delta(action)
        if self._task.residual_control_mode == "desired_body_offset":
            return self._control_from_desired_body_offset(action)
        if self._task.residual_control_mode in {
            "cascaded_velocity",
            "gated_cascaded_velocity",
        }:
            return self._control_from_cascaded_velocity(action)

        nominal_control = self._nominal_control()
        gate, delta = self._task.split_action(action)
        merged = dict(nominal_control)
        merged["cmd_thrust"] = float(nominal_control["cmd_thrust"]) + (
            float(delta[0])
            * self._task.delta_thrust_fraction
            * self._vehicle.hover_thrust
        )
        merged["cmd_w"] = (
            np.asarray(nominal_control["cmd_w"], dtype=np.float64)
            + np.asarray(delta[1:4], dtype=np.float64) * self._task.delta_rate_limits
        )
        control = self._vehicle.clip_control(merged)
        # Cache for context + info dict logging.
        self._last_gate = gate
        self._last_correction_energy = float(np.mean(np.square(delta)))
        self._cached_nominal_control = nominal_control
        return control

    def _control_from_velocity_command(
        self, action: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        """Command a world-frame velocity directly: no reference shifting.

        ``v_cmd = v_ref + dv + k_return * (x_ref - x)``

        The correction is a plain velocity offset, so there is no
        integrator, no offset clip, and no interaction with a position
        controller's gains. The return term is what keeps the drone near
        the path once a dodge ends, and it is capped so it can never
        dominate the command.

        This exists because the SE3 position path can lose the vehicle
        outright: its ``kp_pos`` term can rotate ``F_des`` below horizontal,
        after which commanded thrust collapses to zero and the drone falls
        (measured in ~10-15% of episodes with zero action).
        """
        task = self._task
        state = self.sim.dynamics.state
        dt = float(self.steps_per_action / self.sim.config.world_rate)

        gate, delta = task.split_action(action)
        commanded = np.clip(delta, -1.0, 1.0)
        tau_action = float(task.action_filter_tau_s)
        alpha = 1.0 if tau_action <= dt else dt / tau_action
        self._filtered_action = self._filtered_action + alpha * (
            commanded - self._filtered_action
        )

        delta_body = self._filtered_action * task.delta_velocity_limits_mps
        delta_world = rotate_vector_by_quat(delta_body, state["q"])

        flat = self._nominal_flat()
        pos_err = np.asarray(flat["x"], dtype=np.float64) - np.asarray(
            state["x"], dtype=np.float64
        )
        ret = task.return_gain_hz * pos_err
        speed = float(np.linalg.norm(ret))
        if speed > task.max_return_speed_mps:
            ret = ret * (task.max_return_speed_mps / speed)

        v_cmd = np.asarray(flat["x_dot"], dtype=np.float64) + delta_world + ret
        control = self._vehicle.clip_control({"cmd_v": v_cmd})

        # Deviation from the path is now an outcome, not a commanded state;
        # report it as the offset so observations and metrics still work.
        self._offset_cmd = -pos_err
        self._last_gate = gate
        self._last_correction_energy = float(np.mean(np.square(self._filtered_action)))
        self._last_delta_velocity = delta_world
        self._cached_nominal_control = control
        return control

    def _control_from_velocity_delta(
        self, action: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        """Integrate a body-frame velocity correction into the SE3 reference.

        The offset obeys ``de/dt = dv - e / tau`` and is clipped to
        ``offset_max_m``. The reference velocity handed to SE3 is the
        *exact* derivative of the applied (post-clip) offset rather than
        the analytic ``dv - e/tau``: at saturation the analytic form leaves
        a residual velocity feedforward, which the controller converts into
        an extra (Kd/Kp) * v_residual of deviation, so the clip would stop
        being a real bound on how far the drone leaves the path.
        """
        task = self._task
        state = self.sim.dynamics.state
        dt = float(self.steps_per_action / self.sim.config.world_rate)

        # Low-pass the action first: the offset averages zero-mean action
        # noise away, but the derivative handed to SE3 as reference velocity
        # does not, and Kd turns it straight into acceleration.
        commanded = np.clip(task.split_action(action)[1], -1.0, 1.0)
        tau_action = float(task.action_filter_tau_s)
        alpha = 1.0 if tau_action <= dt else dt / tau_action
        self._filtered_action = self._filtered_action + alpha * (
            commanded - self._filtered_action
        )

        delta_body = self._filtered_action * task.delta_velocity_limits_mps
        delta_world = rotate_vector_by_quat(delta_body, state["q"])

        previous_offset = self._offset_cmd.copy()
        self._offset_cmd = np.clip(
            self._offset_cmd
            + dt * (delta_world - self._offset_cmd / task.offset_tau_s),
            -task.offset_max_m,
            task.offset_max_m,
        )
        applied_velocity = (self._offset_cmd - previous_offset) / dt

        flat = self._nominal_flat()
        shifted_flat = dict(flat)
        shifted_flat["x"] = np.asarray(flat["x"], dtype=np.float64) + self._offset_cmd
        shifted_flat["x_dot"] = (
            np.asarray(flat["x_dot"], dtype=np.float64) + applied_velocity
        )

        nominal_control = self._nominal_controller.update(
            self.sim.time, state, shifted_flat
        )
        control = self._vehicle.clip_control(dict(nominal_control))

        self._last_gate = 1.0
        self._last_correction_energy = float(np.mean(np.square(self._filtered_action)))
        self._last_delta_velocity = delta_world
        self._cached_nominal_control = nominal_control
        return control

    def _control_from_desired_body_offset(
        self, action: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        """Track a smooth, bounded body-frame offset selected by the policy."""
        task = self._task
        state = self.sim.dynamics.state
        dt = float(self.steps_per_action / self.sim.config.world_rate)

        commanded = np.clip(task.split_action(action)[1], -1.0, 1.0)
        desired_body = commanded * task.offset_max_m
        desired_world = rotate_vector_by_quat(desired_body, state["q"])

        previous_offset = self._offset_cmd.copy()
        accel_limit = float(getattr(task, "offset_accel_limit_mps2", 0.0) or 0.0)
        if accel_limit > 0.0:
            # Acceleration-limited (second-order) shaping.
            #
            # The first-order form below steps the offset *velocity*: alpha is
            # applied to the whole position error, so a fresh command starts
            # moving at ~amp/tau immediately -- 2.25 m/s for a 0.45 m dodge at
            # tau 0.2. That velocity step reaches SE3 through
            # shifted_flat["x_dot"] while x_ddot stays at the nominal value,
            # so the reference is internally inconsistent and SE3 answers the
            # unexplained velocity error with attitude. Measured in an empty
            # scene: a 0.45 m sidestep flips the vehicle past 109 deg on 4/4
            # seeds.
            #
            # Clamping offset_rate_limits_mps does not fix it (0.6 m/s still
            # flipped 4/4), because a rate limiter produces bang-bang velocity
            # -- bounded speed, unbounded acceleration. Bounding acceleration
            # is what makes the reference physically realisable, and it lets
            # the velocity start at zero and ramp.
            error = desired_world - self._offset_cmd
            # Approach velocity: never travel faster than can still be
            # braked to zero within the remaining error, sqrt(2*a*|e|).
            # A plain proportional law (error/tau) with an acceleration
            # clamp cannot stop in time -- at a=0.4, v=0.6 the stopping
            # distance is v^2/2a = 0.45 m, exactly the dodge distance, and
            # the command overshot a 0.45 m target to 0.684 m. Overshoot is
            # not harmless here: the planner certifies clearance for the
            # commanded path, so travelling 52% further flies through
            # geometry it validated as clear.
            braking = np.sqrt(2.0 * accel_limit * np.abs(error))
            target_velocity = np.sign(error) * np.minimum(
                braking, task.offset_rate_limits_mps
            )
            dv = np.clip(
                target_velocity - self._offset_vel,
                -accel_limit * dt,
                accel_limit * dt,
            )
            self._offset_vel = np.clip(
                self._offset_vel + dv,
                -task.offset_rate_limits_mps,
                task.offset_rate_limits_mps,
            )
            self._offset_cmd = np.clip(
                self._offset_cmd + self._offset_vel * dt,
                -task.offset_max_m,
                task.offset_max_m,
            )
        else:
            alpha = min(dt / task.offset_command_tau_s, 1.0)
            requested_delta = alpha * (desired_world - self._offset_cmd)
            max_delta = task.offset_rate_limits_mps * dt
            applied_delta = np.clip(requested_delta, -max_delta, max_delta)
            self._offset_cmd = np.clip(
                self._offset_cmd + applied_delta,
                -task.offset_max_m,
                task.offset_max_m,
            )
        applied_velocity = (self._offset_cmd - previous_offset) / dt
        self._offset_vel = applied_velocity.copy()

        flat = self._nominal_flat()
        shifted_flat = dict(flat)
        shifted_flat["x"] = np.asarray(flat["x"], dtype=np.float64) + self._offset_cmd
        shifted_flat["x_dot"] = (
            np.asarray(flat["x_dot"], dtype=np.float64) + applied_velocity
        )
        nominal_control = self._nominal_controller.update(
            self.sim.time, state, shifted_flat
        )
        control = self._vehicle.clip_control(dict(nominal_control))

        self._filtered_action = commanded.copy()
        self._last_gate = 1.0
        self._last_correction_energy = float(np.mean(np.square(commanded)))
        self._last_delta_velocity = applied_velocity
        self._cached_nominal_control = nominal_control
        return control

    def _control_from_cascaded_velocity(
        self, action: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        """Run an explicit position outer loop around SE3 velocity tracking.

        ``v_cmd = v_nom + K_outer * (x_nom - x) + dv_policy``.  SE3 receives
        the measured position as its position reference, making its own
        position error identically zero; it therefore acts as the inner
        velocity/attitude loop instead of fighting the policy correction with
        a second, hidden position loop.
        """
        task = self._task
        state = self.sim.dynamics.state
        dt = float(self.steps_per_action / self.sim.config.world_rate)

        gate, delta = task.split_action(action)
        commanded = np.clip(delta, -1.0, 1.0)
        tau_action = float(task.action_filter_tau_s)
        alpha = 1.0 if tau_action <= dt else dt / tau_action
        self._filtered_action = self._filtered_action + alpha * (
            commanded - self._filtered_action
        )

        delta_body = self._filtered_action * task.delta_velocity_limits_mps
        delta_world = rotate_vector_by_quat(delta_body, state["q"])

        flat = self._nominal_flat()
        nominal_position = np.asarray(flat["x"], dtype=np.float64)
        position = np.asarray(state["x"], dtype=np.float64)
        position_error = nominal_position - position
        # In gated mode the avoidance trajectory owns lateral motion while
        # the gate is open.  Applying the nominal-path restoring loop at full
        # strength at the same time makes the two references contradictory
        # and caps reachable displacement at dv_limit / K_outer.  A soft gate
        # also gives a soft hand-off back to nominal tracking.
        return_scale = (
            1.0 - gate
            if task.residual_control_mode == "gated_cascaded_velocity"
            else 1.0
        )
        return_velocity = np.clip(
            return_scale * task.outer_position_gain_hz * position_error,
            -task.max_outer_return_velocity_mps,
            task.max_outer_return_velocity_mps,
        )
        velocity_command = (
            np.asarray(flat["x_dot"], dtype=np.float64)
            + return_velocity
            + delta_world
        )

        inner_flat = dict(flat)
        inner_flat["x"] = position.copy()
        inner_flat["x_dot"] = velocity_command
        nominal_control = self._nominal_controller.update(
            self.sim.time, state, inner_flat
        )
        control = self._vehicle.clip_control(dict(nominal_control))

        # The state observation reports actual displacement from the nominal
        # path, not a commanded offset, and exposes the filtered policy action.
        self._offset_cmd = position - nominal_position
        self._last_gate = gate
        self._last_correction_energy = float(np.mean(np.square(self._filtered_action)))
        self._last_delta_velocity = delta_world
        self._cached_nominal_control = nominal_control
        return control

    def _effective_previous_action(self) -> np.ndarray | None:
        """Report the *filtered* action as the policy's previous correction.

        The filter carries state between steps, so exposing the raw action
        would leave that state hidden and the observation non-Markov. The
        filtered action is itself the filter state, so reporting it keeps
        the policy fully informed about the correction actually in effect.
        """
        if self._task.residual_control_mode in {
            "velocity_delta_integrated",
            "desired_body_offset",
            "cascaded_velocity",
            "gated_cascaded_velocity",
        }:
            if self._task.residual_control_mode == "gated_cascaded_velocity":
                return np.concatenate(
                    [
                        np.asarray([self._last_gate], dtype=np.float32),
                        self._filtered_action.astype(np.float32, copy=True),
                    ]
                )
            return self._filtered_action.astype(np.float32, copy=True)
        return super()._effective_previous_action()

    def _populate_task_context(
        self,
        *,
        state: dict[str, np.ndarray],
        action: np.ndarray | None,
    ) -> None:
        threats = self._compute_threat_metrics(state)
        lookahead_errors = self._lookahead_errors(state)
        nominal_control_normalized = (
            self._vehicle.control_to_normalized(self._cached_nominal_control)
            if self._cached_nominal_control is not None
            else None
        )

        # Re-sample the reference at the *current* sim time. `_current_flat`
        # was last written when the control was computed, one action-interval
        # of world steps ago, so reusing it would score the post-step state
        # against a pre-step reference -- a systematic v_avg * dt bias in
        # both the tracking error and the observation.
        tracking = {
            "flat": self._nominal_flat(),
            "lookahead_errors": lookahead_errors,
            "nominal_control_normalized": nominal_control_normalized,
        }

        context: dict[str, Any] = {
            "sim_time": float(self.sim.time),
            "previous_action": self._prev_action,
            "offset_cmd": self._offset_cmd.copy(),
            "delta_velocity": self._last_delta_velocity.copy(),
            "active_obstacle_count": len(self.active_obstacles()),
            "obstacle_relative_states": self._obstacle_relative_states(state),
            "bounds_margin_m": self.sim.safety.bounds_margin(
                self.sim.safety.dynamics_to_habitat(np.asarray(state["x"]))
            ),
            "remaining_steps": max(
                (float(self.episode_seconds) - float(self.sim.time))
                / (
                    self.policy_decimation
                    * self.steps_per_action
                    / self.sim.config.world_rate
                ),
                0.0,
            ),
        }
        context.update(threats)
        context.update(tracking)
        if action is not None:
            context["correction_energy"] = self._last_correction_energy
            context["gate_value"] = self._last_gate

        self._task.set_context(context)
        self._last_threats = threats
        self._last_tracking = tracking

    def _step_info_extras(self, reward_terms: dict[str, float]) -> dict[str, Any]:
        task_metrics = {
            key: reward_terms[key] for key in _TASK_METRIC_KEYS if key in reward_terms
        }
        task_context = {**self._last_threats}
        if self._current_flat is not None:
            task_context["sim_time"] = float(self.sim.time)
        if reward_terms:
            if "correction_energy" in reward_terms:
                task_context["correction_energy"] = reward_terms["correction_energy"]
            if "gate_value" in reward_terms:
                task_context["gate_value"] = reward_terms["gate_value"]
        return {
            "task_context": task_context,
            "task_metrics": task_metrics,
        }

    # ---- Public helpers used by tools / scripts --------------------------

    def active_obstacles(self) -> dict[int, Any]:
        """Active dynamic-obstacle dict (id -> item) from the visual backend."""
        manager = getattr(self.sim.visual_backend, "_dynamic_obstacles", None)
        if manager is None:
            return {}
        return dict(getattr(manager, "_active", {}))

    def last_threats(self) -> dict[str, Any]:
        """Threat-metric slice of the most recent step."""
        return {
            key: self._last_threats[key]
            for key in _THREAT_INFO_KEYS
            if key in self._last_threats
        }

    def last_tracking(self) -> dict[str, Any]:
        """Tracking-context slice of the most recent step."""
        return {
            key: self._last_tracking[key]
            for key in _TRACKING_INFO_KEYS
            if key in self._last_tracking
        }

    @staticmethod
    def constant_velocity_closest_approach(
        rel_pos: np.ndarray,
        rel_vel: np.ndarray,
        horizon_s: float,
    ) -> tuple[float, float]:
        return constant_velocity_closest_approach(rel_pos, rel_vel, horizon_s)

    @staticmethod
    def obstacle_velocity(item: Any, sim_time: float | None = None) -> np.ndarray:
        return obstacle_velocity(item, sim_time)

    @staticmethod
    def obstacle_acceleration(item: Any) -> np.ndarray:
        return obstacle_acceleration(item)

    # ---- Nominal controller / trajectory ---------------------------------

    def _build_nominal_controller(self) -> None:
        controller_cfg = dict(getattr(self._task, "controller_config", {}) or {})
        controller_cfg.setdefault("model", "rotorpy_se3")
        controller_cfg.setdefault("vehicle", self._dynamics_config["vehicle"])
        self._nominal_controller = create_controller(**controller_cfg)

    def _build_nominal_trajectory(
        self,
        *,
        hab_start: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        trajectory_cfg = dict(getattr(self._task, "trajectory_config", {}) or {})
        trajectory_cfg.setdefault("model", "habitat_random_minsnap")
        trajectory_cfg.setdefault("target_length", max(5.0, self.episode_seconds))
        trajectory_cfg.setdefault("min_waypoint_distance", 2.0)
        trajectory_cfg.setdefault("max_waypoints", 100)
        trajectory_cfg.setdefault("v_avg", 1.0)
        pathfinder = self.sim.visual_backend._sim.pathfinder
        trajectory_cfg.setdefault(
            "start", self._lift_off_floor(hab_start, pathfinder, trajectory_cfg, rng)
        )
        trajectory_cfg["pathfinder"] = pathfinder
        trajectory_cfg["coord_transform"] = self.sim.coord_trans.inverse_transform_batch
        trajectory_cfg["collision_sim"] = self.sim.visual_backend._sim
        trajectory_cfg["trajectory_to_habitat"] = (
            self.sim.coord_trans.transform_batch
        )
        if "seed" not in trajectory_cfg:
            trajectory_cfg["seed"] = int(rng.integers(0, np.iinfo(np.int32).max))
        self._nominal_trajectory = create_trajectory(**trajectory_cfg)

    @staticmethod
    def _lift_off_floor(
        hab_start: np.ndarray,
        pathfinder: Any,
        trajectory_cfg: dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Raise the sampled start point into the flight-altitude band.

        ``sample_habitat_start`` (base env) draws directly from the
        navmesh, which sits at floor level, with no notion of the altitude
        margin every other waypoint gets from
        ``sample_random_navigable_point_with_height``. Passed straight
        through as the trajectory's ``start``, that meant every episode
        *began* at floor height regardless of ``min_altitude_m`` --
        confirmed directly: one episode's minimum altitude measured 0.023 m
        *below* the scene's median navmesh height.

        Keeping ``hab_start``'s exact (x, z) and only searching for a valid
        height there is too brittle: a single fixed column can be
        unnavigable at every height in the altitude band (a doorway, a low
        overhang), and silently falling back to the unlifted point on that
        one failure is exactly how two of the first three episodes in
        testing stayed at floor level. Delegating to the same
        margin-respecting sampler used for every other waypoint -- which
        retries a fresh (x, z, y) up to 100 times -- is far more robust; the
        cost is that the start (x, z) is no longer necessarily the exact
        point ``sample_habitat_start`` found, which does not matter here
        since nothing has been built from it yet.
        """
        min_altitude = float(trajectory_cfg.get("min_altitude_m", 0.0))
        if min_altitude <= 0.0:
            return hab_start
        lateral_margin = float(trajectory_cfg.get("lateral_margin_m", 0.0))
        pathfinder.seed(int(rng.integers(0, 2**31 - 1)))
        candidate = sample_random_navigable_point_with_height(
            pathfinder,
            min_altitude_m=min_altitude,
            max_altitude_m=trajectory_cfg.get("max_altitude_m", None),
            ceiling_margin_m=float(trajectory_cfg.get("ceiling_margin_m", 0.3)),
            lateral_margin_m=lateral_margin,
        )
        return hab_start if candidate is None else candidate

    def _nominal_flat(self) -> dict[str, Any]:
        """Sample the un-shifted reference at the current sim time."""
        if self._nominal_controller is None or self._nominal_trajectory is None:
            raise RuntimeError("Nominal controller/trajectory requested before reset")
        self._current_flat = self._nominal_trajectory.update(self.sim.time)
        return self._current_flat

    def _nominal_control(self) -> dict[str, np.ndarray | float]:
        return self._nominal_controller.update(
            self.sim.time,
            self.sim.dynamics.state,
            self._nominal_flat(),
        )

    def _lookahead_errors(self, state: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self._nominal_trajectory is None:
            return []
        errors = []
        for dt in getattr(self._task, "lookahead_seconds", ()):
            flat = self._nominal_trajectory.update(self.sim.time + float(dt))
            errors.append(
                np.asarray(state["x"], dtype=np.float32)
                - np.asarray(flat["x"], dtype=np.float32)
            )
        return errors

    def _build_intercept_schedule(
        self, rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        """Schedule throws whose final trajectory is solved at launch time.

        For an intercept at ``t_k`` we take the reference position there,
        pick an approach direction and speed, and place the spawn a lead
        distance back along that direction, launching at ``t_k - R/s``. A
        ``kinematic_line`` obstacle then advances exactly ``d * s * dt``, so
        it passes through the predicted drone position at precisely ``t_k``.
        The launch-time retargeting uses the actual drone position and
        velocity, so a pre-existing policy offset cannot make later throws
        miss merely because they were solved against the nominal path at
        episode reset.

        Everything about the throw is randomised (time, direction,
        elevation, speed, lead distance, and a lateral aim offset so throws
        are not all dead-centre). Randomising *around* a solved intercept
        keeps encounters guaranteed while preventing the policy from
        overfitting one canned attack, and reproducibility comes from the
        episode seed rather than from removing the randomness.
        """
        cfg = dict(self._intercept_config or {})
        if not cfg.get("enabled", False) or self._nominal_trajectory is None:
            return []

        first = float(cfg.get("first_intercept_s", 3.0))
        interval = float(cfg.get("interval_s", 4.0))
        interval_jitter = float(cfg.get("interval_jitter_s", 1.0))
        speed_range = cfg.get("speed_range_mps", [2.0, 4.0])
        lead_range = cfg.get("lead_distance_range_m", [3.0, 5.0])
        azimuth_range = cfg.get("azimuth_range_deg", [-60.0, 60.0])
        elevation_range = cfg.get("elevation_range_deg", [-10.0, 25.0])
        aim_offset_std = float(cfg.get("aim_offset_std_m", 0.15))
        # Coordinated groups: several throws solved to arrive at the SAME
        # t_k, one on the reference point and the rest displaced to one
        # side, so a specific gap has to be threaded.
        #
        # Every single-throw aiming scheme measured (2026-08-26) is
        # blind-solvable, because evading one ballistic projectile needs only
        # being somewhere other than predicted -- direction carries no
        # information. Nominal-aimed fell to one constant offset (best
        # constant cleared 100% of episodes); drone-aimed with leading fell
        # to a 0.5 Hz sine weave (44.0% vs a privileged oracle's 24.0%).
        # A group blocks the reference point AND one side, so the free side
        # is a fact about this encounter that has to be perceived.
        group_size = max(int(cfg.get("group_size", 1)), 1)
        group_spread = float(cfg.get("group_spread_m", 0.0))
        require_los = bool(cfg.get("require_line_of_sight", True))
        max_attempts = int(cfg.get("max_attempts", 8))

        n_templates = max(
            len(
                getattr(
                    self.sim.visual_backend._dynamic_obstacles,
                    "_resolved_templates",
                    [],
                )
            ),
            1,
        )

        # The generated trajectory can be shorter than the episode (path
        # resampling in particular can yield a shorter route), and sampling
        # MinSnap past its final keyframe indexes off the end of the segment
        # arrays -- an IndexError inside reset(), which kills the worker and
        # the whole run.
        trajectory_end = float(
            getattr(self._nominal_trajectory, "t_keyframes", [self.episode_seconds])[-1]
        )
        horizon = min(float(self.episode_seconds), trajectory_end - 1e-3)

        entries: list[dict[str, Any]] = []
        t_k = first + float(rng.uniform(-interval_jitter, interval_jitter))
        while t_k < horizon:
            flat = self._nominal_trajectory.update(float(t_k))
            target = self.sim.safety.dynamics_to_habitat(
                np.asarray(flat["x"], dtype=np.float64)
            )

            travel = self.sim.safety.dynamics_to_habitat_vel(
                np.asarray(flat["x_dot"], dtype=np.float64)
            )
            heading = float(np.arctan2(travel[2], travel[0]))

            # Resample the throw until the drone would actually be able to
            # see it when it launches. Being inside the frustum is not
            # enough -- scene geometry occludes a large fraction of
            # otherwise-valid throws, and an unseeable obstacle trains the
            # actor on something it cannot perceive.
            for _ in range(max(max_attempts, 1)):
                azimuth = heading + np.deg2rad(
                    float(rng.uniform(azimuth_range[0], azimuth_range[1]))
                )
                elevation = np.deg2rad(
                    float(rng.uniform(elevation_range[0], elevation_range[1]))
                )
                approach = np.array(
                    [
                        np.cos(elevation) * np.cos(azimuth),
                        np.sin(elevation),
                        np.cos(elevation) * np.sin(azimuth),
                    ],
                    dtype=np.float64,
                )
                approach /= np.linalg.norm(approach) + 1e-12

                speed = float(rng.uniform(speed_range[0], speed_range[1]))
                lead = float(rng.uniform(lead_range[0], lead_range[1]))
                aim_offset = rng.normal(0.0, aim_offset_std, size=3)
                spawn_position = target + approach * lead + aim_offset
                spawn_time = t_k - lead / speed
                if spawn_time < 0.0:
                    # Launch at t=0 from further out rather than dropping the
                    # slot: distance = speed * t_k still puts the obstacle at
                    # the reference point at exactly t_k, so the intercept
                    # guarantee is preserved. Skipping instead silently
                    # removed every early intercept, which is what capped
                    # episodes at a single encounter.
                    spawn_time = 0.0
                    lead = speed * t_k
                    spawn_position = target + approach * lead + aim_offset

                if not require_los:
                    break
                # Sight-line is checked from where the drone will be when the
                # throw launches, not from where it is now.
                observer = self.sim.safety.dynamics_to_habitat(
                    np.asarray(
                        self._nominal_trajectory.update(float(spawn_time))["x"],
                        dtype=np.float64,
                    )
                )
                if self._has_line_of_sight(observer, spawn_position):
                    break
            else:
                spawn_time = -1.0  # no unoccluded throw found; skip this slot

            if spawn_time >= 0.0:
                # Lateral axis of the reference path: the free side has to be
                # a direction the vehicle can actually use, so it is defined
                # against travel, not against the throw's approach.
                up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                lateral = np.cross(travel, up)
                lateral_norm = float(np.linalg.norm(lateral))
                lateral = (
                    lateral / lateral_norm
                    if lateral_norm > 1e-9
                    else np.array([1.0, 0.0, 0.0], dtype=np.float64)
                )
                # Members are laid across the WHOLE reachable width at
                # group_spread_m intervals, with one randomly chosen slot
                # left open. Stacking them to one side instead (0 and
                # +0.70 m) left the outer band free: a constant offset at
                # the 1.2 m limit cleared the centre by 1.2 m and the side
                # member by 0.5 m, both past the 0.30 m hit radius, whatever
                # side was blocked -- best constant 1.000, every episode.
                # Leaving exactly one gap makes the free slot a per-encounter
                # fact rather than a standing option.
                # group_size members plus one gap = group_size + 1 slots,
                # centred on the reference point. Spacing equal to the
                # combined hit radius doubled (0.6 m) tiles the line without
                # leaving holes between neighbours.
                slots = [
                    (float(i) - 0.5 * group_size) * group_spread
                    for i in range(group_size + 1)
                ]
                gap_index = int(rng.integers(0, len(slots)))
                for member, slot in enumerate(slots):
                    if member == gap_index:
                        continue
                    member_offset = np.asarray(aim_offset, dtype=np.float64)
                    if group_spread > 0.0:
                        member_offset = member_offset + lateral * slot
                    member_spawn = target + approach * lead + member_offset
                    entries.append(
                        {
                            "spawn_time": spawn_time,
                            "spawn_position": member_spawn,
                            "velocity": -approach * speed,
                            "retarget_at_spawn": True,
                            "approach": approach.copy(),
                            "speed_mps": speed,
                            "lead_distance_m": lead,
                            "aim_offset": member_offset,
                            "target_velocity": travel.copy(),
                            "template_index": int(rng.integers(0, n_templates)),
                            "intercept_time": float(t_k),
                        }
                    )

            t_k += interval + float(rng.uniform(-interval_jitter, interval_jitter))

        entries.sort(key=lambda e: e["spawn_time"])
        return entries

    def _has_line_of_sight(
        self, origin: np.ndarray, target: np.ndarray, slack_m: float = 0.05
    ) -> bool:
        """Whether scene geometry blocks the straight path origin -> target.

        Being inside the camera frustum does not mean being *seen*: a throw
        can be solved perfectly and still spawn behind a wall or inside
        geometry, in which case the actor is asked to dodge something it
        cannot perceive while the privileged critic sees it fine -- the
        worst case for advantage variance.
        """
        import habitat_sim

        delta = np.asarray(target, dtype=np.float64) - np.asarray(
            origin, dtype=np.float64
        )
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return True

        ray = habitat_sim.geo.Ray(
            np.asarray(origin, dtype=np.float32),
            (delta / distance).astype(np.float32),
        )
        try:
            hits = self.sim.visual_backend._sim.cast_ray(ray, max_distance=distance)
        except Exception:  # raycasting unavailable -> do not block spawning
            logger.debug("cast_ray unavailable; skipping line-of-sight check")
            return True

        if not hits.has_hits():
            return True
        # Anything struck before the target sits between the two.
        return float(hits.hits[0].ray_distance) >= distance - slack_m

    def _obstacle_relative_states(
        self, state: dict[str, np.ndarray]
    ) -> list[dict[str, Any]]:
        """Threat-first obstacle geometry, expressed in the dynamics frame.

        Habitat-frame quantities are mapped back through the coordinate
        transform so the task can rotate them into the body frame with the
        same convention it uses for the drone's own velocity.
        """
        dynamic_obstacles = getattr(self.sim.visual_backend, "_dynamic_obstacles", None)
        active = getattr(dynamic_obstacles, "_active", {}) if dynamic_obstacles else {}
        if not active or self.sim.safety is None:
            return []

        to_dynamics = self.sim.coord_trans.pos_transform_inv
        agent_pos = self.sim.safety.dynamics_to_habitat(np.asarray(state["x"]))
        agent_vel = self.sim.safety.dynamics_to_habitat_vel(
            np.asarray(state["v"], dtype=np.float64)
        )
        agent_radius = float(getattr(dynamic_obstacles, "_agent_radius", 0.0))
        horizon_s = float(getattr(self._task, "threat_time_horizon_s", 1.0))

        rows: list[dict[str, Any]] = []
        for item in active.values():
            obstacle_pos = np.asarray(item.obj.translation, dtype=np.float64)
            rel_pos = obstacle_pos - agent_pos
            rel_vel = obstacle_velocity(item, self.sim.time) - agent_vel
            combined_radius = agent_radius + float(item.collision_radius)
            predicted_center_distance, tca = constant_velocity_closest_approach(
                rel_pos, rel_vel, horizon_s
            )
            rows.append(
                {
                    "object_id": int(item.object_id),
                    "rel_pos": to_dynamics @ rel_pos,
                    "rel_vel": to_dynamics @ rel_vel,
                    # Obstacle acceleration, dynamics frame. Zero for the
                    # kinematic_line templates and -gravity for
                    # kinematic_parabola. Without it, position and velocity
                    # at a single instant cannot distinguish a straight throw
                    # from a ballistic one -- and a parabola at 3.0 m/s^2
                    # curves ~2.16 m over a 1.2 s flight, against a 0.30 m
                    # contact radius. The agent's own acceleration is not
                    # subtracted: it is a control input the policy already
                    # knows through prev_action, whereas the obstacle's is
                    # the unobservable half.
                    "rel_accel": to_dynamics @ obstacle_acceleration(item),
                    "clearance": float(np.linalg.norm(rel_pos) - combined_radius),
                    "predicted_clearance": float(
                        predicted_center_distance - combined_radius
                    ),
                    "time_to_closest_approach": tca,
                }
            )

        rows.sort(key=obstacle_threat_priority)
        return rows

    # ---- Threat metrics --------------------------------------------------

    def _nominal_counterfactual_collision(
        self,
        *,
        obstacle_pos: np.ndarray,
        obstacle_vel: np.ndarray,
        combined_radius: float,
        horizon_s: float,
        threshold_m: float,
    ) -> bool:
        if self._nominal_trajectory is None:
            return False
        samples = max(int(np.ceil(horizon_s * self.sim.config.control_rate)), 2)
        samples = min(samples, 20)
        for dt in np.linspace(0.0, horizon_s, samples):
            flat = self._nominal_trajectory.update(self.sim.time + float(dt))
            nominal_pos = self.sim.safety.dynamics_to_habitat(
                np.asarray(flat["x"], dtype=np.float64)
            )
            future_obstacle = obstacle_pos + obstacle_vel * float(dt)
            clearance = float(
                np.linalg.norm(future_obstacle - nominal_pos) - combined_radius
            )
            if clearance <= threshold_m:
                return True
        return False

    def _compute_threat_metrics(self, state: dict[str, np.ndarray]) -> dict[str, Any]:
        empty = {
            "min_obstacle_distance": np.inf,
            "predicted_closest_distance": np.inf,
            "time_to_closest_approach": np.inf,
            "obstacle_threat": False,
            "threat_ids": (),
            "near_miss_ids": (),
            "nominal_counterfactual_collision": False,
        }
        if self.sim.safety is None:
            return empty

        dynamic_obstacles = getattr(self.sim.visual_backend, "_dynamic_obstacles", None)
        active = getattr(dynamic_obstacles, "_active", {}) if dynamic_obstacles else {}
        if not active:
            return empty

        agent_pos = self.sim.safety.dynamics_to_habitat(np.asarray(state["x"]))
        agent_vel = self.sim.safety.dynamics_to_habitat_vel(
            np.asarray(state["v"], dtype=np.float64)
        )
        min_distance = np.inf
        min_predicted_distance = np.inf
        min_tca = np.inf
        threat_ids: list[int] = []
        near_miss_ids: list[int] = []
        nominal_counterfactual_collision = False
        agent_radius = float(getattr(dynamic_obstacles, "_agent_radius", 0.0))
        horizon_s = float(getattr(self._task, "threat_time_horizon_s", 1.0))
        threat_distance = float(getattr(self._task, "threat_distance_m", 1.5))
        near_miss_radius = float(getattr(self._task, "near_miss_radius_m", 1.0))

        for item in active.values():
            obstacle_pos = np.asarray(item.obj.translation, dtype=np.float64)
            obstacle_vel = obstacle_velocity(item, self.sim.time)
            rel_pos = obstacle_pos - agent_pos
            rel_vel = obstacle_vel - agent_vel
            combined_radius = agent_radius + float(item.collision_radius)

            clearance = float(np.linalg.norm(rel_pos) - combined_radius)
            predicted_center_distance, tca = constant_velocity_closest_approach(
                rel_pos,
                rel_vel,
                horizon_s,
            )
            predicted_clearance = predicted_center_distance - combined_radius

            min_distance = min(min_distance, clearance)
            if predicted_clearance < min_predicted_distance:
                min_predicted_distance = predicted_clearance
                min_tca = tca

            object_id = int(item.object_id)
            if predicted_clearance <= threat_distance and tca <= horizon_s:
                threat_ids.append(object_id)
            if clearance <= near_miss_radius:
                near_miss_ids.append(object_id)
            if self._nominal_counterfactual_collision(
                obstacle_pos=obstacle_pos,
                obstacle_vel=obstacle_vel,
                combined_radius=combined_radius,
                horizon_s=horizon_s,
                threshold_m=threat_distance,
            ):
                nominal_counterfactual_collision = True

        return {
            "min_obstacle_distance": min_distance,
            "predicted_closest_distance": min_predicted_distance,
            "time_to_closest_approach": min_tca,
            "obstacle_threat": bool(threat_ids),
            "threat_ids": tuple(threat_ids),
            "near_miss_ids": tuple(near_miss_ids),
            "nominal_counterfactual_collision": nominal_counterfactual_collision,
        }
