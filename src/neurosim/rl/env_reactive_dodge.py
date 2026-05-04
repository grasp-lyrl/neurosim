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

from typing import Any

import numpy as np
from gymnasium import spaces

from neurosim.core.control import create_controller
from neurosim.core.trajectory import create_trajectory

from .env import BaseNeurosimRLEnv


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


def obstacle_velocity(item: Any) -> np.ndarray:
    """Habitat-frame linear velocity of a dynamic-obstacle item."""
    obj_vel = getattr(item.obj, "linear_velocity", None)
    if obj_vel is not None:
        return np.asarray(obj_vel, dtype=np.float64)
    return np.asarray(item.velocity, dtype=np.float64)


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
        self._last_threats = {}
        self._last_tracking = {}
        self._last_gate = 1.0
        self._last_correction_energy = 0.0

    def _control_from_action(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
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

        tracking = {
            "flat": self._current_flat,
            "lookahead_errors": lookahead_errors,
            "nominal_control_normalized": nominal_control_normalized,
        }

        context: dict[str, Any] = {
            "sim_time": float(self.sim.time),
            "previous_action": self._prev_action,
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
    def obstacle_velocity(item: Any) -> np.ndarray:
        return obstacle_velocity(item)

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
        trajectory_cfg.setdefault("start", hab_start)
        trajectory_cfg["pathfinder"] = self.sim.visual_backend._sim.pathfinder
        trajectory_cfg["coord_transform"] = self.sim.coord_trans.inverse_transform_batch
        if "seed" not in trajectory_cfg:
            trajectory_cfg["seed"] = int(rng.integers(0, np.iinfo(np.int32).max))
        self._nominal_trajectory = create_trajectory(**trajectory_cfg)

    def _nominal_control(self) -> dict[str, np.ndarray | float]:
        if self._nominal_controller is None or self._nominal_trajectory is None:
            raise RuntimeError("Nominal controller/trajectory requested before reset")
        self._current_flat = self._nominal_trajectory.update(self.sim.time)
        return self._nominal_controller.update(
            self.sim.time,
            self.sim.dynamics.state,
            self._current_flat,
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
        agent_height = float(getattr(dynamic_obstacles, "_agent_height", 0.0))
        agent_radius = float(getattr(dynamic_obstacles, "_agent_radius", 0.0))
        horizon_s = float(getattr(self._task, "threat_time_horizon_s", 1.0))
        threat_distance = float(getattr(self._task, "threat_distance_m", 1.5))
        near_miss_radius = float(getattr(self._task, "near_miss_radius_m", 1.0))

        for item in active.values():
            obstacle_pos = np.asarray(item.obj.translation, dtype=np.float64)
            obstacle_pos[1] -= agent_height
            obstacle_vel = obstacle_velocity(item)
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
