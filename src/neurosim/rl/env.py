"""Gymnasium environment wrapping the synchronous neurosim simulator.

Supports a *hover-stop* task: the drone starts with a random translational
velocity and must decelerate to a stable hover inside the scene without
collisions, using event-camera observations.

Safety checks are delegated to Habitat pathfinder bounds and navigability.

Experiment configs are self-contained: scenes, sensors, a ``dynamics`` block
(model / vehicle / control_abstraction, plus RL-only fields), the full
``simulator`` block (rates, IMU, etc.), and optional
``simulator.domain_randomization`` (``enabled``, ``resample_every`` for how
often to rebuild the sim, and per-sensor specs under ``sensors``) gate
scene/sensor sampling for
:class:`~neurosim.sims.synchronous_simulator.randomized_simulator.RandomizedSimulator`.

Dynamics randomization is optional under ``dynamics.domain_randomization``
(``enabled``, ``resample_every``, ``scales``); omit the block to disable.

Habitat-oriented defaults live in ``_VISUAL_BACKEND_DEFAULTS`` only;
everything tunable for a run is in YAML. The RL policy supplies control via
``sim.step()`` — no ``controller`` or ``trajectory`` entries in settings
(same idea as omitting trajectory for ``sim.run()``-style workflows).
"""

import copy
import time
import logging
import numpy as np
from typing import Any
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

from neurosim.rl.safety import HabitatSafetyChecker
from neurosim.rl.tasks import EventRepresentationManager, RLTask, build_task
from neurosim.rl.vehicles import build_vehicle
from neurosim.core.control import create_controller
from neurosim.core.trajectory import create_trajectory
from neurosim.sims.synchronous_simulator import RandomizedSimulator

# ---------------------------------------------------------------------------
# Defaults for building SynchronousSimulator settings from env_config
# ---------------------------------------------------------------------------

_VISUAL_BACKEND_DEFAULTS: dict[str, Any] = {
    "gpu_id": 0,
    "scene_dataset_config_file": "default",
    "clear_color": [0.0, 0.0, 0.0, 1.0],
    "default_agent": 0,
    "agent_height": 1.0,
    "agent_radius": 0.3,
    "agent_max_climb": 1.0,
    "agent_max_slope": 90.0,
    "enable_hbao": False,
    "frustum_culling": True,
    "seed": 324,
    "physics_config_file": "data/default.physics_config.json",
    "enable_physics": True,
}


class NeurosimRLEnv(gym.Env):
    """Single-drone RL environment with CTBR actions and pluggable tasks.

    Action space - 4-D continuous:
        Normalized policy action in [-1, 1]^4.
        Internally rescaled to [collective_thrust, roll_rate, pitch_rate, yaw_rate].

    Observation modes:
        "events"   - event histogram only  (2, H, W)
        "state"    - privileged state only (13,)
        "combined" - Dict {"events", "state"} (default, for SB3 MultiInputPolicy)

    Safety checks are performed via Habitat pathfinder bounds and
    navigability queries.

    ``train=True`` forces Rerun visualization off regardless of
    ``env_config["enable_visualization"]`` (for SB3 training; rollouts use
    ``train=False``).
    """

    def __init__(self, env_config: dict[str, Any], *, train: bool = False):
        super().__init__()

        # Worker setup --------------------------------------------------------------------
        self._worker_log_setup(env_config)

        # Observation mode -----------------------------------------------------------------
        self.obs_mode = env_config["obs_mode"]
        if self.obs_mode not in {"events", "state", "combined"}:
            raise ValueError("obs_mode must be one of: events, state, combined")

        # Episode specification ------------------------------------------------------------
        self.episode_seconds = env_config["episode_seconds"]
        self.init_speed_range = env_config["init_speed_range"]

        # Event representation -------------------------------------------------------------
        self.event_downsample_factor = env_config.get("event_downsample_factor", 1)
        self.event_representation = env_config.get(
            "event_representation", "time_surface"
        )
        if self.event_representation not in {
            "histogram",
            "event_frame",
            "time_surface",
        }:
            raise ValueError(
                "event_representation must be 'histogram', 'event_frame', or 'time_surface'"
            )
        self.event_log_compression = env_config.get("event_log_compression", 1.0)
        self._event_ts_tau_seconds = env_config.get("event_ts_decay_ms", 10.0) * 1e-3

        # Visualization --------------------------------------------------------------------
        self.enable_visualization = (
            False if train else bool(env_config.get("enable_visualization", False))
        )
        self.visualization_log_every_n_steps = env_config.get(
            "visualization_log_every_n_steps", 1
        )
        self._visualizer_initialized = False
        self._rr_rollout_episode_idx = 0
        self._rr_needs_episode_stream_switch = False

        # Task ----------------------------------------------------------------------------
        self._task: RLTask = build_task(
            task_name=env_config["task"]["name"], **env_config["task"]["config"]
        )

        self.crash_penalty = float(self._task.crash_penalty)
        self._prev_action: np.ndarray | None = None
        self._enable_navigable_check = env_config.get("enable_navigable_check", True)
        self._event_sensor_uuid_cfg = env_config.get("event_sensor_uuid")
        self._dynamics_config = env_config["dynamics"]
        self._nominal_controller = None
        self._nominal_trajectory = None
        self._current_flat: dict[str, Any] | None = None
        self._delta_thrust_fraction = 0.2
        self._delta_rate_limits = np.asarray([2.0, 2.0, 1.0], dtype=np.float64)
        self._residual_control_mode = "ctbr_delta"
        self._last_task_context: dict[str, Any] = {}

        self._safety: HabitatSafetyChecker | None = None
        self.event_sensor_uuid = ""

        # Domain randomization -------------------------------------------------------------
        dr_cfg = env_config.get("simulator", {}).get("domain_randomization", {})
        self._dr_enabled = bool(dr_cfg.get("enabled", False))
        if self._dr_enabled:
            self._resample_every = int(dr_cfg["resample_every"])
            assert self._resample_every > 0, "resample_every must be positive"
        self._episode_count = 0

        base_settings = self._build_simulator_settings(env_config)

        randomization: dict[str, Any] | None = None
        if self._dr_enabled:
            randomization = {
                "scenes": list(env_config.get("scenes", [])),
                "sensors": dict(dr_cfg.get("sensors", {})),
            }

        self._rsim = RandomizedSimulator(
            base_settings=base_settings,
            randomization=randomization,
            visualizer_disabled=not self.enable_visualization,
        )
        self._sync_from_simulator()

        self.steps_per_action = max(
            int(self.sim.config.world_rate / self.sim.config.control_rate), 1
        )

    # ------------------------------------------------------------------
    # Settings builder
    # ------------------------------------------------------------------

    def _worker_log_setup(self, env_config: dict[str, Any]) -> None:
        # Optional per-worker disk log (injected by applications/rl/train_sb3.make_env)
        self._worker_log: logging.Logger | None = None

        wdir = env_config.pop("_neurosim_rl_worker_log_dir", None)
        wrole = str(env_config.pop("_neurosim_rl_worker_log_role", "train"))
        widx = int(env_config.pop("_neurosim_rl_env_idx", 0))

        if wdir:
            from neurosim.rl.disk_logging import attach_worker_env_logger

            vb = env_config.get("visual_backend") or {}
            gid = int(vb.get("gpu_id", _VISUAL_BACKEND_DEFAULTS["gpu_id"]))
            self._worker_log = attach_worker_env_logger(
                Path(wdir), role=wrole, env_idx=widx, gpu_id=gid
            )

    @staticmethod
    def _build_simulator_settings(env_config: dict[str, Any]) -> dict[str, Any]:
        """Construct a SynchronousSimulator-compatible settings dict.

        Expects ``env_config["simulator"]`` to be the full block passed to
        ``SimulationConfig`` (world/control rates, ``additional_sensors``,
        ``sensor_rates``, ``viz_rates``, etc.).  Expects ``env_config["dynamics"]``
        with ``model``, ``vehicle``, and ``control_abstraction`` (same shape as
        top-level ``dynamics`` in full simulator YAMLs).  RL-only keys such as
        ``ctbr_rate_limits`` and ``domain_randomization`` (including
        ``resample_every``) are not forwarded to ``create_dynamics``.  Merges
        optional ``visual_backend`` overrides with
        ``_VISUAL_BACKEND_DEFAULTS``, then sets ``scene`` from the first entry in
        ``scenes`` and ``sensors`` from ``env_config["sensors"]``.  Strips
        ``domain_randomization`` from the ``simulator`` sub-dict before building
        :class:`~neurosim.core.utils.utils_simcfg.SimulationConfig` (unknown keys
        would otherwise be rejected).

        Omits ``controller`` and ``trajectory`` so the simulator runs under
        external RL control via :meth:`~neurosim.sims.synchronous_simulator.simulator.SynchronousSimulator.step`.
        """
        if "simulator" not in env_config:
            raise KeyError(
                'env_config must include a "simulator" block '
                "(world_rate, control_rate, additional_sensors, sensor_rates, viz_rates, …)"
            )

        scenes = env_config["scenes"]
        scene_path = scenes[0]["path"]
        sensors = dict(env_config["sensors"])
        dyn_cfg = env_config["dynamics"]

        sim_settings = copy.deepcopy(env_config["simulator"])
        sim_settings.pop("domain_randomization", None)
        vb_overrides = dict(env_config.get("visual_backend", {}))
        vb_settings = {**_VISUAL_BACKEND_DEFAULTS, **vb_overrides}
        vb_settings["scene"] = scene_path
        vb_settings["sensors"] = sensors

        dynamics_settings: dict[str, Any] = {
            "model": dyn_cfg["model"],
            "vehicle": dyn_cfg["vehicle"],
            "control_abstraction": dyn_cfg["control_abstraction"],
        }

        return {
            "simulator": sim_settings,
            "visual_backend": vb_settings,
            "dynamics": dynamics_settings,
        }

    def _get_default_event_sensor_uuid(self) -> str:
        event_sensors = self.sim.config.sensor_manager.get_sensors_by_type("event")
        if not event_sensors:
            raise RuntimeError(
                "No event sensor configured; RL env expects at least one event sensor"
            )
        return event_sensors[0].uuid

    def _sync_from_simulator(self) -> None:
        """Re-derive spaces, vehicle, safety checker, etc. from the current simulator."""
        self._safety = HabitatSafetyChecker(
            self.sim,
            enable_navigable_check=self._enable_navigable_check,
        )

        self.event_sensor_uuid = (
            self._event_sensor_uuid_cfg or self._get_default_event_sensor_uuid()
        )
        event_cfg = self.sim.config.visual_sensors[self.event_sensor_uuid]
        self._raw_event_height = int(event_cfg["height"])
        self._raw_event_width = int(event_cfg["width"])
        self.event_height = self._raw_event_height // self.event_downsample_factor
        self.event_width = self._raw_event_width // self.event_downsample_factor
        if self.event_height <= 0 or self.event_width <= 0:
            raise ValueError(
                "event_downsample_factor is too large for event sensor resolution"
            )

        self._event_manager = EventRepresentationManager(
            representation=self.event_representation,
            raw_height=self._raw_event_height,
            raw_width=self._raw_event_width,
            downsample_factor=self.event_downsample_factor,
            event_log_compression=self.event_log_compression,
            ts_tau_seconds=self._event_ts_tau_seconds,
            event_device=f"cuda:{int(self.sim.settings['visual_backend']['gpu_id'])}",
        )
        if self._worker_log is not None:
            self._worker_log.info(
                f"event_manager initialized on device={self._event_manager._device}",
            )

        self._vehicle = build_vehicle(
            sim=self.sim,
            dynamics_config=self._dynamics_config,
        )
        if self._task.uses_nominal_controller:
            residual_cfg = getattr(self._task, "residual_control_config", {}) or {}
            self._residual_control_mode = str(residual_cfg.get("mode", "ctbr_delta"))
            self._delta_thrust_fraction = float(
                residual_cfg.get("delta_thrust_fraction", 0.2)
            )
            self._delta_rate_limits = np.asarray(
                residual_cfg.get("delta_rate_limits", [2.0, 2.0, 1.0]),
                dtype=np.float64,
            )
            action_dim = int(self._task.action_dim or 4)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_dim,),
                dtype=np.float32,
            )
        else:
            self.action_space = self._vehicle.action_space

        event_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2, self.event_height, self.event_width),
            dtype=np.float32,
        )
        state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._task.state_observation_dim,),
            dtype=np.float32,
        )

        if self.obs_mode == "events":
            self.observation_space = event_space
        elif self.obs_mode == "state":
            self.observation_space = state_space
        else:
            self.observation_space = spaces.Dict(
                {"events": event_space, "state": state_space}
            )

    @property
    def sim(self):
        """Access the inner SynchronousSimulator."""
        return self._rsim.sim

    @staticmethod
    def _state_vector(state: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(state["x"], dtype=np.float32),
                np.asarray(state["v"], dtype=np.float32),
                np.asarray(state["q"], dtype=np.float32),
                np.asarray(state["w"], dtype=np.float32),
            ],
            axis=0,
        )

    def _compose_observation(
        self,
        event_rep: np.ndarray | None,
        state: dict[str, np.ndarray],
    ) -> np.ndarray | dict[str, np.ndarray]:
        base_state = self._state_vector(state)
        state_vec = self._task.make_state_observation(
            state=state,
            base_state=base_state,
        )
        if self.obs_mode == "state":
            return state_vec

        if event_rep is None:
            raise ValueError(
                f"event representation is required for obs_mode={self.obs_mode!r}"
            )
        if self.obs_mode == "events":
            return event_rep
        return {"events": event_rep, "state": state_vec}

    @staticmethod
    def _quat_from_yaw(yaw: float) -> np.ndarray:
        half_yaw = 0.5 * float(yaw)
        return np.asarray(
            [0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)],
            dtype=np.float32,
        )

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

    def _reset_nominal_tracking(
        self,
        *,
        hab_start: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
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
        if not self._task.uses_nominal_controller or self._nominal_trajectory is None:
            return []
        errors = []
        for dt in getattr(self._task, "lookahead_seconds", ()):
            flat = self._nominal_trajectory.update(self.sim.time + float(dt))
            errors.append(
                np.asarray(state["x"], dtype=np.float32)
                - np.asarray(flat["x"], dtype=np.float32)
            )
        return errors

    @staticmethod
    def _constant_velocity_closest_approach(
        rel_pos: np.ndarray,
        rel_vel: np.ndarray,
        horizon_s: float,
    ) -> tuple[float, float]:
        rel_pos = np.asarray(rel_pos, dtype=np.float64)
        rel_vel = np.asarray(rel_vel, dtype=np.float64)
        speed_sq = float(np.dot(rel_vel, rel_vel))
        if speed_sq < 1e-9:
            return float(np.linalg.norm(rel_pos)), np.inf
        tca = -float(np.dot(rel_pos, rel_vel)) / speed_sq
        tca = float(np.clip(tca, 0.0, horizon_s))
        closest = rel_pos + rel_vel * tca
        return float(np.linalg.norm(closest)), tca

    @staticmethod
    def _obstacle_velocity(item: Any) -> np.ndarray:
        obj_vel = getattr(item.obj, "linear_velocity", None)
        if obj_vel is not None:
            return np.asarray(obj_vel, dtype=np.float64)
        return np.asarray(item.velocity, dtype=np.float64)

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
            nominal_pos = self._safety.dynamics_to_habitat(
                np.asarray(flat["x"], dtype=np.float64)
            )
            future_obstacle = obstacle_pos + obstacle_vel * float(dt)
            clearance = float(
                np.linalg.norm(future_obstacle - nominal_pos) - combined_radius
            )
            if clearance <= threshold_m:
                return True
        return False

    def _obstacle_metrics(self, state: dict[str, np.ndarray]) -> dict[str, Any]:
        empty_metrics = {
            "min_obstacle_distance": np.inf,
            "predicted_closest_distance": np.inf,
            "time_to_closest_approach": np.inf,
            "obstacle_threat": False,
            "threat_ids": (),
            "near_miss_ids": (),
            "nominal_counterfactual_collision": False,
        }
        if self._safety is None:
            return empty_metrics

        dynamic_obstacles = getattr(self.sim.visual_backend, "_dynamic_obstacles", None)
        active = getattr(dynamic_obstacles, "_active", {}) if dynamic_obstacles else {}
        if not active:
            return empty_metrics

        agent_pos = self._safety.dynamics_to_habitat(np.asarray(state["x"]))
        agent_vel = np.asarray(
            self._safety._pos_transform, dtype=np.float64
        ) @ np.asarray(state["v"], dtype=np.float64)
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
            obstacle_vel = self._obstacle_velocity(item)
            rel_pos = obstacle_pos - agent_pos
            rel_vel = obstacle_vel - agent_vel
            combined_radius = agent_radius + float(item.collision_radius)

            clearance = float(np.linalg.norm(rel_pos) - combined_radius)
            predicted_center_distance, tca = self._constant_velocity_closest_approach(
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

    def _action_gate_and_delta(self, action: np.ndarray) -> tuple[float, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._residual_control_mode == "gated_ctbr_delta":
            gate = float(np.clip(action[0], 0.0, 1.0))
            delta = gate * action[1:5]
            return gate, delta.astype(np.float32, copy=False)
        return 1.0, action[:4].astype(np.float32, copy=False)

    def _update_task_context(
        self,
        *,
        state: dict[str, np.ndarray],
        nominal_control: dict[str, np.ndarray | float] | None = None,
        action: np.ndarray | None = None,
    ) -> None:
        context: dict[str, Any] = {
            "flat": self._current_flat,
            "sim_time": self.sim.time,
            "lookahead_errors": self._lookahead_errors(state),
            "previous_action": self._prev_action,
        }
        context.update(self._obstacle_metrics(state))
        if nominal_control is not None:
            context["nominal_control_normalized"] = self._vehicle.control_to_normalized(
                nominal_control
            )
        if action is not None:
            gate, delta = self._action_gate_and_delta(action)
            context["correction_energy"] = float(np.mean(np.square(delta)))
            context["gate_value"] = gate
        self._last_task_context = context
        self._task.set_context(context)

    def _task_context_info(self) -> dict[str, Any]:
        keys = (
            "sim_time",
            "min_obstacle_distance",
            "predicted_closest_distance",
            "time_to_closest_approach",
            "obstacle_threat",
            "threat_ids",
            "near_miss_ids",
            "nominal_counterfactual_collision",
            "correction_energy",
            "gate_value",
        )
        return {
            key: self._last_task_context[key]
            for key in keys
            if key in self._last_task_context
        }

    def _control_from_action(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray | float] | None]:
        if not self._task.uses_nominal_controller:
            return self._vehicle.action_to_control(action), None

        nominal_control = self._nominal_control()
        gate, delta_action = self._action_gate_and_delta(action)
        if self._residual_control_mode == "gated_ctbr_delta":
            control = self._vehicle.apply_gated_ctbr_delta(
                nominal_control,
                gate,
                action[1:5],
                delta_thrust_fraction=self._delta_thrust_fraction,
                delta_rate_limits=self._delta_rate_limits,
            )
        else:
            control = self._vehicle.apply_ctbr_delta(
                nominal_control,
                delta_action,
                delta_thrust_fraction=self._delta_thrust_fraction,
                delta_rate_limits=self._delta_rate_limits,
            )
        return control, nominal_control

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        state: dict[str, np.ndarray],
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        return self._task.compute_reward(
            state=state,
            action=action,
            prev_action=self._prev_action,
            event_manager=self._event_manager,
            obs_mode=self.obs_mode,
        )

    def _check_terminated(self, state: dict[str, np.ndarray]) -> tuple[bool, str]:
        safe, reason = self._safety.check(np.asarray(state["x"]))
        if not safe:
            return True, reason
        task_terminated, task_reason = self._task.check_terminated(state=state)
        if task_terminated:
            return True, task_reason
        return False, ""

    def _check_success(self, state: dict[str, np.ndarray]) -> bool:
        return self._task.check_success(state=state)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _ensure_visualizer(self) -> None:
        if not self.enable_visualization:
            return
        if self._visualizer_initialized:
            return
        if self.sim.visualizer is None:
            raise RuntimeError(
                "Visualization requested but simulator visualizer is unavailable"
            )
        self.sim.visualizer.initialize(display=True)
        self._visualizer_initialized = True

    def _maybe_visualize(
        self,
        measurements: dict[str, Any],
        event_rep: np.ndarray | None = None,
    ) -> None:
        if (
            not self.enable_visualization
            or not self.sim.visualizer
            or self.sim.simsteps % self.visualization_log_every_n_steps != 0
        ):
            return

        if self._rr_needs_episode_stream_switch:
            self._rr_rollout_episode_idx += 1
            self.sim.visualizer.set_episode_index(self._rr_rollout_episode_idx)
            self._rr_needs_episode_stream_switch = False

        if measurements:
            self.sim.visualizer.log_measurements(
                measurements, self.sim.time, self.sim.simsteps
            )
            self.sim.visualizer.log_state(self.sim.dynamics.state)

        if event_rep is not None:
            self.sim.visualizer.log_image(
                path=f"sensors/{self.event_sensor_uuid}/event_representation",
                image=self._event_manager.to_rgb(event_rep),
            )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def _should_domain_randomize_now(self) -> bool:
        """True when ``RandomizedSimulator.randomize`` should run on this reset."""
        if not self._dr_enabled:
            return False
        return self._episode_count % self._resample_every == 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        if self._should_domain_randomize_now():
            if self._worker_log is not None:
                self._worker_log.info(
                    f"domain_randomization_start episode_count={self._episode_count}",
                )
            t0 = time.perf_counter()

            self._rsim.randomize(rng)
            self._sync_from_simulator()

            if self._worker_log is not None:
                self._worker_log.info(
                    f"domain_randomization_done episode_count={self._episode_count} "
                    f"elapsed_s={time.perf_counter() - t0:.3f}",
                )
            self._visualizer_initialized = False

        self._vehicle.randomize(self._episode_count, rng)

        self._ensure_visualizer()
        self._task.on_reset()
        self._prev_action = None
        self._current_flat = None
        if self.enable_visualization:
            self._rr_needs_episode_stream_switch = True

        # Sample a valid navigable starting position in Habitat space,
        # then convert to dynamics frame.
        hab_start = self._safety.sample_habitat_start()
        x0 = (self.sim.coord_trans.pos_transform_inv @ hab_start).astype(np.float32)

        self.sim.time = 0.0
        self.sim.simsteps = 0

        if self._task.uses_nominal_controller:
            self._reset_nominal_tracking(hab_start=hab_start, rng=rng)
        else:
            # Random initial velocity (random direction, speed from range).
            speed = rng.uniform(*self.init_speed_range)
            direction = rng.standard_normal(3).astype(np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            v0 = (direction * speed).astype(np.float32)

            q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            w0 = np.zeros(3, dtype=np.float32)
            self.sim.dynamics.state = {"x": x0, "v": v0, "q": q0, "w": w0}

        position, quaternion = self.sim.coord_trans.transform(
            self.sim.dynamics.state["x"], self.sim.dynamics.state["q"]
        )
        self.sim.visual_backend.update_agent_state(position, quaternion)

        self._event_manager.reset_episode()
        event_rep = self._event_manager.observation()
        reset_nominal_control = (
            self._nominal_control() if self._task.uses_nominal_controller else None
        )
        self._update_task_context(
            state=self.sim.dynamics.state,
            nominal_control=reset_nominal_control,
        )

        observation = self._compose_observation(event_rep, self.sim.dynamics.state)
        info: dict[str, Any] = {
            "time": self.sim.time,
            "simsteps": self.sim.simsteps,
            "task_context": self._task_context_info(),
        }
        self._episode_count += 1
        return observation, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        control, nominal_control = self._control_from_action(action)

        self._event_manager.begin_step()
        events_accum = self.obs_mode != "state"

        for _ in range(self.steps_per_action):
            self.sim.step(control)
            measurements = self.sim._render_sensors()
            if events_accum:
                self._event_manager.accumulate(measurements.get(self.event_sensor_uuid))
            self._maybe_visualize(measurements)

        event_rep = (
            None if self.obs_mode == "state" else self._event_manager.observation()
        )
        self._maybe_visualize(None, event_rep)

        state = self.sim.dynamics.state
        self._update_task_context(
            state=state,
            nominal_control=nominal_control,
            action=action,
        )
        reward, reward_terms = self._compute_reward(state, action)
        self._prev_action = action.copy()

        terminated, term_reason = self._check_terminated(state)
        truncated = self.sim.time >= self.episode_seconds

        is_success = self._check_success(state) if not terminated else False

        if terminated:
            if hasattr(self._task, "set_termination_reason"):
                self._task.set_termination_reason(term_reason)
            reward -= float(self._task.crash_penalty)

        observation = self._compose_observation(event_rep, state)
        info: dict[str, Any] = {
            "time": self.sim.time,
            "simsteps": self.sim.simsteps,
            "reward_terms": reward_terms,
            "task_context": self._task_context_info(),
            "task_metrics": {
                key: reward_terms[key]
                for key in (
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
                if key in reward_terms
            },
            "is_success": is_success,
        }
        if term_reason:
            info["termination_reason"] = term_reason

        return observation, float(reward), terminated, truncated, info

    def close(self):
        if self._worker_log is not None:
            self._worker_log.info("worker_close")
            self._worker_log = None

        if hasattr(self, "_rsim"):
            self._rsim.close()
