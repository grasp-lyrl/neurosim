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

from neurosim.rl.safety import HabitatSafetyChecker, build_safety_checker
from neurosim.rl.tasks import EventRepresentationManager, RLTask, build_task
from neurosim.rl.vehicles import build_vehicle
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
        self._safety = build_safety_checker(
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
        self.action_space = self._vehicle.action_space

        event_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2, self.event_height, self.event_width),
            dtype=np.float32,
        )
        state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
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
        state_vec = self._state_vector(state)
        if self.obs_mode == "state":
            return state_vec

        if event_rep is None:
            raise ValueError(
                f"event representation is required for obs_mode={self.obs_mode!r}"
            )
        if self.obs_mode == "events":
            return event_rep
        return {"events": event_rep, "state": state_vec}

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
        if self.enable_visualization:
            self._rr_needs_episode_stream_switch = True

        # Sample a valid navigable starting position in Habitat space,
        # then convert to dynamics frame.
        hab_start = self._safety.sample_habitat_start()
        x0 = (self.sim.coord_trans.pos_transform_inv @ hab_start).astype(np.float32)

        # Random initial velocity (random direction, speed from range).
        speed = rng.uniform(*self.init_speed_range)
        direction = rng.standard_normal(3).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-8
        v0 = (direction * speed).astype(np.float32)

        q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        w0 = np.zeros(3, dtype=np.float32)

        self.sim.time = 0.0
        self.sim.simsteps = 0
        self.sim.dynamics.state = {"x": x0, "v": v0, "q": q0, "w": w0}

        position, quaternion = self.sim.coord_trans.transform(
            self.sim.dynamics.state["x"], self.sim.dynamics.state["q"]
        )
        self.sim.visual_backend.update_agent_state(position, quaternion)

        self._event_manager.reset_episode()
        event_rep = self._event_manager.observation()

        observation = self._compose_observation(event_rep, self.sim.dynamics.state)
        info: dict[str, Any] = {
            "time": self.sim.time,
            "simsteps": self.sim.simsteps,
        }
        self._episode_count += 1
        return observation, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        control = self._vehicle.action_to_control(action)

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
        reward, reward_terms = self._compute_reward(state, action)
        self._prev_action = action.copy()

        terminated, term_reason = self._check_terminated(state)
        truncated = self.sim.time >= self.episode_seconds

        is_success = self._check_success(state) if not terminated else False

        if terminated:
            reward -= self.crash_penalty

        observation = self._compose_observation(event_rep, state)
        info: dict[str, Any] = {
            "time": self.sim.time,
            "simsteps": self.sim.simsteps,
            "reward_terms": reward_terms,
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
