"""Gymnasium environments for Neurosim RL tasks.

This module owns ``BaseNeurosimRLEnv`` — the boring shared plumbing
(simulator construction, domain randomization, safety checks, event
representation, reward/info plumbing) — plus a thin ``HoverStopEnv`` that
plugs the hover-stop task in with no extras.

Tasks that need extra machinery (e.g. residual control on top of a
nominal trajectory tracker for ``reactive_dodge``) live in their own
module and subclass the base; see ``env_reactive_dodge.py``.

Experiment configs are self-contained: scenes, sensors, ``dynamics``
(model / vehicle / control_abstraction), ``simulator`` (rates), and
``simulator.domain_randomization`` (``enabled`` / ``resample_every`` /
per-sensor specs) gate scene/sensor sampling for
:class:`~neurosim.sims.synchronous_simulator.randomized_simulator.RandomizedSimulator`.
Dynamics randomization is optional under
``dynamics.domain_randomization`` (``enabled`` / ``resample_every`` /
``scales``).

Habitat-oriented defaults live in ``_VISUAL_BACKEND_DEFAULTS`` only;
everything tunable for a run is in YAML. The RL policy supplies control
via ``sim.step()`` — no ``controller`` or ``trajectory`` entries in
settings (same idea as omitting trajectory for ``sim.run()``-style
workflows).
"""

import copy
import time
import logging
import numpy as np
from typing import Any
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

from neurosim.rl.representations import EventRepresentationManager
from neurosim.rl.tasks import RLTask, TaskStep, build_task
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


class BaseNeurosimRLEnv(gym.Env):
    """Shared Gymnasium env skeleton for Neurosim tasks.

    Subclass for a specific task and override the small set of hooks
    documented under "Subclass hooks". The base class handles all the
    Habitat / RotorPy / DR / safety plumbing, plus the per-step substep
    loop and reward bookkeeping.

    Action space (default): the underlying vehicle's action_space.
    Subclasses can override ``_build_action_space`` if the task's policy
    operates over a different shape (e.g. a residual on top of a nominal
    command).

    Observation space (default): obs_mode dispatch — ``state`` (Box of
    ``task.state_observation_dim``), ``events`` (event tensor Box), or
    ``combined`` (Dict of both). The state vector itself is produced by
    the task's :meth:`RLTask.make_state_observation`, so tasks that need
    extra features (tracking errors, lookaheads, etc.) extend it there.

    ``train=True`` forces Rerun visualization off regardless of
    ``env_config["enable_visualization"]`` (for SB3 training; rollouts
    use ``train=False``).
    """

    def __init__(self, env_config: dict[str, Any], *, train: bool = False):
        super().__init__()

        # Worker-side disk log (optional; injected by train_sb3.make_env).
        self._worker_log_setup(env_config)

        # Observation mode -----------------------------------------------------------------
        self.obs_mode = env_config["obs_mode"]
        if self.obs_mode not in {"events", "state", "combined"}:
            raise ValueError("obs_mode must be one of: events, state, combined")

        # Episode specification ------------------------------------------------------------
        self.episode_seconds = env_config["episode_seconds"]
        self.init_speed_range = env_config["init_speed_range"]

        # Event representation block (kwargs match EventRepresentationManager
        # constructor; sensor_uuid is consumed here, not by the manager).
        event_rep_cfg = dict(env_config["event_representation"])
        self._event_sensor_uuid_cfg = event_rep_cfg.pop("sensor_uuid", None)
        self._event_representation_cfg = event_rep_cfg

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
        self._dynamics_config = env_config["dynamics"]

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
    # Construction helpers
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
        ``sensor_rates``, ``viz_rates``, etc.). Expects ``env_config["dynamics"]``
        with ``model``, ``vehicle``, and ``control_abstraction``. RL-only
        keys such as ``ctbr_rate_limits`` and ``domain_randomization`` are
        not forwarded to ``create_dynamics``. Strips ``domain_randomization``
        from the ``simulator`` sub-dict before building
        :class:`~neurosim.core.utils.utils_simcfg.SimulationConfig`.

        Omits ``controller`` and ``trajectory`` so the simulator runs under
        external RL control via :meth:`SynchronousSimulator.step`.
        """
        if "simulator" not in env_config:
            raise KeyError(
                'env_config must include a "simulator" block '
                "(world_rate, control_rate, additional_sensors, sensor_rates, viz_rates, ...)"
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

        settings: dict[str, Any] = {
            "simulator": sim_settings,
            "visual_backend": vb_settings,
            "dynamics": dynamics_settings,
        }
        if "safety" in env_config:
            settings["safety"] = copy.deepcopy(env_config["safety"])
        return settings

    def _get_default_event_sensor_uuid(self) -> str:
        event_sensors = self.sim.config.sensor_manager.get_sensors_by_type("event")
        if not event_sensors:
            raise RuntimeError(
                "No event sensor configured; RL env expects at least one event sensor"
            )
        return event_sensors[0].uuid

    def _sync_from_simulator(self) -> None:
        """Re-derive spaces, vehicle, etc. from the current simulator."""
        self.event_sensor_uuid = (
            self._event_sensor_uuid_cfg or self._get_default_event_sensor_uuid()
        )
        event_cfg = self.sim.config.visual_sensors[self.event_sensor_uuid]
        self._raw_event_height = int(event_cfg["height"])
        self._raw_event_width = int(event_cfg["width"])

        self._event_manager = EventRepresentationManager(
            raw_height=self._raw_event_height,
            raw_width=self._raw_event_width,
            event_device=f"cuda:{int(self.sim.settings['visual_backend']['gpu_id'])}",
            **self._event_representation_cfg,
        )
        self.event_height = self._event_manager.downsampled_height
        self.event_width = self._event_manager.downsampled_width

        if self._worker_log is not None:
            self._worker_log.info(
                f"event_manager initialized on device={self._event_manager._device}",
            )

        self._vehicle = build_vehicle(
            sim=self.sim,
            dynamics_config=self._dynamics_config,
        )
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    def _build_observation_space(self) -> spaces.Space:
        """Build observation_space from obs_mode and the task's state dim."""
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
            return event_space
        if self.obs_mode == "state":
            return state_space
        return spaces.Dict({"events": event_space, "state": state_space})

    # ------------------------------------------------------------------
    # Subclass hooks: THESE ARE OVERRIDDEN BY TASK-SPECIFIC ENVS
    # ------------------------------------------------------------------

    def _build_action_space(self) -> spaces.Space:
        """Action space for the policy. Default: the vehicle's action space.

        Override when the policy operates over a different shape (e.g. a
        residual on top of a nominal command).
        """
        return self._vehicle.action_space

    def _control_from_action(self, action: np.ndarray) -> dict[str, np.ndarray | float]:
        """Convert a raw policy action into the control dict sent to ``sim.step``.

        Default: pass the action straight through the vehicle. Override
        when the action needs to be combined with anything else (e.g.
        added as a residual to a nominal command).
        """
        return self._vehicle.action_to_control(action)

    def _on_episode_reset(
        self,
        *,
        rng: np.random.Generator,
        hab_start: np.ndarray,
    ) -> None:
        """Set the per-episode initial state. Called once per ``reset()``.

        Called after ``self.sim.time`` has been zeroed and a navigable
        ``hab_start`` (Habitat-frame position) has been sampled, but
        before the agent state is pushed to the visual backend, the
        event manager is reset, or the first observation is composed.
        The override is responsible for writing
        ``self.sim.dynamics.state`` — and may also build any per-episode
        components it needs (e.g. a random trajectory or a fresh
        tracking controller).

        Default behaviour: place the drone at the dynamics-frame
        equivalent of ``hab_start`` with a random initial velocity drawn
        from ``self.init_speed_range`` and a level orientation.
        """
        x0 = (self.sim.coord_trans.pos_transform_inv @ hab_start).astype(np.float32)
        speed = rng.uniform(*self.init_speed_range)
        direction = rng.standard_normal(3).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-8
        v0 = (direction * speed).astype(np.float32)
        q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        w0 = np.zeros(3, dtype=np.float32)
        self.sim.dynamics.state = {"x": x0, "v": v0, "q": q0, "w": w0}

    def _populate_task_context(
        self,
        *,
        state: dict[str, np.ndarray],
        action: np.ndarray | None,
    ) -> None:
        """Feed extra environment-side context into ``self._task.set_context``.

        Called once before each call into ``self._task`` (in ``reset()``
        before the first observation with ``action=None``, and in
        ``step()`` before ``compute_reward`` with the current action).
        Default: no-op. Override to push tracking errors, threat metrics,
        or other per-step quantities the task needs but that don't fit
        cleanly inside ``TaskStep`` (e.g. because they are accumulated
        across substeps or read from another component).
        """

    def _step_info_extras(self, reward_terms: dict[str, float]) -> dict[str, Any]:
        """Extra fields to add to the step / reset info dict. Default: none.

        Override to surface task-specific metrics or context (e.g. threat
        info, tracking errors) to callers.
        """
        return {}

    # ------------------------------------------------------------------
    # Public helpers used by tools / scripts
    # ------------------------------------------------------------------

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

    @staticmethod
    def _quat_from_yaw(yaw: float) -> np.ndarray:
        half_yaw = 0.5 * float(yaw)
        return np.asarray(
            [0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Observation assembly
    # ------------------------------------------------------------------

    def _compose_observation(
        self,
        event_rep: np.ndarray | None,
        state: dict[str, np.ndarray],
    ) -> np.ndarray | dict[str, np.ndarray]:
        if self.obs_mode == "events":
            if event_rep is None:
                raise ValueError("event representation required for obs_mode='events'")
            return np.asarray(event_rep, dtype=np.float32)

        state_vec = np.asarray(
            self._task.make_state_observation(
                state=state, base_state=self._state_vector(state)
            ),
            dtype=np.float32,
        )
        if self.obs_mode == "state":
            return state_vec

        if event_rep is None:
            raise ValueError("event representation required for obs_mode='combined'")
        return {
            "events": np.asarray(event_rep, dtype=np.float32),
            "state": state_vec,
        }

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def _build_task_step(
        self,
        *,
        state: dict[str, np.ndarray],
        action: np.ndarray,
    ) -> TaskStep:
        return TaskStep(
            state=state,
            base_state=self._state_vector(state),
            action=np.asarray(action, dtype=np.float32),
            prev_action=self._prev_action,
            sim_time=float(self.sim.time),
            dt=float(self.steps_per_action / self.sim.config.world_rate),
            event_manager=self._event_manager,
            obs_mode=self.obs_mode,
        )

    def _check_terminated(self, state: dict[str, np.ndarray]) -> tuple[bool, str]:
        safe, reason = self.sim.safety.check(np.asarray(state["x"]))
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

        # Sample a valid navigable starting position in Habitat space.
        hab_start = self.sim.safety.sample_habitat_start()

        self.sim.time = 0.0
        self.sim.simsteps = 0

        # Subclass hook fills in self.sim.dynamics.state and any per-episode
        # components (e.g. nominal trajectory).
        self._on_episode_reset(rng=rng, hab_start=hab_start)

        position, quaternion = self.sim.coord_trans.transform(
            self.sim.dynamics.state["x"], self.sim.dynamics.state["q"]
        )
        self.sim.visual_backend.update_agent_state(position, quaternion)

        self._event_manager.reset_episode()
        event_rep = self._event_manager.observation()

        self._populate_task_context(state=self.sim.dynamics.state, action=None)

        observation = self._compose_observation(event_rep, self.sim.dynamics.state)
        info: dict[str, Any] = {
            "time": self.sim.time,
            "simsteps": self.sim.simsteps,
        }
        info.update(self._step_info_extras({}))
        self._episode_count += 1
        return observation, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        control = self._control_from_action(action)

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
        self._populate_task_context(state=state, action=action)

        outcome = self._task.compute_reward(
            self._build_task_step(state=state, action=action)
        )
        reward = float(outcome.reward)
        reward_terms = outcome.terms
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
            "is_success": is_success,
        }
        info.update(self._step_info_extras(reward_terms))
        if term_reason:
            info["termination_reason"] = term_reason

        return observation, float(reward), terminated, truncated, info

    def close(self):
        if self._worker_log is not None:
            self._worker_log.info("worker_close")
            self._worker_log = None
        if hasattr(self, "_rsim"):
            self._rsim.close()


class HoverStopEnv(BaseNeurosimRLEnv):
    """Hover-stop env: drone starts with random velocity, must hover.

    All defaults of ``BaseNeurosimRLEnv`` apply: 4-D CTBR action passed
    straight to the vehicle, 13-D base state observation, random
    initial-velocity reset.
    """
