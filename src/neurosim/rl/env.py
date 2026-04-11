"""Gymnasium environment wrapping the synchronous neurosim simulator.

Supports a *hover-stop* task: the drone starts with a random translational
velocity and must decelerate to a stable hover inside the scene without
collisions, using event-camera observations.

Safety checks are delegated to Habitat pathfinder bounds and navigability.

Experiment configs are self-contained: scenes, sensors, the full
``simulator`` block (rates, IMU, etc.), and optional
``simulator.domain_randomization`` (``enabled``, ``resample_on_reset``, and
per-sensor specs under ``sensors``) gate scene/sensor sampling for
:class:`~neurosim.sims.synchronous_simulator.randomized_simulator.RandomizedSimulator`.
Vehicle dynamics DR is optional under ``vehicle.domain_randomization``; when
omitted, vehicle DR is treated as disabled.

Habitat-oriented defaults live in ``_VISUAL_BACKEND_DEFAULTS`` only;
everything tunable for a run is in YAML. The RL policy supplies control via
``sim.step()`` — no ``controller`` or ``trajectory`` entries in settings
(same idea as omitting trajectory for ``sim.run()``-style workflows).
"""

import copy
import logging
import numpy as np
from typing import Any

import gymnasium as gym
from gymnasium import spaces

from neurosim.rl.safety import HabitatSafetyChecker, build_safety_checker
from neurosim.rl.tasks import RLTask, build_task
from neurosim.rl.vehicles import build_vehicle
from neurosim.sims.synchronous_simulator import RandomizedSimulator

logger = logging.getLogger(__name__)

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

# Reference count level in log normalization: output maps ~linearly near this
# accumulation (replaces the former ``event_clip`` scale in the denominator).
_EVENT_LOG_COUNT_REFERENCE = 10.0


class EventRepresentationManager:
    """Owns event representation state and update logic for RL observations."""

    def __init__(
        self,
        representation: str,
        raw_height: int,
        raw_width: int,
        downsample_factor: int,
        event_log_compression: float,
        ts_tau_seconds: float,
    ):
        self.representation = representation
        self.raw_height = int(raw_height)
        self.raw_width = int(raw_width)
        self.downsample_factor = max(int(downsample_factor), 1)
        self.event_log_compression = float(event_log_compression)
        self.ts_tau_seconds = float(ts_tau_seconds)

        self.raw = np.zeros((2, self.raw_height, self.raw_width), dtype=np.float32)
        self.step_event_count = 0
        self._last_update_time_s: float | None = None

    def reset_episode(self) -> None:
        self.raw.fill(0.0)
        self.step_event_count = 0
        self._last_update_time_s = None

    def begin_step(self) -> None:
        self.step_event_count = 0
        if self.representation != "time_surface":
            self.raw.fill(0.0)

    def _downsample(self, event_rep: np.ndarray) -> np.ndarray:
        if self.downsample_factor == 1:
            return event_rep

        factor = self.downsample_factor
        h = (event_rep.shape[1] // factor) * factor
        w = (event_rep.shape[2] // factor) * factor
        cropped = event_rep[:, :h, :w]
        reshaped = cropped.reshape(2, h // factor, factor, w // factor, factor)
        return reshaped.mean(axis=(2, 4), dtype=np.float32)

    def _normalize(self, event_rep: np.ndarray) -> np.ndarray:
        k = self.event_log_compression
        event_rep = np.log1p(k * event_rep) / np.log1p(k * _EVENT_LOG_COUNT_REFERENCE)
        np.clip(event_rep, 0.0, 1.0, out=event_rep)
        return event_rep

    def accumulate(self, events: Any | None) -> None:
        if events is None:
            return

        x, y, t, p = events.x, events.y, events.t, events.p

        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            t = t.detach().cpu().numpy()
            p = p.detach().cpu().numpy()

        if x.size == 0:
            return

        self.step_event_count += int(x.size)

        x = x.astype(np.int64, copy=False)
        y = y.astype(np.int64, copy=False)
        t = (t * 1e-6).astype(np.float64, copy=False)
        p = p.astype(np.int64, copy=False)

        if self.representation == "histogram":
            flat_idx = p * (self.raw_height * self.raw_width) + y * self.raw_width + x
            counts = np.bincount(
                flat_idx, minlength=2 * self.raw_height * self.raw_width
            ).astype(np.float32)
            self.raw += counts.reshape(2, self.raw_height, self.raw_width)
            return

        if self.representation == "event_frame":
            self.raw[p, y, x] = 1.0
            return

        if self.representation == "time_surface":
            latest_time_s = float(t[-1])
            if self._last_update_time_s is not None:
                dt_seconds = latest_time_s - self._last_update_time_s
                if dt_seconds > 0.0:
                    decay = np.float32(np.exp(-dt_seconds / self.ts_tau_seconds))
                    self.raw *= decay
            self._last_update_time_s = latest_time_s
            self.raw[p, y, x] += 1.0
            return

        raise ValueError(f"Unsupported event_representation: {self.representation}")

    def observation(self) -> np.ndarray:
        if self.representation == "histogram":
            self.raw = self._normalize(self.raw)
        obs = self._downsample(self.raw)
        return obs.astype(np.float32, copy=True)


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

        obs_mode = env_config["obs_mode"]
        episode_seconds = env_config["episode_seconds"]

        event_sensor_uuid = env_config.get("event_sensor_uuid")
        event_downsample_factor = env_config.get("event_downsample_factor", 1)

        init_speed_range = env_config["init_speed_range"]

        event_representation = env_config.get("event_representation", "time_surface")
        event_log_compression = float(env_config.get("event_log_compression", 1.0))
        event_ts_decay_ms = env_config.get("event_ts_decay_ms", 10.0)

        enable_navigable_check = env_config.get("enable_navigable_check", True)

        enable_visualization = bool(env_config.get("enable_visualization", False))
        if train:
            enable_visualization = False
        visualization_log_every_n_steps = env_config.get(
            "visualization_log_every_n_steps", 1
        )

        task_name = env_config["task"]["name"]
        task_kwargs = dict(env_config["task"]["config"])

        dr_cfg = env_config.get("simulator", {}).get("domain_randomization", {})
        self._dr_enabled = bool(dr_cfg.get("enabled", False))
        self._resample_on_reset = bool(dr_cfg.get("resample_on_reset", False))
        self._episode_count = 0

        if obs_mode not in {"events", "state", "combined"}:
            raise ValueError("obs_mode must be one of: events, state, combined")

        self.obs_mode = obs_mode
        self.episode_seconds = float(episode_seconds)
        self.event_downsample_factor = max(int(event_downsample_factor), 1)
        self.event_representation = event_representation
        if event_representation not in {"histogram", "event_frame", "time_surface"}:
            raise ValueError(
                "event_representation must be 'histogram', 'event_frame', or 'time_surface'"
            )
        self.event_log_compression = event_log_compression
        self._event_ts_tau_seconds = float(event_ts_decay_ms) * 1e-3
        self.enable_visualization = bool(enable_visualization)
        self.visualization_log_every_n_steps = max(
            int(visualization_log_every_n_steps), 1
        )
        self._visualizer_initialized = False
        self._rr_rollout_episode_idx = 0
        self._rr_needs_episode_stream_switch = False

        self.init_speed_range = tuple(init_speed_range)

        self._task: RLTask = build_task(task_name=task_name, **task_kwargs)

        self.crash_penalty = float(self._task.crash_penalty)
        self._prev_action: np.ndarray | None = None
        self._enable_navigable_check = bool(enable_navigable_check)
        self._event_sensor_uuid_cfg = event_sensor_uuid
        self._vehicle_config = env_config["vehicle"]

        self._safety: HabitatSafetyChecker | None = None
        self.event_sensor_uuid = ""

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

    # ------------------------------------------------------------------
    # Settings builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_simulator_settings(env_config: dict[str, Any]) -> dict[str, Any]:
        """Construct a SynchronousSimulator-compatible settings dict.

        Expects ``env_config["simulator"]`` to be the full block passed to
        ``SimulationConfig`` (world/control rates, ``additional_sensors``,
        ``sensor_rates``, ``viz_rates``, etc.).  Merges optional
        ``visual_backend`` overrides with ``_VISUAL_BACKEND_DEFAULTS``,
        then sets ``scene`` from the first entry in ``scenes`` and
        ``sensors`` from ``env_config["sensors"]``.  Strips
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
        vehicle_cfg = env_config["vehicle"]

        sim_settings = copy.deepcopy(env_config["simulator"])
        sim_settings.pop("domain_randomization", None)
        vb_overrides = dict(env_config.get("visual_backend", {}))
        vb_settings = {**_VISUAL_BACKEND_DEFAULTS, **vb_overrides}
        vb_settings["scene"] = scene_path
        vb_settings["sensors"] = sensors

        return {
            "simulator": sim_settings,
            "visual_backend": vb_settings,
            "dynamics": {
                "model": vehicle_cfg["dynamics_model"],
                "vehicle": vehicle_cfg["vehicle_name"],
                "control_abstraction": "cmd_ctbr",
            },
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

        world_rate = self.sim.config.world_rate
        control_rate = self.sim.config.control_rate
        self.steps_per_action = max(int(world_rate / control_rate), 1)

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
        )

        self._vehicle = build_vehicle(
            sim=self.sim,
            vehicle_config=self._vehicle_config,
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
        return self._rsim.sim if hasattr(self, "_rsim") else None

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
        if self.obs_mode == "events":
            if event_rep is None:
                raise ValueError("event representation is required for events obs_mode")
            return event_rep
        if self.obs_mode == "state":
            return state_vec
        if event_rep is None:
            raise ValueError("event representation is required for combined obs_mode")
        return {"events": event_rep, "state": state_vec}

    def _event_rep_to_rgb(self, event_rep: np.ndarray, clip: float) -> np.ndarray:
        """Map negative/positive event channels to RGB for quick visual debugging."""
        scale = max(float(clip), 1e-6)
        neg = np.clip(event_rep[0] / scale, 0.0, 1.0)
        pos = np.clip(event_rep[1] / scale, 0.0, 1.0)

        rgb = np.zeros(
            (event_rep.shape[1], event_rep.shape[2], 3),
            dtype=np.uint8,
        )
        rgb[..., 0] = np.round(255.0 * neg).astype(np.uint8)
        rgb[..., 2] = np.round(255.0 * pos).astype(np.uint8)
        return rgb

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
            event_count=self._event_manager.step_event_count,
            event_shape=(self._raw_event_height, self._raw_event_width),
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
        if not self.enable_visualization or self.sim.visualizer is None:
            return
        if self.sim.simsteps % self.visualization_log_every_n_steps != 0:
            return
        if self._rr_needs_episode_stream_switch:
            self._rr_rollout_episode_idx += 1
            self.sim.visualizer.set_episode_index(self._rr_rollout_episode_idx)
            self._rr_needs_episode_stream_switch = False
        self.sim.visualizer.log_measurements(
            measurements, self.sim.time, self.sim.simsteps
        )
        if event_rep is not None and self.sim.config.sensor_manager.should_visualize(
            self.event_sensor_uuid, self.sim.simsteps
        ):
            self.sim.visualizer.log_image(
                f"sensors/{self.event_sensor_uuid}/event_representation",
                self._event_rep_to_rgb(event_rep, clip=1.0),
            )
        self.sim.visualizer.log_state(self.sim.dynamics.state)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        rng = np.random.default_rng(seed)

        should_randomize = self._dr_enabled and (
            self._episode_count == 0 or self._resample_on_reset
        )
        if should_randomize:
            self._rsim.randomize(rng)
            self._sync_from_simulator()
            self._visualizer_initialized = False

        self._ensure_visualizer()
        self._task.on_reset()
        self._prev_action = None
        if self.enable_visualization:
            self._rr_needs_episode_stream_switch = True

        self._vehicle.on_reset(rng)

        # Sample a valid navigable starting position in Habitat space,
        # then convert to dynamics frame.
        hab_start = self._safety.sample_habitat_start(rng)
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

        # Take multiple simulator steps per environment step to match control rate. ###
        if self.obs_mode == "state":
            event_rep = None
            self._event_manager.begin_step()

            for _ in range(self.steps_per_action):
                self.sim.step(control)
                if self.enable_visualization:
                    measurements = self.sim._render_sensors()
                    self._maybe_visualize(measurements)
        else:
            self._event_manager.begin_step()
            event_rep = self._event_manager.raw

            for _ in range(self.steps_per_action):
                self.sim.step(control)
                measurements = self.sim._render_sensors()
                self._event_manager.accumulate(measurements.get(self.event_sensor_uuid))
                self._maybe_visualize(measurements, event_rep)

            event_rep = self._event_manager.observation()
        ###############################################################################

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
        if hasattr(self, "_rsim"):
            self._rsim.close()
