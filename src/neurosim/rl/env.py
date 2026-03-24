"""Gymnasium environment wrapping the synchronous neurosim simulator.

Supports a *hover-stop* task: the drone starts with a random translational
velocity and must decelerate to a stable hover inside the scene without
collisions, using event-camera observations.

Safety checks are delegated to Habitat pathfinder bounds and navigability.
"""

import copy
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from neurosim.rl.safety import HabitatSafetyChecker, build_safety_checker
from neurosim.sims.synchronous_simulator import SynchronousSimulator

logger = logging.getLogger(__name__)


class NeurosimRLEnv(gym.Env):
    """Single-drone RL environment with CTBR actions.

    Action space - 4-D continuous:
        Normalized policy action in [-1, 1]^4.
        Internally rescaled to [collective_thrust, roll_rate, pitch_rate, yaw_rate].

    Observation modes:
        "events"   - event histogram only  (2, H, W)
        "state"    - privileged state only (13,)
        "combined" - Dict {"events", "state"} (default, for SB3 MultiInputPolicy)

    Safety checks are performed via Habitat pathfinder bounds and
    navigability queries.
    """

    def __init__(
        self,
        settings: str | Path | dict[str, Any],
        obs_mode: str = "combined",
        # Episode length
        episode_seconds: float = 10.0,
        # Event sensor
        event_sensor_uuid: str | None = None,
        event_clip: float = 10.0,
        event_downsample_factor: int = 1,
        # Initial-velocity randomisation
        init_speed_range: tuple[float, float] = (0.5, 1.0),
        # Event representation and normalization
        event_representation: str = "histogram",
        event_log_compression: float | None = None,
        # Safety / termination
        enable_navigable_check: bool = True,
        # Reward weights
        w_velocity: float = 1.0,
        w_events: float = 0.001,
        w_angular: float = 0.05,
        w_action: float = 1e-5,
        w_survival: float = 0.2,
        crash_penalty: float = 10.0,
        # Success detection
        success_velocity_threshold: float = 0.15,
        success_steps_required: int = 10,
        # Visualisation
        enable_visualization: bool = False,
        visualization_rrd_path: str | None = None,
        visualization_log_every_n_steps: int = 1,
        # Headless debug dumps
        debug_save_events_png: bool = False,
        debug_png_dir: str | Path | None = None,
        debug_save_every_n_steps: int = 100,
        debug_accumulate_n_steps: int = 20,
    ):
        super().__init__()

        if obs_mode not in {"events", "state", "combined"}:
            raise ValueError("obs_mode must be one of: events, state, combined")

        self.obs_mode = obs_mode
        self.episode_seconds = float(episode_seconds)
        self.event_clip = float(event_clip)
        self.event_downsample_factor = max(int(event_downsample_factor), 1)
        self.event_representation = event_representation
        if event_representation not in {"histogram", "event_frame"}:
            raise ValueError(
                "event_representation must be 'histogram' or 'event_frame'"
            )
        self.event_log_compression = (
            float(event_log_compression) if event_log_compression is not None else None
        )
        self.enable_visualization = bool(enable_visualization)
        self.visualization_rrd_path = visualization_rrd_path
        self.visualization_log_every_n_steps = max(
            int(visualization_log_every_n_steps), 1
        )
        self._visualizer_initialized = False

        self.init_speed_range = tuple(init_speed_range)

        # Fixed CTBR body-rate limits aligned with RotorPy vectorized environments.
        self.max_roll_br = 7.0
        self.max_pitch_br = 7.0
        self.max_yaw_br = 3.0

        self.w_velocity = float(w_velocity)
        self.w_events = float(w_events)
        self.w_angular = float(w_angular)
        self.w_action = float(w_action)
        self.w_survival = float(w_survival)
        self.crash_penalty = float(crash_penalty)

        self.success_velocity_threshold = float(success_velocity_threshold)
        self.success_steps_required = int(success_steps_required)
        self._consecutive_success_steps = 0
        self._prev_action: np.ndarray | None = None

        self.debug_save_events_png = bool(debug_save_events_png)
        self.debug_save_every_n_steps = max(int(debug_save_every_n_steps), 1)
        self.debug_accumulate_n_steps = max(int(debug_accumulate_n_steps), 1)
        self._debug_episode_idx = 0
        self._debug_step_idx = 0
        self._debug_accum_counter = 0
        self._debug_accum_frame: np.ndarray | None = None
        self._debug_png_dir = (
            Path(debug_png_dir)
            if debug_png_dir is not None
            else Path("outputs/rl/debug_events")
        )
        if self.debug_save_events_png:
            self._debug_png_dir.mkdir(parents=True, exist_ok=True)

        # Build simulator — strip trajectory config since RL drives the drone
        # directly via CTBR; trajectory planning is unused and expensive.
        settings_dict = self._load_settings_dict(settings)
        settings_dict.pop("trajectory", None)
        settings_dict.setdefault("dynamics", {})["control_abstraction"] = "cmd_ctbr"

        self.sim = SynchronousSimulator(
            settings_dict,
            visualizer_disabled=not self.enable_visualization,
        )

        # Safety checker — always Habitat-backed when used for RL.
        self._safety: HabitatSafetyChecker = build_safety_checker(
            self.sim,
            enable_navigable_check=enable_navigable_check,
        )

        # Timing
        world_rate = self.sim.config.world_rate
        control_rate = self.sim.config.control_rate
        self.steps_per_action = max(int(world_rate / control_rate), 1)

        # Event sensor resolution
        self.event_sensor_uuid = (
            event_sensor_uuid or self._get_default_event_sensor_uuid()
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

        # Action scaling - CTBR (policy uses normalized [-1, 1]^4 actions)
        mass = float(self.sim.dynamics._multirotor.mass)
        gravity = float(self.sim.dynamics._multirotor.g)
        self._hover_thrust = mass * gravity

        # Prefer rotor-speed-derived thrust limits from RotorPy params.
        rotor_speed_min = np.asarray(self.sim.dynamics._multirotor.rotor_speed_min)
        rotor_speed_max = np.asarray(self.sim.dynamics._multirotor.rotor_speed_max)
        k_eta = np.asarray(self.sim.dynamics._multirotor.k_eta)

        min_thrust_per_rotor = k_eta * np.square(rotor_speed_min)
        max_thrust_per_rotor = k_eta * np.square(rotor_speed_max)
        self.cmd_thrust_min = float(np.sum(min_thrust_per_rotor))
        self.cmd_thrust_max = float(np.sum(max_thrust_per_rotor))

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Observation spaces
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_settings_dict(settings: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(settings, dict):
            return copy.deepcopy(settings)
        with open(Path(settings), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_default_event_sensor_uuid(self) -> str:
        event_sensors = self.sim.config.sensor_manager.get_sensors_by_type("event")
        if not event_sensors:
            raise RuntimeError(
                "No event sensor configured; RL env expects at least one event sensor"
            )
        return event_sensors[0].uuid

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
        self, event_frame: np.ndarray, state: dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        state_vec = self._state_vector(state)
        if self.obs_mode == "events":
            return event_frame
        if self.obs_mode == "state":
            return state_vec
        return {"events": event_frame, "state": state_vec}

    def _downsample_event_frame(self, event_frame: np.ndarray) -> np.ndarray:
        if self.event_downsample_factor == 1:
            return event_frame

        factor = self.event_downsample_factor
        h = (event_frame.shape[1] // factor) * factor
        w = (event_frame.shape[2] // factor) * factor
        cropped = event_frame[:, :h, :w]
        reshaped = cropped.reshape(2, h // factor, factor, w // factor, factor)
        return reshaped.mean(axis=(2, 4), dtype=np.float32)

    def _normalize_event_frame(self, event_frame: np.ndarray) -> np.ndarray:
        """Scale clipped event counts to [0, 1] before policy/CNN consumption."""
        if self.event_log_compression is not None:
            # Apply log compression to boost low-intensity events while compressing peaks.
            # Formula: log(1 + k*x) / log(1 + k*clip) where k controls compression strength.
            k = self.event_log_compression
            event_frame = np.log1p(k * event_frame) / np.log1p(k * self.event_clip)
        else:
            # Linear normalization.
            scale = max(self.event_clip, 1e-6)
            event_frame /= scale
        np.clip(event_frame, 0.0, 1.0, out=event_frame)
        return event_frame

    def _accumulate_events_into_frame(
        self, event_frame: np.ndarray, events: Any | None
    ) -> None:
        if events is None:
            return

        x, y, p = events.x, events.y, events.p

        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            p = p.detach().cpu().numpy()

        if x.size == 0:
            return

        x = x.astype(np.int64, copy=False)
        y = y.astype(np.int64, copy=False)
        p = p.astype(np.int64, copy=False)

        if self.event_representation == "histogram":
            np.add.at(event_frame, (p, y, x), 1.0)
        elif self.event_representation == "event_frame":
            # Mark pixels with events as 1 (binary representation, not accumulated).
            event_frame[p, y, x] = 1.0

    @staticmethod
    def _write_png(path: Path, image: np.ndarray) -> None:
        """Write PNG without GUI/display requirements."""
        try:
            import cv2

            cv2.imwrite(str(path), image)
            return
        except Exception:
            pass

        try:
            from PIL import Image

            Image.fromarray(image).save(path)
            return
        except Exception:
            pass

        logger.warning("Failed to save debug PNG at %s", path)

    def _event_frame_to_rgb(self, event_frame: np.ndarray, clip: float) -> np.ndarray:
        """Map negative/positive event channels to RGB for quick visual debugging."""
        scale = max(float(clip), 1e-6)
        neg = np.clip(event_frame[0] / scale, 0.0, 1.0)
        pos = np.clip(event_frame[1] / scale, 0.0, 1.0)

        rgb = np.zeros((event_frame.shape[1], event_frame.shape[2], 3), dtype=np.uint8)
        rgb[..., 0] = np.round(255.0 * neg).astype(np.uint8)
        rgb[..., 2] = np.round(255.0 * pos).astype(np.uint8)
        return rgb

    def _state_panel(self, state: dict[str, np.ndarray], width: int) -> np.ndarray:
        """Create a compact text panel for state inspection."""
        panel_h = 96
        panel = np.full((panel_h, width, 3), 20, dtype=np.uint8)

        x = np.asarray(state["x"], dtype=np.float32)
        v = np.asarray(state["v"], dtype=np.float32)
        w = np.asarray(state["w"], dtype=np.float32)

        lines = [
            f"step={self._debug_step_idx} t={self.sim.time:.3f}s",
            f"x=[{x[0]:+.2f}, {x[1]:+.2f}, {x[2]:+.2f}]",
            f"|v|={np.linalg.norm(v):.3f}  |w|={np.linalg.norm(w):.3f}",
        ]

        try:
            import cv2

            y = 24
            for text in lines:
                cv2.putText(
                    panel,
                    text,
                    (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                y += 26
        except Exception:
            pass

        return panel

    def _maybe_dump_debug_images(
        self, event_frame: np.ndarray, state: dict[str, np.ndarray]
    ) -> None:
        if not self.debug_save_events_png:
            return

        self._debug_step_idx += 1

        if self._debug_step_idx % self.debug_save_every_n_steps == 0:
            rgb = self._event_frame_to_rgb(event_frame, clip=1.0)
            panel = self._state_panel(state, width=rgb.shape[1])
            frame = np.vstack([rgb, panel])
            path = (
                self._debug_png_dir
                / f"ep{self._debug_episode_idx:04d}_step{self._debug_step_idx:06d}_events.png"
            )
            self._write_png(path, frame)

        if self._debug_accum_frame is None:
            self._debug_accum_frame = np.zeros_like(event_frame, dtype=np.float32)

        self._debug_accum_frame += event_frame
        self._debug_accum_counter += 1

        if self._debug_accum_counter >= self.debug_accumulate_n_steps:
            clip = float(self.debug_accumulate_n_steps)
            rgb_acc = self._event_frame_to_rgb(self._debug_accum_frame, clip=clip)
            panel = self._state_panel(state, width=rgb_acc.shape[1])
            frame = np.vstack([rgb_acc, panel])
            path = self._debug_png_dir / (
                f"ep{self._debug_episode_idx:04d}_"
                f"step{self._debug_step_idx:06d}_acc{self.debug_accumulate_n_steps}.png"
            )
            self._write_png(path, frame)
            self._debug_accum_frame.fill(0.0)
            self._debug_accum_counter = 0

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def _compute_reward(
        self, state: dict[str, np.ndarray], event_frame: np.ndarray, action: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        v = np.asarray(state["v"], dtype=np.float32)
        w = np.asarray(state["w"], dtype=np.float32)

        vel_norm = float(np.linalg.norm(v))
        ang_rate_norm = float(np.linalg.norm(w))
        if self.obs_mode == "state":
            # State-only baseline should not pay event penalties.
            event_activity = 0.0
            event_activity_density = 0.0
        else:
            event_activity = float(event_frame.sum())
            event_activity_density = event_activity / float(event_frame.size)

        if self._prev_action is None:
            action_smoothness = 0.0
        else:
            action_smoothness = float(np.linalg.norm(action - self._prev_action))

        r_velocity = -self.w_velocity * vel_norm
        r_events = -self.w_events * event_activity_density
        r_angular = -self.w_angular * ang_rate_norm
        r_action = -self.w_action * action_smoothness
        r_survival = self.w_survival

        reward = r_velocity + r_events + r_angular + r_action + r_survival

        return reward, {
            "vel_norm": vel_norm,
            "ang_rate_norm": ang_rate_norm,
            "event_activity": event_activity,
            "event_activity_density": event_activity_density,
            "action_smoothness": action_smoothness,
            "r_velocity": r_velocity,
            "r_events": r_events,
            "r_angular": r_angular,
            "r_action": r_action,
            "r_survival": r_survival,
        }

    def _check_terminated(self, state: dict[str, np.ndarray]) -> tuple[bool, str]:
        x = np.asarray(state["x"])
        safe, reason = self._safety.check(x)
        if not safe:
            return True, reason
        return False, ""

    def _check_success(self, state: dict[str, np.ndarray]) -> bool:
        vel_norm = float(np.linalg.norm(np.asarray(state["v"])))
        if vel_norm < self.success_velocity_threshold:
            self._consecutive_success_steps += 1
        else:
            self._consecutive_success_steps = 0
        if self.success_steps_required > 0:
            return self._consecutive_success_steps >= self.success_steps_required
        return False

    @staticmethod
    def _minmax_scale(
        x: np.ndarray,
        min_values: float | np.ndarray,
        max_values: float | np.ndarray,
    ) -> np.ndarray:
        """Scale from [-1, 1] policy space to [min_values, max_values]."""
        x_scaled = (x + 1.0) * 0.5 * (max_values - min_values) + min_values
        return np.clip(x_scaled, min_values, max_values)

    def _rescale_ctbr_action(self, action: np.ndarray) -> tuple[float, np.ndarray]:
        """Convert normalized action in [-1, 1]^4 to physical CTBR commands."""
        cmd_thrust = float(
            self._minmax_scale(
                action[0],
                self.cmd_thrust_min,
                self.cmd_thrust_max,
            )
        )
        cmd_roll_br = float(
            self._minmax_scale(action[1], -self.max_roll_br, self.max_roll_br)
        )
        cmd_pitch_br = float(
            self._minmax_scale(action[2], -self.max_pitch_br, self.max_pitch_br)
        )
        cmd_yaw_br = float(
            self._minmax_scale(action[3], -self.max_yaw_br, self.max_yaw_br)
        )
        cmd_w = np.asarray([cmd_roll_br, cmd_pitch_br, cmd_yaw_br], dtype=np.float64)
        return cmd_thrust, cmd_w

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
        self.sim.visualizer.initialize(
            display=True, rrd_path=self.visualization_rrd_path
        )
        self._visualizer_initialized = True

    def _maybe_visualize(self, measurements: dict[str, Any]) -> None:
        if not self.enable_visualization or self.sim.visualizer is None:
            return
        if self.sim.simsteps % self.visualization_log_every_n_steps != 0:
            return
        self.sim.visualizer.log_measurements(
            measurements, self.sim.time, self.sim.simsteps
        )
        self.sim.visualizer.log_state(self.sim.dynamics.state)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ensure_visualizer()
        self._consecutive_success_steps = 0
        self._prev_action = None
        self._debug_episode_idx += 1
        self._debug_step_idx = 0
        self._debug_accum_counter = 0

        if self.debug_save_events_png and self._debug_accum_frame is not None:
            self._debug_accum_frame.fill(0.0)

        rng = np.random.default_rng(seed)

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

        event_frame = np.zeros(
            (2, self.event_height, self.event_width), dtype=np.float32
        )

        observation = self._compose_observation(event_frame, self.sim.dynamics.state)
        info: dict[str, Any] = {
            "time": self.sim.time,
            "simsteps": self.sim.simsteps,
        }
        return observation, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        cmd_thrust, cmd_w = self._rescale_ctbr_action(action)

        control = {
            "cmd_thrust": cmd_thrust,
            "cmd_w": cmd_w,
        }

        event_frame = np.zeros(
            (2, self.event_height, self.event_width), dtype=np.float32
        )

        for _ in range(self.steps_per_action):
            self.sim.step(control)

            if self.obs_mode == "state":
                # Skip event rendering/accumulation for state-only baseline.
                continue

            measurements = self.sim._render_sensors()
            self._maybe_visualize(measurements)
            self._accumulate_events_into_frame(
                event_frame, measurements.get(self.event_sensor_uuid)
            )

        event_frame = np.clip(event_frame, 0.0, self.event_clip, out=event_frame)
        # Normalize event frame to [0, 1] range for policy consumption if using histogram representation.
        # Event frames are binary by design and don't require normalization.
        if self.event_representation == "histogram":
            event_frame = self._normalize_event_frame(event_frame)
        event_frame = self._downsample_event_frame(event_frame)

        state = self.sim.dynamics.state
        self._maybe_dump_debug_images(event_frame, state)
        reward, reward_terms = self._compute_reward(state, event_frame, action)
        self._prev_action = action.copy()

        terminated, term_reason = self._check_terminated(state)
        truncated = self.sim.time >= self.episode_seconds

        is_success = self._check_success(state) if not terminated else False

        if terminated:
            reward -= self.crash_penalty

        observation = self._compose_observation(event_frame, state)
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
        self.sim.close()
