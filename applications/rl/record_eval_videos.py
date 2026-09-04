"""Record inspection videos of dodge-policy evaluation episodes.

Writes one MP4 per episode showing, side by side, the forward RGB view and
the event representation the policy actually consumes, with a telemetry
overlay (action, tracking error, nearest-obstacle distance, reward).

The point is to catch the class of bug that numeric metrics hide: a camera
pointing the wrong way, an action mapped to the wrong axis, obstacles that
never intersect the flight path, a policy that never moves.

Nominal mode (``--nominal``) runs with a zero action and needs no
checkpoint, so it doubles as a coordinate-frame smoke test: on a correct
build the scene flows outward from the image centre as the drone advances.

Usage::

    # frame / transform check, no policy needed
    python applications/rl/record_eval_videos.py \
        --rollout-config applications/rl/configs/reactive_dodge_nominal_eval.yaml \
        --nominal --episodes 3 --seed0 31000 --out-dir outputs/rl/videos/nominal

    # trained policy
    python applications/rl/record_eval_videos.py \
        --checkpoint outputs/rl/dodge/latest_model.zip \
        --rollout-config applications/rl/configs/reactive_dodge_nominal_eval.yaml \
        --episodes 8 --seed0 47000 --out-dir outputs/rl/videos/step_2p1M

Always vary ``--seed0`` between runs: reusing the default replays an
identical scene/trajectory/obstacle sequence, which makes genuinely
different checkpoints look identical side by side.
"""

import argparse
import sys
import copy
import json
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np

from neurosim.core.utils.utils_gen import deep_update, load_yaml
from neurosim.rl import env_class_for_task

# Sensor injected for the human-viewable pane. Matches the event camera's
# pose/FOV so the video shows the same view the policy is reasoning over.
VIDEO_RGB_UUID = "video_rgb_camera"
PANE_HEIGHT = 480
# Rendered on demand per env step, so the simulator's own sampling of it is
# pure overhead; sample it as rarely as the config schema allows.
VIDEO_RGB_SAMPLE_HZ = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rollout-config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--nominal",
        action="store_true",
        help="Run zero-action (pure nominal tracking); no checkpoint needed",
    )
    p.add_argument(
        "--expert",
        action="store_true",
        help="Run the collision-checked local trajectory expert; no checkpoint needed",
    )
    p.add_argument(
        "--expert-planner",
        choices=("quintic", "sampling_mpc"),
        default="quintic",
        help="Planner used with --expert; default preserves historical videos",
    )
    p.add_argument("--clone", type=str, default=None,
                   help="Behaviour-cloned .pt to fly; no checkpoint needed")
    p.add_argument(
        "--event-geometry-spatial",
        type=str,
        default=None,
        help="Spatial event-localizer checkpoint for the geometry controller",
    )
    p.add_argument(
        "--event-geometry-temporal",
        type=str,
        default=None,
        help="Temporal event-tracker checkpoint for the geometry controller",
    )
    p.add_argument(
        "--event-geometry-blank",
        action="store_true",
        help="Zero the event tensor before the geometry tracker (causal control)",
    )
    p.add_argument(
        "--event-action-checkpoint",
        type=str,
        default=None,
        help="Frozen-tracker event-action checkpoint to fly directly.",
    )
    p.add_argument(
        "--event-action-intervention",
        choices=("real", "blank", "reverse_history"),
        default="real",
    )
    p.add_argument("--event-action-structured-pulse", action="store_true")
    p.add_argument("--event-action-pulse-hold-steps", type=int, default=20)
    p.add_argument("--event-action-pulse-ramp-steps", type=int, default=3)
    p.add_argument(
        "--event-pulse-fixed-hold",
        action="store_true",
        help="Disable the PPO wrapper's opt-in inbound-conditioned early release.",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--sim-gpu-id", type=int, default=None)
    p.add_argument("--clone-downsample", type=int, default=4)
    p.add_argument("--vecnormalize", type=str, default=None)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument(
        "--seed0",
        type=int,
        default=None,
        help="First episode seed; episode i uses seed0 + i. Vary between runs.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit episode seeds; overrides --seed0/--episodes",
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "Playback frame rate. Defaults to the env's control rate, which "
            "makes playback real-time (one frame is emitted per policy step)."
        ),
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use the deterministic policy action (default true)",
    )
    p.add_argument(
        "--obstacles",
        dest="obstacles",
        action="store_true",
        default=None,
        help="Force dynamic obstacles on (default: whatever the config says)",
    )
    p.add_argument(
        "--no-obstacles",
        dest="obstacles",
        action="store_false",
        help="Force dynamic obstacles off",
    )
    p.add_argument(
        "--deterministic-obstacle-aim",
        dest="deterministic_obstacle_aim",
        action="store_true",
        default=False,
        help=(
            "Aim obstacle throws at the nominal trajectory instead of the "
            "live drone position, making the same --seed0 reproducible "
            "across checkpoints. NOT the default: the residual controller "
            "holds a standing offset off the nominal path, so aiming there "
            "systematically throws at where a dodging drone is not -- "
            "measured at 5/6 success vs 0/8 for the same checkpoint and "
            "seeds. Use for A/B-ing two checkpoints against an identical "
            "scenario, never for an absolute success-rate number."
        ),
    )
    return p.parse_args()


def load_rollout_config(path: str | Path) -> dict[str, Any]:
    """Load a rollout config, resolving the ``experiment_config`` indirection.

    Mirrors ``evaluate_reactive_dodge_nominal.load_eval_config`` so the same
    eval YAML drives both the metrics run and the video run.
    """
    cfg = load_yaml(path)
    if "experiment_config" in cfg:
        # Curriculum configs may themselves inherit another experiment.
        # Use the trainer's recursive resolver so videos run with exactly
        # the environment that produced the checkpoint.
        from train_sb3 import load_experiment_config

        cfg = load_experiment_config(path)
    if "env" not in cfg:
        raise ValueError("rollout config must define 'env' or 'experiment_config'")
    return cfg


class _ClonePolicy:
    """Wrap a behaviour-cloned network so it matches the predict() interface.

    The clone is trained on pooled events, so the same pooling has to be
    applied here -- feeding it full-resolution frames would silently give it
    an input it has never seen.
    """

    def __init__(self, path: str, cfg, env, downsample: int):
        import torch  # noqa: PLC0415

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_velocity_dodge_bc import build_clone_net  # noqa: PLC0415

        self._device = str(cfg["ppo"].get("device", "cuda:0"))
        payload = torch.load(path, map_location=self._device, weights_only=False)
        preprocessing = payload.get("preprocessing", {})
        self._torch = torch
        self._downsample = int(preprocessing.get("downsample_events", downsample))
        self._discrete = bool(preprocessing.get("discrete_actions", False))
        recurrent_type = preprocessing.get("recurrent_type")
        self._recurrent = recurrent_type in {"gru", "compact_gru"}
        self._compact = recurrent_type == "compact_gru"
        self._hidden = None
        self._controller = None
        common = {
            "blank_previous_action": bool(
                preprocessing.get("blank_previous_action", False)
            ),
            "events_only_state": bool(
                preprocessing.get("events_only_state", False)
            ),
            "blank_tracking_error": bool(
                preprocessing.get("blank_tracking_error", False)
            ),
        }
        if self._compact:
            from train_velocity_dodge_compact_gru import (  # noqa: PLC0415
                CommitmentController,
                build_compact_model,
            )

            model = build_compact_model(
                cfg,
                env,
                self._downsample,
                int(preprocessing.get("hidden_size", 128)),
            )
            self._controller = CommitmentController(
                **preprocessing.get("controller", {})
            )
        elif self._recurrent:
            from train_velocity_dodge_gru_bc import (  # noqa: PLC0415
                build_gru_clone_net,
            )

            model, _ = build_gru_clone_net(
                cfg,
                env,
                self._downsample,
                int(preprocessing.get("hidden_size", 128)),
                **common,
            )
        else:
            model, _ = build_clone_net(
                cfg, env, self._downsample, self._discrete, **common
            )
        model.load_state_dict(payload["model"])
        self._model = model.to(self._device).eval()

    def predict(self, obs, deterministic: bool = True):
        from train_velocity_dodge_bc import pool_events  # noqa: PLC0415

        if isinstance(obs, dict):
            batch = {}
            for key, value in obs.items():
                array = np.asarray(value)
                if key == "events":
                    array = pool_events(array, self._downsample)
                batch[key] = self._torch.from_numpy(array[None]).to(self._device)
        else:
            batch = self._torch.from_numpy(np.asarray(obs)[None]).to(self._device)
        with self._torch.no_grad():
            if self._recurrent:
                output, self._hidden = self._model.recurrent_step(
                    batch, self._hidden
                )
                if self._compact:
                    output = self._controller.action(output)
                else:
                    output = output.cpu().numpy()[0]
            else:
                output = self._model(batch).cpu().numpy()[0]
            if self._discrete:
                action = np.zeros(3, dtype=np.float32)
                action[1] = float(int(np.argmax(output)) - 1)
                output = action
            return output, None

    def reset(self):
        self._hidden = None
        if self._controller is not None:
            self._controller.reset()


def inject_rgb_sensor(env_config: dict[str, Any]) -> dict[str, Any]:
    """Add an RGB sensor mirroring the event camera's pose and FOV.

    Mirroring the mount is what makes the video diagnostic: if the event
    camera's ``orientation`` is wrong, the RGB pane shows it wrong too,
    rather than quietly showing a correct view of a broken setup.
    """
    env_config = copy.deepcopy(env_config)
    sensors = env_config["sensors"]

    event_sensors = [s for s in sensors.values() if s.get("type") == "event"]
    if not event_sensors:
        raise ValueError("rollout config defines no event sensor to mirror")
    src = event_sensors[0]

    sensors[VIDEO_RGB_UUID] = {
        "type": "color",
        "position": list(src["position"]),
        "orientation": list(src["orientation"]),
        "width": src["width"],
        "height": src["height"],
        "hfov": src["hfov"],
        "zfar": src.get("zfar", 20.0),
    }

    # SensorConfig construction indexes sensor_rates by uuid, so an injected
    # sensor without a rate is a KeyError at simulator build time.
    simulator = env_config.setdefault("simulator", {})
    simulator.setdefault("sensor_rates", {})[VIDEO_RGB_UUID] = VIDEO_RGB_SAMPLE_HZ
    simulator.setdefault("viz_rates", {})[VIDEO_RGB_UUID] = VIDEO_RGB_SAMPLE_HZ

    # Domain randomization resamples per-sensor specs by uuid; the video
    # sensor must stay fixed or the RGB pane stops matching the event pane.
    dr_sensors = simulator.get("domain_randomization", {}).get("sensors", {})
    dr_sensors.pop(VIDEO_RGB_UUID, None)
    return env_config


def apply_obstacle_override(env_config: dict[str, Any], enabled: bool) -> None:
    env_config.setdefault("visual_backend", {}).setdefault("dynamic_obstacles", {})[
        "enabled"
    ] = enabled


def _to_uint8_rgb(frame: Any) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return frame[..., :3]


def _fit_pane(img: np.ndarray, height: int = PANE_HEIGHT) -> np.ndarray:
    scale = height / img.shape[0]
    width = max(round(img.shape[1] * scale), 1)
    interp = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (width, height), interpolation=interp)


def _draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    y = 22
    for line in lines:
        cv2.putText(
            out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20
    return out


def _fmt_vec(vec: Any, precision: int = 2) -> str:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return "[" + " ".join(f"{v:+.{precision}f}" for v in arr) + "]"


def _draw_body_vector(frame, vec, color, label, scale_px, y_label):
    """Draw a body-frame vector as an arrow from the image centre.

    The camera looks along body-forward, so the lateral/vertical components
    are what appear in the image: body +y is left (image -x) and body +z is
    up (image -y). The forward component cannot be drawn as a direction and
    is printed instead.

    The commanded delta and the obstacle bearing are both mapped through
    this same function on purpose -- so even if the absolute convention is
    off, their *relative* orientation is still read correctly, which is the
    thing worth seeing.
    """
    if vec is None:
        return frame
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size < 3 or not np.all(np.isfinite(v)):
        return frame
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    dx = int(round(-v[1] * scale_px))
    dy = int(round(-v[2] * scale_px))
    if abs(dx) + abs(dy) >= 3:
        cv2.arrowedLine(
            frame, (cx, cy), (cx + dx, cy + dy), (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.25
        )
        cv2.arrowedLine(
            frame, (cx, cy), (cx + dx, cy + dy), color, 2, cv2.LINE_AA, tipLength=0.25
        )
    cv2.putText(frame, f"{label} fwd {v[0]:+.2f}", (8, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"{label} fwd {v[0]:+.2f}", (8, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame


def compose_frame(
    rgb: np.ndarray,
    event_rgb: np.ndarray,
    overlay_lines: list[str],
    delta_body: np.ndarray | None = None,
    obstacle_body: np.ndarray | None = None,
) -> np.ndarray:
    left = _draw_overlay(_fit_pane(rgb), overlay_lines)
    h = left.shape[0]
    # Neither body-frame arrow is drawn, by standing request: they clutter
    # the view without adding anything the scene does not already show. Both
    # parameters are kept so callers need not change, and the numeric
    # commanded action still appears in the text overlay.
    cv2.circle(left, (left.shape[1] // 2, h // 2), 3, (255, 255, 255), -1, cv2.LINE_AA)
    right = _fit_pane(event_rgb)
    return np.concatenate([left, right], axis=1)


def episode_overlay_lines(
    *,
    episode: int,
    seed: int,
    step: int,
    sim_time: float,
    action: np.ndarray,
    reward: float,
    total_reward: float,
    terms: dict[str, float],
    acceleration_mps2: float,
    residual_command_acceleration_mps2: float,
) -> list[str]:
    lines = [
        f"ep {episode} seed {seed} step {step} t={sim_time:5.2f}s",
        f"action {_fmt_vec(action)}",
        (
            f"accel {acceleration_mps2:.2f}m/s^2  "
            f"residual_cmd {residual_command_acceleration_mps2:.2f}m/s^2"
        ),
        f"reward {reward:+.3f} total {total_reward:+.1f}",
    ]
    if "pos_error" in terms:
        lines.append(
            f"pos_err {terms['pos_error']:.2f}m  vel_err {terms.get('vel_error', 0.0):.2f}"
        )
    min_dist = terms.get("min_obstacle_distance")
    if min_dist is not None and np.isfinite(min_dist):
        threat = "THREAT" if terms.get("obstacle_threat", 0.0) > 0.5 else ""
        lines.append(f"obstacle {min_dist:.2f}m {threat}")
    return lines


def record_episode(
    env,
    model,
    *,
    episode: int,
    seed: int,
    out_path: Path,
    fps: int,
    deterministic: bool,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    if model is not None and hasattr(model, "reset"):
        model.reset()
    raw_env = env.unwrapped
    backend = raw_env.sim.visual_backend

    total_reward = 0.0
    step = 0
    min_obstacle_distance = np.inf
    policy_dt = float(
        raw_env.policy_decimation
        * raw_env.steps_per_action
        / raw_env.sim.config.world_rate
    )
    previous_velocity = np.asarray(
        raw_env.sim.dynamics.state["v"], dtype=np.float64
    ).copy()
    previous_delta_velocity = np.asarray(
        getattr(raw_env, "_last_delta_velocity", np.zeros(3)), dtype=np.float64
    ).copy()
    acceleration_norms = []
    command_acceleration_norms = []
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)

    try:
        while True:
            if model is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32).reshape(
                env.action_space.shape
            )

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            total_reward += float(reward)

            # Residual-policy actions live in an internal coordinate system.
            # Show and diagnose the command that actually reached the task.
            pulse_residual = info.get("event_pulse_residual") or {}
            executed_action = np.asarray(
                pulse_residual.get("executed_action", action), dtype=np.float32
            ).reshape(env.action_space.shape)

            velocity = np.asarray(
                raw_env.sim.dynamics.state["v"], dtype=np.float64
            ).copy()
            acceleration_norm = float(
                np.linalg.norm(velocity - previous_velocity) / policy_dt
            )
            previous_velocity = velocity
            acceleration_norms.append(acceleration_norm)
            delta_velocity = np.asarray(
                getattr(raw_env, "_last_delta_velocity", np.zeros(3)),
                dtype=np.float64,
            ).copy()
            command_acceleration_norm = float(
                np.linalg.norm(delta_velocity - previous_delta_velocity) / policy_dt
            )
            previous_delta_velocity = delta_velocity
            command_acceleration_norms.append(command_acceleration_norm)

            terms = info.get("reward_terms", {}) or {}
            dist = terms.get("min_obstacle_distance", np.inf)
            if np.isfinite(dist):
                min_obstacle_distance = min(min_obstacle_distance, float(dist))

            rgb = _to_uint8_rgb(backend.render_color(VIDEO_RGB_UUID))
            event_rgb = raw_env._event_manager.to_rgb(
                raw_env._event_manager.observation()
            )

            # What the policy is actually commanding, in the same body frame
            # the camera looks along, plus where the nearest threat is. Seen
            # together these answer the question the numbers keep dodging:
            # is the correction pointed away from the obstacle, or not?
            delta_body = obstacle_body = None
            try:
                delta_body = np.asarray(
                    raw_env._task.split_action(executed_action)[1],
                    dtype=np.float64,
                ).reshape(-1)[:3]
            except Exception:
                delta_body = None
            try:
                rows = raw_env._obstacle_relative_states(
                    raw_env.sim.dynamics.state
                ) or []
                if rows:
                    from neurosim.core.coord_trans import rotate_vector_by_quat

                    obstacle_body = rotate_vector_by_quat(
                        np.asarray(rows[0]["rel_pos"], dtype=np.float64),
                        np.asarray(raw_env.sim.dynamics.state["q"], dtype=np.float64),
                        inverse=True,
                    )
            except Exception:
                obstacle_body = None

            overlay_lines = episode_overlay_lines(
                episode=episode,
                seed=seed,
                step=step,
                sim_time=float(info.get("time", 0.0)),
                action=executed_action,
                reward=float(reward),
                total_reward=total_reward,
                terms=terms,
                acceleration_mps2=acceleration_norm,
                residual_command_acceleration_mps2=command_acceleration_norm,
            )
            prediction = getattr(model, "last_prediction", None)
            controller = getattr(model, "controller", None)
            pulse_controller = getattr(model, "pulse_controller", None)
            if pulse_controller is None:
                wrapped_event_policy = getattr(env, "policy", None)
                pulse_controller = getattr(
                    wrapped_event_policy, "pulse_controller", None
                )
            if prediction is not None:
                overlay_lines.append(
                    f"tracker p {prediction.probability:.3f}  "
                    f"inbound {prediction.inbound_probability:.3f}  "
                    f"uv ({prediction.centre[0]:.3f}, {prediction.centre[1]:.3f})"
                )
            elif pulse_residual:
                overlay_lines.append(
                    f"tracker p {pulse_residual['tracker_probability']:.3f}  "
                    f"inbound {pulse_residual['inbound_probability']:.3f}  "
                    f"PPO enabled {bool(pulse_residual['gate'])}"
                )
                overlay_lines.append(
                    f"trigger base {pulse_residual['base_gate_probability']:.3f}  "
                    f"adjusted {pulse_residual['adjusted_gate_probability']:.3f}  "
                    f"commit {bool(pulse_residual['committed'])}"
                )
            if controller is not None:
                release = (
                    controller.last_release_reason
                    if getattr(model, "last_committed", False)
                    else "-"
                )
                overlay_lines.append(
                    f"track age {controller.track_age}  "
                    f"commitments {controller.commitments}  release {release}"
                )
            elif pulse_controller is not None:
                overlay_lines.append(
                    f"learned pulse commitments {pulse_controller.commitments}  "
                    f"early releases {pulse_controller.early_releases}  "
                    f"active {pulse_controller.pulse_step is not None}  "
                    f"refractory {pulse_controller.refractory}"
                )

            writer.append_data(
                compose_frame(
                    rgb,
                    event_rgb,
                    overlay_lines,
                    delta_body=delta_body,
                    obstacle_body=obstacle_body,
                )
            )

            if terminated or truncated:
                return {
                    "episode": episode,
                    "seed": seed,
                    "steps": step,
                    "total_reward": total_reward,
                    "sim_time": float(info.get("time", 0.0)),
                    "is_success": bool(info.get("is_success", False)),
                    "termination_reason": info.get(
                        "termination_reason", "truncated" if truncated else "none"
                    ),
                    "min_obstacle_distance": (
                        None
                        if not np.isfinite(min_obstacle_distance)
                        else min_obstacle_distance
                    ),
                    "peak_acceleration_mps2": float(max(acceleration_norms)),
                    "p95_acceleration_mps2": float(
                        np.percentile(acceleration_norms, 95)
                    ),
                    "peak_residual_command_acceleration_mps2": float(
                        max(command_acceleration_norms)
                    ),
                    "video": str(out_path),
                }
    finally:
        writer.close()


def load_model(args: argparse.Namespace, cfg: dict[str, Any], env):
    """Load the SB3 policy, wrapping the env's normalizer if one was saved."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    checkpoint = Path(args.checkpoint)
    vecnorm_path = args.vecnormalize or (checkpoint.parent / "vecnormalize.pkl")

    obs_norm = None
    if Path(vecnorm_path).exists():
        dummy = DummyVecEnv([lambda: env])
        obs_norm = VecNormalize.load(str(vecnorm_path), dummy)
        obs_norm.training = False
        obs_norm.norm_reward = False
        print(f"loaded VecNormalize from {vecnorm_path}")
    else:
        print(f"no vecnormalize.pkl at {vecnorm_path}; using raw observations")

    device = str(cfg.get("policy", {}).get("device", "auto"))
    model = PPO.load(str(checkpoint), device=device)
    return model, obs_norm


class _NormalizedPolicy:
    """Applies the saved VecNormalize obs statistics before predicting."""

    def __init__(self, model, obs_norm):
        self._model = model
        self._obs_norm = obs_norm

    def predict(self, obs, deterministic: bool = True):
        if self._obs_norm is not None:
            obs = self._obs_norm.normalize_obs(obs)
        if isinstance(obs, dict):
            batched = {k: np.asarray(v)[None] for k, v in obs.items()}
        else:
            batched = np.asarray(obs)[None]
        action, state = self._model.predict(batched, deterministic=deterministic)
        return np.asarray(action)[0], state


class _TrajectoryExpertPolicy:
    """Adapter exposing the privileged trajectory expert as a video policy."""

    def __init__(self, env, planner: str = "quintic"):
        self._env = env
        self._planner = str(planner)

    def reset(self):
        # expert_for_task, not a bare LocalTrajectoryExpert: the default
        # config carries a hardcoded 1.5/1.0 speed envelope, while the task
        # derives its own from offset_rate_limits_mps (now 0.6). Constructing
        # the default here would record an expert planning dodges the
        # configured vehicle cannot fly, so the video would not show the
        # oracle that the metrics describe.
        from evaluate_velocity_dodge_oracle import expert_for_task, mpc_expert_for_env

        if self._planner == "sampling_mpc":
            self._env._receding_horizon_dodge_expert = mpc_expert_for_env(self._env)
        else:
            self._env._trajectory_dodge_expert = expert_for_task(self._env._task)
        self._env._trajectory_expert_plans = 0
        self._env._trajectory_expert_failed_plans = 0
        self._env._mpc_slack_replans = 0
        self._env._mpc_visibility_released_ids = set()
        self._env._mpc_visibility_release_events = []
        self._env._trajectory_expert_plan_diagnostics = []

    def predict(self, obs, deterministic: bool = True):
        from evaluate_velocity_dodge_oracle import mpc_oracle_action, oracle_action

        action = (
            mpc_oracle_action(self._env)
            if self._planner == "sampling_mpc"
            else oracle_action(self._env)
        )
        return action, None


def main() -> None:
    args = parse_args()
    event_geometry = bool(
        args.event_geometry_spatial or args.event_geometry_temporal
    )
    event_action = bool(args.event_action_checkpoint)
    if event_geometry and not (
        args.event_geometry_spatial and args.event_geometry_temporal
    ):
        raise SystemExit(
            "--event-geometry-spatial and --event-geometry-temporal are both required"
        )
    if sum((bool(args.nominal), bool(args.expert), bool(args.checkpoint),
            bool(args.clone), event_geometry, event_action)) != 1:
        raise SystemExit(
            "pass exactly one policy mode: --checkpoint, --nominal, --expert, "
            "--clone, --event-action-checkpoint, or both --event-geometry checkpoints"
        )
    if args.seeds is None and args.seed0 is None:
        raise SystemExit("pass --seed0 or an explicit --seeds list")

    cfg = load_rollout_config(args.rollout_config)
    env_config = inject_rgb_sensor(cfg["env"])
    pulse_residual_cfg = env_config.pop("event_pulse_residual", None)
    if args.event_pulse_fixed_hold and pulse_residual_cfg is not None:
        pulse_residual_cfg["pulse_kwargs"] = {}
    env_config["enable_visualization"] = False
    if args.sim_gpu_id is not None:
        env_config.setdefault("visual_backend", {})["gpu_id"] = args.sim_gpu_id
    env_config["deterministic_obstacle_aim"] = bool(args.deterministic_obstacle_aim)
    if args.obstacles is not None:
        apply_obstacle_override(env_config, args.obstacles)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cls = env_class_for_task(env_config["task"]["name"])
    env = env_cls(env_config=env_config, train=False)
    if args.checkpoint and pulse_residual_cfg is not None:
        from event_pulse_residual import EventPulseResidualWrapper

        pulse_residual_cfg = dict(pulse_residual_cfg)
        if pulse_residual_cfg.pop("enabled", True):
            configured_device = str(pulse_residual_cfg.pop("device", "auto"))
            device = args.device or configured_device
            if device == "auto":
                device = str(cfg["ppo"].get("device", "cuda:0"))
            env = EventPulseResidualWrapper(
                env, device=device, **pulse_residual_cfg
            )

    # One video frame per policy step, so real-time playback means matching
    # the *policy's* decision rate, not the inner controller rate: each
    # env.step() advances policy_decimation * steps_per_action world steps
    # before a frame is written. Omitting policy_decimation here understated
    # the elapsed sim time per frame by that factor, so at decimation=2
    # playback ran 2x too fast.
    raw_env = env.unwrapped
    policy_hz = raw_env.sim.config.world_rate / (
        raw_env.policy_decimation * raw_env.steps_per_action
    )
    fps = int(args.fps) if args.fps else max(int(round(policy_hz)), 1)
    print(f"recording at {fps} fps (policy runs at {policy_hz:.1f} Hz -> real-time)")

    model = None
    if args.clone:
        model = _ClonePolicy(args.clone, cfg, env, args.clone_downsample)
    elif event_action:
        from train_event_tracker_action_head import EventActionPolicy

        model = EventActionPolicy(
            args.event_action_checkpoint,
            device=args.device or str(cfg["ppo"].get("device", "cuda:0")),
            intervention=args.event_action_intervention,
            structured_pulse=(
                True if args.event_action_structured_pulse else None
            ),
            pulse_kwargs=(
                {
                    "hold_steps": args.event_action_pulse_hold_steps,
                    "ramp_steps": args.event_action_pulse_ramp_steps,
                }
                if args.event_action_structured_pulse
                else None
            ),
        )
    elif event_geometry:
        from evaluate_event_geometry_controller import EventGeometryPolicy

        model = EventGeometryPolicy(
            args.event_geometry_spatial,
            args.event_geometry_temporal,
            device=str(cfg["ppo"].get("device", "cuda:0")),
            blank_events=args.event_geometry_blank,
        )
    elif args.expert:
        model = _TrajectoryExpertPolicy(env, args.expert_planner)
    elif not args.nominal:
        loaded, obs_norm = load_model(args, cfg, env)
        model = _NormalizedPolicy(loaded, obs_norm)

    results = []
    try:
        seeds = (
            [int(seed) for seed in args.seeds]
            if args.seeds is not None
            else [int(args.seed0) + i for i in range(args.episodes)]
        )
        for i, seed in enumerate(seeds):
            out_path = out_dir / f"episode_{i:02d}_seed{seed}.mp4"
            summary = record_episode(
                env,
                model,
                episode=i,
                seed=seed,
                out_path=out_path,
                fps=fps,
                deterministic=args.deterministic,
            )
            results.append(summary)
            print(
                f"ep {i} seed {seed} steps={summary['steps']} "
                f"reward={summary['total_reward']:+.1f} "
                f"success={summary['is_success']} "
                f"term={summary['termination_reason']} -> {out_path.name}"
            )
    finally:
        env.close()

    summary_path = out_dir / "summary.json"
    successes = [r["is_success"] for r in results]
    payload = {
        "mode": (
            "nominal" if args.nominal
            else f"expert_{args.expert_planner}" if args.expert
            else "clone" if args.clone
            else f"event_action_{args.event_action_intervention}" if event_action
            else "event_geometry_blank" if args.event_geometry_blank
            else "event_geometry"
            if event_geometry else "policy"
        ),
        "checkpoint": args.checkpoint or args.event_action_checkpoint,
        "rollout_config": args.rollout_config,
        "seed0": args.seed0,
        "seeds": [int(seed) for seed in (args.seeds or [])],
        "episodes": len(results),
        "success_rate": (float(np.mean(successes)) if successes else 0.0),
        "results": results,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    print(
        f"\n{len(results)} episodes, success_rate={payload['success_rate']:.2f}\n"
        f"videos + summary in {out_dir}"
    )


if __name__ == "__main__":
    main()
