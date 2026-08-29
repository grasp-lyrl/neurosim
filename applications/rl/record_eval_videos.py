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
    p.add_argument("--clone", type=str, default=None,
                   help="Behaviour-cloned .pt to fly; no checkpoint needed")
    p.add_argument("--clone-downsample", type=int, default=4)
    p.add_argument("--vecnormalize", type=str, default=None)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument(
        "--seed0",
        type=int,
        required=True,
        help="First episode seed; episode i uses seed0 + i. Vary between runs.",
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

        self._torch = torch
        self._downsample = int(downsample)
        self._device = str(cfg["ppo"].get("device", "cuda:0"))
        model, _ = build_clone_net(cfg, env, self._downsample)
        payload = torch.load(path, map_location=self._device)
        model.load_state_dict(payload["model"])
        self._model = model.to(self._device).eval()

    def predict(self, obs, deterministic: bool = True):
        from train_velocity_dodge_bc import pool_events  # noqa: PLC0415

        batch = {}
        for key, value in obs.items():
            array = np.asarray(value)
            if key == "events":
                array = pool_events(array, self._downsample)
            batch[key] = self._torch.from_numpy(array[None]).to(self._device)
        with self._torch.no_grad():
            return self._model(batch).cpu().numpy()[0], None


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
) -> list[str]:
    lines = [
        f"ep {episode} seed {seed} step {step} t={sim_time:5.2f}s",
        f"action {_fmt_vec(action)}",
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
    backend = env.sim.visual_backend

    total_reward = 0.0
    step = 0
    min_obstacle_distance = np.inf
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

            terms = info.get("reward_terms", {}) or {}
            dist = terms.get("min_obstacle_distance", np.inf)
            if np.isfinite(dist):
                min_obstacle_distance = min(min_obstacle_distance, float(dist))

            rgb = _to_uint8_rgb(backend.render_color(VIDEO_RGB_UUID))
            event_rgb = env._event_manager.to_rgb(env._event_manager.observation())

            # What the policy is actually commanding, in the same body frame
            # the camera looks along, plus where the nearest threat is. Seen
            # together these answer the question the numbers keep dodging:
            # is the correction pointed away from the obstacle, or not?
            delta_body = obstacle_body = None
            try:
                delta_body = np.asarray(
                    env._task.split_action(np.asarray(action, dtype=np.float32))[1],
                    dtype=np.float64,
                ).reshape(-1)[:3]
            except Exception:
                delta_body = None
            try:
                rows = env._obstacle_relative_states(env.sim.dynamics.state) or []
                if rows:
                    from neurosim.core.coord_trans import rotate_vector_by_quat

                    obstacle_body = rotate_vector_by_quat(
                        np.asarray(rows[0]["rel_pos"], dtype=np.float64),
                        np.asarray(env.sim.dynamics.state["q"], dtype=np.float64),
                        inverse=True,
                    )
            except Exception:
                obstacle_body = None

            writer.append_data(
                compose_frame(
                    rgb,
                    event_rgb,
                    episode_overlay_lines(
                        episode=episode,
                        seed=seed,
                        step=step,
                        sim_time=float(info.get("time", 0.0)),
                        action=action,
                        reward=float(reward),
                        total_reward=total_reward,
                        terms=terms,
                    ),
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

    def __init__(self, env):
        self._env = env

    def reset(self):
        # expert_for_task, not a bare LocalTrajectoryExpert: the default
        # config carries a hardcoded 1.5/1.0 speed envelope, while the task
        # derives its own from offset_rate_limits_mps (now 0.6). Constructing
        # the default here would record an expert planning dodges the
        # configured vehicle cannot fly, so the video would not show the
        # oracle that the metrics describe.
        from evaluate_velocity_dodge_oracle import expert_for_task

        self._env._trajectory_dodge_expert = expert_for_task(self._env._task)
        self._env._trajectory_expert_plans = 0
        self._env._trajectory_expert_failed_plans = 0
        self._env._trajectory_expert_plan_diagnostics = []

    def predict(self, obs, deterministic: bool = True):
        from evaluate_velocity_dodge_oracle import oracle_action

        return oracle_action(self._env), None


def main() -> None:
    args = parse_args()
    if sum((bool(args.nominal), bool(args.expert), bool(args.checkpoint),
            bool(args.clone))) != 1:
        raise SystemExit("pass exactly one of --checkpoint, --nominal, or --expert")

    cfg = load_rollout_config(args.rollout_config)
    env_config = inject_rgb_sensor(cfg["env"])
    env_config["enable_visualization"] = False
    env_config["deterministic_obstacle_aim"] = bool(args.deterministic_obstacle_aim)
    if args.obstacles is not None:
        apply_obstacle_override(env_config, args.obstacles)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cls = env_class_for_task(env_config["task"]["name"])
    env = env_cls(env_config=env_config, train=False)

    # One video frame per policy step, so real-time playback means matching
    # the *policy's* decision rate, not the inner controller rate: each
    # env.step() advances policy_decimation * steps_per_action world steps
    # before a frame is written. Omitting policy_decimation here understated
    # the elapsed sim time per frame by that factor, so at decimation=2
    # playback ran 2x too fast.
    policy_hz = env.sim.config.world_rate / (env.policy_decimation * env.steps_per_action)
    fps = int(args.fps) if args.fps else max(int(round(policy_hz)), 1)
    print(f"recording at {fps} fps (policy runs at {policy_hz:.1f} Hz -> real-time)")

    model = None
    if args.clone:
        model = _ClonePolicy(args.clone, cfg, env, args.clone_downsample)
    elif args.expert:
        model = _TrajectoryExpertPolicy(env)
    elif not args.nominal:
        loaded, obs_norm = load_model(args, cfg, env)
        model = _NormalizedPolicy(loaded, obs_norm)

    results = []
    try:
        for i in range(args.episodes):
            seed = int(args.seed0) + i
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
            else "expert" if args.expert
            else "clone" if args.clone
            else "policy"
        ),
        "checkpoint": args.checkpoint,
        "rollout_config": args.rollout_config,
        "seed0": args.seed0,
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
