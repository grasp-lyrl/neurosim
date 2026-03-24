"""Run a trained SB3 PPO policy in neurosim.

Usage:
    python run_policy.py --checkpoint outputs/rl/hover_sb3/best_model.zip \
                         --rollout-config applications/rl/configs/hover_sb3_rollout.yaml

If the checkpoint was trained with VecNormalize, place ``vecnormalize.pkl``
next to the model or pass ``--vecnormalize`` explicitly.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from neurosim.rl import NeurosimRLEnv


def load_rollout_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Rollout config must be a YAML mapping")

    required_keys = [
        "settings",
        "obs_mode",
        "episode_seconds",
        "body_rate_limit",
        "event_downsample_factor",
        "init_speed_min",
        "init_speed_max",
        "enable_navigable_check",
        "event_representation",
        "event_log_compression",
        "device",
        "deterministic",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required rollout config keys: {missing}")

    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a trained SB3 PPO policy in neurosim")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .zip model")
    p.add_argument(
        "--rollout-config",
        type=str,
        default=None,
        help="YAML file containing rollout/env parameters",
    )
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--visualization-rrd-path", type=str, default=None)
    p.add_argument(
        "--debug-save-events-png",
        action="store_true",
        help="Save periodic event-frame PNGs for headless debugging",
    )
    p.add_argument(
        "--debug-png-dir",
        type=str,
        default="outputs/rl/hover_sb3/debug_events_rollout",
    )
    p.add_argument("--debug-save-every-n-steps", type=int, default=25)
    p.add_argument("--debug-accumulate-n-steps", type=int, default=10)
    p.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help="Path to vecnormalize.pkl (auto-detected from checkpoint dir if omitted)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_rollout_config(args.rollout_config)
    init_speed_range = (float(cfg["init_speed_min"]), float(cfg["init_speed_max"]))

    def _make_env():
        return NeurosimRLEnv(
            settings=str(cfg["settings"]),
            obs_mode=str(cfg["obs_mode"]),
            episode_seconds=float(cfg["episode_seconds"]),
            body_rate_limit=float(cfg["body_rate_limit"]),
            init_speed_range=init_speed_range,
            event_downsample_factor=int(cfg["event_downsample_factor"]),
            enable_navigable_check=bool(cfg["enable_navigable_check"]),
            enable_visualization=args.visualize,
            visualization_rrd_path=args.visualization_rrd_path,
            debug_save_events_png=args.debug_save_events_png,
            debug_png_dir=args.debug_png_dir,
            debug_save_every_n_steps=args.debug_save_every_n_steps,
            debug_accumulate_n_steps=args.debug_accumulate_n_steps,
            event_representation=str(cfg["event_representation"]),
            event_log_compression=cfg["event_log_compression"],
        )

    vec_env = DummyVecEnv([_make_env])

    # Restore observation / reward normalizer if available
    vecnorm_path = args.vecnormalize
    if vecnorm_path is None:
        candidate = Path(args.checkpoint).parent / "vecnormalize.pkl"
        if candidate.exists():
            vecnorm_path = str(candidate)

    if vecnorm_path is not None:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize from {vecnorm_path}")

    model = PPO.load(args.checkpoint, env=vec_env, device=str(cfg["device"]))

    for ep in range(1, args.episodes + 1):
        obs = vec_env.reset()
        total_reward = 0.0
        step = 0

        while True:
            step += 1
            action, _ = model.predict(obs, deterministic=bool(cfg["deterministic"]))
            obs, reward, done, infos = vec_env.step(action)
            total_reward += float(reward[0])

            if done[0]:
                info = infos[0]
                print(
                    f"episode={ep} steps={step} total_reward={total_reward:.3f} "
                    f"sim_time={info.get('time', 0):.3f} "
                    f"is_success={info.get('is_success', False)} "
                    f"termination={info.get('termination_reason', 'none')}"
                )
                break

    vec_env.close()


if __name__ == "__main__":
    main()
