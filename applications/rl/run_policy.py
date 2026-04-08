"""Run a trained SB3 PPO policy in neurosim.

Usage:
    python run_policy.py --checkpoint outputs/rl/hover_sb3/best_model.zip \
                         --rollout-config applications/rl/configs/hover_sb3_rollout.yaml

If the checkpoint was trained with VecNormalize, place ``vecnormalize.pkl``
next to the model or pass ``--vecnormalize`` explicitly.
"""

import argparse
import copy
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

    required_keys = ["env", "policy"]
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
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help=("Base rollout seed. If set, episode i uses seed + (i-1)"),
    )
    p.add_argument("--visualize", action="store_true")
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
    parent = Path(args.checkpoint).parent

    base_env_config = copy.deepcopy(cfg["env"])
    if args.visualize:
        base_env_config["enable_visualization"] = True

    def _make_env():
        return NeurosimRLEnv(env_config=base_env_config)

    vec_env = DummyVecEnv([_make_env])

    # Restore observation / reward normalizer if available
    vecnorm_path = args.vecnormalize
    if vecnorm_path is None:
        candidate = parent / "vecnormalize.pkl"
        if candidate.exists():
            vecnorm_path = str(candidate)

    if vecnorm_path is not None:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize from {vecnorm_path}")

    model = PPO.load(args.checkpoint, env=vec_env, device=str(cfg["policy"]["device"]))

    for ep in range(1, args.episodes + 1):
        episode_seed = int(args.seed) + (ep - 1)
        vec_env.seed(episode_seed)
        print(f"episode={ep} seed={episode_seed}")

        obs = vec_env.reset()
        total_reward = 0.0
        step = 0

        while True:
            step += 1
            action, _ = model.predict(
                obs,
                deterministic=bool(cfg["policy"]["deterministic"]),
            )
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
