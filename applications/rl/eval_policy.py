"""Evaluate and compare a PPO policy against a zero-action classical trajectory baseline.

Usage:
    python eval_policy.py \
        --checkpoint outputs/rl/<run_name>/best_model.zip \
        --rollout-config applications/rl/configs/dodge_sb3_combined_rollout.yaml \
        --episodes 10 \
        --seed 42
"""

import copy
import argparse
from pathlib import Path
from typing import Any
from collections import defaultdict
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from neurosim.rl import NeurosimRLEnv
import yaml

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
    p = argparse.ArgumentParser(description="Evaluate policy vs classical baseline")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained SB3 model .zip file",
    )
    p.add_argument(
        "--rollout-config",
        type=str,
        required=True,
        help="YAML file containing rollout/env parameters",
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base rollout seed. If set, episode i uses seed + (i-1)",
    )
    p.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help="Path to vecnormalize.pkl (auto-detected from checkpoint dir if omitted)",
    )
    return p.parse_args()


def run_evaluation_pass(vec_env, model, episodes, base_seed, use_policy=True):
    stats = {
        "success": 0,
        "obstacle_collision": 0,
        "not_navigable": 0,
        "out_of_bounds": 0,
        "max_tracking_error_exceeded": 0,
        "other": 0,
        "rewards": [],
    }

    for ep in range(1, episodes + 1):
        episode_seed = int(base_seed) + (ep - 1)
        vec_env.seed(episode_seed)
        
        obs = vec_env.reset()
        total_reward = 0.0

        while True:
            if use_policy:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Classical mode: zero velocity correction
                # Note: DummyVecEnv wraps the action space so it expects batched actions
                action = np.zeros((1,) + vec_env.action_space.shape)

            obs, reward, done, infos = vec_env.step(action)
            total_reward += float(reward[0])

            if done[0]:
                info = infos[0]
                term_reason = info.get("termination_reason", "none")
                
                if info.get("is_success", False) or term_reason == "success":
                    stats["success"] += 1
                elif term_reason in stats:
                    stats[term_reason] += 1
                else:
                    stats["other"] += 1
                
                stats["rewards"].append(total_reward)
                break

    return stats


def print_comparison(episodes, classical_stats, policy_stats):
    def pct(count):
        return f"{(count / episodes) * 100:>5.1f}%"

    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY ({episodes} episodes)")
    print("="*60)
    print(f"{'Metric':<30} | {'Classical':<12} | {'Policy':<12}")
    print("-" * 60)
    
    print(f"{'Success Rate':<30} | {pct(classical_stats['success']):<12} | {pct(policy_stats['success']):<12}")
    print(f"{'Crash: Dynamic Obstacles':<30} | {pct(classical_stats['obstacle_collision']):<12} | {pct(policy_stats['obstacle_collision']):<12}")
    print(f"{'Crash: Static Walls':<30} | {pct(classical_stats['not_navigable']):<12} | {pct(policy_stats['not_navigable']):<12}")
    print(f"{'Crash: Out of Bounds':<30} | {pct(classical_stats['out_of_bounds']):<12} | {pct(policy_stats['out_of_bounds']):<12}")
    print(f"{'Crash: Tracking Error':<30} | {pct(classical_stats['max_tracking_error_exceeded']):<12} | {pct(policy_stats['max_tracking_error_exceeded']):<12}")
    
    mean_rew_class = np.mean(classical_stats['rewards']) if classical_stats['rewards'] else 0.0
    mean_rew_pol = np.mean(policy_stats['rewards']) if policy_stats['rewards'] else 0.0
    print(f"{'Average Reward':<30} | {mean_rew_class:<12.1f} | {mean_rew_pol:<12.1f}")
    print("="*60 + "\n")


def main():
    args = parse_args()
    cfg = load_rollout_config(args.rollout_config)
    parent = Path(args.checkpoint).parent

    base_env_config = copy.deepcopy(cfg["env"])
    base_env_config["enable_visualization"] = False

    def _make_env():
        return NeurosimRLEnv(env_config=base_env_config, train=False)

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

    print(f"\nRunning CLASSICAL evaluation pass...")
    classical_stats = run_evaluation_pass(vec_env, model, args.episodes, args.seed, use_policy=False)

    print(f"Running POLICY evaluation pass...")
    policy_stats = run_evaluation_pass(vec_env, model, args.episodes, args.seed, use_policy=True)

    print_comparison(args.episodes, classical_stats, policy_stats)
    vec_env.close()


if __name__ == "__main__":
    main()
