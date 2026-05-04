"""Evaluate the reactive_dodge nominal SE3 tracker with zero residual action.

This is a fast sanity check for the residual-control setup: with dynamic
obstacles disabled, zero action should leave the SE3 + minsnap stack tracking
without scene crashes or tracking failures.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neurosim.rl import NeurosimRLEnv


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_eval_config(path: str | Path) -> dict[str, Any]:
    cfg = _load_yaml(path)
    if "experiment_config" in cfg:
        exp_path = Path(cfg["experiment_config"])
        if not exp_path.is_absolute():
            exp_path = Path.cwd() / exp_path
        exp_cfg = _load_yaml(exp_path)
        cfg["env"] = copy.deepcopy(exp_cfg["env"])
        _deep_update(cfg, cfg.get("overrides", {}))
    if "env" not in cfg:
        raise ValueError("Nominal eval config must define env or experiment_config")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="applications/rl/configs/reactive_dodge_nominal_eval.yaml",
        help="Nominal evaluation YAML config",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_eval_config(args.config)
    episodes = int(
        args.episodes if args.episodes is not None else cfg.get("episodes", 5)
    )
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    env_cfg = copy.deepcopy(cfg["env"])
    if cfg.get("disable_obstacles", True):
        env_cfg.setdefault("visual_backend", {}).setdefault("dynamic_obstacles", {})[
            "enabled"
        ] = False
    env_cfg["task"]["config"]["require_obstacle_encounter"] = False

    env = NeurosimRLEnv(env_config=env_cfg, train=False)
    results: list[dict[str, Any]] = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            _ = obs
            total_reward = 0.0
            terminated = truncated = False
            info: dict[str, Any] = {}

            while not (terminated or truncated):
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

            metrics = info.get("task_metrics", {})
            row = {
                "episode": ep + 1,
                "seed": seed + ep,
                "reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
                "success": bool(info.get("is_success", False)),
                "termination_reason": info.get("termination_reason", "none"),
                "pos_error": info.get("reward_terms", {}).get("pos_error", np.nan),
                "vel_error": info.get("reward_terms", {}).get("vel_error", np.nan),
                "correction_energy": metrics.get("correction_energy", np.nan),
            }
            results.append(row)
            print(
                "episode={episode} seed={seed} reward={reward:.3f} "
                "success={success} termination={termination_reason} "
                "pos_error={pos_error:.3f} vel_error={vel_error:.3f}".format(**row)
            )
    finally:
        env.close()

    successes = sum(1 for row in results if row["success"])
    terminations = [row["termination_reason"] for row in results]
    print(
        f"summary episodes={episodes} success_rate={successes / max(episodes, 1):.3f} "
        f"terminations={terminations}"
    )


if __name__ == "__main__":
    main()
