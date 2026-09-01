"""Evaluate reproducible blind-action baselines on velocity-dodge tasks."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from neurosim.rl import env_class_for_task

from train_sb3 import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument(
        "--arm", choices=("control", "constant", "weave", "random"), required=True
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=9001)
    parser.add_argument("--amplitude", type=float, default=0.8)
    parser.add_argument("--axis", type=int, default=1)
    parser.add_argument("--frequency-hz", type=float, default=0.5)
    parser.add_argument("--hold-s", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def arm_label(args: argparse.Namespace) -> str:
    if args.arm == "control":
        return "control"
    if args.arm == "constant":
        return f"const+{args.amplitude:.2f}"
    if args.arm == "weave":
        return f"weave{args.amplitude:.1f}@{args.frequency_hz:.1f}Hz"
    return f"random{args.amplitude:.1f}@{args.hold_s:.2f}s"


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.amplitude < 0.0 or args.amplitude > 1.0:
        raise ValueError("amplitude must lie in [0, 1]")
    if args.hold_s <= 0.0:
        raise ValueError("hold-s must be positive")

    cfg = load_experiment_config(args.experiment_config)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    env = env_class_for_task(env_cfg["task"]["name"])(
        env_config=env_cfg, train=False
    )
    if not 0 <= args.axis < int(np.prod(env.action_space.shape)):
        raise ValueError(f"axis {args.axis} outside action shape {env.action_space.shape}")

    rows: list[dict] = []
    label = arm_label(args)
    try:
        for episode in range(args.episodes):
            seed = args.seed0 + episode
            obs, _ = env.reset(seed=seed)
            del obs
            random_generator = np.random.default_rng(seed)
            random_bucket = -1
            held_random = np.zeros(env.action_space.shape, dtype=np.float32)
            reward_sum = 0.0
            action_energy_sum = 0.0
            steps = 0
            final_info: dict = {}

            while True:
                sim_time = float(env.sim.time)
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                if args.arm == "constant":
                    action[args.axis] = args.amplitude
                elif args.arm == "weave":
                    action[args.axis] = args.amplitude * np.sin(
                        2.0 * np.pi * args.frequency_hz * sim_time
                    )
                elif args.arm == "random":
                    bucket = int(np.floor((sim_time + 1e-9) / args.hold_s))
                    if bucket != random_bucket:
                        held_random = random_generator.uniform(
                            -args.amplitude,
                            args.amplitude,
                            size=env.action_space.shape,
                        ).astype(np.float32)
                        random_bucket = bucket
                    action = held_random.copy()

                _, reward, terminated, truncated, final_info = env.step(action)
                reward_sum += float(reward)
                action_energy_sum += float(np.mean(np.square(action)))
                steps += 1
                if terminated or truncated:
                    break

            terms = final_info.get("reward_terms", {}) or {}
            row = {
                "episode": episode,
                "seed": seed,
                "reward": reward_sum,
                "steps": steps,
                "success": bool(final_info.get("is_success", False)),
                "termination_reason": final_info.get(
                    "termination_reason", "timeout"
                ),
                "action_energy": action_energy_sum / max(steps, 1),
                "encounters_total": int(terms.get("encounters_total", 0)),
                "encounters_cleared": int(terms.get("encounters_cleared", 0)),
                "min_clearance": float(terms.get("episode_min_clearance", np.inf)),
            }
            rows.append(row)
            print(row, flush=True)
    finally:
        env.close()

    total_steps = sum(row["steps"] for row in rows)
    total_encounters = sum(row["encounters_total"] for row in rows)
    total_cleared = sum(row["encounters_cleared"] for row in rows)
    payload = {
        "experiment_config": args.experiment_config,
        "arm": label,
        "episodes": len(rows),
        "seed0": args.seed0,
        "mean_return": float(np.mean([row["reward"] for row in rows])),
        "return_per_step": sum(row["reward"] for row in rows)
        / max(total_steps, 1),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_action_energy": float(
            np.mean([row["action_energy"] for row in rows])
        ),
        "encounter_clear_rate": total_cleared / max(total_encounters, 1),
        "encounters_total": total_encounters,
        "encounters_cleared": total_cleared,
        "termination_reasons": dict(
            Counter(row["termination_reason"] for row in rows)
        ),
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(
        f"{label:>18} {payload['mean_return']:+9.1f} "
        f"success={payload['success_rate']:.1%} "
        f"enc_clear={payload['encounter_clear_rate']:.1%} "
        f"terminations={payload['termination_reasons']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
