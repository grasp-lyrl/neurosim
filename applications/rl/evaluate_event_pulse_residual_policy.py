"""Matched evaluation for a PPO-corrected frozen event-BC pulse policy."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_sb3 import load_experiment_config, make_env  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--seed0", type=int, default=47001)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=1)
    parser.add_argument(
        "--fixed-hold",
        action="store_true",
        help="Disable opt-in inbound-conditioned release for an exact A/B arm.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.experiment_config)
    env_cfg = copy.deepcopy(cfg["env"])
    if args.fixed_hold:
        env_cfg["event_pulse_residual"]["pulse_kwargs"] = {}
    env = make_env(
        env_cfg,
        train=False,
        gpu_id=args.sim_gpu_id,
    )()
    normalizer_owner = DummyVecEnv([lambda: env])
    normalizer = VecNormalize.load(args.vecnormalize, normalizer_owner)
    normalizer.training = False
    normalizer.norm_reward = False
    model = PPO.load(args.checkpoint, device=args.device)

    seeds = args.seeds or [args.seed0 + i for i in range(args.episodes)]
    rows = []
    try:
        for episode, seed in enumerate(seeds):
            observation, _ = env.reset(seed=int(seed))
            raw_env = env.unwrapped
            previous_velocity = np.asarray(
                raw_env.sim.dynamics.state["v"], dtype=np.float64
            ).copy()
            dt = (
                raw_env.policy_decimation
                * raw_env.steps_per_action
                / raw_env.sim.config.world_rate
            )
            total_reward = 0.0
            steps = 0
            peak_acceleration = 0.0
            enabled_steps = 0
            adjusted_trigger_delta = []
            active_steps = 0
            eligible_inbound = []
            release_probe_thresholds = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
            release_probe_current = {threshold: 0 for threshold in release_probe_thresholds}
            release_probe_max = {threshold: 0 for threshold in release_probe_thresholds}
            while True:
                normalized = normalizer.normalize_obs(observation)
                action, _ = model.predict(normalized, deterministic=True)
                event_policy = env.get_wrapper_attr("policy")
                controller_before = event_policy.pulse_controller
                eligible = bool(
                    controller_before.pulse_step is not None
                    # Probe the proposed eight-step release point even in
                    # the fixed-hold A/B arm, whose configured minimum is 20.
                    and controller_before.pulse_step >= 8
                    and event_policy.last_event_active
                    and event_policy.last_prediction.probability >= 0.7
                )
                inbound_probability = float(
                    event_policy.last_prediction.inbound_probability
                )
                if eligible:
                    eligible_inbound.append(inbound_probability)
                for threshold in release_probe_thresholds:
                    release_probe_current[threshold] = (
                        release_probe_current[threshold] + 1
                        if eligible and inbound_probability <= threshold
                        else 0
                    )
                    release_probe_max[threshold] = max(
                        release_probe_max[threshold],
                        release_probe_current[threshold],
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                pulse = info["event_pulse_residual"]
                enabled_steps += int(bool(pulse["gate"]))
                if pulse["gate"]:
                    adjusted_trigger_delta.append(
                        abs(
                            float(pulse["adjusted_gate_probability"])
                            - float(pulse["base_gate_probability"])
                        )
                    )
                active_steps += int(
                    np.linalg.norm(pulse["executed_action"]) > 1e-6
                )
                total_reward += float(reward)
                steps += 1
                velocity = np.asarray(
                    raw_env.sim.dynamics.state["v"], dtype=np.float64
                ).copy()
                peak_acceleration = max(
                    peak_acceleration,
                    float(np.linalg.norm(velocity - previous_velocity) / dt),
                )
                previous_velocity = velocity
                if terminated or truncated:
                    terms = info.get("reward_terms", {}) or {}
                    controller = env.get_wrapper_attr("policy").pulse_controller
                    row = {
                        "episode": episode,
                        "seed": int(seed),
                        "success": bool(info.get("is_success", False)),
                        "termination_reason": info.get(
                            "termination_reason", "timeout"
                        ),
                        "steps": steps,
                        "reward": total_reward,
                        "commitments": int(controller.commitments),
                        "early_releases": int(controller.early_releases),
                        "early_release_pulse_steps": list(
                            controller.early_release_pulse_steps
                        ),
                        "renewals": int(controller.renewals),
                        "renewal_pulse_steps": list(
                            controller.renewal_pulse_steps
                        ),
                        "active_steps": active_steps,
                        "adjustment_enabled_fraction": enabled_steps / max(steps, 1),
                        "mean_abs_trigger_probability_delta": (
                            float(np.mean(adjusted_trigger_delta))
                            if adjusted_trigger_delta
                            else 0.0
                        ),
                        "eligible_inbound_min": (
                            min(eligible_inbound) if eligible_inbound else None
                        ),
                        "eligible_inbound_median": (
                            float(np.median(eligible_inbound))
                            if eligible_inbound
                            else None
                        ),
                        "max_consecutive_eligible_below": {
                            str(threshold): release_probe_max[threshold]
                            for threshold in release_probe_thresholds
                        },
                        "encounters_total": int(terms.get("encounters_total", 0)),
                        "encounters_cleared": int(
                            terms.get("encounters_cleared", 0)
                        ),
                        "episode_min_clearance": float(
                            terms.get("episode_min_clearance", np.inf)
                        ),
                        "peak_acceleration_mps2": peak_acceleration,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    break
    finally:
        normalizer_owner.close()

    encounters = sum(row["encounters_total"] for row in rows)
    cleared = sum(row["encounters_cleared"] for row in rows)
    payload = {
        "experiment_config": args.experiment_config,
        "checkpoint": args.checkpoint,
        "vecnormalize": args.vecnormalize,
        "fixed_hold": bool(args.fixed_hold),
        "seeds": [int(seed) for seed in seeds],
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "termination_reasons": dict(
            Counter(row["termination_reason"] for row in rows)
        ),
        "encounters_total": encounters,
        "encounters_cleared": cleared,
        "encounter_clear_rate": cleared / max(encounters, 1),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_active_steps": float(np.mean([row["active_steps"] for row in rows])),
        "commitments": sum(row["commitments"] for row in rows),
        "early_releases": sum(row["early_releases"] for row in rows),
        "renewals": sum(row["renewals"] for row in rows),
        "max_peak_acceleration_mps2": max(
            row["peak_acceleration_mps2"] for row in rows
        ),
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
