"""Evaluate a velocity-dodge policy under perception/control ablations."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from neurosim.rl import env_class_for_task
from neurosim.rl.trajectory_dodge_expert import LocalTrajectoryExpert

from evaluate_velocity_dodge_oracle import oracle_action
from train_sb3 import load_experiment_config


def effective_dodge_action(env, action: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the gate and controller-applied residual for diagnostics."""
    gate, residual = env._task.split_action(np.asarray(action, dtype=np.float32))
    return float(gate), np.asarray(residual, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed0", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zero-events", action="store_true")
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument(
        "--oracle-diagnostics",
        action="store_true",
        help="Compare policy actions with the privileged oracle at each state",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.experiment_config)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    if args.no_obstacles:
        env_cfg["visual_backend"]["dynamic_obstacles"]["enabled"] = False
    env = env_class_for_task(env_cfg["task"]["name"])(
        env_config=env_cfg, train=False
    )
    model = PPO.load(args.policy, env=None, device=str(cfg["ppo"]["device"]))

    rows = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed0 + episode)
            env._trajectory_dodge_expert = LocalTrajectoryExpert()
            env._trajectory_expert_plans = 0
            env._trajectory_expert_failed_plans = 0
            env._trajectory_expert_plan_diagnostics = []
            action_norms = []
            gates = []
            threat_steps = 0
            threat_detected_steps = 0
            threat_cosines = []
            quiet_action_norms = []
            peak_position_error = 0.0
            while True:
                policy_obs = obs
                if args.zero_events:
                    if isinstance(obs, dict):
                        policy_obs = dict(obs)
                        policy_obs["events"] = np.zeros_like(obs["events"])
                    else:
                        policy_obs = np.zeros_like(obs)
                action, _ = model.predict(policy_obs, deterministic=True)
                policy_gate, applied_action = effective_dodge_action(env, action)
                gates.append(policy_gate)
                if args.oracle_diagnostics:
                    expert = oracle_action(env)
                    _, applied_expert = effective_dodge_action(env, expert)
                    expert_norm = float(np.linalg.norm(applied_expert))
                    policy_norm = float(np.linalg.norm(applied_action))
                    if expert_norm > 0.1:
                        threat_steps += 1
                        threat_detected_steps += int(policy_norm > 0.1)
                        if policy_norm > 1e-6:
                            threat_cosines.append(
                                float(
                                    np.dot(applied_action, applied_expert)
                                    / (policy_norm * expert_norm)
                                )
                            )
                    else:
                        quiet_action_norms.append(policy_norm)
                action_norms.append(float(np.linalg.norm(applied_action)))
                obs, _, terminated, truncated, info = env.step(action)
                terms = info.get("reward_terms", {}) or {}
                peak_position_error = max(
                    peak_position_error, float(terms.get("pos_error", 0.0))
                )
                if terminated or truncated:
                    row = {
                        "episode": episode,
                        "seed": args.seed0 + episode,
                        "success": bool(info.get("is_success", False)),
                        "termination_reason": info.get("termination_reason", "timeout"),
                        "mean_action_norm": float(np.mean(action_norms)),
                        "peak_action_norm": float(np.max(action_norms)),
                        "mean_gate": float(np.mean(gates)),
                        "peak_position_error": peak_position_error,
                        "threat_steps": threat_steps,
                        "threat_detected_steps": threat_detected_steps,
                        "mean_threat_cosine": (
                            float(np.mean(threat_cosines)) if threat_cosines else None
                        ),
                        "mean_quiet_action_norm": (
                            float(np.mean(quiet_action_norms))
                            if quiet_action_norms
                            else None
                        ),
                    }
                    rows.append(row)
                    print(row, flush=True)
                    break
    finally:
        env.close()

    reasons = Counter(row["termination_reason"] for row in rows)
    payload = {
        "experiment_config": args.experiment_config,
        "policy": args.policy,
        "zero_events": args.zero_events,
        "no_obstacles": args.no_obstacles,
        "oracle_diagnostics": args.oracle_diagnostics,
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "termination_reasons": dict(reasons),
        "mean_action_norm": float(np.mean([row["mean_action_norm"] for row in rows])),
        "mean_gate": float(np.mean([row["mean_gate"] for row in rows])),
        "mean_peak_action_norm": float(
            np.mean([row["peak_action_norm"] for row in rows])
        ),
        "mean_peak_position_error": float(
            np.mean([row["peak_position_error"] for row in rows])
        ),
        "results": rows,
    }
    if args.oracle_diagnostics:
        total_threat = sum(row["threat_steps"] for row in rows)
        payload.update(
            threat_steps=total_threat,
            threat_detection_rate=(
                sum(row["threat_detected_steps"] for row in rows) / total_threat
                if total_threat
                else None
            ),
            mean_threat_cosine=float(
                np.mean(
                    [
                        row["mean_threat_cosine"]
                        for row in rows
                        if row["mean_threat_cosine"] is not None
                    ]
                )
            ) if any(row["mean_threat_cosine"] is not None for row in rows) else None,
            mean_quiet_action_norm=float(
                np.mean(
                    [
                        row["mean_quiet_action_norm"]
                        for row in rows
                        if row["mean_quiet_action_norm"] is not None
                    ]
                )
            ) if any(row["mean_quiet_action_norm"] is not None for row in rows) else None,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}))


if __name__ == "__main__":
    main()
