"""Collect compact oracle labels on states visited by an event-action policy.

The rollout action is sampled from a mixture of the frozen event actor and
sampling-MPC expert. The actor sees only events and legitimate ego state;
privileged simulator geometry is used solely to query the expert label.
Collisions are intentionally retained because recovery and pre-collision
states are the purpose of DAgger. Frames where MPC has no feasible label are
kept in the recurrent input sequence but masked out of the imitation loss.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_velocity_dodge_oracle import (  # noqa: E402
    effective_expert_action,
    expert_for_task,
    mpc_expert_for_env,
    mpc_oracle_action,
)
from train_event_tracker_action_head import (  # noqa: E402
    INPUT_DIM,
    EventActionPolicy,
    causal_visibility_mask,
)
from train_sb3 import load_experiment_config  # noqa: E402
from neurosim.rl import env_class_for_task  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--rollout-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--seed0", type=int, required=True)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--visibility-threshold", type=float, default=0.7)
    parser.add_argument("--confirm-steps", type=int, default=2)
    parser.add_argument("--reset-absence-steps", type=int, default=3)
    return parser.parse_args()


def choose_rollout_action(expert, learner, *, beta: float, draw: float):
    """Return one DAgger mixture action; factored out for boundary tests."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    return np.asarray(expert if draw < beta else learner, dtype=np.float32)


def save(output: Path, summary: Path, args, episodes, results):
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": INPUT_DIM,
            "collection_type": "event_policy_dagger",
            "rollout_checkpoint": args.rollout_checkpoint,
            "beta": args.beta,
            "visibility_threshold": args.visibility_threshold,
            "confirm_steps": args.confirm_steps,
            "reset_absence_steps": args.reset_absence_steps,
            "episodes": episodes,
        },
        output,
    )
    payload = {
        "attempts": len(results),
        "episodes": len(episodes),
        "beta": args.beta,
        "success_rate": float(np.mean([row["success"] for row in results]))
        if results
        else 0.0,
        "samples": int(sum(len(row["inputs"]) for row in episodes)),
        "valid_labels": int(
            sum(np.asarray(row["label_valid"]).sum() for row in episodes)
        ),
        "active_visible_valid": int(
            sum(
                (
                    (np.linalg.norm(row["actions"][:, 1:], axis=1) > 0.1)
                    & np.asarray(row["label_valid"], dtype=bool)
                ).sum()
                for row in episodes
            )
        ),
        "results": results,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--beta must lie in [0, 1]")
    cfg = load_experiment_config(args.experiment_config)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    env_cfg.setdefault("visual_backend", {})["gpu_id"] = args.sim_gpu_id
    env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
    policy = EventActionPolicy(args.rollout_checkpoint, device=args.device)
    rng = np.random.default_rng(args.seed0)
    output, summary = Path(args.output), Path(args.summary)
    episodes, results = [], []
    try:
        for episode_index in range(args.attempts):
            seed = args.seed0 + episode_index
            observation, _ = env.reset(seed=seed)
            policy.reset()
            env._trajectory_dodge_expert = expert_for_task(env._task)
            env._trajectory_expert_plans = 0
            env._trajectory_expert_failed_plans = 0
            env._trajectory_expert_plan_diagnostics = []
            env._mpc_slack_replans = 0
            env._mpc_visibility_released_ids = set()
            env._mpc_visibility_release_events = []
            env._receding_horizon_dodge_expert = mpc_expert_for_env(env)
            inputs, labels, probabilities, activity, label_valid = [], [], [], [], []
            expert_steps = 0
            steps = 0
            while True:
                learner_action, _ = policy.predict(observation)
                inputs.append(np.asarray(policy.last_input, dtype=np.float32).copy())
                probabilities.append(float(policy.last_prediction.probability))
                events = np.asarray(observation["events"])
                activity.append(float(np.any(np.abs(events) > 1e-6)))

                expert_action = mpc_oracle_action(env)
                valid = not bool(env._trajectory_expert_frame_unlabelled)
                label_valid.append(valid)
                labels.append(effective_expert_action(env, expert_action))
                draw = float(rng.random())
                executed = choose_rollout_action(
                    expert_action, learner_action, beta=args.beta, draw=draw
                )
                expert_steps += int(draw < args.beta)
                observation, _, terminated, truncated, info = env.step(executed)
                steps += 1
                if terminated or truncated:
                    terms = info.get("reward_terms", {}) or {}
                    row = {
                        "episode": episode_index,
                        "seed": seed,
                        "success": bool(info.get("is_success", False)),
                        "termination_reason": info.get("termination_reason", "timeout"),
                        "steps": steps,
                        "encounters_total": int(terms.get("encounters_total", 0)),
                        "encounters_cleared": int(terms.get("encounters_cleared", 0)),
                        "episode_min_clearance": float(
                            terms.get("episode_min_clearance", np.inf)
                        ),
                        "expert_steps": expert_steps,
                        "learner_steps": steps - expert_steps,
                        "valid_labels": int(np.asarray(label_valid).sum()),
                        "expert_failed_plans": int(env._trajectory_expert_failed_plans),
                    }
                    results.append(row)
                    original = np.asarray(labels, dtype=np.float32)
                    visible = causal_visibility_mask(
                        np.asarray(probabilities),
                        np.asarray(activity),
                        threshold=args.visibility_threshold,
                        confirm_steps=args.confirm_steps,
                        reset_absence_steps=args.reset_absence_steps,
                    )
                    gated = original.copy()
                    gated[~visible] = 0.0
                    gated[:, 0] = 0.0
                    episodes.append(
                        {
                            "key": f"dagger:seed{seed}",
                            "inputs": np.asarray(inputs, dtype=np.float32),
                            "actions": gated,
                            "original_actions": original,
                            "probabilities": np.asarray(probabilities, dtype=np.float32),
                            "label_valid": np.asarray(label_valid, dtype=bool),
                        }
                    )
                    save(output, summary, args, episodes, results)
                    print(json.dumps(row), flush=True)
                    break
    finally:
        save(output, summary, args, episodes, results)
        env.close()
    print(f"saved {len(episodes)} DAgger episodes to {output}", flush=True)


if __name__ == "__main__":
    main()
