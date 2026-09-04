"""Collect compact causal event-tracker features under the sampling-MPC oracle.

Unlike the raw-event HDF5 collector, this stores about 140 floats per frame,
making it practical to scale action-head BC while perception remains frozen.
The actor inputs are computed from events online; privileged simulator state is
used only by sampling MPC to create the action label.
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

from evaluate_event_geometry_controller import EventObstacleTracker  # noqa: E402
from evaluate_velocity_dodge_oracle import (  # noqa: E402
    effective_expert_action,
    expert_for_task,
    mpc_expert_for_env,
    mpc_oracle_action,
)
from train_event_tracker_action_head import (  # noqa: E402
    EventActionPolicy,
    INPUT_DIM,
    causal_visibility_mask,
    tracker_input,
)
from train_sb3 import load_experiment_config  # noqa: E402
from neurosim.rl import env_class_for_task  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--spatial-checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--seed0", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--visibility-threshold", type=float, default=0.7)
    parser.add_argument("--confirm-steps", type=int, default=2)
    parser.add_argument("--reset-absence-steps", type=int, default=3)
    parser.add_argument(
        "--zero-action-labels",
        action="store_true",
        help=(
            "Collect nominal ego/scene-motion features with an all-zero action "
            "label. Intended for no-obstacle negative-control trajectories."
        ),
    )
    parser.add_argument(
        "--rollout-checkpoint",
        help=(
            "Optional event-policy checkpoint used to drive zero-label negative "
            "collection. This captures the closed-loop event distribution after "
            "a false dodge while retaining zero supervision."
        ),
    )
    return parser.parse_args()


def save(output: Path, summary: Path, args, episodes, results):
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": INPUT_DIM,
            "spatial_checkpoint": args.spatial_checkpoint,
            "temporal_checkpoint": args.temporal_checkpoint,
            "visibility_threshold": args.visibility_threshold,
            "confirm_steps": args.confirm_steps,
            "reset_absence_steps": args.reset_absence_steps,
            "zero_action_labels": bool(args.zero_action_labels),
            "rollout_checkpoint": args.rollout_checkpoint,
            "episodes": episodes,
        },
        output,
    )
    payload = {
        "zero_action_labels": bool(args.zero_action_labels),
        "rollout_checkpoint": args.rollout_checkpoint,
        "attempts": len(results),
        "accepted": len(episodes),
        "success_rate": float(np.mean([row["success"] for row in results]))
        if results
        else 0.0,
        "samples": int(sum(len(row["inputs"]) for row in episodes)),
        "active_original": int(
            sum(
                (np.linalg.norm(row["original_actions"], axis=1) > 0.1).sum()
                for row in episodes
            )
        ),
        "active_visible": int(
            sum(
                (np.linalg.norm(row["actions"][:, 1:], axis=1) > 0.1).sum()
                for row in episodes
            )
        ),
        "results": results,
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    cfg = load_experiment_config(args.experiment_config)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    env_cfg.setdefault("visual_backend", {})["gpu_id"] = args.sim_gpu_id
    env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
    tracker = EventObstacleTracker(
        args.spatial_checkpoint, args.temporal_checkpoint, device=args.device
    )
    if args.rollout_checkpoint and not args.zero_action_labels:
        raise ValueError("--rollout-checkpoint requires --zero-action-labels")
    rollout_policy = (
        EventActionPolicy(args.rollout_checkpoint, device=args.device)
        if args.rollout_checkpoint
        else None
    )
    output, summary = Path(args.output), Path(args.summary)
    episodes, results = [], []
    for episode_index in range(args.attempts):
        seed = args.seed0 + episode_index
        observation, _ = env.reset(seed=seed)
        tracker.reset()
        if rollout_policy is not None:
            rollout_policy.reset()
        if not args.zero_action_labels:
            env._trajectory_dodge_expert = expert_for_task(env._task)
            env._trajectory_expert_plans = 0
            env._trajectory_expert_failed_plans = 0
            env._trajectory_expert_plan_diagnostics = []
            env._mpc_slack_replans = 0
            env._mpc_visibility_released_ids = set()
            env._mpc_visibility_release_events = []
            env._receding_horizon_dodge_expert = mpc_expert_for_env(env)
        inputs, actions, probabilities, activity = [], [], [], []
        plan_failure = False
        commitments = 0
        steps = 0
        while True:
            prediction = tracker.step(observation["events"])
            inputs.append(tracker_input(prediction, observation["state"]))
            probabilities.append(prediction.probability)
            activity.append(float(np.any(np.abs(observation["events"]) > 1e-6)))
            if args.zero_action_labels:
                expert_action = np.zeros(env.action_space.shape, dtype=np.float32)
                actions.append(expert_action.copy())
                rollout_action = (
                    rollout_policy.predict(observation)[0]
                    if rollout_policy is not None
                    else expert_action
                )
                if rollout_policy is not None:
                    commitments += int(rollout_policy.last_committed)
            else:
                expert_action = mpc_oracle_action(env)
                plan_failure |= bool(env._trajectory_expert_frame_unlabelled)
                actions.append(effective_expert_action(env, expert_action))
                rollout_action = expert_action
            observation, _, terminated, truncated, info = env.step(rollout_action)
            steps += 1
            if terminated or truncated:
                terms = info.get("reward_terms", {}) or {}
                success = bool(info.get("is_success", False))
                row = {
                    "episode": episode_index,
                    "seed": seed,
                    "success": success,
                    "termination_reason": info.get("termination_reason", "timeout"),
                    "steps": steps,
                    "rollout_commitments": commitments,
                    "episode_min_clearance": float(
                        terms.get("episode_min_clearance", np.inf)
                    ),
                    "expert_plans": int(
                        getattr(env, "_trajectory_expert_plans", 0)
                    ),
                    "expert_failed_plans": int(
                        getattr(env, "_trajectory_expert_failed_plans", 0)
                    ),
                }
                results.append(row)
                accepted = (
                    truncated
                    if args.zero_action_labels
                    else success and not plan_failure
                )
                if accepted:
                    original = np.asarray(actions, dtype=np.float32)
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
                            "key": f"seed{seed}",
                            "inputs": np.asarray(inputs, dtype=np.float32),
                            "actions": gated,
                            "original_actions": original,
                            "probabilities": np.asarray(
                                probabilities, dtype=np.float32
                            ),
                        }
                    )
                save(output, summary, args, episodes, results)
                print(json.dumps(row), flush=True)
                break
    save(output, summary, args, episodes, results)
    print(f"saved {len(episodes)}/{len(results)} accepted episodes to {output}")
    env.close()


if __name__ == "__main__":
    main()
