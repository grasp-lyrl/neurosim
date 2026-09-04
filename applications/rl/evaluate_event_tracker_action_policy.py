"""Closed-loop evaluation for the frozen-tracker recurrent event actor."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_event_tracker_action_head import EventActionPolicy  # noqa: E402
from train_sb3 import load_experiment_config  # noqa: E402
from neurosim.rl import env_class_for_task  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--intervention", choices=("real", "blank", "reverse_history"), default="real"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--seed0", type=int, default=36001)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument(
        "--gate-threshold",
        type=float,
        help="Override the checkpoint deployment trigger threshold.",
    )
    parser.add_argument("--structured-pulse", action="store_true")
    parser.add_argument("--pulse-confirm-steps", type=int, default=2)
    parser.add_argument("--pulse-hold-steps", type=int, default=20)
    parser.add_argument("--pulse-ramp-steps", type=int, default=3)
    parser.add_argument("--pulse-refractory-absence-steps", type=int, default=8)
    parser.add_argument("--pulse-magnitude", type=float, default=1.0)
    parser.add_argument("--track-presence-threshold", type=float, default=0.7)
    parser.add_argument("--minimum-inbound-probability", type=float)
    return parser.parse_args()


def write_results(path, args, rows):
    encounters = sum(int(row["encounters_total"]) for row in rows)
    cleared = sum(int(row["encounters_cleared"]) for row in rows)
    payload = {
        "experiment_config": args.experiment_config,
        "checkpoint": args.checkpoint,
        "intervention": args.intervention,
        "gate_threshold": args.gate_threshold,
        "track_presence_threshold": args.track_presence_threshold,
        "minimum_inbound_probability": args.minimum_inbound_probability,
        "structured_pulse": args.structured_pulse,
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "mean_steps": float(np.mean([row["steps"] for row in rows])) if rows else 0.0,
        "encounters_total": encounters,
        "encounters_cleared": cleared,
        "encounter_clear_rate": cleared / max(encounters, 1),
        "max_peak_acceleration_mps2": max(
            (float(row["peak_acceleration_mps2"]) for row in rows), default=0.0
        ),
        "results": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    cfg = load_experiment_config(args.experiment_config)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    env_cfg.setdefault("visual_backend", {})["gpu_id"] = args.sim_gpu_id
    env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
    pulse_kwargs = {
        "confirm_steps": args.pulse_confirm_steps,
        "hold_steps": args.pulse_hold_steps,
        "ramp_steps": args.pulse_ramp_steps,
        "refractory_absence_steps": args.pulse_refractory_absence_steps,
        "magnitude": args.pulse_magnitude,
    }
    policy = EventActionPolicy(
        args.checkpoint,
        device=args.device,
        intervention=args.intervention,
        structured_pulse=True if args.structured_pulse else None,
        pulse_kwargs=pulse_kwargs if args.structured_pulse else None,
        track_presence_threshold=args.track_presence_threshold,
        minimum_inbound_probability=args.minimum_inbound_probability,
    )
    if args.gate_threshold is not None:
        policy.gate_threshold = float(args.gate_threshold)
        if policy.pulse_controller is not None:
            policy.pulse_controller.gate_threshold = float(args.gate_threshold)
    args.structured_pulse = policy.structured_pulse
    seeds = args.seeds or [args.seed0 + index for index in range(args.episodes)]
    output = Path(args.output)
    rows = []
    for episode, seed in enumerate(seeds):
        observation, _ = env.reset(seed=int(seed))
        policy.reset()
        total_reward = 0.0
        steps = 0
        gate_steps = 0
        commitments = 0
        previous_velocity = np.asarray(env.sim.dynamics.state["v"], dtype=float).copy()
        dt = env.policy_decimation * env.steps_per_action / env.sim.config.world_rate
        peak_acceleration = 0.0
        while True:
            action, _ = policy.predict(observation)
            gate_steps += int(policy.last_gate_probability >= policy.gate_threshold)
            commitments += int(policy.last_committed)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            velocity = np.asarray(env.sim.dynamics.state["v"], dtype=float).copy()
            peak_acceleration = max(
                peak_acceleration,
                float(np.linalg.norm(velocity - previous_velocity) / dt),
            )
            previous_velocity = velocity
            steps += 1
            if terminated or truncated:
                terms = info.get("reward_terms", {}) or {}
                row = {
                    "episode": episode,
                    "seed": int(seed),
                    "success": bool(info.get("is_success", False)),
                    "termination_reason": info.get("termination_reason", "timeout"),
                    "steps": steps,
                    "reward": total_reward,
                    "gate_steps": gate_steps,
                    "gate_fraction": gate_steps / max(steps, 1),
                    "commitments": commitments,
                    "encounters_total": int(terms.get("encounters_total", 0)),
                    "encounters_cleared": int(terms.get("encounters_cleared", 0)),
                    "episode_min_clearance": float(
                        terms.get("episode_min_clearance", np.inf)
                    ),
                    "peak_acceleration_mps2": peak_acceleration,
                }
                rows.append(row)
                write_results(output, args, rows)
                print(json.dumps(row), flush=True)
                break
    write_results(output, args, rows)
    print(f"saved {output}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
