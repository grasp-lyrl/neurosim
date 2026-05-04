"""Generate heuristic reactive_dodge oracle rollouts for warm-start training.

The oracle uses privileged obstacle state to choose a small evasive CTBR
residual. It is meant for behavior-cloning data or debugging only; obstacle
state should not be part of the deployed policy observation.
"""

import copy
import argparse
from typing import Any
from pathlib import Path

import numpy as np

from neurosim.core.utils.utils_gen import load_yaml
from neurosim.rl import ReactiveDodgeEnv


def _oracle_delta_direction(env: ReactiveDodgeEnv) -> np.ndarray:
    active = env.active_obstacles()
    if not active:
        return np.zeros(3, dtype=np.float32)

    state = env.sim.dynamics.state
    agent_pos = env.dynamics_to_habitat_pos(np.asarray(state["x"]))
    agent_vel = env.dynamics_to_habitat_vel(np.asarray(state["v"], dtype=np.float64))
    manager = getattr(env.sim.visual_backend, "_dynamic_obstacles", None)
    agent_height = float(getattr(manager, "_agent_height", 0.0))

    best_score = np.inf
    best_rel_pos = None
    best_rel_vel = None
    for item in active.values():
        obstacle_pos = np.asarray(item.obj.translation, dtype=np.float64)
        obstacle_pos[1] -= agent_height
        obstacle_vel = env.obstacle_velocity(item)
        rel_pos = obstacle_pos - agent_pos
        rel_vel = obstacle_vel - agent_vel
        closest, tca = env.constant_velocity_closest_approach(rel_pos, rel_vel, 1.0)
        score = closest + 0.25 * tca
        if score < best_score:
            best_score = score
            best_rel_pos = rel_pos
            best_rel_vel = rel_vel

    if best_rel_pos is None or best_rel_vel is None:
        return np.zeros(3, dtype=np.float32)

    approach = -best_rel_vel
    approach_norm = np.linalg.norm(approach)
    if approach_norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    approach /= approach_norm

    # Choose the horizontal perpendicular that increases distance from the object.
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    lateral = np.cross(up, approach)
    if np.dot(lateral, -best_rel_pos) < 0.0:
        lateral *= -1.0
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm < 1e-6:
        lateral = up
    else:
        lateral /= lateral_norm

    # Convert Habitat direction back to dynamics coordinates.
    dodge_dyn = env.sim.coord_trans.pos_transform_inv @ lateral
    norm = np.linalg.norm(dodge_dyn)
    if norm > 1e-6:
        dodge_dyn /= norm
    return dodge_dyn.astype(np.float32)


def oracle_action(env: ReactiveDodgeEnv) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    threats = env.last_threats()
    if not bool(threats.get("obstacle_threat", False)):
        return action

    dodge = _oracle_delta_direction(env)
    if action.shape[0] == 5:
        action[0] = 1.0
        offset = 1
    else:
        offset = 0

    action[offset + 0] = 0.05 * float(np.clip(dodge[2], -1.0, 1.0))
    action[offset + 1] = float(np.clip(-dodge[1], -1.0, 1.0))
    action[offset + 2] = float(np.clip(dodge[0], -1.0, 1.0))
    action[offset + 3] = 0.0
    return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default="applications/rl/configs/reactive_dodge_sb3_combined_experiment.yaml",
    )
    parser.add_argument("--output", default="outputs/rl/reactive_dodge_oracle.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-events", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.experiment_config)
    env = ReactiveDodgeEnv(env_config=copy.deepcopy(cfg["env"]), train=False)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    events: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            terminated = truncated = False
            while not (terminated or truncated):
                if isinstance(obs, dict):
                    states.append(np.asarray(obs["state"], dtype=np.float32))
                    if args.save_events:
                        events.append(np.asarray(obs["events"], dtype=np.float32))
                else:
                    states.append(np.asarray(obs, dtype=np.float32).reshape(-1))

                action = oracle_action(env)
                actions.append(action.astype(np.float32))
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                metrics.append(dict(info.get("task_metrics", {})))
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "metrics": np.asarray(metrics, dtype=object),
    }
    if args.save_events:
        payload["events"] = np.asarray(events, dtype=np.float32)
    np.savez_compressed(output, **payload)
    print(f"saved {len(actions)} oracle samples to {output}")


if __name__ == "__main__":
    main()
