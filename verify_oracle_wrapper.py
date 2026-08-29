"""Does the vectorized oracle wrapper produce IDENTICAL behavior to the
already-validated single-env approach?

pretrain_encoder_oracle.py calls oracle_action(raw_env) in the main process,
immediately before vec_env.step(action) -- already used successfully (60+
round runs with sensible R2 growth). pretrain_encoder_oracle_vec.py moves
that SAME call inside a wrapper's step(), executed once per step just like
the original. If both call oracle_action() exactly once per step on the same
env state, with the same seed, a deterministic env should produce IDENTICAL
action sequences and resulting trajectories -- this checks that directly
rather than trusting the refactor by inspection alone.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "applications" / "rl"))
import numpy as np

from train_sb3 import load_experiment_config
from neurosim.rl import env_class_for_task
from evaluate_velocity_dodge_oracle import oracle_action, expert_for_task
from pretrain_encoder_oracle_vec import _OracleDrivenEnv

CFG = "applications/rl/configs/velocity_dodge_dynamic_v16_effnet.yaml"
SEED = 9001
N_STEPS = 200

exp = load_experiment_config(CFG)
env_config = copy.deepcopy(exp["env"])
env_config["enable_visualization"] = False


def run_direct():
    """Exactly pretrain_encoder_oracle.py's pattern: call in the main loop."""
    env = env_class_for_task(env_config["task"]["name"])(env_config=env_config, train=True)
    env.reset(seed=SEED)
    env._trajectory_dodge_expert = expert_for_task(env._task)
    actions, positions = [], []
    for _ in range(N_STEPS):
        a = np.asarray(oracle_action(env), dtype=np.float32).copy()
        actions.append(a)
        positions.append(np.asarray(env.sim.dynamics.state["x"], dtype=np.float64).copy())
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset(seed=SEED)
            env._trajectory_dodge_expert = expert_for_task(env._task)
    return np.asarray(actions), np.asarray(positions)


def run_wrapped():
    """Exactly pretrain_encoder_oracle_vec.py's pattern: oracle inside step()."""
    env = env_class_for_task(env_config["task"]["name"])(env_config=env_config, train=True)
    env = _OracleDrivenEnv(env)
    env.reset(seed=SEED)
    actions, positions = [], []
    for _ in range(N_STEPS):
        positions.append(
            np.asarray(env._raw.sim.dynamics.state["x"], dtype=np.float64).copy()
        )
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        actions.append(info["oracle_action"].copy())
        if term or trunc:
            env.reset(seed=SEED)
    return np.asarray(actions), np.asarray(positions)


a_direct, p_direct = run_direct()
a_wrapped, p_wrapped = run_wrapped()

n = min(len(a_direct), len(a_wrapped))
action_diff = np.abs(a_direct[:n] - a_wrapped[:n])
pos_diff = np.abs(p_direct[:n] - p_wrapped[:n])
print(f"\nsteps compared: {n}")
print(f"action diff:   max={action_diff.max():.6f}  mean={action_diff.mean():.6f}")
print(f"position diff: max={pos_diff.max():.6f}  mean={pos_diff.mean():.6f}")
print(f"action_a nonzero frac: {(np.linalg.norm(a_direct,axis=1)>1e-4).mean():.3f}")
print(f"action_b nonzero frac: {(np.linalg.norm(a_wrapped,axis=1)>1e-4).mean():.3f}")
identical = action_diff.max() < 1e-4 and pos_diff.max() < 1e-4
print(f"\n{'IDENTICAL' if identical else 'DIVERGED'} within tolerance")
