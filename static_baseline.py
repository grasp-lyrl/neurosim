"""Static-task baseline under the current (post-shaping-fix) config.

Static is the one regime where vision has measurably worked here: aux R2
0.334, and today's ablation put vision at +11.5 per-encounter pts against
-1.3 on dynamic. It also matches the reference paper's setup (static
cylinders flown past), where parallax gives an event camera the signal it
actually responds to.

Two things to establish before collecting a dataset:
  1. Is it non-degenerate? Obstacles sit along the nominal path, so earlier a
     constant lateral offset cleared everything. slalom_offset_m 0.9 should
     defeat that -- the fixed-offset arms test it directly.
  2. What is the headroom (oracle - control) under the shaping fix applied
     today, which invalidated the previously-trained clone?
"""
import collections
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from evaluate_velocity_dodge_oracle import oracle_action, expert_for_task
from train_sb3 import load_experiment_config

CFG = "applications/rl/configs/velocity_dodge_static.yaml"
SEED0, EPISODES = 9001, 40


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return 100 * (c - h), 100 * (c + h)


cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
dim = int(np.prod(env.action_space.shape))
print(f"STATIC task, n={EPISODES}, seeds {SEED0}+ (post shaping fix)\n")
print(f"{'arm':>12} {'episode':>8} {'per-enc':>9} {'enc/ep':>7}  terminations", flush=True)


def run(mode, const=0.0):
    wins = tot = cleared = 0
    term = collections.Counter()
    for i in range(EPISODES):
        env.reset(seed=SEED0 + i)
        if mode == "oracle":
            env._trajectory_dodge_expert = expert_for_task(env._task)
            env._trajectory_expert_plans = 0
            env._trajectory_expert_failed_plans = 0
            env._trajectory_expert_plan_diagnostics = []
        while True:
            if mode == "oracle":
                action = oracle_action(env)
            else:
                action = np.zeros(dim, dtype=np.float32)
                if const:
                    action[-2] = const
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                t = info.get("reward_terms", {}) or {}
                tot += int(t.get("encounters_total", 0))
                cleared += int(t.get("encounters_cleared", 0))
                wins += bool(info.get("is_success", False))
                term[info.get("termination_reason") or "timeout"] += 1
                break
    lo, hi = wilson(cleared, tot)
    label = mode if not const else f"const{const:+.2f}"
    print(f"{label:>12} {100*wins/EPISODES:>7.1f}% {100*cleared/max(tot,1):>8.1f}%"
          f" {tot/EPISODES:>7.2f}  [{lo:.1f},{hi:.1f}] {dict(term.most_common(2))}", flush=True)


run("control")
for c in (0.4, -0.4, 0.8, -0.8):
    run("const", c)
run("oracle")
env.close()
