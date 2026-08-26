"""Static-task baseline with UNCENSORED per-encounter clearance.

static_baseline.py compares arms that do not see the same number of
obstacles: control terminated on obstacle_collision in 39/40 episodes and
logged 1.35 encounters per episode, while const+0.40 survived longer and
logged 2.83. Per-encounter clearance is then conditioned on having survived
that far, so the two numbers describe different obstacle populations and the
gap between them is partly survivor bias rather than skill. A blind arm that
dies on obstacle 1 is never tested against obstacles 2-6.

Removing the collision TERMINATION fixes this: clearance is derived from
per-obstacle closest approach, which is recorded whether or not the vehicle
survives, so with termination off every episode presents all 6 obstacles and
every arm is scored on the same set. Flying through a sphere is unphysical,
but this is a measurement of the blind-guessing ceiling, not a flight test.

Episode success is still reported from the same rollouts (conjunctive over
encounters), so the headline number keeps its physical meaning.
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
HIT_RADIUS_M = 0.30


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

# Bounds and navigability still terminate; only the obstacle sphere stops
# stopping the episode.
env.sim.safety.has_obstacle_collision = lambda habitat_pos: False

print("STATIC task, collision termination DISABLED (uncensored encounters)")
print(f"n={EPISODES}, seeds {SEED0}+, all 6 obstacles scored every episode\n")
print("%12s %8s %9s %7s %10s  %s"
      % ("arm", "episode", "per-enc", "enc/ep", "wilson", "terminations"), flush=True)


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
    label = mode if not const else "const%+.2f" % const
    print("%12s %7.1f%% %8.1f%% %7.2f  [%4.1f,%4.1f] %s"
          % (label, 100 * wins / EPISODES, 100 * cleared / max(tot, 1),
             tot / EPISODES, lo, hi, dict(term.most_common(2))), flush=True)


run("control")
for c in (0.4, -0.4, 0.8, -0.8):
    run("const", c)
run("oracle")
env.close()
