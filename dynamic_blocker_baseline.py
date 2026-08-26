"""Dynamic dodging scored against an exogenous denominator.

The per-encounter clearance built into velocity_dodge.py cannot support a
blind ablation. Its denominator counts an obstacle only if its closest
approach came within clearance_threshold_m -- a quantity the policy controls,
so dodging well DELETES an obstacle from the denominator instead of scoring
it as cleared. On the static task that machinery reported a 41.4% do-nothing
floor where exogenous scoring showed 14.1%, and it over-reported genuine
threats by 4-9x.

Here the denominator is fixed before the policy acts, exactly as in
blocker_baseline.py but for moving obstacles: replay each obstacle's world
track, compare it against the NOMINAL trajectory at matching times, and count
it as a threat if the nominal path would have been hit. That set depends only
on the seed. The numerator is how many of those threats the actual flight kept
clear of.

Obstacles that never threatened the nominal path are reported separately as
"bystanders" -- hitting one is a real failure (the policy dodged INTO it) and
pooling them with threats hides exactly that, which is what inflated every
constant-offset arm on the static task.

Collision termination is disabled so every threat is scored in every episode
rather than only those an arm survives long enough to meet.
"""
import collections
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from evaluate_velocity_dodge_oracle import oracle_action, expert_for_task
from train_sb3 import load_experiment_config

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "applications/rl/configs/velocity_dodge_dynamic_v2.yaml"
SEED0, EPISODES = 9001, 30


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
env.sim.safety.has_obstacle_collision = lambda habitat_pos: False
manager = env.sim.visual_backend._dynamic_obstacles
agent_r = float(manager._agent_radius)
dodge_clear = float(env._task.dodge_clearance_m)

print("config: %s" % CFG)
print("DYNAMIC, exogenous denominator, collision termination disabled")
print("threat := the NOMINAL path would have been hit by this throw\n")
print("%12s %8s %9s %8s %11s %9s  %s"
      % ("arm", "episode", "threats", "cleared", "bystanders", "byst_hit", "wilson"),
      flush=True)


def episode(mode, const, seed):
    env.reset(seed=seed)
    if mode == "oracle":
        env._trajectory_dodge_expert = expert_for_task(env._task)
        env._trajectory_expert_plans = 0
        env._trajectory_expert_failed_plans = 0
        env._trajectory_expert_plan_diagnostics = []
    traj = env._nominal_trajectory
    end = float(getattr(traj, "t_keyframes", [float(env.episode_seconds)])[-1])
    duration = min(float(env.episode_seconds), end - 1e-3)

    tracks = collections.defaultdict(list)
    actual = collections.defaultdict(lambda: np.inf)
    success = False
    while True:
        now = float(env.sim.time)
        hp = env.sim.safety.dynamics_to_habitat(
            np.asarray(env.sim.dynamics.state["x"], dtype=np.float64))
        for oid, item in manager._active.items():
            opos = np.asarray(item.obj.translation, dtype=np.float64).copy()
            tracks[oid].append((now, opos, float(item.collision_radius)))
            actual[oid] = min(
                actual[oid],
                float(np.linalg.norm(hp - opos)) - agent_r - float(item.collision_radius))
        if mode == "oracle":
            action = oracle_action(env)
        else:
            action = np.zeros(dim, dtype=np.float32)
            if const:
                action[-2] = const
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            success = bool(info.get("is_success", False))
            break

    threats = cleared = bystanders = byst_hit = 0
    for oid, samples in tracks.items():
        if len(samples) < 2:
            continue
        times = np.asarray([s[0] for s in samples])
        pos = np.asarray([s[1] for s in samples])
        rad = float(samples[0][2])
        nom = np.asarray([env.sim.safety.dynamics_to_habitat(
            np.asarray(traj.update(float(min(t, duration)))["x"], dtype=np.float64))
            for t in times])
        nominal_clear = float(np.linalg.norm(pos - nom, axis=1).min()) - agent_r - rad
        if nominal_clear <= 0.0:
            threats += 1
            if actual[oid] > dodge_clear:
                cleared += 1
        else:
            bystanders += 1
            if actual[oid] <= 0.0:
                byst_hit += 1
    return threats, cleared, bystanders, byst_hit, success


def run(mode, const=0.0):
    T = C = B = BH = W = 0
    for i in range(EPISODES):
        t, c, b, bh, s = episode(mode, const, SEED0 + i)
        T += t; C += c; B += b; BH += bh; W += int(s)
    lo, hi = wilson(C, T)
    label = mode if not const else "const%+.2f" % const
    print("%12s %7.1f%% %9.2f %7.1f%% %11.2f %8.1f%%  [%4.1f,%4.1f]"
          % (label, 100 * W / EPISODES, T / EPISODES, 100 * C / max(T, 1),
             B / EPISODES, 100 * BH / max(B, 1), lo, hi), flush=True)


run("control")
for c in (0.4, -0.4, 0.8, -0.8):
    run("const", c)
run("oracle")
env.close()
