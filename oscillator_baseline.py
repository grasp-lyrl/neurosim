"""Is TIMING informative, or does a blind weave do just as well?

Every blind baseline measured so far held a CONSTANT displacement. That was
the right adversary for nominal-aimed configs (v2/v3), where one offset
clears everything. It is the wrong adversary for drone-aimed configs (v4),
where each throw is solved against wherever the vehicle actually is, so a
held offset is simply followed and cannot work.

What a drone-aimed throw still permits is evasion by moving AFTER launch --
and any direction will do, so direction carries no information. The open
question is whether TIMING does: moving when a throw is genuinely incoming
requires perceiving it, and a policy that cannot perceive must move blindly.

This measures the blind alternative directly. A sinusoidal weave moves
constantly, at a fixed amplitude and frequency, with no knowledge of any
obstacle. If it matches the oracle, then timing is not informative either --
the task is solvable by moving on a clock, and no perception is needed. If
the oracle clearly beats every weave, timing is a real perceptual
requirement and the task is worth training on.

Scored with the exogenous threat definition (would the throw have hit the
path the vehicle was actually flying) and the bystander column, since a
weave that thrashes into other obstacles is not succeeding.
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
    "applications/rl/configs/velocity_dodge_dynamic_v4.yaml"
SEED0, EPISODES = 9001, 25
WEAVES = [(0.4, 0.5), (0.4, 1.0), (0.8, 0.5), (0.8, 1.0), (0.8, 2.0)]


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
print("blind weave vs oracle; threat := the FLOWN path would have been hit\n")
print("%14s %9s %9s %8s %11s %9s  %s"
      % ("arm", "episode", "threats", "cleared", "bystanders", "byst_hit", "wilson"),
      flush=True)


def episode(mode, seed, amp=0.0, freq=0.0):
    env.reset(seed=seed)
    if mode == "oracle":
        env._trajectory_dodge_expert = expert_for_task(env._task)
        env._trajectory_expert_plans = 0
        env._trajectory_expert_failed_plans = 0
        env._trajectory_expert_plan_diagnostics = []
    tracks = collections.defaultdict(list)
    actual = collections.defaultdict(lambda: np.inf)
    flown = []
    success = False
    while True:
        now = float(env.sim.time)
        hp = env.sim.safety.dynamics_to_habitat(
            np.asarray(env.sim.dynamics.state["x"], dtype=np.float64))
        flown.append((now, hp.copy()))
        for oid, item in manager._active.items():
            opos = np.asarray(item.obj.translation, dtype=np.float64).copy()
            tracks[oid].append((now, opos, float(item.collision_radius)))
            actual[oid] = min(actual[oid], float(np.linalg.norm(hp - opos))
                              - agent_r - float(item.collision_radius))
        if mode == "oracle":
            action = oracle_action(env)
        else:
            action = np.zeros(dim, dtype=np.float32)
            if mode == "weave":
                action[-2] = amp * np.sin(2.0 * np.pi * freq * now)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            success = bool(info.get("is_success", False))
            break

    # A throw counts as a threat if it would have hit the NOMINAL path, which
    # is policy-independent; scoring against the flown path instead would make
    # the denominator endogenous again (dodging would delete the threat).
    traj = env._nominal_trajectory
    end = float(getattr(traj, "t_keyframes", [float(env.episode_seconds)])[-1])
    duration = min(float(env.episode_seconds), end - 1e-3)
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
        if float(np.linalg.norm(pos - nom, axis=1).min()) - agent_r - rad <= 0.0:
            threats += 1
            if actual[oid] > dodge_clear:
                cleared += 1
        else:
            bystanders += 1
            if actual[oid] <= 0.0:
                byst_hit += 1
    return threats, cleared, bystanders, byst_hit, success


def run(mode, amp=0.0, freq=0.0, label=None):
    T = C = B = BH = W = 0
    for i in range(EPISODES):
        t, c, b, bh, s = episode(mode, SEED0 + i, amp, freq)
        T += t; C += c; B += b; BH += bh; W += int(s)
    lo, hi = wilson(C, T)
    name = label or mode
    print("%14s %8.1f%% %9.2f %7.1f%% %11.2f %8.1f%%  [%4.1f,%4.1f]"
          % (name, 100 * W / EPISODES, T / EPISODES, 100 * C / max(T, 1),
             B / EPISODES, 100 * BH / max(B, 1), lo, hi), flush=True)


run("control")
for amp, freq in WEAVES:
    run("weave", amp, freq, label="weave%.1f@%.1fHz" % (amp, freq))
run("oracle")
env.close()
