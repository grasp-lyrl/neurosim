"""Escape set over BOTH along-track and cross-track displacement.

escape_set_probe.py swept only cross-track offsets, so it could not see the
degenerate strategy this task is most exposed to: braking. A throw solved to
arrive at a point at a time is defeated by simply lagging along the path --
no perception needed, since it works whichever side the obstacle comes from,
and a ballistic obstacle cannot re-aim to compensate. Aiming throws at the
nominal path (v2) makes each throw MORE committed to a fixed arrival, so it
may have made braking more effective rather than less.

Sweeps a 2-D grid of constant displacements -- along the path tangent and
across it -- and reports three fractions:

    escape_any    any displacement in the grid clears
    escape_cross  cross-track only (along = 0): requires choosing a SIDE,
                  so it carries directional information
    escape_along  along-track only (cross = 0): pure braking/hurrying,
                  carries no information about where the obstacle is

If escape_along is large, the task is solvable without perception and no
amount of reward shaping on the cross-track term will change that; the fix
has to remove the along-track degree of freedom or make throws robust to it.
"""
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "applications/rl/configs/velocity_dodge_dynamic_v2.yaml"
SEEDS = list(range(9001, 9021))
ALONG = np.linspace(-1.2, 1.2, 25)
CROSS = np.linspace(-1.2, 1.2, 25)
DODGE_CLEAR_M = 0.10


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
env.sim.safety.has_obstacle_collision = lambda habitat_pos: False
manager = env.sim.visual_backend._dynamic_obstacles
agent_r = float(manager._agent_radius)
dim = int(np.prod(env.action_space.shape))

any_f, cross_f, along_f, n_enc = [], [], [], 0
for seed in SEEDS:
    env.reset(seed=seed)
    traj = env._nominal_trajectory
    end = float(getattr(traj, "t_keyframes", [float(env.episode_seconds)])[-1])
    duration = min(float(env.episode_seconds), end - 1e-3)

    tracks = {}
    while True:
        now = float(env.sim.time)
        for oid, item in manager._active.items():
            tracks.setdefault(oid, []).append(
                (now, np.asarray(item.obj.translation, dtype=np.float64).copy(),
                 float(item.collision_radius)))
        _, _, term, trunc, _ = env.step(np.zeros(dim, dtype=np.float32))
        if term or trunc:
            break

    for oid, samples in tracks.items():
        if len(samples) < 3:
            continue
        times = np.asarray([s[0] for s in samples])
        pos = np.asarray([s[1] for s in samples])
        rad = float(samples[0][2])
        nom = np.asarray([env.sim.safety.dynamics_to_habitat(
            np.asarray(traj.update(float(min(t, duration)))["x"], dtype=np.float64))
            for t in times])
        if float(np.linalg.norm(pos - nom, axis=1).min()) - agent_r - rad > 0.0:
            continue
        n_enc += 1

        # Local frame from the nominal velocity at closest approach.
        k = int(np.argmin(np.linalg.norm(pos - nom, axis=1)))
        vel = env.sim.safety.dynamics_to_habitat_vel(
            np.asarray(traj.update(float(min(times[k], duration)))["x_dot"],
                       dtype=np.float64))
        tangent = unit(vel)
        cross = unit(np.cross(tangent, np.array([0.0, 1.0, 0.0])))
        if not np.any(tangent) or not np.any(cross):
            continue

        grid = np.zeros((len(ALONG), len(CROSS)), dtype=bool)
        for i, a in enumerate(ALONG):
            for j, c in enumerate(CROSS):
                disp = nom + a * tangent + c * cross
                clr = float(np.linalg.norm(pos - disp, axis=1).min()) - agent_r - rad
                grid[i, j] = clr > DODGE_CLEAR_M
        mid = len(ALONG) // 2
        any_f.append(grid.mean())
        cross_f.append(grid[mid, :].mean())    # along = 0
        along_f.append(grid[:, mid].mean())    # cross = 0
env.close()

print("\nconfig: %s" % CFG)
print("encounters threatening the nominal path: %d over %d seeds (%.2f/episode)"
      % (n_enc, len(SEEDS), n_enc / len(SEEDS)))
if any_f:
    print("\n%-14s %8s %8s %8s" % ("escape route", "mean", "median", "p90"))
    for name, arr in (("any (2-D)", any_f), ("cross only", cross_f),
                      ("along only", along_f)):
        a = np.asarray(arr)
        print("%-14s %8.3f %8.3f %8.3f"
              % (name, a.mean(), np.median(a), np.percentile(a, 90)))
    al = np.asarray(along_f)
    print("\nbraking alone clears >=50%% of displacements in %.0f%% of encounters"
          % (100 * float((al >= 0.5).mean())))
    print("braking alone clears NOTHING in %.0f%% of encounters"
          % (100 * float((al == 0.0).mean())))
