"""Per-encounter escape set: which displacements would have cleared?

"Did the policy dodge the correct way" is ill-posed. For an obstacle
crossing left-to-right, going behind it and beating it across can both
clear, and which is better depends on timing -- so scoring against a single
correct direction penalises good dodges.

The escape SET is well-posed. For an encounter the geometry is known: the
nominal path p(t), the obstacle path o(t), and the combined radius r. For a
candidate constant lateral displacement d, clearance is

    min_t | p(t) + d - o(t) | - r

so sweeping d over the cross-track plane gives exactly the set of
displacements that would have cleared. From it:

  * escape_frac -- the fraction of the grid that clears. This IS the
    probability a blind guess succeeds, measured from the geometry of each
    encounter instead of argued as 2^-traps. Every blind-ceiling figure in
    this project so far has been the hand-derived kind.
  * degenerate encounters -- escape_frac near 1 means almost any
    displacement works, so the encounter cannot distinguish perception from
    guessing. That is the quantitative form of the aimed-obstacle
    degeneracy that twice let a blind clone match a sighted one.

Pure geometry, no rollout: the vehicle is displaced from the nominal path
rather than resimulated, which is the same approximation the residual
offset controller implements.
"""
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "applications/rl/configs/velocity_dodge_dynamic.yaml"
SEEDS = list(range(9001, 9021))
GRID = np.linspace(-1.2, 1.2, 25)      # cross-track displacement grid, metres
DODGE_CLEAR_M = 0.10                   # clearance counted as a successful escape


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

fracs, n_enc, n_degenerate, n_impossible = [], 0, 0, 0
for seed in SEEDS:
    env.reset(seed=seed)
    traj = env._nominal_trajectory
    end = float(getattr(traj, "t_keyframes", [float(env.episode_seconds)])[-1])
    duration = min(float(env.episode_seconds), end - 1e-3)
    ts = np.linspace(0.0, duration, 300)
    nominal = np.asarray([env.sim.safety.dynamics_to_habitat(
        np.asarray(traj.update(float(t))["x"], dtype=np.float64)) for t in ts])
    tangent = unit(nominal[-1] - nominal[0])
    up = np.array([0.0, 1.0, 0.0])
    cross = unit(np.cross(tangent, up))
    if not np.any(cross):
        continue

    # Record each obstacle's world track over the episode.
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
        # Nominal position at each recorded time.
        nom = np.asarray([env.sim.safety.dynamics_to_habitat(
            np.asarray(traj.update(float(min(t, duration)))["x"],
                       dtype=np.float64)) for t in times])
        base = float(np.linalg.norm(pos - nom, axis=1).min()) - agent_r - rad
        if base > 0.0:
            continue          # never threatened the nominal path
        n_enc += 1
        cleared = 0
        for d in GRID:
            disp = nom + d * cross
            c = float(np.linalg.norm(pos - disp, axis=1).min()) - agent_r - rad
            if c > DODGE_CLEAR_M:
                cleared += 1
        frac = cleared / len(GRID)
        fracs.append(frac)
        if frac >= 0.9:
            n_degenerate += 1
        if frac == 0.0:
            n_impossible += 1
env.close()

f = np.asarray(fracs)
print("\nconfig: %s" % CFG)
print("encounters that threatened the nominal path: %d over %d seeds" % (n_enc, len(SEEDS)))
if len(f):
    print("escape fraction (= blind-guess success probability):")
    print("   mean %.3f   median %.3f   p10 %.3f   p90 %.3f"
          % (f.mean(), np.median(f), np.percentile(f, 10), np.percentile(f, 90)))
    print("   degenerate encounters (>=90%% of displacements clear): %d (%.0f%%)"
          % (n_degenerate, 100 * n_degenerate / len(f)))
    print("   impossible encounters (no displacement clears):       %d (%.0f%%)"
          % (n_impossible, 100 * n_impossible / len(f)))
    print("\nA task that can distinguish perception from guessing needs the mean")
    print("escape fraction well below 1, and few impossible encounters.")
else:
    print("no threatening encounters found")
