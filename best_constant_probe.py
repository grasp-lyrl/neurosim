"""Can ONE fixed displacement clear a whole episode?

The mean escape fraction was the wrong statistic. It averages over the
displacement grid, which models a policy guessing at random -- but the
strategy PPO actually finds is a single saturated offset held for the whole
episode. Measured on velocity_dodge_dynamic_v2: mean escape fraction 0.560
predicted a ~17% blind ceiling, while a constant +0.40 m offset cleared 95%
of threats and took episode success from 3.3% to 36.7%. Averaging hid that
completely.

The right question is adversarial, not average: over the 2-D grid of
constant displacements (along-track x cross-track), what does the BEST one
achieve against every threat in an episode? A task where perception is
necessary must have no single displacement that survives -- the threats have
to cover the reachable offset space, not converge on one point in it.

Reports, per episode:
    best_all    clearance of the best single constant displacement
    best_cross  same, restricted to cross-track only (no braking)
and the fraction of episodes some constant displacement clears completely.
A task ready for training wants best_all well below 1 and the
fully-cleared fraction near 0.
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
ALONG = np.linspace(-1.2, 1.2, 17)
CROSS = np.linspace(-1.2, 1.2, 17)
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

best_all, best_cross, full, n_ep, threats_per = [], [], 0, 0, []
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

    # One boolean grid per threat: does this constant displacement clear it?
    grids = []
    for oid, samples in tracks.items():
        if len(samples) < 3:
            continue
        times = np.asarray([s[0] for s in samples])
        pos = np.asarray([s[1] for s in samples])
        rad = float(samples[0][2])
        nom = np.asarray([env.sim.safety.dynamics_to_habitat(
            np.asarray(traj.update(float(min(t, duration)))["x"], dtype=np.float64))
            for t in times])
        # Include any throw that could threaten SOME reachable displacement,
        # not only those aimed at the nominal path. Filtering on
        # nominal-path threat silently discards exactly the throws that
        # punish a policy sitting at an offset -- which is the whole point
        # of spreading aim across the reachable set. With that filter, a
        # config whose throws cover the offset space would still report a
        # surviving constant offset, because the throws defeating it were
        # never counted.
        reach = float(max(abs(ALONG).max(), abs(CROSS).max()))
        if float(np.linalg.norm(pos - nom, axis=1).min()) - agent_r - rad > reach:
            continue
        k = int(np.argmin(np.linalg.norm(pos - nom, axis=1)))
        vel = env.sim.safety.dynamics_to_habitat_vel(
            np.asarray(traj.update(float(min(times[k], duration)))["x_dot"],
                       dtype=np.float64))
        tangent = unit(vel)
        cross = unit(np.cross(tangent, np.array([0.0, 1.0, 0.0])))
        if not np.any(tangent) or not np.any(cross):
            continue
        g = np.zeros((len(ALONG), len(CROSS)), dtype=bool)
        for i, a in enumerate(ALONG):
            for j, c in enumerate(CROSS):
                disp = nom + a * tangent + c * cross
                clr = float(np.linalg.norm(pos - disp, axis=1).min()) - agent_r - rad
                g[i, j] = clr > DODGE_CLEAR_M
        grids.append(g)

    if not grids:
        continue
    n_ep += 1
    threats_per.append(len(grids))
    stack = np.stack(grids)                 # (threats, along, cross)
    per_disp = stack.mean(axis=0)           # clearance of each constant displacement
    best_all.append(float(per_disp.max()))
    mid = len(ALONG) // 2
    best_cross.append(float(per_disp[mid, :].max()))
    full += int(per_disp.max() >= 1.0)
env.close()

print("\nconfig: %s" % CFG)
print("episodes with threats: %d, mean threats/episode %.2f"
      % (n_ep, float(np.mean(threats_per)) if threats_per else 0.0))
if best_all:
    ba, bc = np.asarray(best_all), np.asarray(best_cross)
    print("\n%-24s %8s %8s %8s" % ("strategy", "mean", "median", "min"))
    print("%-24s %8.3f %8.3f %8.3f" % ("best constant (2-D)", ba.mean(), np.median(ba), ba.min()))
    print("%-24s %8.3f %8.3f %8.3f" % ("best constant (cross)", bc.mean(), np.median(bc), bc.min()))
    print("\nepisodes fully cleared by ONE constant displacement: %d/%d (%.0f%%)"
          % (full, n_ep, 100 * full / n_ep))
    print("A task needing perception wants these well below 1.0 and near 0%.")
