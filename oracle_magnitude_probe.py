"""Is the oracle's 1.00 m candidate cap what makes it collide?

The expert collides in 13/40 v20 episodes. Two very different explanations:

  magnitude  -- the dodge it is allowed to plan is too small. Its menu is
                candidate_offsets_m = (0.35, 0.50, 0.65, 0.80, 1.00), which
                was never rescaled when offset_rate_limits_mps went
                0.8 -> 1.5 and reachable displacement went 0.707 -> 1.275 m.
  timing     -- the displacement it needs is available, but it commits too
                late or tracks the plan too poorly to realise it.

Only the first is answered here, and it is answered geometrically, with no
oracle in the loop: for every obstacle that actually threatens the nominal
path, sweep displacement over direction AND magnitude and report the SMALLEST
magnitude that clears in any direction.

That number is a lower bound on what any displacement-based expert must be
allowed to command. If it sits comfortably under 1.00 m for nearly every
encounter, the cap is not the binding constraint and raising it will not fix
the oracle -- attention belongs on commit time and tracking instead
(oracle_commit_probe.py, offset_tracking_probe.py).

Geometry and encounter definition are copied from escape_set_probe.py so the
denominators match: an encounter counts only if the obstacle would have come
within the combined radius of the UNDISPLACED nominal path.

Unlike escape_set_probe.py, which sweeps one cross-track axis, this sweeps a
full sphere of directions -- the expert searches 12, including 8 in the plane
perpendicular to the incoming relative velocity, so restricting to one axis
would understate what it can reach and overstate the required magnitude.
Magnitudes are frame-independent, so this deliberately avoids the
habitat(y-up) / dynamics(z-up) frame question entirely.

Usage:
    python oracle_magnitude_probe.py <config.yaml> [n_seeds]
"""
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "applications/rl/configs/velocity_dodge_dynamic_v20_effnet.yaml"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEEDS = list(range(9001, 9001 + N_SEEDS))

DODGE_CLEAR_M = 0.10        # v20 dodge_clearance_m; an encounter "clears" above this
MAGNITUDES = np.arange(0.05, 2.001, 0.05)
N_DIRECTIONS = 64

# The expert's own limits, for reporting only.
ORACLE_MAX_CANDIDATE_M = 1.00     # max(candidate_offsets_m)
REACHABLE_M = 1.275               # measured lateral reach in the 1.5 s threat window


def fibonacci_sphere(n):
    """n roughly-uniform unit vectors -- direction coverage without a frame."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


DIRS = fibonacci_sphere(N_DIRECTIONS)

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
# Let the episode run its full length: we are measuring the geometry of every
# encounter, not surviving them, and a collision would truncate the record.
env.sim.safety.has_obstacle_collision = lambda habitat_pos: False
manager = env.sim.visual_backend._dynamic_obstacles
agent_r = float(manager._agent_radius)
dim = int(np.prod(env.action_space.shape))

required = []       # min magnitude that clears, per encounter (inf = impossible)
baselines = []      # how deep the undisplaced miss distance was
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
            np.asarray(traj.update(float(min(t, duration)))["x"],
                       dtype=np.float64)) for t in times])

        rel = pos - nom                                   # (T, 3)
        base = float(np.linalg.norm(rel, axis=1).min()) - agent_r - rad
        if base > 0.0:
            continue                                      # never threatened
        baselines.append(base)

        # clearance(m, u) = min_t |rel_t - m*u| - (agent_r + rad)
        # rel (T,3), DIRS (D,3), MAGNITUDES (M,) -> (M, D, T)
        proj = rel @ DIRS.T                               # (T, D)
        sq = np.sum(rel * rel, axis=1)[:, None]           # (T, 1)
        m = MAGNITUDES[:, None, None]
        d2 = sq.T[None, :, :] - 2.0 * m * proj.T[None, :, :] + m ** 2
        clearance = np.sqrt(np.maximum(d2, 0.0)).min(axis=2) - (agent_r + rad)
        ok = (clearance > DODGE_CLEAR_M).any(axis=1)      # (M,) any direction clears
        idx = np.flatnonzero(ok)
        required.append(float(MAGNITUDES[idx[0]]) if idx.size else np.inf)

env.close()

req = np.asarray(required)
finite = req[np.isfinite(req)]
n = len(req)
print("\nconfig: %s   seeds: %d" % (CFG, len(SEEDS)))
print("threatening encounters: %d  (%.2f per episode)" % (n, n / max(len(SEEDS), 1)))
if n == 0:
    print("no threatening encounters found")
    raise SystemExit(0)

print("undisplaced miss distance (negative = would hit): median %.3f m, min %.3f m"
      % (float(np.median(baselines)), float(np.min(baselines))))
print("\nsmallest displacement that clears by >= %.2f m, over %d directions:"
      % (DODGE_CLEAR_M, N_DIRECTIONS))
if finite.size:
    print("   median %.2f m   p75 %.2f m   p90 %.2f m   max %.2f m"
          % (float(np.median(finite)), float(np.percentile(finite, 75)),
             float(np.percentile(finite, 90)), float(finite.max())))
print("   impossible within %.2f m: %d (%.0f%%)"
      % (MAGNITUDES[-1], int((~np.isfinite(req)).sum()),
         100.0 * (~np.isfinite(req)).sum() / n))

over_cap = int((req > ORACLE_MAX_CANDIDATE_M).sum())
over_reach = int((req > REACHABLE_M).sum())
print("\nagainst the expert's limits:")
print("   need more than %.2f m (max candidate_offsets_m): %d (%.0f%%)"
      % (ORACLE_MAX_CANDIDATE_M, over_cap, 100.0 * over_cap / n))
print("   need more than %.2f m (reachable in the window): %d (%.0f%%)"
      % (REACHABLE_M, over_reach, 100.0 * over_reach / n))

print("\nVERDICT:")
if 100.0 * over_cap / n >= 15.0:
    print("   The 1.00 m cap IS binding -- it blocks %.0f%% of encounters." % (
        100.0 * over_cap / n))
    print("   Extending candidate_offsets_m is worth doing. Note %.0f%% need more"
          % (100.0 * over_reach / n))
    print("   than is reachable at all, which no expert change can fix.")
else:
    print("   The 1.00 m cap is NOT the binding constraint: only %.0f%% of" % (
        100.0 * over_cap / n))
    print("   encounters need more than it allows, against a %.0f%% oracle failure"
          % 32.5)
    print("   rate. Raising it cannot explain the collisions -- look at commit")
    print("   time and plan tracking instead.")
