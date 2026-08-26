"""Can an open-scene regime be had by flying ABOVE skokloster?

The navmesh openness probe measured only the walkable floor and found the
scene's maximum clearance anywhere is 3.55 m -- too tight for either task.
But the navmesh is a 2-D walking surface: it says nothing about the volume
above the roof, which for an outdoor-ish scene may be wide open. If it is,
the reference paper's regime (obstacles near, background far) is reachable
without building any new asset -- just raise the flight volume and disable
the navmesh-based path sampling.

True 3-D free space is measured by raycasting, not by the navmesh: from
sampled points at a range of altitudes, cast rays over a sphere of
directions and record how far they travel before hitting geometry. The
median ray distance is the effective background depth an event camera would
see from there.
"""
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
import habitat_sim
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG = "applications/rl/configs/velocity_dodge_static.yaml"
MAX_RAY_M = 60.0
N_DIRS = 26
N_POINTS = 20

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
print("env constructed, resetting...", flush=True)
env.reset(seed=9001)
print("reset done", flush=True)
sim = env.sim.visual_backend._sim
pf = sim.pathfinder

lo, hi = pf.get_bounds()
lo = np.asarray(lo, dtype=np.float64)
hi = np.asarray(hi, dtype=np.float64)
print("\nnavmesh AABB: x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f]"
      % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))

# Fibonacci sphere: even direction coverage without pole clustering.
i = np.arange(N_DIRS) + 0.5
phi = np.arccos(1 - 2 * i / N_DIRS)
theta = np.pi * (1 + 5 ** 0.5) * i
dirs = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi),
                 np.sin(theta) * np.sin(phi)], axis=-1)

samples = [np.asarray(pf.get_random_navigable_point(), dtype=np.float64)
           for _ in range(200)]
ys = np.asarray([p[1] for p in samples if np.all(np.isfinite(p))])
print("navigable point heights: min %.2f median %.2f max %.2f"
      % (ys.min(), np.median(ys), ys.max()))
print("\n%10s %10s %12s %12s %10s %12s"
      % ("alt_above", "n_pts", "median_ray", "p10_ray", "frac_open", "down_ray"))

rng = np.random.default_rng(0)
for alt in (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0):
    dists, downs, opens, n = [], [], 0, 0
    for _ in range(N_POINTS):
        p = np.asarray(pf.get_random_navigable_point(), dtype=np.float64)
        if not np.all(np.isfinite(p)):
            continue
        # Raise each point above ITS OWN floor. Using a scene-wide median
        # floor put samples above the roof, where every ray escapes to the
        # 60 m cap -- which is what "100% open at 0.5 m" really meant.
        p = p.copy()
        p[1] = p[1] + alt
        d = []
        for dvec in dirs:
            ray = habitat_sim.geo.Ray(p.astype(np.float32), dvec.astype(np.float32))
            hits = sim.cast_ray(ray, max_distance=MAX_RAY_M)
            d.append(float(hits.hits[0].ray_distance) if hits.has_hits() else MAX_RAY_M)
        d = np.asarray(d)
        dists.append(np.median(d))
        # Sanity check: a ray straight down must hit the floor at about the
        # altitude we claim to be at. If it does not, the cast is not
        # reaching the stage geometry and every other number here is void.
        dray = habitat_sim.geo.Ray(p.astype(np.float32),
                                   np.asarray([0.0, -1.0, 0.0], dtype=np.float32))
        dh = sim.cast_ray(dray, max_distance=MAX_RAY_M)
        downs.append(float(dh.hits[0].ray_distance) if dh.has_hits() else np.nan)
        # "open" = at least half the sphere sees past 10 m
        opens += int(np.median(d) >= 10.0)
        n += 1
    if n:
        dn = np.asarray(downs, dtype=np.float64)
        dn = dn[np.isfinite(dn)]
        print("%9.1fm %10d %11.1fm %11.1fm %9.0f%% %11s"
              % (alt, n, np.median(dists), np.percentile(dists, 10),
                 100 * opens / n,
                 ("%.2fm" % np.median(dn)) if len(dn) else "NO HIT"))
print("\nReference-paper regime needs background depth >> obstacle range (~1-4 m).")
env.close()
