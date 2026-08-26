"""How much of skokloster is open enough to host the reference paper's regime?

The depth probe found background geometry at 1.04 m median against obstacles
at 0.73 m -- a depth ratio of 1.42, where open terrain would be >10. Static
obstacles are therefore competing against clutter at their own depth, which
explains the weak 3.45 sigma contrast.

Before proposing a synthetic open scene, check whether the existing one can
be used better: the path sampler currently accepts any navmesh shortest
path, corridors included, but a castle also has halls. If enough navigable
volume carries several metres of clearance, constraining the sampler to it
is a far smaller change than building a new scene.

Reports the clearance distribution over navigable space, and how large a
connected open region exists at each threshold -- a handful of scattered
open points is useless, since a 15 m path has to fit inside one.
"""
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG = "applications/rl/configs/velocity_dodge_static.yaml"
N_SAMPLES = 4000

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
env.reset(seed=9001)
pf = env.sim.visual_backend._sim.pathfinder

pts, clear = [], []
for _ in range(N_SAMPLES):
    p = pf.get_random_navigable_point()
    if not np.all(np.isfinite(p)):
        continue
    d = float(pf.distance_to_closest_obstacle(p, max_search_radius=20.0))
    if np.isfinite(d):
        pts.append(np.asarray(p, dtype=np.float64))
        clear.append(d)
pts = np.asarray(pts)
clear = np.asarray(clear)

print("\nskokloster navigable space, %d sampled points" % len(clear))
print("clearance to nearest geometry: median %.2f m  p90 %.2f  max %.2f"
      % (np.median(clear), np.percentile(clear, 90), clear.max()))
print()
print("%9s %9s %12s %14s" % ("thresh", "frac", "n_points", "max_extent_m"))
for thr in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
    keep = pts[clear >= thr]
    if len(keep) < 2:
        print("%8.1fm %8.1f%% %12d %14s" % (thr, 100 * len(keep) / len(clear), len(keep), "--"))
        continue
    # Largest pairwise separation among open points bounds the longest path
    # that could stay inside the open region.
    idx = np.random.default_rng(0).choice(len(keep), size=min(len(keep), 400), replace=False)
    sub = keep[idx]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    print("%8.1fm %8.1f%% %12d %14.1f"
          % (thr, 100 * len(keep) / len(clear), len(keep), d.max()))
print()
print("A 15 m path at 1 m/s needs max_extent >= 15 m at the chosen threshold.")
env.close()
