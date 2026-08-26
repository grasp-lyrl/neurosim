"""Is the static task's weak obstacle contrast a depth-separation problem?

Static obstacles are visible only through parallax, whose angular rate goes
as v/d. Since background parallax scales with speed identically, flight
speed cannot change the obstacle-to-background CONTRAST -- only the depth
ratio can. An obstacle at 2 m against background at 50 m produces a large
excursion; the same obstacle against walls at 3 m produces almost none.

That is the structural difference from the reference paper, which flies past
tree cylinders in open terrain (obstacles near, background effectively at
infinity) while this project flies inside skokloster castle.

Measured here: along the nominal path, the distance to the nearest scene
geometry (navmesh obstacle surface, a proxy for wall depth) against the
distance at which obstacles are actually encountered. If the two are
comparable, obstacles are competing against clutter at their own depth and
the measured 3.45 sigma is explained -- and the lever for the static task is
the scene, not the network, the speed, or the time-surface window.
"""
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG = "applications/rl/configs/velocity_dodge_static.yaml"
SEEDS = list(range(9001, 9021))

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
pathfinder = env.sim.visual_backend._sim.pathfinder
manager = env.sim.visual_backend._dynamic_obstacles

wall_d, obs_d = [], []
for seed in SEEDS:
    env.reset(seed=seed)
    traj = env._nominal_trajectory
    end = float(getattr(traj, "t_keyframes", [float(env.episode_seconds)])[-1])
    duration = min(float(env.episode_seconds), end - 1e-3)
    obstacles = [np.asarray(i.obj.translation, dtype=np.float64)
                 for i in manager._active.values()]
    for t in np.linspace(0.0, duration, 120):
        p = env.sim.safety.dynamics_to_habitat(
            np.asarray(traj.update(float(t))["x"], dtype=np.float64))
        d = float(pathfinder.distance_to_closest_obstacle(p, max_search_radius=15.0))
        if np.isfinite(d):
            wall_d.append(d)
        if obstacles:
            obs_d.append(float(min(np.linalg.norm(o - p) for o in obstacles)))
env.close()

w = np.asarray(wall_d)
o = np.asarray(obs_d)
print("\nnominal path, %d seeds" % len(SEEDS))
print("distance to nearest scene geometry:  median %.2f m  p10 %.2f  p90 %.2f"
      % (np.median(w), np.percentile(w, 10), np.percentile(w, 90)))
print("distance to nearest obstacle:        median %.2f m  p10 %.2f  p90 %.2f"
      % (np.median(o), np.percentile(o, 10), np.percentile(o, 90)))
print("\ndepth ratio (background / obstacle), median: %.2f" % (np.median(w) / np.median(o)))
print("The reference paper's open-terrain setting is effectively >10.")
