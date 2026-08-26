"""How far off is the straight-line lead prediction?

Throws are aimed at drone_position + drone_velocity * lead_t. The vehicle is
flying a curved MinSnap reference, so extrapolating current velocity over the
~1.4 s of flight lands somewhere the vehicle never goes. Against a 0.30 m
combined hit radius, even a modest curvature error makes the throw a
bystander by construction.

Measured on v5, where tightening aim_scatter to 0.25 m -- which by itself
should have put 100% of throws inside the hit radius -- lifted the threat
fraction only from 16% to 27%. That residual has to come from somewhere, and
the lead prediction is the candidate.

Reports, for a passive vehicle on the nominal path, the distance between the
straight-line prediction and where it ACTUALLY is lead_t later. If that
median is comfortably above 0.30 m, the prediction is the miss source and no
scatter setting can fix it.
"""
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "applications/rl/configs/velocity_dodge_dynamic_v5.yaml"
SEEDS = list(range(9001, 9016))
LEADS = [0.5, 1.0, 1.4, 2.0]

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)

err = {L: [] for L in LEADS}
for seed in SEEDS:
    env.reset(seed=seed)
    traj = env._nominal_trajectory
    end = float(getattr(traj, "t_keyframes", [float(env.episode_seconds)])[-1])
    dur = min(float(env.episode_seconds), end - 1e-3)
    for t in np.linspace(0.0, dur * 0.7, 40):
        flat = traj.update(float(t))
        p = env.sim.safety.dynamics_to_habitat(
            np.asarray(flat["x"], dtype=np.float64))
        v = env.sim.safety.dynamics_to_habitat_vel(
            np.asarray(flat["x_dot"], dtype=np.float64))
        for L in LEADS:
            tf = min(t + L, dur)
            actual = env.sim.safety.dynamics_to_habitat(
                np.asarray(traj.update(float(tf))["x"], dtype=np.float64))
            predicted = p + v * (tf - t)
            err[L].append(float(np.linalg.norm(predicted - actual)))
env.close()

print("\nconfig: %s" % CFG)
print("straight-line lead prediction error vs combined hit radius 0.30 m\n")
print("%8s %10s %10s %10s %14s" % ("lead_s", "median", "p10", "p90", "within 0.30m"))
for L in LEADS:
    e = np.asarray(err[L])
    print("%8.1f %9.2fm %9.2fm %9.2fm %13.0f%%"
          % (L, np.median(e), np.percentile(e, 10), np.percentile(e, 90),
             100 * float((e < 0.30).mean())))
