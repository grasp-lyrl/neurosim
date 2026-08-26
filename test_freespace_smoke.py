import sys
sys.path.insert(0, "/home/odexter/neurosim/src")
import numpy as np
from neurosim.core.trajectory import create_trajectory

lo = np.array([-8.0, 2.0, 2.0])
hi = np.array([8.0, 10.0, 20.0])
for seed in (7, 11, 23):
    traj = create_trajectory(
        model="freespace_minsnap", seed=seed, box_lo=lo, box_hi=hi,
        target_length=15.0, min_waypoint_distance=2.0, v_avg=1.0)
    ts = np.linspace(0.0, float(traj.t_keyframes[-1]), 300)
    P = np.asarray([traj.update(float(t))["x"] for t in ts])
    sp = np.linalg.norm(
        np.asarray([traj.update(float(t))["x_dot"] for t in ts]), axis=1)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    print("seed %2d duration %5.2fs path %5.2fm mean_sp %.2f max_sp %.2f finite %s"
          % (seed, traj.t_keyframes[-1], d, sp.mean(), sp.max(),
             bool(np.all(np.isfinite(P)))))
    print("     bbox min", P.min(axis=0).round(2), "max", P.max(axis=0).round(2))
