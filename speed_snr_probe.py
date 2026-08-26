"""Does making dynamic obstacles faster help or hurt detectability?

The dodge needs a fixed warning time set by the acceleration budget, not by
the obstacle: clearing the 0.30 m hit radius at 0.8 m/s^2 takes
sqrt(2*0.3/0.8) = 0.87 s. The obstacle must therefore be detected at
d = v_obs * 0.87, and at that range

    angular rate  w = v_obs/d = 1/t_warn        -- independent of v_obs
    angular size  th = 2r/d   = 2r/(v_obs*t_w)  -- falls as 1/v_obs

so event mass at the decision point should scale as 1/v_obs while background
egomotion stays pinned by the 1 m/s cruise. Prediction: faster obstacles are
strictly WORSE at the moment the decision has to be made, even though they
produce more events in total over the throw.

A whole-episode average cannot see this -- it pools frames where the
obstacle is 6 m away with frames where it is already past. Signal is
therefore bucketed by time to closest approach, and the bucket straddling
t_warn is the one that decides the question.

Uses the paired obstacles-on/off difference (identical zero-action paths),
so no projection of the obstacle into the image is needed.
"""
import collections
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG_OBS = "applications/rl/configs/velocity_dodge_dynamic.yaml"
CFG_QUIET = "applications/rl/configs/velocity_dodge_dynamic_quiet.yaml"
SEEDS = list(range(9001, 9011))
SPEEDS = [2.5, 5.0, 10.0]
BLOCK = 4
POS_TOL_M = 0.02
BUCKETS = [(0.0, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 4.0)]
T_WARN = 0.87


def set_key(node, key, value):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                node[k] = value
            else:
                set_key(v, key, value)
    elif isinstance(node, list):
        for v in node:
            set_key(v, key, value)


def blocks(ev):
    amp = np.abs(np.asarray(ev, dtype=np.float32)).sum(axis=0)
    H, W = amp.shape
    h, w = H // BLOCK, W // BLOCK
    return amp[: h * BLOCK, : w * BLOCK].reshape(h, BLOCK, w, BLOCK).sum(axis=(1, 3))


def rollout(env, seed, dim, want_priv):
    o, _ = env.reset(seed=seed)
    maps, pos, priv = [], [], []
    while True:
        maps.append(blocks(o["events"]))
        pos.append(np.asarray(env.sim.dynamics.state["x"], dtype=np.float64).copy())
        if want_priv:
            priv.append(np.asarray(o.get("privileged", np.zeros(9)),
                                   dtype=np.float64).copy())
        o, _, term, trunc, _ = env.step(np.zeros(dim, dtype=np.float32))
        if term or trunc:
            break
    return maps, np.asarray(pos), priv


def collect(cfg_path, speed, seeds, want_priv):
    cfg = load_experiment_config(cfg_path)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    set_key(env_cfg, "kinematic_speed_mps", speed)
    env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
    dim = int(np.prod(env.action_space.shape))
    env.sim.safety.has_obstacle_collision = lambda habitat_pos: False
    out = {s: rollout(env, s, dim, want_priv) for s in seeds}
    env.close()
    return out


print("obstacle-attributable signal by TIME TO CLOSEST APPROACH, per obstacle speed")
print("paired zero-action rollouts, %d seeds, ts_decay_ms as configured" % len(SEEDS))
print("dodge at 0.8 m/s^2 needs t_warn = %.2f s -- read the bucket containing it\n"
      % T_WARN)
hdr = "%7s %9s" % ("v_obs", "metric")
for lo, hi in BUCKETS:
    hdr += " %11s" % ("%.1f-%.1fs" % (lo, hi))
print(hdr, flush=True)

for speed in SPEEDS:
    with_obs = collect(CFG_OBS, speed, SEEDS, True)
    without = collect(CFG_QUIET, speed, SEEDS, False)
    pk = collections.defaultdict(list)
    npx = collections.defaultdict(list)
    for s in SEEDS:
        ma, pa, pv = with_obs[s]
        mb, pb, _ = without[s]
        n = min(len(ma), len(mb), len(pv))
        if n < 5 or float(np.abs(pa[:n] - pb[:n]).max()) > POS_TOL_M:
            continue
        for i in range(n):
            p = pv[i]
            if p[0] <= 0.5:
                continue
            tca = float(p[8])
            if not np.isfinite(tca) or tca < 0:
                continue
            d = np.abs(ma[i] - mb[i])
            sd = float(mb[i].std())
            for lo, hi in BUCKETS:
                if lo <= tca < hi:
                    pk[(lo, hi)].append(float(d.max()) / max(sd, 1e-9))
                    # blocks holding any obstacle-attributable mass at all
                    npx[(lo, hi)].append(float((d > 0.5 * sd).sum()))
                    break
    for name, table in (("peak_z", pk), ("blocks", npx)):
        row = "%7.1f %9s" % (speed, name)
        for b in BUCKETS:
            v = table.get(b, [])
            row += " %11s" % ("%.2f(%d)" % (np.mean(v), len(v)) if v else "--")
        print(row, flush=True)
