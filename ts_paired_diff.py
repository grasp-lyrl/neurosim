"""Obstacle signal vs static-scene background, by time-surface window.

Directly answers: does a shorter window make obstacles stand out more
against the static scene?

Both earlier attempts tried to LOCATE the obstacle in the image (project its
body-frame bearing to a pixel, or take the argmax of an event profile) and
both failed -- calib_proj measured corr(bearing, peak_col) = -0.04, i.e. the
whole-frame event peak is castle structure, not a 3-pixel sphere. Any metric
built on a guessed pixel inherits that error.

This needs no pixel. velocity_dodge_dynamic_quiet.yaml differs from the
dynamic config in exactly one key (dynamic_obstacles.enabled), so flying the
SAME seed with the SAME open-loop action sequence in both gives two event
streams whose only difference is the obstacle. Then

    signal     = sum |A - B|      obstacle-attributable event mass
    background = sum  B           static-scene event mass
    frac       = signal / background

and a local salience term: the largest single block of obstacle-attributable
mass, in units of the block-to-block spread of the background, which is
closer to what a conv filter keys on than any global ratio.

Actions are zeros so the two runs fly the same path by construction. That
assumption is CHECKED, not assumed: max position deviation is reported, and
a run whose paths diverge is discarded rather than silently averaged in.

Event maps are reduced by 4x4 block SUM before storing -- exact for mass,
and 16x cheaper to hold for a whole episode.
"""
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG_OBS = sys.argv[1] if len(sys.argv) > 1 else \
    "applications/rl/configs/velocity_dodge_dynamic.yaml"
CFG_QUIET = sys.argv[2] if len(sys.argv) > 2 else \
    "applications/rl/configs/velocity_dodge_dynamic_quiet.yaml"
SEEDS = list(range(9001, 9011))
WINDOWS = [5.0, 10.0, 20.0, 50.0, 100.0]
BLOCK = 4
POS_TOL_M = 0.02          # paths must agree this closely to be comparable


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
    """4x4 block sums of |events|, summed over polarity channels."""
    amp = np.abs(np.asarray(ev, dtype=np.float32)).sum(axis=0)
    H, W = amp.shape
    h, w = H // BLOCK, W // BLOCK
    return amp[: h * BLOCK, : w * BLOCK].reshape(h, BLOCK, w, BLOCK).sum(axis=(1, 3))


def rollout(env, seed, dim):
    """Open-loop zero-action rollout; returns block maps and positions."""
    o, _ = env.reset(seed=seed)
    maps, pos = [], []
    while True:
        maps.append(blocks(o["events"]))
        pos.append(np.asarray(env.sim.dynamics.state["x"], dtype=np.float64).copy())
        o, _, term, trunc, _ = env.step(np.zeros(dim, dtype=np.float32))
        if term or trunc:
            break
    return maps, np.asarray(pos)


def collect(cfg_path, ts, seeds):
    cfg = load_experiment_config(cfg_path)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    set_key(env_cfg, "ts_decay_ms", ts)
    env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
    dim = int(np.prod(env.action_space.shape))
    # Without this the obstacle run ends on the first collision while the
    # quiet run flies on, so the pair would be compared over a handful of
    # early frames -- and on the static task a zero-action rollout hits an
    # obstacle almost immediately (39/40 episodes).
    env.sim.safety.has_obstacle_collision = lambda habitat_pos: False
    out = {s: rollout(env, s, dim) for s in seeds}
    env.close()
    return out


print(f"obstacle-attributable event mass vs time-surface window, {len(SEEDS)} seeds")
print("paired same-seed zero-action rollouts, obstacles on vs off\n")
print(f"{'ts_ms':>7} {'pairs':>6} {'steps':>7} {'signal':>10} {'bg':>10}"
      f" {'frac':>7} {'peak_z':>7} {'maxdev_m':>9}", flush=True)

for ts in WINDOWS:
    with_obs = collect(CFG_OBS, ts, SEEDS)
    without = collect(CFG_QUIET, ts, SEEDS)
    sig, bg, pk, devs, nstep, npair = [], [], [], [], 0, 0
    for s in SEEDS:
        ma, pa = with_obs[s]
        mb, pb = without[s]
        n = min(len(ma), len(mb))
        if n < 5:
            continue
        dev = float(np.abs(pa[:n] - pb[:n]).max()) if pa.size and pb.size else 0.0
        devs.append(dev)
        if dev > POS_TOL_M:
            # Paths diverged: the difference is no longer attributable to the
            # obstacle alone, so this pair cannot be used.
            continue
        npair += 1
        for i in range(n):
            d = np.abs(ma[i] - mb[i])
            b = mb[i]
            sig.append(float(d.sum()))
            bg.append(float(b.sum()))
            sd = float(b.std())
            pk.append(float(d.max()) / max(sd, 1e-9))
            nstep += 1
    if npair and nstep:
        S, B = float(np.mean(sig)), float(np.mean(bg))
        print(f"{ts:>7.0f} {npair:>6d} {nstep:>7d} {S:>10.0f} {B:>10.0f}"
              f" {S/max(B,1e-9):>7.3f} {np.mean(pk):>7.2f}"
              f" {max(devs) if devs else 0.0:>9.4f}", flush=True)
    else:
        worst = max(devs) if devs else float("nan")
        note = "--- no usable pairs, max path dev %.3f m ---" % worst
        print("%7.0f %6d %50s" % (ts, npair, note), flush=True)
