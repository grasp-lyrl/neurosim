"""Why does a privileged oracle survive less than a sine wave?

Across v4, v7 and v8 the oracle -- which sees every obstacle's position and
velocity -- is out-survived by a blind 0.5 Hz weave (v7: 30% vs 55%; v8: 45%
vs 60%). That disqualifies it as a BC expert and as a performance ceiling,
and no reward reweighting fixes it: the oracle crashes MORE often (11/20 vs
8/20 on v7), so any weighting that ranks by survival ranks the weave first.

The physics says it should do better. Clearing the 0.30 m combined hit radius
at offset_accel_limit 1.5 m/s^2 under the 0.8 m/s rate cap takes ~0.64 s,
against 1.1-1.5 s of flight. The oracle has roughly twice the time it needs
and full state knowledge, yet clears only 39-54%.

Two candidate causes, both measurable:
  commit time  -- when does the offset command actually start moving,
                  relative to closest approach? If it starts at 0.3 s when
                  0.64 s is needed, the dodge cannot finish in time.
  direction thrash -- does the commanded cross-track direction flip during
                  an encounter? Re-solving every step with no commitment
                  produces a command that averages to nothing.

Reports both against what was achieved: peak cross-track displacement by the
moment of closest approach, versus the 0.30 m required.
"""
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from evaluate_velocity_dodge_oracle import oracle_action, expert_for_task
from train_sb3 import load_experiment_config

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "applications/rl/configs/velocity_dodge_dynamic_v7.yaml"
SEED0, EPISODES = 9001, 20
MOVE_THRESHOLD_M = 0.05     # command counts as "started" past this
REQUIRED_S = 0.64           # 0.30 m at 1.5 m/s^2 under an 0.8 m/s rate cap

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
dim = int(np.prod(env.action_space.shape))

commits, flips, peaks, tcas = [], [], [], []
for i in range(EPISODES):
    # Per-step trace of (sim_time, privileged, offset command)
    trace = []
    obs, _ = env.reset(seed=SEED0 + i)
    env._trajectory_dodge_expert = expert_for_task(env._task)
    env._trajectory_expert_plans = 0
    env._trajectory_expert_failed_plans = 0
    env._trajectory_expert_plan_diagnostics = []
    while True:
        # Read the privileged channel off the observation, as the working
        # probes do. Calling the task method behind a bare except silently
        # yielded None for every step and traced zero encounters.
        obs_priv = np.asarray(obs["privileged"], dtype=np.float64)
        act = oracle_action(env)
        cmd = np.asarray(getattr(env, "_offset_cmd", np.zeros(3)), dtype=np.float64).copy()
        trace.append((float(env.sim.time), obs_priv, cmd))
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break

    # Segment the trace into encounters: contiguous runs with a live threat
    # flag, ending at the local minimum of time-to-closest-approach.
    seg = []
    for t, priv, cmd in trace:
        live = priv[0] > 0.5
        if live:
            seg.append((t, float(priv[8]), cmd))
        elif seg:
            if len(seg) >= 3:
                times = np.asarray([s[0] for s in seg])
                tca = np.asarray([s[1] for s in seg])
                cmds = np.asarray([s[2] for s in seg])
                k = int(np.argmin(tca))
                impact_t = times[k]
                # Lateral component: project onto the frame's horizontal axes
                lateral = cmds[:, [0, 2]]
                mag = np.linalg.norm(lateral, axis=1)
                started = np.nonzero(mag[: k + 1] > MOVE_THRESHOLD_M)[0]
                if started.size:
                    commits.append(float(impact_t - times[started[0]]))
                else:
                    commits.append(0.0)
                # Direction flips: sign changes of the dominant lateral axis
                dom = lateral[: k + 1, 0]
                sgn = np.sign(dom[np.abs(dom) > MOVE_THRESHOLD_M])
                flips.append(int(np.sum(np.abs(np.diff(sgn)) > 1)) if sgn.size > 1 else 0)
                peaks.append(float(mag[: k + 1].max()))
                tcas.append(float(tca[0]))
            seg = []
env.close()

c = np.asarray(commits); f = np.asarray(flips); p = np.asarray(peaks)
print("\nconfig: %s" % CFG)
print("encounters traced: %d" % len(c))
if len(c):
    print("\ncommit lead time (offset command starts, before closest approach)")
    print("   median %.2f s   p10 %.2f   p90 %.2f   required %.2f s"
          % (np.median(c), np.percentile(c, 10), np.percentile(c, 90), REQUIRED_S))
    print("   encounters with LESS lead than required: %.0f%%"
          % (100 * float((c < REQUIRED_S).mean())))
    print("\ndirection flips before closest approach")
    print("   median %.1f   mean %.2f   encounters with >=1 flip: %.0f%%"
          % (np.median(f), f.mean(), 100 * float((f >= 1).mean())))
    print("\npeak lateral displacement achieved by closest approach")
    print("   median %.2f m  p90 %.2f m   needed ~0.30 m"
          % (np.median(p), np.percentile(p, 90)))
    print("   encounters reaching 0.30 m: %.0f%%" % (100 * float((p >= 0.30).mean())))
