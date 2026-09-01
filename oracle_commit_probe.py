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
SEED0 = 9001
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MOVE_THRESHOLD_M = 0.05     # command counts as "started" past this

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
dim = int(np.prod(env.action_space.shape))

# Size the timing check from the expert and task actually under test.  The
# old 0.64 s constant described v7 (0.30 m at 1.5 m/s^2); it became stale as
# soon as v4 raised acceleration to 5.0 m/s^2.  For a symmetric accelerate /
# decelerate move, distance d takes 2*sqrt(d/a) until the rate cap binds, and
# d/v + v/a afterwards.
probe_expert = expert_for_task(env._task)
TARGET_M = float(probe_expert.config.candidate_offsets_m[0])
accel = float(env._task.offset_accel_limit_mps2)
rate = float(np.min(np.asarray(env._task.offset_rate_limits_mps, dtype=np.float64)[:2]))
REQUIRED_S = (
    2.0 * np.sqrt(TARGET_M / accel)
    if TARGET_M <= rate**2 / accel
    else TARGET_M / rate + rate / accel
)


def privileged_channel(observation):
    """Return the first obstacle slot from old dict or flat actor layouts."""
    if isinstance(observation, dict):
        privileged = observation["privileged"]
    else:
        # With privileged_actor=true the obstacle slots are appended to the
        # state vector. v4 is this flat 30-D layout (18 state + one 12-D
        # obstacle slot), while the older probe only accepted a dict.
        width = int(env._task.privileged_channel_dim)
        privileged = np.asarray(observation).reshape(-1)[-width:]
    return np.asarray(privileged, dtype=np.float64).reshape(-1)

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
        obs_priv = privileged_channel(obs)
        act = oracle_action(env)
        cmd = np.asarray(getattr(env, "_offset_cmd", np.zeros(3)), dtype=np.float64).copy()
        trace.append((float(env.sim.time), obs_priv, cmd))
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break

    # Segment the trace into encounters: contiguous runs with a live threat
    # flag, ending at the local minimum of time-to-closest-approach.
    seg = []
    # A false-live sentinel flushes an encounter that lasts until episode
    # termination (notably obstacle collisions).
    empty_priv = np.zeros(env._task.privileged_slot_dim, dtype=np.float64)
    for t, priv, cmd in trace + [(float("inf"), empty_priv, np.zeros(3))]:
        live = priv[0] > 0.5
        if live:
            # TCA is the final value in both the legacy 9-D slot and v4's
            # acceleration-aware 12-D slot.
            seg.append((t, float(priv[env._task.privileged_slot_dim - 1]), cmd))
        elif seg:
            if len(seg) >= 3:
                times = np.asarray([s[0] for s in seg])
                tca = np.asarray([s[1] for s in seg])
                cmds = np.asarray([s[2] for s in seg])
                k = int(np.argmin(tca))
                impact_t = times[k]
                # Dynamics coordinates use z as vertical, so x/y are the
                # horizontal plane. The old [0,2] slice mixed lateral and
                # vertical motion and understated v4's achieved dodge.
                lateral = cmds[:, :2]
                mag = np.linalg.norm(lateral, axis=1)
                started = np.nonzero(mag[: k + 1] > MOVE_THRESHOLD_M)[0]
                if started.size:
                    commits.append(float(impact_t - times[started[0]]))
                else:
                    commits.append(0.0)
                # Direction flips: sign changes along whichever horizontal
                # axis dominates this encounter.
                dominant_axis = int(np.argmax(np.max(np.abs(lateral[: k + 1]), axis=0)))
                dom = lateral[: k + 1, dominant_axis]
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
    print("   median %.2f m  p90 %.2f m   first candidate %.2f m"
          % (np.median(p), np.percentile(p, 90), TARGET_M))
    print("   encounters reaching %.2f m: %.0f%%"
          % (TARGET_M, 100 * float((p >= TARGET_M).mean())))
