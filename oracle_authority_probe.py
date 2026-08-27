"""Does the oracle ever command a dodge large enough to clear?

Direct inspection of one v7 episode showed the oracle's commanded lateral
offset peaking at 0.083 m. Clearing the 0.30 m combined hit radius needs
0.30 m. If that holds across seeds, the expert is not failing at timing or
committing to a direction -- it is barely dodging at all, which explains
39-54% clearance and being out-survived by a 0.8 m sine weave.

Reports per episode: peak commanded offset, peak achieved lateral deviation
from the reference, episode length, and termination -- against the 0.30 m the
geometry requires.
"""
import collections
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
NEEDED_M = 0.30

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
dim = int(np.prod(env.action_space.shape))

peak_cmd, peak_act, lengths = [], [], []
term = collections.Counter()
for i in range(EPISODES):
    obs, _ = env.reset(seed=SEED0 + i)
    env._trajectory_dodge_expert = expert_for_task(env._task)
    env._trajectory_expert_plans = 0
    env._trajectory_expert_failed_plans = 0
    env._trajectory_expert_plan_diagnostics = []
    pc = pa = 0.0
    n = 0
    while True:
        act = oracle_action(env)
        pc = max(pc, float(np.linalg.norm(
            np.asarray(getattr(env, "_offset_cmd", np.zeros(3)), dtype=np.float64))))
        # Achieved deviation from the reference, which is what actually
        # clears an obstacle -- the command is only the request.
        flat = env._nominal_flat()
        dev = (np.asarray(env.sim.dynamics.state["x"], dtype=np.float64)
               - np.asarray(flat["x"], dtype=np.float64))
        pa = max(pa, float(np.linalg.norm(dev)))
        obs, _, t, tr, info = env.step(act)
        n += 1
        if t or tr:
            term[info.get("termination_reason") or "timeout"] += 1
            break
    peak_cmd.append(pc); peak_act.append(pa); lengths.append(n)

pc = np.asarray(peak_cmd); pa = np.asarray(peak_act); ln = np.asarray(lengths)
print("\nconfig: %s" % CFG)
print("oracle authority over %d episodes (need %.2f m to clear)\n" % (EPISODES, NEEDED_M))
print("peak COMMANDED offset : median %.3f m  p90 %.3f m  max %.3f m"
      % (np.median(pc), np.percentile(pc, 90), pc.max()))
print("peak ACHIEVED deviation: median %.3f m  p90 %.3f m  max %.3f m"
      % (np.median(pa), np.percentile(pa, 90), pa.max()))
print("episodes commanding >= %.2f m : %.0f%%" % (NEEDED_M, 100 * float((pc >= NEEDED_M).mean())))
print("episode length          : median %.0f steps  min %d  max %d"
      % (np.median(ln), ln.min(), ln.max()))
print("terminations: %s" % dict(term.most_common()))
env.close()
