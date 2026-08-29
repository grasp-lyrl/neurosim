"""Does the oracle collide with obstacles it PREDICTED it would clear?

The expert collides in 13/40 v20 episodes despite full state knowledge. The
two candidate causes make opposite predictions about one measurable quantity:

  magnitude / feasibility -- the planner cannot find a safe dodge at all, so
      it returns None and the vehicle flies the nominal path into the throw.
      Signature: high failed-plan rate; collisions on UNPLANNED obstacles.

  prediction / tracking   -- the planner finds a dodge, predicts a healthy
      clearance, and the vehicle still gets hit. Signature: low failed-plan
      rate, but predicted_min_clearance >> achieved clearance.

Both are already recorded and just never joined. `oracle_action` appends
`{object_id, peak_offset_m, predicted_min_clearance_m, rise_time_s, ...}` per
plan, and the task keeps `_encounter_min_clearance[object_id]` -- the
clearance actually achieved against that same obstacle. Joining on object_id
turns "the oracle is weak" into a per-encounter prediction error.

Reported:
  * failed plans as a fraction of planning attempts
  * peak_offset_m against max(candidate_offsets_m) = 1.00 m, i.e. is the
    planner actually pressed against its magnitude cap
  * predicted vs achieved clearance, and the gap between them
  * for collision episodes, whether the obstacle hit was one the planner had
    a plan for and predicted clear

`clearance` throughout is SURFACE-TO-SURFACE (the 0.30 m combined radius is
already subtracted), so clearance == 0 is contact, not a 0.30 m miss.

Usage:
    python oracle_prediction_probe.py <config.yaml> [episodes]
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
    "applications/rl/configs/velocity_dodge_dynamic_v20_effnet.yaml"
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEED0 = 9001                       # same seeds as return_baseline.py / the gates
ORACLE_MAX_CANDIDATE_M = 1.00      # max(candidate_offsets_m)

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
dim = int(np.prod(env.action_space.shape))
dodge_clear_m = float(getattr(env._task, "dodge_clearance_m", 0.10))

plans_total, plans_failed = 0, 0
peak_offsets, predicted, achieved, gaps = [], [], [], []
rise_times = []
term = collections.Counter()
collided_with_plan, collided_no_plan, collisions = 0, 0, 0

for i in range(EPISODES):
    env.reset(seed=SEED0 + i)
    env._trajectory_dodge_expert = expert_for_task(env._task)
    env._trajectory_expert_plans = 0
    env._trajectory_expert_failed_plans = 0
    env._trajectory_expert_plan_diagnostics = []
    while True:
        action = oracle_action(env)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            reason = info.get("termination_reason") or "timeout"
            term[reason] += 1
            break

    plans_total += int(getattr(env, "_trajectory_expert_plans", 0))
    plans_failed += int(getattr(env, "_trajectory_expert_failed_plans", 0))

    # Best (last) plan per obstacle: the planner replans as the encounter
    # develops, and the operative prediction is the one it acted on last.
    best = {}
    for d in getattr(env, "_trajectory_expert_plan_diagnostics", []):
        best[int(d["object_id"])] = d
    actual = dict(getattr(env._task, "_encounter_min_clearance", {}) or {})

    for oid, d in best.items():
        peak_offsets.append(float(d["peak_offset_m"]))
        rise_times.append(float(d["rise_time_s"]))
        pred = float(d["predicted_min_clearance_m"])
        if oid in actual and np.isfinite(actual[oid]) and np.isfinite(pred):
            predicted.append(pred)
            achieved.append(float(actual[oid]))
            gaps.append(pred - float(actual[oid]))

    if reason == "obstacle_collision":
        collisions += 1
        # Which obstacle was hit: the one whose achieved clearance is lowest.
        if actual:
            hit = min(actual, key=lambda k: actual[k])
            if hit in best:
                collided_with_plan += 1
            else:
                collided_no_plan += 1

env.close()

pk = np.asarray(peak_offsets)
pr, ac, gp = np.asarray(predicted), np.asarray(achieved), np.asarray(gaps)

print("\nconfig: %s   episodes: %d   dodge_clearance_m: %.2f"
      % (CFG, EPISODES, dodge_clear_m))
print("terminations: %s" % dict(term.most_common()))

print("\nplanning:")
attempts = plans_total + plans_failed
print("   plans made %d, failed %d  -> failure rate %.1f%%"
      % (plans_total, plans_failed, 100.0 * plans_failed / max(attempts, 1)))
if pk.size:
    at_cap = int((pk >= ORACLE_MAX_CANDIDATE_M - 1e-6).sum())
    print("   peak_offset_m: median %.2f  p90 %.2f  max %.2f"
          % (float(np.median(pk)), float(np.percentile(pk, 90)), float(pk.max())))
    print("   plans pressed against the %.2f m cap: %d/%d (%.0f%%)"
          % (ORACLE_MAX_CANDIDATE_M, at_cap, pk.size, 100.0 * at_cap / pk.size))
    print("   rise_time_s: median %.2f" % float(np.median(rise_times)))

if pr.size:
    print("\npredicted vs achieved clearance (%d planned encounters):" % pr.size)
    print("   predicted: median %+.3f m   p10 %+.3f m" % (
        float(np.median(pr)), float(np.percentile(pr, 10))))
    print("   achieved:  median %+.3f m   p10 %+.3f m" % (
        float(np.median(ac)), float(np.percentile(ac, 10))))
    print("   gap (predicted - achieved): median %+.3f m   p90 %+.3f m   max %+.3f m"
          % (float(np.median(gp)), float(np.percentile(gp, 90)), float(gp.max())))
    betrayed = int(((pr > dodge_clear_m) & (ac <= 0.0)).sum())
    print("   predicted a clear but actually made CONTACT: %d/%d (%.0f%%)"
          % (betrayed, pr.size, 100.0 * betrayed / pr.size))

print("\ncollision episodes: %d/%d" % (collisions, EPISODES))
print("   hit obstacle HAD a plan:    %d" % collided_with_plan)
print("   hit obstacle had NO plan:   %d" % collided_no_plan)

print("\nVERDICT:")
fail_rate = 100.0 * plans_failed / max(attempts, 1)
if collided_no_plan > collided_with_plan:
    print("   Feasibility-limited: most collisions are on obstacles the planner")
    print("   never found a dodge for (failure rate %.1f%%). Widening" % fail_rate)
    print("   candidate_offsets_m / relaxing safety_margin_m is the lever.")
elif pr.size and float(np.median(gp)) > dodge_clear_m:
    print("   Prediction/tracking-limited: the planner predicts clearances it")
    print("   does not achieve (median gap %+.3f m). A bigger dodge menu will"
          % float(np.median(gp)))
    print("   not help; the plan is not being realised. Look at commit time")
    print("   (oracle_commit_probe.py) and offset tracking")
    print("   (offset_tracking_probe.py).")
else:
    print("   Neither signature dominates. Read the numbers above rather than")
    print("   taking a verdict from this line.")
