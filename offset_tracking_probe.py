"""Where do 4 metres of deviation come from when 0.5 m was commanded?

Oracle authority on v7: peak COMMANDED offset median 0.351 m (p90 0.501),
peak ACHIEVED deviation median 0.422 m but p90 3.454 m and max 5.321 m. The
expert asks for half a metre and the vehicle ends up several metres off the
reference. Raising offset_rate_limits_mps 0.8 -> 1.5 made it worse (p90
4.891, max 8.949, out_of_bounds 3 -> 5), so it is not a rate limit.

That divergence -- not the choice of dodge -- is the likely reason a
privileged oracle is out-survived by a blind sine weave: the weave's command
is smooth by construction, while the expert's is step-like, beginning
abruptly when a threat is detected. The code already records this failure
mode: a commanded sidestep produces a velocity error the SE3 reference cannot
explain and the controller answers with attitude.

Traces the worst episode step by step -- commanded offset, achieved
deviation, tracking error against the SHIFTED reference, attitude, and the
command's step size -- so the point where the gap opens is visible rather
than inferred.
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

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)

worst = None
for i in range(EPISODES):
    obs, _ = env.reset(seed=SEED0 + i)
    env._trajectory_dodge_expert = expert_for_task(env._task)
    env._trajectory_expert_plans = 0
    env._trajectory_expert_failed_plans = 0
    env._trajectory_expert_plan_diagnostics = []
    rows, prev_cmd = [], np.zeros(3)
    while True:
        act = oracle_action(env)
        cmd = np.asarray(getattr(env, "_offset_cmd", np.zeros(3)), dtype=np.float64).copy()
        flat = env._nominal_flat()
        x = np.asarray(env.sim.dynamics.state["x"], dtype=np.float64)
        dev = x - np.asarray(flat["x"], dtype=np.float64)
        # Tracking error against the reference the controller was actually
        # given: nominal shifted by the commanded offset.
        track = float(np.linalg.norm(dev - cmd))
        q = np.asarray(env.sim.dynamics.state["q"], dtype=np.float64)
        # Tilt from level: angle between body z and world z.
        w, xq, yq, zq = q[3], q[0], q[1], q[2]
        bz = np.array([2*(xq*zq + w*yq), 2*(yq*zq - w*xq), 1 - 2*(xq*xq + yq*yq)])
        tilt = float(np.degrees(np.arccos(np.clip(bz[2], -1, 1))))
        rows.append((float(env.sim.time), float(np.linalg.norm(cmd)),
                     float(np.linalg.norm(dev)), track, tilt,
                     float(np.linalg.norm(cmd - prev_cmd))))
        prev_cmd = cmd
        obs, _, t, tr, info = env.step(act)
        if t or tr:
            reason = info.get("termination_reason") or "timeout"
            break
    peak = max(r[2] for r in rows)
    if worst is None or peak > worst[0]:
        worst = (peak, SEED0 + i, rows, reason)
env.close()

peak, seed, rows, reason = worst
print("\nworst episode: seed %d, peak deviation %.2f m, ended %s" % (seed, peak, reason))
print("\n%7s %9s %9s %9s %8s %9s"
      % ("t_s", "|cmd|", "|dev|", "track_err", "tilt_deg", "cmd_step"))
k = int(np.argmax([r[2] for r in rows]))
lo = max(0, k - 25)
for r in rows[lo:k + 5]:
    print("%7.2f %9.3f %9.3f %9.3f %8.1f %9.3f" % r)
allrows = np.asarray(rows)
print("\nepisode summary: max |cmd| %.3f  max |dev| %.3f  max track_err %.3f  max tilt %.1f deg"
      % (allrows[:, 1].max(), allrows[:, 2].max(), allrows[:, 3].max(), allrows[:, 4].max()))
print("largest single-step command change: %.3f m" % allrows[:, 5].max())
