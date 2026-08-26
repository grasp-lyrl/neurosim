"""At what obstacle density can the oracle actually fly v4?

v4 (drone-aimed, ballistic throws, 0.4 m scatter, 12 s episodes) is
infeasible as configured: the privileged oracle cleared 39.4% of threats and
completed 0% of episodes, worse than the do-nothing control's 4.0%. Every
blind weave also scored 0%, so that comparison said nothing -- when nothing
succeeds, "blind does not beat sighted" is not evidence about perception.

Re-aiming is NOT the cause: reaim_commit_time_s 10.0 exceeds the ~1.4 s
flight time, so `range/speed <= commit` always holds and throws are ballistic
from launch (dynamic_obstacles.py:1135). Nor is authority obviously the
cause: deviating the 0.30 m hit radius at 1.5 m/s^2 takes 0.63 s against
~1.4 s of flight.

What is left is density. The field runs 16 concurrent obstacles, ~13 of them
bystanders, so every dodge is constrained by things the vehicle must not hit,
and each dodge changes its velocity, which changes how later throws lead it.

Sweeps max_concurrent and reports oracle vs control. The operating point we
want is the lowest density at which the ORACLE succeeds clearly -- there is
no point measuring blind baselines above it.
"""
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from evaluate_velocity_dodge_oracle import oracle_action, expert_for_task
from train_sb3 import load_experiment_config

CFG = sys.argv[1] if len(sys.argv) > 1 else "applications/rl/configs/velocity_dodge_dynamic_v4.yaml"
SEED0, EPISODES = 9001, 20
DENSITIES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2, 4, 8, 16]


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


print("config: %s" % CFG)
print("lowest density at which the ORACLE clearly succeeds is the target\n")
print("%8s %10s %9s %9s %11s  %s"
      % ("concur", "arm", "episode", "cleared", "bystanders", "threats/ep"), flush=True)

for concur in DENSITIES:
    cfg = load_experiment_config(CFG)
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["enable_visualization"] = False
    set_key(env_cfg, "max_concurrent", concur)
    env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
    dim = int(np.prod(env.action_space.shape))
    env.sim.safety.has_obstacle_collision = lambda habitat_pos: False
    manager = env.sim.visual_backend._dynamic_obstacles
    agent_r = float(manager._agent_radius)
    dodge_clear = float(env._task.dodge_clearance_m)

    for mode in ("control", "oracle"):
        T = C = B = BH = W = 0
        for i in range(EPISODES):
            env.reset(seed=SEED0 + i)
            if mode == "oracle":
                env._trajectory_dodge_expert = expert_for_task(env._task)
                env._trajectory_expert_plans = 0
                env._trajectory_expert_failed_plans = 0
                env._trajectory_expert_plan_diagnostics = []
            tracks, actual = {}, {}
            while True:
                now = float(env.sim.time)
                hp = env.sim.safety.dynamics_to_habitat(
                    np.asarray(env.sim.dynamics.state["x"], dtype=np.float64))
                for oid, item in manager._active.items():
                    opos = np.asarray(item.obj.translation, dtype=np.float64).copy()
                    tracks.setdefault(oid, []).append((now, opos, float(item.collision_radius)))
                    d = float(np.linalg.norm(hp - opos)) - agent_r - float(item.collision_radius)
                    actual[oid] = min(actual.get(oid, np.inf), d)
                act = oracle_action(env) if mode == "oracle" else np.zeros(dim, dtype=np.float32)
                _, _, term, trunc, info = env.step(act)
                if term or trunc:
                    W += int(bool(info.get("is_success", False)))
                    break
            traj = env._nominal_trajectory
            end = float(getattr(traj, "t_keyframes", [float(env.episode_seconds)])[-1])
            dur = min(float(env.episode_seconds), end - 1e-3)
            for oid, samples in tracks.items():
                if len(samples) < 2:
                    continue
                times = np.asarray([s[0] for s in samples])
                pos = np.asarray([s[1] for s in samples])
                rad = float(samples[0][2])
                nom = np.asarray([env.sim.safety.dynamics_to_habitat(
                    np.asarray(traj.update(float(min(t, dur)))["x"], dtype=np.float64))
                    for t in times])
                if float(np.linalg.norm(pos - nom, axis=1).min()) - agent_r - rad <= 0.0:
                    T += 1
                    if actual[oid] > dodge_clear:
                        C += 1
                else:
                    B += 1
                    if actual[oid] <= 0.0:
                        BH += 1
        print("%8d %10s %8.1f%% %8.1f%% %11.2f  %.2f"
              % (concur, mode, 100 * W / EPISODES, 100 * C / max(T, 1),
                 B / EPISODES, T / EPISODES), flush=True)
    env.close()
