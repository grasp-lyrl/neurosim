"""Static-task baseline scored on BLOCKERS only, with an exogenous denominator.

The per-encounter clearance metric used so far is confounded three ways:

  1. Censored by termination. An arm that dies on obstacle 1 is never scored
     against obstacles 2-6. Measured: enc/ep ran 1.35 (control) to 3.92
     (const+0.80) and per-encounter clearance tracked it almost perfectly,
     27.8% -> 77.1%, which is the shape of survival, not skill.
  2. Diluted by free obstacles. The layout alternates blocking obstacles
     (on the nominal path) with trap obstacles (offset to one side). A
     do-nothing policy clears every trap for free simply by flying the
     nominal path, so including traps drags every arm toward the trap
     fraction. Uncensored control scored 41.4%, and 3 free traps out of 6
     obstacles is 50%.
  3. Endogenous denominator. velocity_dodge.py counts an obstacle as an
     "encounter" only if its closest approach came within
     clearance_threshold_m -- but closest approach is what the policy
     controls. Dodging well pushes an obstacle out of the denominator
     instead of scoring it as cleared, so a perfect dodge tends to 0/0.

Here the denominator is fixed by the environment before the policy acts:
sample the NOMINAL trajectory, and count an obstacle as a blocker if the
nominal path would have passed within the combined hit radius of it. That
set depends only on the seed. The numerator is how many of those blockers
the actual flight kept clear of. Traps are reported separately -- a policy
that dodges into a trap should be penalised, and that is invisible in any
metric that pools the two.

Collision termination is disabled so every obstacle is scored in every
episode.
"""
import collections
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from evaluate_velocity_dodge_oracle import oracle_action, expert_for_task
from train_sb3 import load_experiment_config

CFG = "applications/rl/configs/velocity_dodge_static.yaml"
SEED0, EPISODES = 9001, 40


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return 100 * (c - h), 100 * (c + h)


cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
dim = int(np.prod(env.action_space.shape))
env.sim.safety.has_obstacle_collision = lambda habitat_pos: False

manager = env.sim.visual_backend._dynamic_obstacles
dodge_clear = float(env._task.dodge_clearance_m)
agent_r = float(manager._agent_radius)
print("STATIC task, blockers-only scoring, exogenous denominator")
print("n=%d, seeds %d+, collision termination disabled" % (EPISODES, SEED0))
print("blocker := nominal path passes within combined hit radius\n")
print("%12s %8s %10s %8s %10s %8s  %s"
      % ("arm", "episode", "blockers", "cleared", "traps", "trap_hit", "wilson"),
      flush=True)


def episode(mode, const, seed):
    """One rollout; returns (blockers, cleared, traps, trap_hits, success)."""
    env.reset(seed=seed)
    if mode == "oracle":
        env._trajectory_dodge_expert = expert_for_task(env._task)
        env._trajectory_expert_plans = 0
        env._trajectory_expert_failed_plans = 0
        env._trajectory_expert_plan_diagnostics = []
    seen = {}
    actual_min = collections.defaultdict(lambda: np.inf)
    success = False
    while True:
        for oid, item in manager._active.items():
            if oid not in seen:
                seen[oid] = (np.asarray(item.obj.translation, dtype=np.float64).copy(),
                             float(item.collision_radius))
        hp = env.sim.safety.dynamics_to_habitat(
            np.asarray(env.sim.dynamics.state["x"], dtype=np.float64))
        for oid, (opos, orad) in seen.items():
            actual_min[oid] = min(actual_min[oid],
                                  float(np.linalg.norm(hp - opos)) - agent_r - orad)
        if mode == "oracle":
            action = oracle_action(env)
        else:
            action = np.zeros(dim, dtype=np.float32)
            if const:
                action[-2] = const
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            success = bool(info.get("is_success", False))
            break

    # Nominal-path clearance per obstacle: fixed by the seed, not the policy.
    traj = env._nominal_trajectory
    # The spline can end before the episode does; sampling past its last
    # keyframe extrapolates, which would invent path far from the real one
    # and mislabel obstacles near the end. Clamp exactly as the env does.
    trajectory_end = float(
        getattr(traj, "t_keyframes", [float(env.episode_seconds)])[-1])
    duration = min(float(env.episode_seconds), trajectory_end - 1e-3)
    ts = np.linspace(0.0, duration, 400)
    nominal = np.asarray(
        [env.sim.safety.dynamics_to_habitat(
            np.asarray(traj.update(float(t))["x"], dtype=np.float64)) for t in ts])
    blockers = cleared = traps = trap_hits = 0
    for oid, (opos, orad) in seen.items():
        nom_clear = float(np.linalg.norm(nominal - opos, axis=1).min()) - agent_r - orad
        if nom_clear <= 0.0:
            blockers += 1
            if actual_min[oid] > dodge_clear:
                cleared += 1
        else:
            traps += 1
            if actual_min[oid] <= 0.0:
                trap_hits += 1
    return blockers, cleared, traps, trap_hits, success


def run(mode, const=0.0):
    B = C = T = TH = W = 0
    for i in range(EPISODES):
        b, c, t, th, s = episode(mode, const, SEED0 + i)
        B += b; C += c; T += t; TH += th; W += int(s)
    lo, hi = wilson(C, B)
    label = mode if not const else "const%+.2f" % const
    print("%12s %7.1f%% %10.2f %7.1f%% %10.2f %7.1f%%  [%4.1f,%4.1f]"
          % (label, 100 * W / EPISODES, B / EPISODES, 100 * C / max(B, 1),
             T / EPISODES, 100 * TH / max(T, 1), lo, hi), flush=True)


run("control")
for c in (0.4, -0.4, 0.8, -0.8):
    run("const", c)
run("oracle")
env.close()
