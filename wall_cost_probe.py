"""Do walls price the constant-offset exploit?

Every probe today disabled collision termination to get clean encounter
statistics, which made holding a lateral offset FREE. Measured that way, a
constant +0.80 m offset on velocity_dodge_dynamic_v2 cleared 98.8% of
threats and lifted episode success from 3.3% to 40.0% -- while the
privileged oracle managed 35.8%, because it returns to the nominal path and
that is where the throws are aimed.

Indoors the offset is not free. depth_contrast_probe measured scene geometry
at 1.04 m median from the nominal path (p10 0.08 m), and the reachable
offset is +/-1.2 m, so a saturated offset flies into walls. This run leaves
ALL terminations active -- obstacle collision, bounds, navigability,
tracking -- and asks whether that cost cancels the exploit's benefit.

Reports the termination breakdown per arm, which is the actual answer: if
the constant-offset arms die on out_of_bounds / not_navigable often enough,
the environment already prices the exploit and the aiming scheme does not
have to defeat it. Clearance is reported too but is CENSORED here by design
(an arm that dies early meets fewer throws), so it is not comparable across
arms -- episode success and the termination mix are.
"""
import collections
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "applications/rl/configs/velocity_dodge_dynamic_v2.yaml"
SEED0, EPISODES = 9001, 30

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
dim = int(np.prod(env.action_space.shape))
# NOTE: no monkeypatch of has_obstacle_collision here -- that is the point.

print("config: %s" % CFG)
print("ALL terminations active; offset is no longer free\n")
print("%12s %9s %9s %10s  %s"
      % ("arm", "episode", "steps", "wall_term", "terminations"), flush=True)


def run(const):
    wins = 0
    steps_all = []
    term = collections.Counter()
    for i in range(EPISODES):
        env.reset(seed=SEED0 + i)
        n = 0
        while True:
            action = np.zeros(dim, dtype=np.float32)
            if const:
                action[-2] = const
            _, _, terminated, truncated, info = env.step(action)
            n += 1
            if terminated or truncated:
                reason = info.get("termination_reason") or "timeout"
                term[reason] += 1
                wins += bool(info.get("is_success", False))
                steps_all.append(n)
                break
    wall = term["out_of_bounds"] + term["not_navigable"]
    label = "control" if not const else "const%+.2f" % const
    print("%12s %8.1f%% %9.1f %9.1f%%  %s"
          % (label, 100 * wins / EPISODES, float(np.mean(steps_all)),
             100 * wall / EPISODES, dict(term.most_common(4))), flush=True)


run(0.0)
for c in (0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
    run(c)
env.close()
