"""Which reward terms carry the return, and which carry its VARIANCE?

ev_ceiling_probe.py showed discounted return-to-go is a survival-time proxy:
`steps_remaining` alone explains 0.771 of its variance, the crash outcome
adds nothing once length is known, and the full 54-dim Markov state reaches
~0.0. So the critic cannot see dodging outcomes -- they are a small
perturbation on a large, near-constant per-step drift.

This asks which term IS the drift. Two different quantities, and the second
is the one that matters:

  * share of total return   -- what the policy is paid for on average. A big
                               constant term dominates this but teaches
                               nothing, because a constant has no gradient.
  * share of return VARIANCE across episodes -- what actually distinguishes a
                               good episode from a bad one. This is what the
                               critic must predict and what the policy
                               gradient can act on.

A term can dominate the mean while contributing nothing to the variance
(a per-step drift every episode pays equally), or be tiny in the mean while
carrying most of the variance (a rare clear bonus). Reporting only the mean
is how a reward gets tuned in the wrong direction.

Usage:
    python reward_composition_probe.py <config.yaml> [episodes] [mode]
      mode: zero (default) | oracle
"""
import collections
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "applications/rl/configs/velocity_dodge_privileged_teacher_v1.yaml"
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 30
MODE = sys.argv[3] if len(sys.argv) > 3 else "zero"
SEED0 = 9001

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
dim = int(np.prod(env.action_space.shape))

if MODE == "oracle":
    from evaluate_velocity_dodge_oracle import oracle_action, expert_for_task

ep_term_totals = []      # per episode: {term: summed value}
ep_lengths, ep_returns = [], []
terms_seen = set()
term_counts = collections.Counter()

for i in range(EPISODES):
    env.reset(seed=SEED0 + i)
    if MODE == "oracle":
        env._trajectory_dodge_expert = expert_for_task(env._task)
    totals = collections.defaultdict(float)
    total_r, n = 0.0, 0
    while True:
        if MODE == "oracle":
            action = oracle_action(env)
        else:
            action = np.zeros(dim, dtype=np.float32)
        _, reward, term, trunc, info = env.step(action)
        total_r += float(reward)
        n += 1
        for k, v in (info.get("reward_terms", {}) or {}).items():
            v = float(v)
            totals[k] += v
            terms_seen.add(k)
            if abs(v) > 1e-9:
                term_counts[k] += 1
        if term or trunc:
            break
    ep_term_totals.append(dict(totals))
    ep_lengths.append(n)
    ep_returns.append(total_r)

env.close()

names = sorted(terms_seen)
M = np.array([[ep.get(k, 0.0) for k in names] for ep in ep_term_totals])
R = np.asarray(ep_returns)
L = np.asarray(ep_lengths)

print("\nconfig: %s   mode: %s   episodes: %d" % (CFG, MODE, EPISODES))
print("episode return: mean %.2f  sd %.2f" % (R.mean(), R.std()))
print("episode length: mean %.1f  sd %.1f" % (L.mean(), L.std()))
if L.std() > 1e-9:
    print("corr(return, length) = %+.3f   <- 1.0 means return IS length"
          % float(np.corrcoef(R, L)[0, 1]))

print("\n%-28s %10s %10s %9s %9s" % ("term", "mean/ep", "sd/ep", "|mean|%", "var%"))
print("-" * 70)
abs_mean_total = float(np.abs(M.mean(axis=0)).sum()) or 1.0
# Variance share via covariance with the total: sum_j cov(term_j, R) = var(R),
# so cov(term_j, R)/var(R) is term j's signed contribution to return variance.
var_R = float(np.var(R)) or 1.0
order = np.argsort(-np.abs(M.mean(axis=0)))
for j in order:
    col = M[:, j]
    share_mean = 100.0 * abs(col.mean()) / abs_mean_total
    share_var = 100.0 * float(np.cov(col, R, bias=True)[0, 1]) / var_R
    print("%-28s %10.3f %10.3f %8.1f%% %8.1f%%"
          % (names[j], col.mean(), col.std(), share_mean, share_var))

print("\nvar%% columns sum to ~100 by construction (cov decomposition of return).")
print("A term with a large |mean|%% but ~0 var%% is a CONSTANT TOLL: it sets the")
print("return level, contributes no gradient, and -- because it is paid per")
print("step -- makes return track episode LENGTH. That is the mechanism behind")
print("steps_remaining explaining 0.771 of return-to-go variance while the full")
print("Markov state explains ~0.")
