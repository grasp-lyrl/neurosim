"""Blind baselines scored on RETURN, not success rate.

Every blind baseline measured today used episode SUCCESS -- did the arm clear
the throws. PPO does not optimise success; it optimises return. The task
already prices the behaviours those baselines rely on: correction_energy
penalises action magnitude, _split_offset prices along-track lag (braking)
separately from cross-track, and tracking error penalises deviation from the
reference. A 0.8 m sine weave that wins on success may lose badly on return.

If the oracle has the highest RETURN while blind arms have higher success,
the task is not degenerate in the terms the policy is trained against, and
the "blind beats sighted" results are an artifact of scoring the wrong
quantity.

Reports mean episode return alongside success for the same arms, so the two
can be compared directly.
"""
import collections
import copy
import sys
from dataclasses import replace

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from neurosim.rl import env_class_for_task
from evaluate_velocity_dodge_oracle import oracle_action, expert_for_task
from train_sb3 import load_experiment_config

CFG = sys.argv[1] if len(sys.argv) > 1 else \
    "applications/rl/configs/velocity_dodge_dynamic_v7.yaml"
SEED0 = 9001
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
# Optional third argument: a comma-separated subset of arm labels to run.
# The six arms are independent -- each resets the env to the same seeds and
# shares no state -- so running them as separate processes is exact, not an
# approximation, and cuts a gate from ~6 sequential arms to one wall-clock
# arm. This project re-gates on every task change (obstacle count, speed,
# spawn distance and the crash penalty have all moved), and a serial gate has
# repeatedly been the slowest thing in the loop.
#
#   python return_baseline.py <cfg> 40 control
#   python return_baseline.py <cfg> 40 oracle
#   ... run concurrently, then concatenate the logs.
#
# Names: control, const+0.40, const+0.80, weave0.4@0.5Hz, weave0.8@0.5Hz,
# oracle. Omit to run all six in one process, as before. An optional fourth
# argument applies an experimental peak advance to the oracle only:
#
#   python return_baseline.py <cfg> 40 oracle 0.25
#
# An optional fifth argument replaces the oracle displacement menu, allowing
# same-seed feasibility sweeps without changing the task or global expert:
#
#   python return_baseline.py <cfg> 40 oracle 0.0 0.25,0.30,0.35,0.50,0.65,0.80,1.00
ONLY = None
if len(sys.argv) > 3 and sys.argv[3].strip():
    ONLY = {a.strip() for a in sys.argv[3].split(",") if a.strip()}
ORACLE_PEAK_ADVANCE_S = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
ORACLE_CANDIDATES = (
    tuple(float(value) for value in sys.argv[5].split(",") if value.strip())
    if len(sys.argv) > 5
    else None
)
WEAVES = [(0.4, 0.5), (0.8, 0.5)]
CONSTS = [0.4, 0.8]

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False
env = env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)
dim = int(np.prod(env.action_space.shape))

print("config: %s" % CFG)
print("return is what PPO maximises; success is not\n")
print("%16s %12s %10s %9s %10s  %s"
      % ("arm", "return", "return/step", "success", "act_energy", "terminations"),
      flush=True)


def run(mode, amp=0.0, freq=0.0, const=0.0, label=None):
    rets, steps_all, energies, wins = [], [], [], 0
    term = collections.Counter()
    for i in range(EPISODES):
        env.reset(seed=SEED0 + i)
        if mode == "oracle":
            expert = expert_for_task(env._task)
            expert.config = replace(
                expert.config,
                peak_advance_s=ORACLE_PEAK_ADVANCE_S,
                candidate_offsets_m=(
                    ORACLE_CANDIDATES
                    if ORACLE_CANDIDATES is not None
                    else expert.config.candidate_offsets_m
                ),
            )
            env._trajectory_dodge_expert = expert
            env._trajectory_expert_plans = 0
            env._trajectory_expert_failed_plans = 0
            env._trajectory_expert_plan_diagnostics = []
        total, n, energy = 0.0, 0, 0.0
        while True:
            now = float(env.sim.time)
            if mode == "oracle":
                action = oracle_action(env)
            else:
                action = np.zeros(dim, dtype=np.float32)
                if mode == "weave":
                    action[-2] = amp * np.sin(2.0 * np.pi * freq * now)
                elif mode == "const":
                    action[-2] = const
            _, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            # mean(action^2) -- the quantity w_correction prices, and the
            # actual behavioural difference between an expert that acts only
            # during encounters and a weave that acts every step. Both reward
            # levers tried so far were aimed at this without measuring it.
            terms = info.get("reward_terms", {}) or {}
            energy += float(terms.get("correction_energy", 0.0))
            n += 1
            if terminated or truncated:
                term[info.get("termination_reason") or "timeout"] += 1
                wins += bool(info.get("is_success", False))
                break
        rets.append(total)
        steps_all.append(n)
        energies.append(energy / max(n, 1))
    name = label or mode
    print("%16s %12.1f %10.3f %8.1f%% %10.4f  %s"
          % (name, float(np.mean(rets)),
             float(np.sum(rets)) / max(float(np.sum(steps_all)), 1.0),
             100 * wins / EPISODES, float(np.mean(energies)),
             dict(term.most_common(3))), flush=True)


def maybe(_arm, *args, **kwargs):
    # _arm (not "label") is deliberate: run()'s own label kwarg is forwarded
    # through **kwargs for const/weave arms, and a same-named selector
    # parameter here collides with it -- Python binds the keyword to THIS
    # function's parameter before it ever reaches **kwargs, raising
    # "maybe() got multiple values for argument 'label'". Caught the hard
    # way: it silently killed every arm except control and oracle (the two
    # calls that don't pass label=...), and the per-arm log files made that
    # look like slow arms rather than five simultaneous crashes.
    if ONLY is None or _arm in ONLY:
        run(*args, **kwargs)


maybe("control", "control")
for c in CONSTS:
    maybe("const%+.2f" % c, "const", const=c, label="const%+.2f" % c)
for amp, freq in WEAVES:
    maybe(
        "weave%.1f@%.1fHz" % (amp, freq),
        "weave", amp=amp, freq=freq, label="weave%.1f@%.1fHz" % (amp, freq),
    )
maybe("oracle", "oracle")
env.close()
