"""Is the trained policy using the event stream, or weaving blind?

At 213k steps the run reached 44% success -- level with a blind 0.5 Hz sine
weave (45%) -- while the auxiliary distance head sat at R2 = -1.57, i.e. it
predicts obstacle distance WORSE than guessing the mean despite a 5.0
weighting. If the encoder cannot extract obstacle geometry, that success
cannot be perception-driven, and the simplest policy achieving it is an
oscillation.

Two measurements settle it:
  blind ablation -- run the same checkpoint with the event tensor zeroed. A
    policy using perception should drop; one that has learned a fixed motor
    pattern should not.
  act_energy + zero-crossing rate -- the weave's signature is mean(action^2)
    ~0.104 for amplitude 0.8 at 0.5 Hz, with a regular sign flip about once
    per second. A learned weave should look similar.
"""
import collections
import copy
import sys

sys.path.insert(0, "/home/odexter/neurosim/applications/rl")
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from neurosim.rl import env_class_for_task
from train_sb3 import load_experiment_config

CFG = "applications/rl/configs/velocity_dodge_dynamic_v11.yaml"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs/rl/v11_ppo_direct/best_model.zip"
VECN = sys.argv[2] if len(sys.argv) > 2 else "outputs/rl/v11_ppo_direct/vecnormalize.pkl"
SEED0, EPISODES = 9001, 25

cfg = load_experiment_config(CFG)
env_cfg = copy.deepcopy(cfg["env"])
env_cfg["enable_visualization"] = False


def make():
    return env_class_for_task(env_cfg["task"]["name"])(env_config=env_cfg, train=False)


venv = DummyVecEnv([make])
venv = VecNormalize.load(VECN, venv)
venv.training = False
venv.norm_reward = False
model = PPO.load(CKPT, device="cuda")
print("loaded %s\n" % CKPT)
print("%10s %12s %9s %11s %12s  %s"
      % ("mode", "return", "success", "act_energy", "flips/s", "terminations"),
      flush=True)


def run(blind):
    rets, wins, energies, flips_all = [], 0, [], []
    term = collections.Counter()
    base = venv.envs[0]
    for i in range(EPISODES):
        obs = venv.env_method("reset", seed=SEED0 + i)[0][0]
        obs = {k: np.asarray(v)[None] for k, v in obs.items()}
        total, n, energy, signs, dur = 0.0, 0, 0.0, [], 0.0
        while True:
            o = {k: v.copy() for k, v in obs.items()}
            if blind and "events" in o:
                o["events"] = np.zeros_like(o["events"])
            norm = venv.normalize_obs(o)
            act, _ = model.predict(norm, deterministic=True)
            a = np.asarray(act).reshape(-1)
            energy += float(np.mean(np.square(a)))
            signs.append(np.sign(a[1]) if abs(a[1]) > 0.05 else 0.0)
            nxt, reward, done, info = venv.step(act)
            total += float(reward[0]); n += 1
            dur = float(base.sim.time)
            obs = {k: np.asarray(v) for k, v in nxt.items()}
            if done[0]:
                inf = info[0]
                term[inf.get("termination_reason") or "timeout"] += 1
                wins += bool(inf.get("is_success", False))
                break
        rets.append(total); energies.append(energy / max(n, 1))
        s = np.asarray([x for x in signs if x != 0.0])
        flips = int(np.sum(np.abs(np.diff(s)) > 1)) if s.size > 1 else 0
        flips_all.append(flips / max(dur, 1e-6))
    print("%10s %12.1f %8.1f%% %11.4f %12.2f  %s"
          % ("blind" if blind else "sighted", float(np.mean(rets)),
             100 * wins / EPISODES, float(np.mean(energies)),
             float(np.mean(flips_all)), dict(term.most_common(3))), flush=True)


run(False)
run(True)
venv.close()
