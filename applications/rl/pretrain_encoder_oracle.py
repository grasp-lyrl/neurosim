"""Same as pretrain_encoder.py, but ORACLE-driven instead of random actions.

pretrain_encoder.py showed a growth curve with fast early gains (0 -> 0.09 in
the first 32k steps) then a 10x slower crawl (0.09 -> 0.15 over the next
380k) -- diminishing returns from THAT data, not obviously a hard ceiling.
The likely cause: random actions rarely put the vehicle near an obstacle.
Density/escape-set measurements earlier in this session found a passive-ish
policy encounters only ~1.5-3 genuine threats per ~150-200 step episode, so
most random-action frames have no obstacle within the 4 m supervision range
-- diluting the aux signal with easy all-clear frames instead of
concentrating it on the hard, informative near-obstacle ones.

An oracle-driven rollout is the opposite: dodging requires flying NEAR
obstacles, so it should generate far more informative frames per step. This
tests that directly, at a much smaller step budget, to see whether R2 climbs
faster per step -- which would mean the fix is DATA COMPOSITION (mix in
oracle rollouts), not just running pretrain_encoder.py longer.

oracle_action() needs the raw (non-vectorized) env -- it reads env.sim,
env._task, env._trajectory_dodge_expert directly, not through a VecEnv
wrapper. So this uses DummyVecEnv with num_envs=1 (single Habitat/GL
context, which is what actually crashed DummyVecEnv at num_envs=8 earlier --
not present here) rather than the 8-way SubprocVecEnv the random run used.
Lower throughput per wall-clock minute, traded for much higher information
density per environment-step if the hypothesis is right.

Otherwise reuses the same machinery as pretrain_encoder.py: the real
PPO/policy construction (build_policy_config) so architecture matches
production exactly, and DistanceAuxCallback directly -- same loss, same
near-range gate, same occupancy target -- so whatever R2 this reaches is
what the encoder is actually capable of on this data, not an easier proxy
task.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch as th
from stable_baselines3 import PPO
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv

from train_sb3 import (
    DistanceAuxCallback,
    PRIVILEGED_KEY,
    build_policy_config,
    load_experiment_config,
    make_env,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_velocity_dodge_oracle import oracle_action, expert_for_task

def main():
    CFG = sys.argv[1] if len(sys.argv) > 1 else \
        "applications/rl/configs/velocity_dodge_dynamic_v16.yaml"
    N_ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    CKPT_OUT = sys.argv[3] if len(sys.argv) > 3 else \
        "outputs/rl/pretrained_encoder_oracle.pt"
    AUX_CALLS_PER_ROUND = 4   # ~1 epoch over the round's buffer (4 x 512 ~= 2048)
    GPU_ID_FOR_ENVS = 1       # keep render workers off the learner's GPU

    exp = load_experiment_config(CFG)
    env_config = copy.deepcopy(exp["env"])
    env_config["enable_visualization"] = False
    num_envs = int(exp["num_envs"])
    n_steps = int(exp["ppo"]["n_steps"])
    base_seed = int(exp["seed"])

    # oracle_action() needs direct access to env internals (env.sim,
    # env._task, env._trajectory_dodge_expert), which only exists for a raw
    # env in the SAME process -- so this is a single env via DummyVecEnv, not
    # the 8-way SubprocVecEnv the random run used. num_envs=1 sidesteps the
    # earlier DummyVecEnv crash (that was 8 GL contexts sharing one process).
    num_envs = 1
    env_fns = [
        make_env(env_config, seed=base_seed, env_idx=0, train=True,
                 gpu_id=GPU_ID_FOR_ENVS)
    ]
    vec_env = DummyVecEnv(env_fns)
    raw_env = vec_env.envs[0]
    while hasattr(raw_env, "env"):
        raw_env = raw_env.env
    raw_env._trajectory_dodge_expert = expert_for_task(raw_env._task)

    privileged = (
        isinstance(vec_env.observation_space, spaces.Dict)
        and PRIVILEGED_KEY in vec_env.observation_space.spaces
    )
    policy, policy_kwargs = build_policy_config(
        str(exp["env"]["obs_mode"]),
        float(exp["ppo"]["log_std_init"]),
        privileged=privileged,
        event_presence_features=bool(exp["ppo"].get("event_presence_features", False)),
        event_high_resolution=bool(exp["ppo"].get("event_high_resolution", False)),
        event_backbone=str(exp["ppo"].get("event_backbone", "small")),
        recurrent=False,
    )

    model = PPO(
        policy,
        vec_env,
        n_steps=n_steps,
        batch_size=int(exp["ppo"]["batch_size"]),
        device=str(exp["ppo"]["device"]),
        seed=base_seed,
        verbose=0,
        # policy_kwargs sets squash_output=True (build_policy_config), which SB3
        # asserts requires use_sde=True regardless of whether we ever call
        # .learn() -- match the main config's setting so construction succeeds.
        use_sde=bool(exp["ppo"].get("use_sde", False)),
        policy_kwargs=policy_kwargs,
    )
    print(f"policy constructed: {policy}, privileged={privileged}, "
          f"num_envs={num_envs}, n_steps={n_steps}", flush=True)

    # SB3 only initialises .logger inside .learn() -- construct one directly
    # since this script never calls it, and DistanceAuxCallback.logger reads
    # model.logger to record metrics.
    from stable_baselines3.common.utils import configure_logger
    model.set_logger(configure_logger(verbose=0))

    callback = DistanceAuxCallback(
        weight=float(exp["ppo"]["distance_aux_weight"]),
        near_range_m=float(exp["ppo"].get("distance_aux_near_range_m", 0.0)),
        occupancy_only=bool(exp["ppo"].get("distance_aux_occupancy_only", False)),
        batch=int(exp["ppo"].get("distance_aux_batch", 64)),
    )
    callback.stop_after = 0
    callback.init_callback(model)
    print(f"aux config: weight={callback.weight} near_range_m={callback.near_range_m} "
          f"occupancy_only={callback.occupancy_only} batch={callback.batch}", flush=True)

    buffer = model.rollout_buffer
    obs = vec_env.reset()
    episode_starts = np.ones(num_envs, dtype=bool)
    zero_value = th.zeros(num_envs, 1, device=model.device)
    zero_log_prob = th.zeros(num_envs, device=model.device)

    for round_idx in range(N_ROUNDS):
        buffer.reset()
        for _ in range(n_steps):
            action = np.asarray(oracle_action(raw_env), dtype=np.float32).reshape(1, -1)
            next_obs, reward, done, info = vec_env.step(action)
            if done[0]:
                raw_env._trajectory_dodge_expert = expert_for_task(raw_env._task)
            buffer.add(obs, action, reward, episode_starts, zero_value, zero_log_prob)
            obs = next_obs
            episode_starts = done
        buffer.full = True
        r2s_pi, r2s_vf = [], []
        for _ in range(AUX_CALLS_PER_ROUND):
            callback._on_rollout_end()
            if not np.isnan(callback.last_r2):
                r2s_pi.append(callback.last_r2)
            if not np.isnan(callback.last_r2_critic):
                r2s_vf.append(callback.last_r2_critic)
        steps_so_far = (round_idx + 1) * num_envs * n_steps
        print(f"round {round_idx:3d}  env_steps={steps_so_far:7d}  "
              f"r2_pi={np.mean(r2s_pi) if r2s_pi else float('nan'):+.4f}  "
              f"r2_vf={np.mean(r2s_vf) if r2s_vf else float('nan'):+.4f}",
              flush=True)
        if (round_idx + 1) % 20 == 0:
            Path(CKPT_OUT).parent.mkdir(parents=True, exist_ok=True)
            th.save(
                {
                    "pi_features_extractor": model.policy.pi_features_extractor.state_dict(),
                    "vf_features_extractor": model.policy.vf_features_extractor.state_dict(),
                    "env_steps": steps_so_far,
                },
                CKPT_OUT,
            )
            print(f"  checkpoint saved to {CKPT_OUT} at round {round_idx}", flush=True)

    vec_env.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
