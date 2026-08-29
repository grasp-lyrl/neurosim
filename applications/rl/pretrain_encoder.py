"""Standalone aux-only pretraining: is the occupancy task learnable at all?

Decoupled from PPO entirely -- no policy loss, no value loss, no advantage
computation. Just: step the env with random actions to generate diverse
obstacle geometry, and run the auxiliary occupancy loss against BOTH feature
extractors (actor's and critic's -- see the two-tower fix in
DistanceAuxCallback) as fast as possible.

This exists to separate two questions that the joint PPO run cannot: is the
occupancy target learnable from the event stream at all, and is PPO's own
(noisy, policy-gradient) objective helping or interfering with fitting it.
The joint run showed r2_pi and r2_vf swapping which was ahead of the other
between readings (0.0002/0.047, then 0.037/0.018) -- plausibly PPO's policy
gradient perturbing the actor's tower more than the critic's. This isolates
the aux term with a completely clean gradient to find out what R2 is
reachable in principle, unconfounded by anything else touching the encoder.

Reuses the real PPO/policy construction (build_policy_config) so the
architecture matches production exactly, and reuses DistanceAuxCallback
directly -- same loss, same near-range gate, same occupancy target -- so
whatever R2 this reaches is what the encoder is actually capable of on this
data, not an easier proxy task.

Random actions only (no oracle mixing): building an oracle-compatible
vectorized rollout is real additional complexity, and the resulting event
geometry from random exploration is close to what PPO sees in its own early
training anyway.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch as th
from stable_baselines3 import PPO
from gymnasium import spaces

from train_sb3 import (
    DistanceAuxCallback,
    PRIVILEGED_KEY,
    build_policy_config,
    build_train_vec_env,
    load_experiment_config,
    make_env,
)

def main():
    CFG = sys.argv[1] if len(sys.argv) > 1 else \
        "applications/rl/configs/velocity_dodge_dynamic_v16.yaml"
    N_ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    CKPT_OUT = sys.argv[3] if len(sys.argv) > 3 else \
        "outputs/rl/pretrained_encoder.pt"
    AUX_CALLS_PER_ROUND = 4   # ~1 epoch over the round's buffer (4 x 512 ~= 2048)
    GPU_ID_FOR_ENVS = 1       # keep render workers off the learner's GPU

    exp = load_experiment_config(CFG)
    env_config = copy.deepcopy(exp["env"])
    env_config["enable_visualization"] = False
    num_envs = int(exp["num_envs"])
    n_steps = int(exp["ppo"]["n_steps"])
    base_seed = int(exp["seed"])

    env_fns = [
        make_env(env_config, seed=base_seed, env_idx=i, train=True,
                 gpu_id=GPU_ID_FOR_ENVS)
        for i in range(num_envs)
    ]
    # SubprocVecEnv, not DummyVecEnv: 8 Habitat/GL contexts cannot safely share
    # one process (measured -- DummyVecEnv crashed with a CUDA illegal memory
    # access inside render_events on the first rollout). The main training
    # script already resolves num_envs>1 to SubprocVecEnv for this reason;
    # reuse that exact path rather than a separately-debugged one.
    vec_env = build_train_vec_env(
        env_fns,
        num_envs=num_envs,
        vec_env_type=str(exp["ppo"].get("vec_env_type", "auto")),
        start_method=str(exp["ppo"].get("vec_env_start_method", "spawn")),
    )

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
            action = np.stack([vec_env.action_space.sample() for _ in range(num_envs)])
            next_obs, reward, done, info = vec_env.step(action)
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
