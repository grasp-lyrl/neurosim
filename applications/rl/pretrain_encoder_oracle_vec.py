"""Oracle-driven pretraining with FULL environment parallelism.

pretrain_encoder_oracle.py runs a single environment because oracle_action()
needs direct access to env internals (env.sim, env._task,
env._trajectory_dodge_expert) that a SubprocVecEnv worker keeps in its own
process -- the main process only gets obs/reward/done/info over the pipe,
not arbitrary attribute access. So oracle-quality data was paying an ~8x
throughput tax relative to the random-action run, which used all 8 workers.

Habitat rendering, not the aux gradient step, is the bottleneck (single-env
oracle pretraining ran at ~1.7 rounds/min = ~7 env-steps/sec; a 64-sample
EfficientNet-B0 forward+backward at 240x320 is far faster than that). So the
fix has to parallelise environment stepping, not speed up training.

The trick: put the oracle INSIDE each worker instead of computing it in the
main process. _OracleDrivenEnv.step() ignores the action it's given and
computes its own via oracle_action(self) -- run where env.sim/._task
actually live. The main process sends dummy actions to N parallel workers
and gets back oracle-driven transitions, with the same
DummyVecEnv-vs-SubprocVecEnv throughput characteristics as any other
vectorised rollout. Expected ~8x wall-clock speedup at num_envs=8.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces

from train_sb3 import (
    DistanceAuxCallback,
    PRIVILEGED_KEY,
    build_policy_config,
    build_train_vec_env,
    load_experiment_config,
)
from neurosim.rl import env_class_for_task
from evaluate_velocity_dodge_oracle import oracle_action, expert_for_task


class _OracleDrivenEnv(gym.Wrapper):
    """Ignore the action given, drive with the oracle instead.

    Runs inside the SAME process as the raw env (a SubprocVecEnv worker),
    so env.sim/._task are directly available -- unlike the main process,
    which only sees this env through a pipe.
    """

    def __init__(self, env):
        super().__init__(env)
        raw = env
        while hasattr(raw, "env"):
            raw = raw.env
        self._raw = raw
        self._raw._trajectory_dodge_expert = expert_for_task(self._raw._task)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._raw._trajectory_dodge_expert = expert_for_task(self._raw._task)
        return obs, info

    def step(self, action):
        del action  # ignored -- the oracle chooses instead
        real_action = oracle_action(self._raw)
        obs, reward, terminated, truncated, info = self.env.step(real_action)
        # Surface what was ACTUALLY executed. Without this the main loop
        # recorded the caller's dummy zero action into the rollout buffer
        # instead of the real one -- harmless for aux training (it only
        # reads observations) but a landmine for any future reuse of this
        # buffer, and it removes the only cheap way to verify empirically
        # that the oracle is actually driving each worker rather than
        # silently no-op'ing.
        info["oracle_action"] = np.asarray(real_action, dtype=np.float32)
        return obs, reward, terminated, truncated, info


def _make_oracle_env(env_config, seed, env_idx, gpu_id):
    """Module-level factory (must be picklable for spawn-based SubprocVecEnv)."""

    def _init():
        cfg = copy.deepcopy(env_config)
        cfg.setdefault("visual_backend", {})["gpu_id"] = gpu_id
        env_cls = env_class_for_task(cfg["task"]["name"])
        env = env_cls(env_config=cfg, train=True)
        env = Monitor(env)
        env = _OracleDrivenEnv(env)
        env.reset(seed=seed + env_idx)
        return env

    return _init


def main():
    CFG = sys.argv[1] if len(sys.argv) > 1 else \
        "applications/rl/configs/velocity_dodge_dynamic_v16_effnet.yaml"
    N_ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    CKPT_OUT = sys.argv[3] if len(sys.argv) > 3 else \
        "outputs/rl/pretrained_encoder_effnet_vec.pt"
    AUX_CALLS_PER_ROUND = 4
    GPU_ID_FOR_ENVS = 1

    exp = load_experiment_config(CFG)
    env_config = copy.deepcopy(exp["env"])
    env_config["enable_visualization"] = False
    num_envs = int(exp["num_envs"])
    n_steps = int(exp["ppo"]["n_steps"])
    base_seed = int(exp["seed"])

    env_fns = [
        _make_oracle_env(env_config, base_seed, i, GPU_ID_FOR_ENVS)
        for i in range(num_envs)
    ]
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
        use_sde=bool(exp["ppo"].get("use_sde", False)),
        policy_kwargs=policy_kwargs,
    )
    print(f"policy constructed: {policy}, privileged={privileged}, "
          f"num_envs={num_envs}, n_steps={n_steps}", flush=True)

    resume_from = sys.argv[4] if len(sys.argv) > 4 else None
    start_round = 0
    if resume_from:
        ckpt = th.load(resume_from, map_location=model.device)
        model.policy.pi_features_extractor.load_state_dict(ckpt["pi_features_extractor"])
        model.policy.vf_features_extractor.load_state_dict(ckpt["vf_features_extractor"])
        prior_steps = int(ckpt.get("env_steps", 0))
        print(f"resumed encoder weights from {resume_from} "
              f"({prior_steps} prior env-steps)", flush=True)

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
    dummy_action = np.zeros((num_envs,) + vec_env.action_space.shape, dtype=np.float32)

    action_norms = []
    for round_idx in range(N_ROUNDS):
        buffer.reset()
        for _ in range(n_steps):
            # The action sent here is discarded by every worker's
            # _OracleDrivenEnv.step() -- each computes its own oracle action
            # locally and reports it back via info["oracle_action"].
            next_obs, reward, done, info = vec_env.step(dummy_action)
            real_actions = np.stack(
                [i.get("oracle_action", np.zeros(dummy_action.shape[1:], dtype=np.float32))
                 for i in info]
            )
            action_norms.extend(np.linalg.norm(real_actions, axis=1).tolist())
            buffer.add(obs, real_actions, reward, episode_starts, zero_value, zero_log_prob)
            obs = next_obs
            episode_starts = done
        buffer.full = True
        if round_idx < 5 or (round_idx + 1) % 20 == 0:
            an = np.asarray(action_norms[-2048:])
            print(f"  [verify] action_norm: mean={an.mean():.3f} "
                  f"p50={np.median(an):.3f} p90={np.percentile(an,90):.3f} "
                  f"max={an.max():.3f} frac_nonzero={(an > 1e-4).mean():.2f}",
                  flush=True)
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
        if (round_idx + 1) % 10 == 0:
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
