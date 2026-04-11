"""Train a PPO policy for neurosim tasks using Stable-Baselines3.

Usage:
    python train_sb3.py \
        --experiment-config applications/rl/configs/hover_sb3_state_experiment.yaml

The script saves:
    outputs/rl/<run_name>/best_model.zip   - best checkpoint (by eval reward)
    outputs/rl/<run_name>/final_model.zip  - final checkpoint
    outputs/rl/<run_name>/vecnormalize.pkl - observation / reward normalizer state
    W&B run logs                    - training metrics and config

Notes for vectorized training:
    - `num_envs` controls PPO rollout collection parallelism.
    - `eval_freq` in config is interpreted in environment steps;
        converted to callback frequency so evaluation stays consistent.
    - Training uses subprocess vectorization for heavy simulators (Habitat).
    - Experiment configs are self-contained: scenes, sensors, ``dynamics``,
        the full ``simulator`` block, and domain randomization are in the YAML
        (no external settings file).  Training passes ``train=True`` into the RL
        env, which disables Rerun visualization regardless of YAML until
        short-episode logging is sorted out; use ``run_policy.py --visualize``
        for rollout logging.
    - Each vec-env worker is seeded with ``seed + env_idx`` so workers get
        distinct randomized configurations when DR is enabled.
    - Set ``simulator.domain_randomization.resample_on_reset: true`` to
        re-randomize on every episode reset.  Optional dynamics DR lives under
        ``env.dynamics.domain_randomization``.
    - Eval uses ``SubprocVecEnv`` (spawn) so Habitat teardown does not share a
        process with the training ``DummyVecEnv`` when ``num_envs == 1``.
"""

import yaml
import argparse
import numpy as np
from typing import Any
from pathlib import Path
from datetime import datetime

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from neurosim.rl import CombinedEventStateExtractor, EventCnnExtractor, NeurosimRLEnv


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Experiment config must be a YAML mapping")

    required_keys = [
        "seed",
        "num_envs",
        "total_timesteps",
        "eval",
        "env",
        "ppo",
        "vecnormalize",
        "vec_env",
        "wandb",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required experiment config keys: {missing}")

    return cfg


def make_env(
    env_config: dict[str, Any],
    seed: int | None = None,
    env_idx: int = 0,
    *,
    train: bool = True,
):
    """Factory callable for DummyVecEnv / SubprocVecEnv.

    Each worker resets with ``seed + env_idx`` so that parallel workers
    get distinct initial randomizations (when DR is enabled in the config).

    ``train`` is forwarded to :class:`~neurosim.rl.env.NeurosimRLEnv`; when
    true, Rerun visualization is forced off regardless of YAML.
    """

    def _init():
        import copy

        cfg = copy.deepcopy(env_config)
        env = NeurosimRLEnv(env_config=cfg, train=train)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed + env_idx)
        return env

    return _init


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SB3 PPO training for neurosim tasks")
    p.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="YAML file containing experiment/training hyperparameters",
    )
    p.add_argument("--wandb-project", type=str, default="neurosim-rl")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging and WandbCallback",
    )
    return p.parse_args()


def build_policy_config(
    obs_mode: str,
    log_std_init: float,
) -> tuple[str, dict[str, Any]]:
    # Normalized action space [-1, 1] requires lower initial policy std
    # to prevent excessive early exploration.
    base_policy_kwargs = {"log_std_init": float(log_std_init)}

    if obs_mode == "state":
        return "MlpPolicy", base_policy_kwargs
    if obs_mode == "events":
        return "CnnPolicy", {
            "features_extractor_class": EventCnnExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "normalize_images": False,
            **base_policy_kwargs,
        }
    return "MultiInputPolicy", {
        "features_extractor_class": CombinedEventStateExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
        "normalize_images": False,
        **base_policy_kwargs,
    }


def build_vecnormalize_kwargs(
    obs_mode: str,
    normalize_obs: bool,
    normalize_reward: bool,
    training: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "norm_obs": normalize_obs,
        "norm_reward": normalize_reward,
        "clip_obs": 10.0,
        "training": training,
    }
    if normalize_obs and obs_mode == "events":
        # Event tensors are normalized in the RL env.
        kwargs["norm_obs"] = False
    elif normalize_obs and obs_mode == "combined":
        # Normalize privileged state only, not event frames.
        kwargs["norm_obs_keys"] = ["state"]
    return kwargs


def maybe_wrap_vecnormalize(
    vec_env,
    *,
    obs_mode: str,
    normalize_obs: bool,
    normalize_reward: bool,
    training: bool,
    enabled: bool | None = None,
):
    """Return ``VecNormalize(vec_env, ...)`` when normalization is active.

    ``enabled`` defaults to ``normalize_obs or normalize_reward``.  Eval envs
    pass an explicit ``enabled`` from the experiment vecnormalize block so a
    reward-only training setup still wraps eval (``norm_reward=False``) for SB3.
    """
    if enabled is None:
        enabled = normalize_obs or normalize_reward
    if not enabled:
        return vec_env
    return VecNormalize(
        vec_env,
        **build_vecnormalize_kwargs(
            obs_mode=obs_mode,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
            training=training,
        ),
    )


def build_train_vec_env(
    env_fns: list,
    num_envs: int,
    vec_env_type: str,
    start_method: str,
):
    """Construct training VecEnv; use subprocess workers for heavy environments."""
    resolved_type = vec_env_type.strip().lower()
    if resolved_type not in {"auto", "dummy", "subproc"}:
        raise ValueError("vec_env_type must be one of: auto, dummy, subproc")

    if resolved_type == "dummy":
        return DummyVecEnv(env_fns)

    if resolved_type == "subproc" or (resolved_type == "auto" and num_envs > 1):
        resolved_start = start_method.strip().lower()
        if resolved_start not in {"fork", "forkserver", "spawn"}:
            raise ValueError(
                "vec_env_start_method must be one of: fork, forkserver, spawn"
            )
        return SubprocVecEnv(env_fns, start_method=resolved_start)

    return DummyVecEnv(env_fns)


def main():
    args = parse_args()
    exp = load_experiment_config(args.experiment_config)

    np.random.seed(int(exp["seed"]))
    num_envs = int(exp["num_envs"])
    n_steps = int(exp["ppo"]["n_steps"])
    batch_size = int(exp["ppo"]["batch_size"])
    rollout_batch = n_steps * num_envs
    if batch_size > rollout_batch:
        raise ValueError(
            f"batch_size ({batch_size}) must be <= n_steps * num_envs ({rollout_batch})"
        )
    if rollout_batch % batch_size != 0:
        print(
            "Warning: n_steps * num_envs is not divisible by batch_size; "
            "SB3 will use a truncated minibatch."
        )

    run_name = args.run_name
    if run_name is None:
        run_name = f"neurosim_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output = Path("outputs/rl") / run_name
    output.mkdir(parents=True, exist_ok=True)

    run = None
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={"cli": vars(args), "experiment": exp},
            sync_tensorboard=True,
            save_code=True,
            dir=str(output),
        )

    vec_env_type = str(exp["vec_env"]["type"])
    vec_env_start_method = str(exp["vec_env"]["start_method"])

    env_config = dict(exp["env"])

    # Training env (use multiprocessing for heavy simulators when num_envs > 1).
    # Each worker gets a distinct seed so domain randomization produces
    # different initial simulator configurations across workers.
    base_seed = int(exp["seed"])
    train_env_fns = [
        make_env(env_config, seed=base_seed, env_idx=env_idx, train=True)
        for env_idx in range(num_envs)
    ]
    train_vec = build_train_vec_env(
        train_env_fns,
        num_envs=num_envs,
        vec_env_type=vec_env_type,
        start_method=vec_env_start_method,
    )
    vn = exp["vecnormalize"]
    train_vec = maybe_wrap_vecnormalize(
        train_vec,
        obs_mode=str(exp["env"]["obs_mode"]),
        normalize_obs=bool(vn["normalize_obs"]),
        normalize_reward=bool(vn["normalize_reward"]),
        training=True,
    )

    # Eval env (separate instance, shared normalization stats)
    eval_num_envs = int(exp["eval"]["num_envs"])
    if eval_num_envs <= 0:
        raise ValueError("eval_num_envs must be >= 1")

    # Run eval in a subprocess so Habitat/OpenGL is not in the same process as a
    # training DummyVecEnv (avoids double GL context and Magnum teardown aborts).
    eval_vec = SubprocVecEnv(
        [
            make_env(env_config, seed=base_seed + 1000, env_idx=env_idx, train=True)
            for env_idx in range(eval_num_envs)
        ],
        start_method="spawn",
    )
    eval_vec = maybe_wrap_vecnormalize(
        eval_vec,
        obs_mode=str(exp["env"]["obs_mode"]),
        normalize_obs=bool(vn["normalize_obs"]),
        normalize_reward=False,
        training=False,
        enabled=bool(vn["normalize_obs"] or vn["normalize_reward"]),
    )

    policy, policy_kwargs = build_policy_config(
        str(exp["env"]["obs_mode"]),
        float(exp["ppo"]["log_std_init"]),
    )

    model = PPO(
        policy,
        train_vec,
        learning_rate=float(exp["ppo"]["learning_rate"]),
        n_steps=int(exp["ppo"]["n_steps"]),
        batch_size=int(exp["ppo"]["batch_size"]),
        n_epochs=int(exp["ppo"]["n_epochs"]),
        gamma=float(exp["ppo"]["gamma"]),
        gae_lambda=float(exp["ppo"]["gae_lambda"]),
        clip_range=float(exp["ppo"]["clip_range"]),
        ent_coef=float(exp["ppo"]["ent_coef"]),
        vf_coef=float(exp["ppo"]["vf_coef"]),
        max_grad_norm=float(exp["ppo"]["max_grad_norm"]),
        seed=int(exp["seed"]),
        device=str(exp["ppo"]["device"]),
        verbose=1,
        tensorboard_log=str(output / "tensorboard"),
        policy_kwargs=policy_kwargs,
    )

    eval_callback = EvalCallback(
        eval_vec,
        best_model_save_path=str(output),
        log_path=str(output),
        eval_freq=max(int(exp["eval"]["freq"]) // num_envs, 1),
        n_eval_episodes=int(exp["eval"]["episodes"]),
        deterministic=True,
    )

    learn_callbacks = [eval_callback]
    if not args.no_wandb:
        learn_callbacks.append(
            WandbCallback(
                gradient_save_freq=int(exp["wandb"]["log_freq"]),
                model_save_path=str(output / "wandb_models"),
                model_save_freq=max(int(exp["eval"]["freq"]) // num_envs, 1),
                verbose=2,
            )
        )

    model.learn(
        total_timesteps=int(exp["total_timesteps"]),
        callback=learn_callbacks,
    )

    model.save(str(output / "final_model"))
    if isinstance(train_vec, VecNormalize):
        train_vec.save(str(output / "vecnormalize.pkl"))

    print(f"Training complete. Artifacts saved to {output}")

    eval_vec.close()
    train_vec.close()

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
