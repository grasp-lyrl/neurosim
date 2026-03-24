"""Train a PPO policy for the neurosim hover-stop task using Stable-Baselines3.

Usage:
    python train_hover_sb3.py \
        --experiment-config applications/rl/configs/hover_sb3_experiment.yaml

The script saves:
    outputs/rl/<run_name>/best_model.zip   - best checkpoint (by eval reward)
    outputs/rl/<run_name>/final_model.zip  - final checkpoint
    outputs/rl/<run_name>/vecnormalize.pkl - observation / reward normalizer state
    W&B run logs                    - training metrics and config

Notes for vectorized training:
    - `num_envs` controls PPO rollout collection parallelism.
    - `eval_freq` in config is interpreted in environment steps;
        converted to callback frequency so evaluation stays consistent
    - training uses subprocess vectorization for heavy simulators (Habitat).
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
        "settings",
        "obs_mode",
        "seed",
        "num_envs",
        "total_timesteps",
        "eval_freq",
        "eval_episodes",
        "episode_seconds",
        "event_downsample_factor",
        "init_speed_min",
        "init_speed_max",
        "enable_navigable_check",
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "device",
        "normalize_obs",
        "normalize_reward",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required experiment config keys: {missing}")

    return cfg


def make_env(
    settings: str,
    obs_mode: str = "combined",
    episode_seconds: float = 10.0,
    init_speed_range: tuple[float, float] = (0.5, 2.0),
    event_downsample_factor: int = 1,
    enable_navigable_check: bool = True,
    seed: int | None = None,
    visualize: bool = False,
    debug_save_events_png: bool = False,
    debug_png_dir: str | None = None,
    debug_save_every_n_steps: int = 100,
    debug_accumulate_n_steps: int = 20,
    event_representation: str = "histogram",
    event_log_compression: float | None = None,
):
    """Factory callable for DummyVecEnv / SubprocVecEnv."""

    def _init():
        env = NeurosimRLEnv(
            settings=settings,
            obs_mode=obs_mode,
            episode_seconds=episode_seconds,
            init_speed_range=init_speed_range,
            event_downsample_factor=event_downsample_factor,
            enable_navigable_check=enable_navigable_check,
            enable_visualization=visualize,
            debug_save_events_png=debug_save_events_png,
            debug_png_dir=debug_png_dir,
            debug_save_every_n_steps=debug_save_every_n_steps,
            debug_accumulate_n_steps=debug_accumulate_n_steps,
            event_representation=event_representation,
            event_log_compression=event_log_compression,
        )
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _init


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SB3 PPO hover-stop training")
    p.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="YAML file containing experiment/training hyperparameters",
    )
    p.add_argument("--wandb-project", type=str, default="neurosim-rl")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--visualize", action="store_true")
    p.add_argument(
        "--debug-save-events-png",
        action="store_true",
        help="Save periodic event-frame PNGs for headless debugging",
    )
    p.add_argument(
        "--debug-png-dir",
        type=str,
        default=None,
        help="Directory for debug PNGs (defaults to <output-dir>/debug_events)",
    )
    p.add_argument("--debug-save-every-n-steps", type=int, default=100)
    p.add_argument("--debug-accumulate-n-steps", type=int, default=20)
    return p.parse_args()


def build_policy_config(obs_mode: str) -> tuple[str, dict[str, Any]]:
    if obs_mode == "state":
        return "MlpPolicy", {}
    if obs_mode == "events":
        return "CnnPolicy", {
            "features_extractor_class": EventCnnExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "normalize_images": False,
        }
    return "MultiInputPolicy", {
        "features_extractor_class": CombinedEventStateExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
        "normalize_images": False,
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
    n_steps = int(exp["n_steps"])
    batch_size = int(exp["batch_size"])
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

    debug_png_dir = args.debug_png_dir
    if args.debug_save_events_png and debug_png_dir is None:
        debug_png_dir = str(output / "debug_events")

    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={"cli": vars(args), "experiment": exp},
        sync_tensorboard=True,
        save_code=True,
        dir=str(output),
    )

    init_speed_range = (float(exp["init_speed_min"]), float(exp["init_speed_max"]))

    vec_env_type = str(exp.get("vec_env_type", "auto"))
    vec_env_start_method = str(exp.get("vec_env_start_method", "spawn"))

    # Training env (use multiprocessing for heavy simulators when num_envs > 1)
    train_env_fns = [
        make_env(
            settings=str(exp["settings"]),
            obs_mode=str(exp["obs_mode"]),
            episode_seconds=float(exp["episode_seconds"]),
            init_speed_range=init_speed_range,
            event_downsample_factor=int(exp["event_downsample_factor"]),
            enable_navigable_check=bool(exp["enable_navigable_check"]),
            seed=int(exp["seed"]) + env_idx,
            visualize=args.visualize,
            debug_save_events_png=args.debug_save_events_png,
            debug_png_dir=(
                str(Path(debug_png_dir) / f"env_{env_idx}")
                if debug_png_dir is not None
                else None
            ),
            debug_save_every_n_steps=args.debug_save_every_n_steps,
            debug_accumulate_n_steps=args.debug_accumulate_n_steps,
            event_representation=str(exp.get("event_representation", "histogram")),
            event_log_compression=exp.get("event_log_compression"),
        )
        for env_idx in range(num_envs)
    ]
    train_vec = build_train_vec_env(
        train_env_fns,
        num_envs=num_envs,
        vec_env_type=vec_env_type,
        start_method=vec_env_start_method,
    )
    if bool(exp["normalize_obs"]) or bool(exp["normalize_reward"]):
        train_vec = VecNormalize(
            train_vec,
            **build_vecnormalize_kwargs(
                obs_mode=str(exp["obs_mode"]),
                normalize_obs=bool(exp["normalize_obs"]),
                normalize_reward=bool(exp["normalize_reward"]),
                training=True,
            ),
        )

    # Eval env (separate instance, shared normalization stats)
    eval_num_envs = int(exp.get("eval_num_envs", 1))
    if eval_num_envs <= 0:
        raise ValueError("eval_num_envs must be >= 1")

    eval_vec = DummyVecEnv(
        [
            make_env(
                settings=str(exp["settings"]),
                obs_mode=str(exp["obs_mode"]),
                episode_seconds=float(exp["episode_seconds"]),
                init_speed_range=init_speed_range,
                event_downsample_factor=int(exp["event_downsample_factor"]),
                enable_navigable_check=bool(exp["enable_navigable_check"]),
                seed=int(exp["seed"]) + 1000 + env_idx,
                event_representation=str(exp.get("event_representation", "histogram")),
                event_log_compression=exp.get("event_log_compression"),
            )
            for env_idx in range(eval_num_envs)
        ]
    )
    if bool(exp["normalize_obs"]) or bool(exp["normalize_reward"]):
        eval_vec = VecNormalize(
            eval_vec,
            **build_vecnormalize_kwargs(
                obs_mode=str(exp["obs_mode"]),
                normalize_obs=bool(exp["normalize_obs"]),
                normalize_reward=False,
                training=False,
            ),
        )

    policy, policy_kwargs = build_policy_config(str(exp["obs_mode"]))

    model = PPO(
        policy,
        train_vec,
        learning_rate=float(exp["learning_rate"]),
        n_steps=int(exp["n_steps"]),
        batch_size=int(exp["batch_size"]),
        n_epochs=int(exp["n_epochs"]),
        gamma=float(exp["gamma"]),
        gae_lambda=float(exp["gae_lambda"]),
        clip_range=float(exp["clip_range"]),
        ent_coef=float(exp["ent_coef"]),
        vf_coef=float(exp["vf_coef"]),
        max_grad_norm=float(exp["max_grad_norm"]),
        seed=int(exp["seed"]),
        device=str(exp["device"]),
        verbose=1,
        tensorboard_log=str(output / "tensorboard"),
        policy_kwargs=policy_kwargs,
    )

    eval_callback = EvalCallback(
        eval_vec,
        best_model_save_path=str(output),
        log_path=str(output),
        eval_freq=max(int(exp["eval_freq"]) // num_envs, 1),
        n_eval_episodes=int(exp["eval_episodes"]),
        deterministic=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=int(exp.get("wandb_log_freq", 100)),
        model_save_path=str(output / "wandb_models"),
        model_save_freq=max(int(exp["eval_freq"]) // num_envs, 1),
        verbose=2,
    )

    model.learn(
        total_timesteps=int(exp["total_timesteps"]),
        callback=[eval_callback, wandb_callback],
    )

    model.save(str(output / "final_model"))
    if isinstance(train_vec, VecNormalize):
        train_vec.save(str(output / "vecnormalize.pkl"))

    print(f"Training complete. Artifacts saved to {output}")
    train_vec.close()
    eval_vec.close()

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
