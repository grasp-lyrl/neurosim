"""Train a PPO policy for the neurosim hover-stop task using Stable-Baselines3.

Usage:
    python train_hover_sb3.py \
        --experiment-config applications/rl/configs/hover_sb3_experiment.yaml

The script saves:
    <output_dir>/best_model.zip     - best checkpoint (by eval reward)
    <output_dir>/final_model.zip    - final checkpoint
    <output_dir>/vecnormalize.pkl   - observation / reward normalizer state
    W&B run logs                    - training metrics and config
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
        "total_timesteps",
        "eval_freq",
        "eval_episodes",
        "episode_seconds",
        "body_rate_limit",
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
    body_rate_limit: float = 8.0,
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
            body_rate_limit=body_rate_limit,
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
    p.add_argument("--output-dir", type=str, default="outputs/rl/hover_sb3")
    p.add_argument("--wandb-project", type=str, default="neurosim-rl")
    p.add_argument("--wandb-run-name", type=str, default=None)
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


def main():
    args = parse_args()
    exp = load_experiment_config(args.experiment_config)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    debug_png_dir = args.debug_png_dir
    if args.debug_save_events_png and debug_png_dir is None:
        debug_png_dir = str(output / "debug_events")

    np.random.seed(int(exp["seed"]))

    run_name = args.wandb_run_name
    if run_name is None:
        run_name = f"neurosim_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={"cli": vars(args), "experiment": exp},
        sync_tensorboard=True,
        save_code=True,
        dir=str(output),
    )

    init_speed_range = (float(exp["init_speed_min"]), float(exp["init_speed_max"]))

    # Training env
    train_vec = DummyVecEnv(
        [
            make_env(
                settings=str(exp["settings"]),
                obs_mode=str(exp["obs_mode"]),
                episode_seconds=float(exp["episode_seconds"]),
                body_rate_limit=float(exp["body_rate_limit"]),
                init_speed_range=init_speed_range,
                event_downsample_factor=int(exp["event_downsample_factor"]),
                enable_navigable_check=bool(exp["enable_navigable_check"]),
                seed=int(exp["seed"]),
                visualize=args.visualize,
                debug_save_events_png=args.debug_save_events_png,
                debug_png_dir=debug_png_dir,
                debug_save_every_n_steps=args.debug_save_every_n_steps,
                debug_accumulate_n_steps=args.debug_accumulate_n_steps,
                event_representation=str(exp.get("event_representation", "histogram")),
                event_log_compression=exp.get("event_log_compression"),
            )
        ]
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
    eval_vec = DummyVecEnv(
        [
            make_env(
                settings=str(exp["settings"]),
                obs_mode=str(exp["obs_mode"]),
                episode_seconds=float(exp["episode_seconds"]),
                body_rate_limit=float(exp["body_rate_limit"]),
                init_speed_range=init_speed_range,
                event_downsample_factor=int(exp["event_downsample_factor"]),
                enable_navigable_check=bool(exp["enable_navigable_check"]),
                seed=int(exp["seed"]) + 1000,
                event_representation=str(exp.get("event_representation", "histogram")),
                event_log_compression=exp.get("event_log_compression"),
            )
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
        eval_freq=int(exp["eval_freq"]),
        n_eval_episodes=int(exp["eval_episodes"]),
        deterministic=True,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=int(exp.get("wandb_log_freq", 100)),
        model_save_path=str(output / "wandb_models"),
        model_save_freq=int(exp["eval_freq"]),
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
