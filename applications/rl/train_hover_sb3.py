"""Train a PPO policy for the neurosim hover-stop task using Stable-Baselines3.

Usage:
    python train_hover_sb3.py --settings configs/apartment_1-rl-settings.yaml

The script saves:
    <output_dir>/best_model.zip     - best checkpoint (by eval reward)
    <output_dir>/final_model.zip    - final checkpoint
    <output_dir>/vecnormalize.pkl   - observation / reward normalizer state
    <output_dir>/tensorboard/       - TensorBoard logs
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from neurosim.rl import CombinedEventStateExtractor, EventCnnExtractor, NeurosimRLEnv


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
    p.add_argument("--settings", type=str, required=True)
    p.add_argument(
        "--obs-mode",
        type=str,
        default="combined",
        choices=["events", "state", "combined"],
    )
    p.add_argument(
        "--demo-profile",
        type=str,
        default="none",
        choices=["none", "state-baseline", "events-ablation"],
        help="Optional fast-demo profile that overrides selected hyperparameters",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--episode-seconds", type=float, default=10.0)
    p.add_argument("--body-rate-limit", type=float, default=8.0)
    p.add_argument("--event-downsample-factor", type=int, default=1)
    p.add_argument("--init-speed-min", type=float, default=0.5)
    p.add_argument("--init-speed-max", type=float, default=1.0)
    p.add_argument("--enable-navigable-check", action="store_true", default=True)
    p.add_argument(
        "--no-navigable-check", dest="enable_navigable_check", action="store_false"
    )
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--normalize-obs", action="store_true", default=True)
    p.add_argument("--no-normalize-obs", dest="normalize_obs", action="store_false")
    p.add_argument("--normalize-reward", action="store_true", default=True)
    p.add_argument(
        "--no-normalize-reward", dest="normalize_reward", action="store_false"
    )
    p.add_argument("--eval-freq", type=int, default=2048)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="outputs/rl/hover_sb3")
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
    p.add_argument(
        "--event-representation",
        type=str,
        default="histogram",
        choices=["histogram", "event_frame"],
        help="Event representation: 'histogram' accumulates counts, 'event_frame' marks events as 1",
    )
    p.add_argument(
        "--event-log-compression",
        type=float,
        default=None,
        help="Log compression factor (e.g., 10.0) for boosting low-intensity events; None for linear",
    )
    return p.parse_args()


def apply_demo_profile(args: argparse.Namespace) -> argparse.Namespace:
    if args.demo_profile == "none":
        return args

    if args.demo_profile == "state-baseline":
        args.obs_mode = "state"
        args.total_timesteps = max(args.total_timesteps, 100_000)
        args.n_steps = 512
        args.batch_size = 128
        args.n_epochs = 4
        args.normalize_obs = True
        args.normalize_reward = True

    if args.demo_profile == "events-ablation":
        args.obs_mode = "events"
        args.event_downsample_factor = max(args.event_downsample_factor, 4)
        args.total_timesteps = max(args.total_timesteps, 150_000)
        args.n_steps = 1024
        args.batch_size = 128
        args.n_epochs = 4
        args.normalize_obs = True
        args.normalize_reward = True

    return args


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
        # Event tensors are normalized inside the CNN extractor.
        kwargs["norm_obs"] = False
    elif normalize_obs and obs_mode == "combined":
        # Normalize privileged state only, not event frames.
        kwargs["norm_obs_keys"] = ["state"]
    return kwargs


def main():
    args = apply_demo_profile(parse_args())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    debug_png_dir = args.debug_png_dir
    if args.debug_save_events_png and debug_png_dir is None:
        debug_png_dir = str(output / "debug_events")

    np.random.seed(args.seed)

    init_speed_range = (args.init_speed_min, args.init_speed_max)

    # Training env
    train_vec = DummyVecEnv(
        [
            make_env(
                settings=args.settings,
                obs_mode=args.obs_mode,
                episode_seconds=args.episode_seconds,
                body_rate_limit=args.body_rate_limit,
                init_speed_range=init_speed_range,
                event_downsample_factor=args.event_downsample_factor,
                enable_navigable_check=args.enable_navigable_check,
                seed=args.seed,
                visualize=args.visualize,
                debug_save_events_png=args.debug_save_events_png,
                debug_png_dir=debug_png_dir,
                debug_save_every_n_steps=args.debug_save_every_n_steps,
                debug_accumulate_n_steps=args.debug_accumulate_n_steps,
                event_representation=args.event_representation,
                event_log_compression=args.event_log_compression,
            )
        ]
    )
    if args.normalize_obs or args.normalize_reward:
        train_vec = VecNormalize(
            train_vec,
            **build_vecnormalize_kwargs(
                obs_mode=args.obs_mode,
                normalize_obs=args.normalize_obs,
                normalize_reward=args.normalize_reward,
                training=True,
            ),
        )

    # Eval env (separate instance, shared normalization stats)
    eval_vec = DummyVecEnv(
        [
            make_env(
                settings=args.settings,
                obs_mode=args.obs_mode,
                episode_seconds=args.episode_seconds,
                body_rate_limit=args.body_rate_limit,
                init_speed_range=init_speed_range,
                event_downsample_factor=args.event_downsample_factor,
                enable_navigable_check=args.enable_navigable_check,
                seed=args.seed + 1000,
                event_representation=args.event_representation,
                event_log_compression=args.event_log_compression,
            )
        ]
    )
    if args.normalize_obs or args.normalize_reward:
        eval_vec = VecNormalize(
            eval_vec,
            **build_vecnormalize_kwargs(
                obs_mode=args.obs_mode,
                normalize_obs=args.normalize_obs,
                normalize_reward=False,
                training=False,
            ),
        )

    policy, policy_kwargs = build_policy_config(args.obs_mode)

    model = PPO(
        policy,
        train_vec,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=str(output / "tensorboard"),
        policy_kwargs=policy_kwargs,
    )

    eval_callback = EvalCallback(
        eval_vec,
        best_model_save_path=str(output),
        log_path=str(output),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    model.save(str(output / "final_model"))
    if isinstance(train_vec, VecNormalize):
        train_vec.save(str(output / "vecnormalize.pkl"))

    print(f"Training complete. Artifacts saved to {output}")
    train_vec.close()
    eval_vec.close()


if __name__ == "__main__":
    main()
