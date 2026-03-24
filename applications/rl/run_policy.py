"""Run a trained SB3 PPO policy in neurosim.

Usage:
    python run_policy.py --checkpoint outputs/rl/hover_sb3/best_model.zip \
                         --settings configs/apartment_1-rl-settings.yaml

If the checkpoint was trained with VecNormalize, place ``vecnormalize.pkl``
next to the model or pass ``--vecnormalize`` explicitly.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from neurosim.rl import NeurosimRLEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a trained SB3 PPO policy in neurosim")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .zip model")
    p.add_argument("--settings", type=str, required=True)
    p.add_argument(
        "--obs-mode",
        type=str,
        default="combined",
        choices=["events", "state", "combined"],
    )
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--episode-seconds", type=float, default=8.0)
    p.add_argument("--body-rate-limit", type=float, default=8.0)
    p.add_argument("--event-downsample-factor", type=int, default=1)
    p.add_argument("--init-speed-min", type=float, default=0.5)
    p.add_argument("--init-speed-max", type=float, default=2.0)
    p.add_argument("--enable-navigable-check", action="store_true", default=True)
    p.add_argument(
        "--no-navigable-check", dest="enable_navigable_check", action="store_false"
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--visualization-rrd-path", type=str, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--debug-save-events-png",
        action="store_true",
        help="Save periodic event-frame PNGs for headless debugging",
    )
    p.add_argument(
        "--debug-png-dir",
        type=str,
        default="outputs/rl/hover_sb3/debug_events_rollout",
    )
    p.add_argument("--debug-save-every-n-steps", type=int, default=25)
    p.add_argument("--debug-accumulate-n-steps", type=int, default=10)
    p.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help="Path to vecnormalize.pkl (auto-detected from checkpoint dir if omitted)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    init_speed_range = (args.init_speed_min, args.init_speed_max)

    def _make_env():
        return NeurosimRLEnv(
            settings=args.settings,
            obs_mode=args.obs_mode,
            episode_seconds=args.episode_seconds,
            body_rate_limit=args.body_rate_limit,
            init_speed_range=init_speed_range,
            event_downsample_factor=args.event_downsample_factor,
            enable_navigable_check=args.enable_navigable_check,
            enable_visualization=args.visualize,
            visualization_rrd_path=args.visualization_rrd_path,
            debug_save_events_png=args.debug_save_events_png,
            debug_png_dir=args.debug_png_dir,
            debug_save_every_n_steps=args.debug_save_every_n_steps,
            debug_accumulate_n_steps=args.debug_accumulate_n_steps,
        )

    vec_env = DummyVecEnv([_make_env])

    # Restore observation / reward normalizer if available
    vecnorm_path = args.vecnormalize
    if vecnorm_path is None:
        candidate = Path(args.checkpoint).parent / "vecnormalize.pkl"
        if candidate.exists():
            vecnorm_path = str(candidate)

    if vecnorm_path is not None:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize from {vecnorm_path}")

    model = PPO.load(args.checkpoint, env=vec_env, device=args.device)

    for ep in range(1, args.episodes + 1):
        obs = vec_env.reset()
        total_reward = 0.0
        step = 0

        while True:
            step += 1
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, infos = vec_env.step(action)
            total_reward += float(reward[0])

            if done[0]:
                info = infos[0]
                print(
                    f"episode={ep} steps={step} total_reward={total_reward:.3f} "
                    f"sim_time={info.get('time', 0):.3f} "
                    f"is_success={info.get('is_success', False)} "
                    f"termination={info.get('termination_reason', 'none')}"
                )
                break

    vec_env.close()


if __name__ == "__main__":
    main()
