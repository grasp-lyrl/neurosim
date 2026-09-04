"""Measure a telegraph dither's causal signature in event-camera observations.

Each candidate is replayed with the same episode seed and compared with an
otherwise identical zero-dither rollout.  Only the pre-launch warning window
is summarized, so projectile motion after launch cannot inflate the result.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from neurosim.rl import env_class_for_task

from train_sb3 import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--seed0", type=int, default=10001)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--amplitudes-m", type=float, nargs="+", required=True)
    parser.add_argument("--frequencies-hz", type=float, nargs="+", required=True)
    parser.add_argument("--visual-scales-m", type=float, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-amplitude-m", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-frequency-hz", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-visual-scale-m", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(args.amplitudes_m) != len(args.frequencies_hz):
        parser.error("--amplitudes-m and --frequencies-hz must have equal length")
    if args.visual_scales_m is not None and len(args.visual_scales_m) != len(
        args.amplitudes_m
    ):
        parser.error("--visual-scales-m must match the number of amplitudes")
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    return args


def _rollout(
    base_env_cfg: dict,
    *,
    seed: int,
    amplitude_m: float,
    frequency_hz: float,
    visual_scale_m: float,
    warning_frames: int,
    worker_output: str | None = None,
) -> np.ndarray:
    env_cfg = copy.deepcopy(base_env_cfg)
    env_cfg["enable_visualization"] = False
    obstacles = env_cfg.setdefault("visual_backend", {}).setdefault(
        "dynamic_obstacles", {}
    )
    obstacles["telegraph_dither_amplitude_m"] = float(amplitude_m)
    obstacles["telegraph_dither_frequency_hz"] = float(frequency_hz)
    for template in obstacles.get("templates", []):
        template["scale"] = [float(visual_scale_m)] * 3
    env = env_class_for_task(env_cfg["task"]["name"])(
        env_config=env_cfg, train=False
    )
    frames: list[np.ndarray] = []
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(warning_frames):
            frames.append(np.asarray(observation["events"], dtype=np.float32).copy())
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"episode seed {seed} ended inside the warning window"
                )
        stacked = np.stack(frames)
        if worker_output is not None:
            # Habitat can segfault while destroying a second simulator in the
            # same process. Persist first, then bypass native destructors; the
            # parent gives every matched rollout a fresh process.
            with open(worker_output, "wb") as stream:
                np.save(stream, stacked)
                stream.flush()
                os.fsync(stream.fileno())
            os._exit(0)
    finally:
        env.close()
    return stacked


def _worker_command(
    args: argparse.Namespace,
    *,
    seed: int,
    amplitude_m: float,
    frequency_hz: float,
    visual_scale_m: float,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--experiment-config",
        args.experiment_config,
        "--amplitudes-m",
        str(amplitude_m),
        "--frequencies-hz",
        str(frequency_hz),
        "--visual-scales-m",
        str(visual_scale_m),
        "--output",
        args.output,
        "--worker-seed",
        str(seed),
        "--worker-amplitude-m",
        str(amplitude_m),
        "--worker-frequency-hz",
        str(frequency_hz),
        "--worker-visual-scale-m",
        str(visual_scale_m),
        "--worker-output",
        str(output),
    ]


def _isolated_rollout(
    args: argparse.Namespace,
    *,
    seed: int,
    amplitude_m: float,
    frequency_hz: float,
    visual_scale_m: float,
    output: Path,
) -> np.ndarray:
    completed = subprocess.run(
        _worker_command(
            args,
            seed=seed,
            amplitude_m=amplitude_m,
            frequency_hz=frequency_hz,
            visual_scale_m=visual_scale_m,
            output=output,
        ),
        check=False,
    )
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(
            f"isolated rollout failed for seed={seed}, amplitude={amplitude_m}, "
            f"frequency={frequency_hz}: exit {completed.returncode}"
        )
    return np.load(output)


def _summarize(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.abs(candidate - reference)
    reference_mass = float(np.abs(reference).sum())
    candidate_mass = float(np.abs(candidate).sum())
    delta_mass = float(delta.sum())
    return {
        "changed_elements": int(np.count_nonzero(delta > 1e-6)),
        "delta_l1": delta_mass,
        "reference_l1": reference_mass,
        "candidate_l1": candidate_mass,
        "delta_over_reference": delta_mass / max(reference_mass, 1e-12),
        "delta_peak": float(delta.max(initial=0.0)),
        "active_frames": int(np.count_nonzero(delta.reshape(delta.shape[0], -1).sum(1))),
    }


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.experiment_config)
    base_env_cfg = copy.deepcopy(cfg["env"])
    obstacle_cfg = base_env_cfg["visual_backend"]["dynamic_obstacles"]
    dt = float(base_env_cfg.get("control_dt", base_env_cfg.get("dt", 0.05)))
    warning_s = float(obstacle_cfg.get("telegraph_time_s", 0.0))
    warning_frames = max(1, int(np.ceil(warning_s / dt)))

    if args.worker_output is not None:
        if (
            args.worker_seed is None
            or args.worker_amplitude_m is None
            or args.worker_frequency_hz is None
            or args.worker_visual_scale_m is None
        ):
            raise ValueError("incomplete internal worker arguments")
        _rollout(
            base_env_cfg,
            seed=args.worker_seed,
            amplitude_m=args.worker_amplitude_m,
            frequency_hz=args.worker_frequency_hz,
            visual_scale_m=args.worker_visual_scale_m,
            warning_frames=warning_frames,
            worker_output=args.worker_output,
        )
        raise AssertionError("worker returned unexpectedly")

    inherited_scale_m = float(obstacle_cfg["templates"][0]["scale"][0])
    visual_scales_m = args.visual_scales_m or [inherited_scale_m] * len(
        args.amplitudes_m
    )
    candidates = list(
        zip(args.amplitudes_m, args.frequencies_hz, visual_scales_m, strict=True)
    )
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="telegraph-cue-") as temp_dir:
        temp_root = Path(temp_dir)
        for seed in range(args.seed0, args.seed0 + args.seeds):
            reference = _isolated_rollout(
                args,
                seed=seed,
                amplitude_m=0.0,
                frequency_hz=candidates[0][1],
                visual_scale_m=inherited_scale_m,
                output=temp_root / f"{seed}_reference.npy",
            )
            for candidate_index, (
                amplitude_m,
                frequency_hz,
                visual_scale_m,
            ) in enumerate(candidates):
                candidate = _isolated_rollout(
                    args,
                    seed=seed,
                    amplitude_m=amplitude_m,
                    frequency_hz=frequency_hz,
                    visual_scale_m=visual_scale_m,
                    output=temp_root / f"{seed}_{candidate_index}.npy",
                )
                row = {
                    "seed": seed,
                    "amplitude_m": amplitude_m,
                    "frequency_hz": frequency_hz,
                    "visual_scale_m": visual_scale_m,
                    **_summarize(reference, candidate),
                }
                results.append(row)
                print(row, flush=True)

    summaries: list[dict] = []
    for amplitude_m, frequency_hz, visual_scale_m in candidates:
        selected = [
            row
            for row in results
            if row["amplitude_m"] == amplitude_m
            and row["frequency_hz"] == frequency_hz
            and row["visual_scale_m"] == visual_scale_m
        ]
        summary = {
            "amplitude_m": amplitude_m,
            "frequency_hz": frequency_hz,
            "visual_scale_m": visual_scale_m,
            "seeds": len(selected),
        }
        for key in (
            "changed_elements",
            "delta_l1",
            "reference_l1",
            "candidate_l1",
            "delta_over_reference",
            "delta_peak",
            "active_frames",
        ):
            values = np.asarray([row[key] for row in selected], dtype=np.float64)
            summary[f"{key}_median"] = float(np.median(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
        summaries.append(summary)

    payload = {
        "experiment_config": args.experiment_config,
        "seed0": args.seed0,
        "warning_s": warning_s,
        "warning_frames": warning_frames,
        "event_shape": list(reference.shape[1:]),
        "summaries": summaries,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
