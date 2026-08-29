"""Iterate DAgger: train a clone, fly it, relabel what it visits, repeat.

Plain BC on this task fails for a reason no amount of data collected from the
*expert's* states can fix. The clone makes a small error, reaches a state the
expert never visited, predicts worse there, and compounds -- measured at 0%
success with 7 of 20 episodes ending in tracking_failure, the vehicle drifting
past the 3.0 m limit. DAgger closes that loop by labelling the states the
learner actually reaches.

It also weakens a second problem specific to these demonstrations. On the
expert's own trajectories a linear fit of the state vector alone predicts the
expert's command at R2 = 0.990 -- the action is nearly determined by where the
vehicle already is in a maneuver it began earlier, so BC can score well without
reading the event camera at all. Off-distribution states break that
relationship: when the clone is somewhere the expert would not be, the
corrective label no longer follows from the state, and the obstacle's position
has to come from somewhere else.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"failed ({proc.returncode}): {' '.join(cmd)}\nsee {log}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-config", required=True)
    p.add_argument("--seed-datasets", nargs="+", required=True)
    p.add_argument("--outdir", default="outputs/rl/bc/dagger")
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--episodes-per-iter", type=int, default=40)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--samples-per-epoch", type=int, default=30000)
    p.add_argument("--collect-seed0", type=int, default=300000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-seed0", type=int, default=9001)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Freeze the config. The driver re-reads it for every training and
    # collection subprocess, so editing the file mid-run silently changes the
    # task between iterations -- one run reported 50%, 0%, 0%, 80% across four
    # iterations that were not evaluating the same environment, and none of
    # those numbers could be attributed.
    frozen = outdir / "frozen_experiment_config.yaml"
    shutil.copyfile(args.experiment_config, frozen)
    args.experiment_config = str(frozen)
    print(f"[dagger] frozen config -> {frozen}", flush=True)

    datasets = list(args.seed_datasets)
    history = []

    for it in range(args.iterations):
        clone = outdir / f"clone_iter{it}.json"
        run(
            [
                PYTHON, "applications/rl/train_velocity_dodge_bc.py",
                "--experiment-config", args.experiment_config,
                "--dataset", *datasets,
                "--output", str(clone),
                "--epochs", str(args.epochs),
                "--samples-per-epoch", str(args.samples_per_epoch),
                "--blank-previous-action",
                "--eval-episodes", str(args.eval_episodes),
                "--eval-every", str(max(args.epochs // 2, 1)),
                "--eval-seed0", str(args.eval_seed0),
                "--device", args.device,
            ],
            outdir / f"train_iter{it}.log",
        )
        summary = json.loads(clone.read_text())
        evals = [row for row in summary.get("history", []) if "success_rate" in row]
        best = max((row["success_rate"] for row in evals), default=None)
        history.append({"iteration": it, "datasets": len(datasets), "best_success": best})
        print(f"[dagger] iter {it}: {len(datasets)} datasets, best success {best}", flush=True)
        (outdir / "history.json").write_text(json.dumps(history, indent=2))

        if it == args.iterations - 1:
            break

        # Beta anneals expert control to zero, so later iterations collect
        # states the clone reaches unaided -- which is the distribution it
        # will actually be evaluated on.
        beta = max(0.0, 1.0 - (it + 1) / max(args.iterations - 1, 1))
        shard = outdir / f"dagger_iter{it}.h5"
        run(
            [
                PYTHON, "applications/rl/evaluate_velocity_dodge_oracle.py",
                "--experiment-config", args.experiment_config,
                "--episodes", str(args.episodes_per_iter),
                "--seed0", str(args.collect_seed0 + it * 1000),
                "--include-all-episodes", "--include-plan-failures",
                "--rollout-policy", str(clone.with_suffix(".pt")),
                "--beta", f"{beta:g}",
                "--dataset", str(shard),
                "--output", str(outdir / f"collect_iter{it}.json"),
            ],
            outdir / f"collect_iter{it}.log",
        )
        datasets.append(str(shard))
        print(f"[dagger] iter {it}: collected {shard.name} at beta={beta:g}", flush=True)


if __name__ == "__main__":
    main()
