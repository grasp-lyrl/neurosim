"""Average a run's trailing checkpoints into one set of weights.

Training at a constant learning rate wanders: on a fixed eval set adjacent checkpoints
differ by several points of d1. Averaging a trailing window cancels that wander and lands
better than any checkpoint in it, so this is worth running before shipping a model.
"""

import argparse
import re
from pathlib import Path

import torch
from torch import Tensor


def checkpoint_epochs(models: Path) -> list[int]:
    """Epoch numbers of the `checkpoint_*.pth` files in a run, ascending."""
    epochs = [
        int(re.search(r"checkpoint_(\d+)\.pth$", str(p)).group(1))
        for p in models.glob("checkpoint_*.pth")
    ]
    assert epochs, f"no checkpoint_*.pth under {models}"
    return sorted(epochs)


def blend(weights: list[float], mode: str, decay: float) -> list[float]:
    """Normalized per-checkpoint weights, oldest first."""
    n = len(weights)
    if mode == "uniform":
        raw = [1.0] * n
    else:
        raw = [decay ** (n - 1 - i) for i in range(n)]
    return [w / sum(raw) for w in raw]


def average_weights(states: list[dict[str, Tensor]], weights: list[float]) -> dict:
    """Weighted mean of the float tensors; integer buffers come from the last state.

    Accumulates in float64 because the tail of a run differs in the low bits, which is
    exactly the part being averaged.
    """
    last = states[-1]
    averaged = {}
    for key, reference in last.items():
        if not reference.is_floating_point():
            averaged[key] = reference
            continue
        total = torch.zeros_like(reference, dtype=torch.float64)
        for weight, state in zip(weights, states, strict=True):
            total += weight * state[key].double()
        averaged[key] = total.to(reference.dtype)
    return averaged


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average a run's trailing checkpoints into one set of weights."
    )
    parser.add_argument(
        "--run", required=True, help="Training output dir (has models/)"
    )
    parser.add_argument("--last", type=int, default=6, help="How many checkpoints")
    parser.add_argument("--mode", default="uniform", choices=("uniform", "ema"))
    parser.add_argument(
        "--decay", type=float, default=0.6, help="EMA decay, oldest first"
    )
    parser.add_argument(
        "--out", default=None, help="Name under models/ (default avg_<n>.pth)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = Path(args.run) / "models"
    epochs = checkpoint_epochs(models)[-args.last :]
    states = [
        torch.load(
            models / f"checkpoint_{e}.pth", map_location="cpu", weights_only=False
        )
        for e in epochs
    ]
    states = [s.get("model", s) for s in states]

    weights = blend(states, args.mode, args.decay)
    out = models / (args.out or f"avg_{len(epochs)}.pth")
    torch.save(average_weights(states, weights), out)

    print(f"averaged {len(epochs)} checkpoints ({args.mode}) from epochs {epochs}")
    print("weights " + ", ".join(f"{w:.3f}" for w in weights))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
