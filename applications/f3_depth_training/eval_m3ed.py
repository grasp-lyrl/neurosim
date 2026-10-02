"""Score one or more of a run's checkpoints against the M3ED LiDAR validation set.

The same loader, metrics and autocast the trainer validates under, so numbers here are
comparable to the ``Validation (M3ED)`` lines in a run's log.
"""

import argparse

import torch
import yaml

from .data import build_m3ed_loader, evaluate_m3ed
from .nets import EventFFDepthAnythingV2, load_depth_weights
from .train_depth_nonrec import predict_full_frame
from .utils import build_mode
from .utils.experiment import setup_torch

REPORT = ("abs_rel", "d1", "d2", "d3", "rmse", "silog", "abs_rel_near", "d1_near")


def load_model(run: str, ckpt: str, device):
    """Rebuild the model from the run's saved config and load one checkpoint."""
    model = EventFFDepthAnythingV2.init_from_config(f"{run}/models/depth_config.yml")
    state = torch.load(
        f"{run}/models/{ckpt}", map_location="cpu", weights_only=False, mmap=True
    )
    load_depth_weights(model, state.get("model", state))
    return model.to(device).eval()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score checkpoints against the M3ED LiDAR validation set."
    )
    parser.add_argument(
        "--run", required=True, help="Training output dir (has models/)"
    )
    parser.add_argument("--conf", required=True, help="Training config YAML")
    parser.add_argument(
        "--ckpt", nargs="+", required=True, help="Checkpoint names under models/"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--amp", action="store_true", help="bf16, as training validates"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.conf, "r") as f:
        conf = yaml.safe_load(f)

    setup_torch()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    loader = None
    for ckpt in args.ckpt:
        model = load_model(args.run, ckpt, device)
        if loader is None:
            # Built once: the frame shape and the mode are properties of the run.
            mode = build_mode(conf, model.dav2.head == "sigmoid")
            aligned = [k for k in mode.metric_names if k.endswith("_aligned")]
            report = (mode.loss_fn.name, *REPORT, *aligned)
            width, height, window_ms = model.eventff.frame_sizes
            loader = build_m3ed_loader(conf["data"]["m3ed"], width, height, window_ms)
            assert loader is not None, "data.m3ed has no sequences"

        with torch.autocast("cuda", enabled=args.amp, dtype=torch.bfloat16):
            scores = evaluate_m3ed(model, loader, predict_full_frame, device, mode)
        print(
            f"{ckpt:26s} " + "  ".join(f"{name} {scores[name]:.4f}" for name in report),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
