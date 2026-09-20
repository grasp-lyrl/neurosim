"""Step a trained depth model over simulated episodes, three panels per sample.

Same loader the trainer uses, so what you see is what it trains on. Press ``n``/Enter for
the next sample and ``q`` to quit, or pass ``--video`` to write an mp4 instead.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from matplotlib import colormaps

from .data import build_online_loader, process_batch, usable_sample_filter
from .nets import EventFFDepthAnythingV2, load_depth_weights
from .train_depth_nonrec import predict_full_frame
from .utils import (
    build_mode,
    ev_to_frames_with_polarity,
    get_depth_image,
    get_disparity_image,
)

HEADER_H = 96
COLUMNS = ("events", "ground truth", "prediction")


def load_model(run: str, ckpt: str, device):
    """Rebuild the model from its saved config and load weights."""
    model = EventFFDepthAnythingV2.init_from_config(f"{run}/models/depth_config.yml")
    state = torch.load(
        f"{run}/models/{ckpt}", map_location="cpu", weights_only=False, mmap=True
    )
    load_depth_weights(model, state.get("model", state))
    return model.to(device).eval()


def header(width: int, lines: list[str]) -> np.ndarray:
    """Caption strip: two rows of stats, then a label under each panel."""
    strip = np.zeros((HEADER_H, width, 3), np.uint8)
    for row, text in enumerate(lines):
        cv2.putText(
            strip,
            text,
            (14, 28 + 26 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    for column, label in enumerate(COLUMNS):
        cv2.putText(
            strip,
            label,
            (14 + column * width // 3, HEADER_H - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
    return strip


def panels(events, counts, truth, prediction, valid, to_image, lines):
    """Stats strip over events | ground truth | prediction, in BGR."""
    height, width = truth.shape
    frames = ev_to_frames_with_polarity(events, counts, width, height).cpu().numpy()
    images = np.hstack(
        [
            frames[0],
            to_image(truth, valid),
            to_image(prediction, torch.ones_like(prediction, dtype=torch.bool)),
        ]
    )
    return np.vstack(
        [
            header(images.shape[1], lines),
            cv2.cvtColor(images, cv2.COLOR_RGB2BGR),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay simulated episodes through a trained depth model."
    )
    parser.add_argument(
        "--run", required=True, help="Training output dir (has models/)"
    )
    parser.add_argument("--conf", required=True, help="Training config YAML")
    parser.add_argument(
        "--ckpt", default="best_d1.pth", help="Checkpoint under models/"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--video", help="Write an mp4 instead of opening a window")
    parser.add_argument("--frames", type=int, default=200, help="Samples to show")
    parser.add_argument(
        "--depth-range",
        default="0.5,20",
        help="near,far metres of the fixed colour scale (metric runs only)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.conf, "r") as f:
        conf = yaml.safe_load(f)
    for key, value in conf.items():
        setattr(args, key, value)

    data_cfg = conf["data"]
    roles = data_cfg["online_data"]["roles"]
    args.depth_sensor, args.event_sensor = roles["anchor"][0], roles["stream"][0]
    args.color_sensor = data_cfg.get("color_sensor")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    cmap = colormaps["magma"]

    model = load_model(args.run, args.ckpt, device)
    metric = model.dav2.head == "sigmoid"
    mode = build_mode(conf, metric)
    near, far = (float(v) for v in args.depth_range.split(","))
    # A metric run is worth seeing on one scale for the whole clip; a relative one has
    # no scale to hold fixed, so it keeps the per-frame stretch.
    to_image = (
        (lambda x, m: get_depth_image(x, m, cmap, near, far))
        if metric
        else (lambda x, m: get_disparity_image(x, m, cmap))
    )
    width, height, window_ms = model.eventff.frame_sizes
    args.event_norm = (width, height, window_ms * 1000)
    args.max_events = int(data_cfg.get("max_events_per_sample", 0))

    # One producer: this is a viewer, and episodes should arrive in order.
    data_cfg = dict(data_cfg)
    data_cfg["online_data"] = dict(data_cfg["online_data"]) | {
        "num_producers": 1,
        "gpu_ids": [args.gpu],
    }
    loader = build_online_loader(
        data_cfg,
        batch_size=1,
        sample_filter=usable_sample_filter(
            args.depth_sensor,
            args.event_sensor,
            mode.min_depth,
            int(data_cfg.get("min_events_per_sample", 500)),
        ),
    )

    writer = None
    if args.video:
        writer = cv2.VideoWriter(
            args.video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            1000 / window_ms,
            (3 * width, height + HEADER_H),
        )
        print(f"writing {args.frames} samples to {args.video}")
    else:
        print("n/Enter: next sample   q: quit")

    try:
        for index, batch in enumerate(loader):
            if index >= args.frames:
                break
            events, counts, depth, focal, _ = process_batch(batch, args, device)
            with torch.no_grad():
                prediction = predict_full_frame(
                    model, events, counts, *depth.shape[1:]
                ).float()

            target, valid = mode.target(depth, focal)
            scores = mode.metrics(prediction, target, valid, focal)
            truth_m = mode.to_metres(target, focal)
            pred_m = mode.to_metres(prediction, focal)
            depths = truth_m[0][valid[0]]
            ratio = (pred_m[0][valid[0]] / depths).median().item()
            meta = batch.meta
            lines = [
                f"sample {index}   episode {meta.episode_id[0]} step {meta.step_idx[0]}"
                f"   scene {Path(str(meta.scene[0])).stem.split('.')[0]}"
                f"   window {meta.window_us[0] / 1000:.0f} ms"
                f"   {int(counts[0]):,} events",
                f"abs_rel {scores['abs_rel']:.3f}   d1 {scores['d1']:.1f}"
                f"   rmse {scores['rmse']:.2f} m   silog {scores['silog']:.1f}"
                f"   gt depth {depths.min():.1f}-{depths.max():.1f} m"
                f"   valid {100 * valid[0].float().mean():.0f}%"
                + (
                    f"   pred/gt {ratio:.2f}   colour {near:g}-{far:g} m"
                    if metric
                    else ""
                ),
            ]

            shown = (
                (truth_m[0], pred_m[0])
                if metric
                else (1.0 / truth_m[0], 1.0 / pred_m[0])
            )
            panel = panels(events, counts, *shown, valid[0], to_image, lines)
            print(lines[0])
            if writer is not None:
                writer.write(panel)
            else:
                cv2.imshow("events | depth | prediction", panel)
                if (cv2.waitKey(0) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        loader.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
