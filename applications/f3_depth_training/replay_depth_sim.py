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
from .utils import ev_to_frames_with_polarity, eval_relative_depth, get_disparity_image

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


def panels(events, counts, disparity, prediction, valid, cmap, lines):
    """Stats strip over events | ground-truth disparity | prediction, in BGR."""
    height, width = disparity.shape
    frames = ev_to_frames_with_polarity(events, counts, width, height).cpu().numpy()
    images = np.hstack(
        [
            frames[0],
            get_disparity_image(disparity, valid, cmap),
            get_disparity_image(
                prediction, torch.ones_like(prediction, dtype=torch.bool), cmap
            ),
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

    model = load_model(args.run, args.ckpt, device)
    width, height, window_ms = model.eventff.frame_sizes
    args.event_norm = (width, height, window_ms * 1000)
    cmap = colormaps["magma"]

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
            args.max_disparity,
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
            events, counts, disparity, _ = process_batch(batch, args, device)
            with torch.no_grad():
                prediction = predict_full_frame(
                    model, events, counts, *disparity.shape[1:]
                ).float()

            valid = disparity < args.max_disparity
            scores = eval_relative_depth(
                prediction, disparity, valid, args.min_disparity
            )
            depths = 1.0 / disparity[0][valid[0]]
            meta = batch.meta
            lines = [
                f"sample {index}   episode {meta.episode_id[0]} step {meta.step_idx[0]}"
                f"   scene {Path(str(meta.scene[0])).stem.split('.')[0]}"
                f"   window {meta.window_us[0] / 1000:.0f} ms"
                f"   {int(counts[0]):,} events",
                f"abs_rel {scores['abs_rel']:.3f}   d1 {scores['d1']:.1f}"
                f"   rmse {scores['rmse']:.2f} m   silog {scores['silog']:.1f}"
                f"   gt depth {depths.min():.1f}-{depths.max():.1f} m"
                f"   valid {100 * valid[0].float().mean():.0f}%",
            ]

            panel = panels(
                events, counts, disparity[0], prediction[0], valid[0], cmap, lines
            )
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
