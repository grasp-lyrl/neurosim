"""Step a trained depth model over simulated episodes, three panels per sample.

Same loader the trainer uses, so what you see is what it trains on. Press ``n``/Enter for
the next sample and ``q`` to quit, or pass ``--video`` to write an mp4 instead.
"""

import argparse

import cv2
import numpy as np
import torch
import yaml
from matplotlib import colormaps

from .data import build_online_loader, process_batch, usable_sample_filter
from .nets import EventFFDepthAnythingV2, load_depth_weights
from .train_depth_nonrec import predict_full_frame
from .utils import ev_to_frames_with_polarity, get_disparity_image


def load_model(run: str, ckpt: str, device):
    """Rebuild the model from its saved config and load weights."""
    model = EventFFDepthAnythingV2.init_from_config(f"{run}/models/depth_config.yml")
    state = torch.load(
        f"{run}/models/{ckpt}", map_location="cpu", weights_only=False, mmap=True
    )
    load_depth_weights(model, state.get("model", state))
    return model.to(device).eval()


def panels(events, counts, disparity, prediction, index, cmap):
    """Events | ground-truth disparity | prediction, side by side in BGR."""
    height, width = disparity.shape
    valid = disparity < disparity.max()
    frames = ev_to_frames_with_polarity(events, counts, width, height).cpu().numpy()
    images = [
        frames[index],
        get_disparity_image(disparity, valid, cmap),
        get_disparity_image(
            prediction, torch.ones_like(prediction, dtype=torch.bool), cmap
        ),
    ]
    return cv2.cvtColor(np.hstack(images), cv2.COLOR_RGB2BGR)


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
            (3 * width, height),
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

            panel = panels(events, counts, disparity[0], prediction[0], 0, cmap)
            print(f"sample {index:5d}  {int(counts[0]):9,} events")
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
