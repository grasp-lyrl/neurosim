"""Step a trained depth model over recorded events, one window at a time.

Press ``n``/Enter for the next window and ``q`` to quit, or pass ``--video`` to write
an mp4 instead (much faster than forwarding a window over ssh). Events are
centre-cropped to the model's frame, never resampled: rescaling event coordinates
makes some target pixels collect more source pixels than others, and the feature
field reads that fixed density pattern as structure.
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from matplotlib import colormaps

from .nets import EventFFDepthAnythingV2, load_depth_weights
from .utils import get_disparity_image

# dataset -> (event group, ms->index dataset, sensor width, height)
LAYOUTS = {
    "m3ed": ("prophesee/left", "prophesee/left/ms_map_idx", 1280, 720),
    "tumvie": ("events", "ms_to_idx", 1280, 720),
}


def load_model(run: Path, ckpt: str, device):
    """Rebuild the model from its saved config and load weights."""
    model = EventFFDepthAnythingV2.init_from_config(f"{run}/models/depth_config.yml")
    # best/last also carry optimizer+scheduler (3x the model); mmap skips what we drop.
    state = torch.load(
        f"{run}/models/{ckpt}", map_location="cpu", weights_only=False, mmap=True
    )
    load_depth_weights(model, state.get("model", state))
    return model.to(device).eval()


def read_window(grp, ms_idx, t0_ms: int, window_ms: int, crop):
    """Events in ``[t0, t0 + window)`` inside the crop, as ``[x, y, t_rel, p]`` floats."""
    x0, y0, w, h = crop
    i0, i1 = int(ms_idx[t0_ms]), int(ms_idx[t0_ms + window_ms])
    x = grp["x"][i0:i1].astype(np.int32)
    y = grp["y"][i0:i1].astype(np.int32)
    inside = (x >= x0) & (x < x0 + w) & (y >= y0) & (y < y0 + h)

    t_anchor = (t0_ms + window_ms) * 1000
    events = np.empty((int(inside.sum()), 4), np.float32)
    events[:, 0] = (x[inside] - x0) / w
    events[:, 1] = (y[inside] - y0) / h
    events[:, 2] = (t_anchor - grp["t"][i0:i1][inside]) / (window_ms * 1000)
    events[:, 3] = grp["p"][i0:i1][inside]
    return events


def event_frame(events, w: int, h: int):
    """Polarity frame, positive red and negative blue, in BGR."""
    frame = np.zeros((h, w, 3), np.uint8)
    x = np.rint(events[:, 0] * w).astype(np.int32).clip(0, w - 1)
    y = np.rint(events[:, 1] * h).astype(np.int32).clip(0, h - 1)
    pos = events[:, 3] > 0
    frame[y[pos], x[pos], 2] = 255
    frame[y[~pos], x[~pos], 0] = 255
    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay recorded events through a trained depth model, window by window."
    )
    parser.add_argument(
        "--run", required=True, help="Training output dir (has models/)"
    )
    parser.add_argument("--h5", required=True, help="Event sequence HDF5")
    parser.add_argument("--dataset", default="m3ed", choices=sorted(LAYOUTS))
    parser.add_argument("--ckpt", default="best.pth", help="Checkpoint under models/")
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--video", help="Write an mp4 instead of opening a window")
    parser.add_argument("--frames", type=int, default=200, help="Frames for --video")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    model = load_model(Path(args.run), args.ckpt, device)

    model_w, model_h, model_t = model.eventff.frame_sizes
    window_ms = model_t
    group, ms_key, src_w, src_h = LAYOUTS[args.dataset]
    crop = ((src_w - model_w) // 2, (src_h - model_h) // 2, model_w, model_h)

    # Whole frame: infer_image resizes the short edge to dav2's size and keeps aspect,
    # so unlike training there is no square crop and the prediction covers the input.
    cparams = torch.tensor([0, 0, model_h, model_w], dtype=torch.int32, device=device)
    cmap = colormaps["magma"]

    with h5py.File(args.h5, "r") as f:
        grp, ms_idx = f[group], f[ms_key]
        print(f"{args.h5}: {grp['x'].shape[0]:,} events, {len(ms_idx)} ms")
        print(
            f"crop x0={crop[0]} y0={crop[1]} {model_w}x{model_h}, window {window_ms} ms"
        )
        writer = None
        if args.video:
            writer = cv2.VideoWriter(
                args.video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                1000 / window_ms,  # windows are contiguous, so this plays at real speed
                (2 * model_w, model_h),
            )
            print(f"writing {args.frames} frames to {args.video}")
        else:
            print("n/Enter: next window   q: quit")

        end_ms = args.start_ms + args.frames * window_ms if writer else len(ms_idx)
        t0_ms = args.start_ms
        while t0_ms + window_ms < min(end_ms, len(ms_idx)):
            events = read_window(grp, ms_idx, t0_ms, window_ms, crop)
            ff_events = torch.from_numpy(events).to(device)
            counts = torch.tensor([len(events)], dtype=torch.int32, device=device)

            disparity = model.infer_image(ff_events, counts, cparams)[0]

            # get_disparity_image returns matplotlib RGB; cv2 wants BGR.
            pred = cv2.cvtColor(
                get_disparity_image(
                    disparity, torch.ones_like(disparity, dtype=torch.bool), cmap
                ),
                cv2.COLOR_RGB2BGR,
            )
            panel = np.hstack([event_frame(events, model_w, model_h), pred])
            print(f"t={t0_ms:6d} ms  {len(events):8,} events")

            if writer is not None:
                writer.write(panel)
            else:
                cv2.imshow("events | disparity", panel)
                if (cv2.waitKey(0) & 0xFF) in (ord("q"), 27):
                    break
            t0_ms += window_ms

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
