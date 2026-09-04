"""Live view of an ``OnlineDataLoader`` stream: one panel per sample in the batch,
grayscale depth with that sample's events composited on top. Producers use `spawn`,
so this lives behind a __main__ guard.

    python scripts/visualize_online_batches.py --conf configs/online_data_hm3d_2gpu.yaml
"""

import os
import argparse
import itertools
from pathlib import Path

import numpy as np
import yaml

from neurosim.online_data import OnlineDataLoader

POS_RGB = np.array([1.00, 0.27, 0.23], dtype=np.float32)
NEG_RGB = np.array([0.16, 0.55, 1.00], dtype=np.float32)
DEPTH_DIM = 0.35


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live depth+events view of loader batches.")
    p.add_argument("--conf", required=True, help="OnlineDataLoader YAML config")
    p.add_argument("--batch-size", type=int, default=8, help="Samples = panels")
    p.add_argument("--num-producers", type=int, default=2)
    p.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    p.add_argument("--sim-time", type=float, default=None, help="Episode length (s)")
    p.add_argument("--batches", type=int, default=0, help="0 = until window closed")
    p.add_argument("--save-dir", default=None, help="Also write batch_NNN.png here")
    return p.parse_args()


def depth_gray(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Percentile-stretched grayscale depth (invalid 0 m pixels stay dark), with its range."""
    valid = depth > 0
    if not valid.any():  # camera saw no geometry at all, e.g. facing open sky
        return np.full(depth.shape, 0.12, np.float32), 0.0, 0.0
    lo, hi = np.percentile(depth[valid], [1, 99])
    gray = np.clip((depth - lo) / max(hi - lo, 1e-3), 0.0, 1.0).astype(np.float32)
    gray[~valid] = 0.12
    return gray, float(lo), float(hi)


def event_rgb(
    ev: np.ndarray, height: int, width: int, gamma: float = 0.8
) -> np.ndarray:
    """Per-pixel event counts as RGB: warm positive, cool negative, magenta where both fired."""
    # Each polarity gets its own channel. A signed sum would show only the net polarity,
    # and over a 50 ms window most pixels fire both ways — a mildly imbalanced sample
    # then renders as a solid wall of one color.
    if ev.shape[0] == 0:  # the assembler emits a sample even when its window is empty
        return np.zeros((height, width, 3), np.float32)
    flat = ev[:, 1].astype(np.int64) * width + ev[:, 0].astype(np.int64)
    is_pos = ev[:, 3] > 0
    shape = (height, width)
    pos = np.bincount(flat, weights=is_pos, minlength=height * width).reshape(shape)
    neg = np.bincount(flat, weights=~is_pos, minlength=height * width).reshape(shape)
    peak = np.maximum(pos, neg)
    scale = max(1.0, float(np.percentile(peak[peak > 0], 99.5)))
    lit_pos = (np.clip(pos / scale, 0.0, 1.0) ** gamma).astype(np.float32)
    lit_neg = (np.clip(neg / scale, 0.0, 1.0) ** gamma).astype(np.float32)
    return np.clip(
        POS_RGB * lit_pos[..., None] + NEG_RGB * lit_neg[..., None], 0.0, 1.0
    )


def panel(depth: np.ndarray, ev: np.ndarray) -> tuple[np.ndarray, float, float]:
    """One sample composited: dimmed grayscale depth under its event frame."""
    gray, lo, hi = depth_gray(depth)
    base = np.repeat(gray[:, :, None], 3, axis=2) * DEPTH_DIM
    events = event_rgb(ev, *depth.shape)
    alpha = events.max(axis=2, keepdims=True)
    return base * (1.0 - alpha) + events * alpha, lo, hi


def scene_name(path: str) -> str:
    """`.../kfPV7w3FaU5.basis.glb` -> `kfPV7w3FaU5`."""
    return Path(path).stem.split(".")[0]


def build_loader(args: argparse.Namespace) -> tuple[OnlineDataLoader, str, str, dict]:
    """Loader plus the anchor/stream UUIDs and the spec_id -> gpu_id map."""
    cfg = yaml.safe_load(open(args.conf))
    if args.sim_time is not None:
        cfg.setdefault("simulator", {})["sim_time"] = args.sim_time
    overrides = {"batch_size": args.batch_size, "num_producers": args.num_producers}
    if args.gpu_ids is not None:
        overrides["gpu_ids"] = args.gpu_ids

    loader = OnlineDataLoader.from_config(cfg, **overrides)
    roles = cfg["online_data"]["roles"]
    spec_gpu = {s.spec_id: s.gpu_id for s in loader._specs}
    return loader, roles["anchor"][0], roles["stream"][0], spec_gpu


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    interactive = bool(os.environ.get("DISPLAY"))
    if not interactive:
        if save_dir is None:
            raise SystemExit("no DISPLAY: pass --save-dir to write PNGs instead")
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loader, depth_uuid, event_uuid, spec_gpu = build_loader(args)

    cols = min(4, args.batch_size)
    rows = -(-args.batch_size // cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.8 * cols, 3.1 * rows),
        facecolor="#111",
        layout="constrained",
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[args.batch_size :]:
        ax.remove()
    axes = axes[: args.batch_size]

    blank = np.zeros((2, 2, 3), np.float32)
    images = [ax.imshow(blank, interpolation="nearest") for ax in axes]
    titles = [ax.set_title("", color="w", fontsize=8) for ax in axes]
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    suptitle = fig.suptitle("waiting for the first batch...", color="w")

    closed = False

    def on_close(_event):
        nonlocal closed
        closed = True

    if interactive:
        fig.canvas.mpl_connect("close_event", on_close)
        plt.show(block=False)

    stream = loader if args.batches <= 0 else itertools.islice(loader, args.batches)
    try:
        for i, batch in enumerate(stream):
            depth = batch[depth_uuid]
            counts, events = batch[event_uuid]
            meta = batch.meta
            offsets = np.concatenate([[0], np.cumsum(counts)])

            for j, (image, title) in enumerate(zip(images, titles)):
                ev = events[offsets[j] : offsets[j + 1]]
                rgb, lo, hi = panel(depth[j], ev)
                image.set_data(rgb)
                image.set_extent((0, rgb.shape[1], rgb.shape[0], 0))
                title.set_text(
                    f"[{j}] {scene_name(meta.scene[j])} · gpu{spec_gpu[int(meta.spec_id[j])]} "
                    f"· step {int(meta.step_idx[j])}\n"
                    f"{int(counts[j]):,} ev · {float((ev[:, 3] > 0).mean()) * 100:.0f}% pos "
                    f"· {lo:.1f}–{hi:.1f} m"
                )
            suptitle.set_text(
                f"batch {i} · {int(counts.sum()):,} events "
                f"· {int(meta.window_us[0]) / 1000:.0f} ms window"
            )

            if save_dir:
                fig.savefig(save_dir / f"batch_{i:03d}.png", dpi=80, facecolor="#111")
            if interactive:
                fig.canvas.draw_idle()
                plt.pause(0.001)
                if closed:
                    break
    finally:
        loader.close()


if __name__ == "__main__":
    main()
