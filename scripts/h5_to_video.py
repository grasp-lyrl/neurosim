"""Render a recorded episode H5 into an rgb | depth | events video.

WHY: comparing how a scene *looks* to a sensor (flat vs lit vs baked lighting)
needs the same trajectory replayed side by side, and the repo's existing replay
tools all need a trained depth checkpoint.

Events are splatted binary (red positive, blue negative, magenta both) so clips can
be compared by how much of the frame fires at all.

HOW (neurosim env, repo root):

    python scripts/h5_to_video.py outputs/ep.h5 --out /tmp/ep.mp4 --bin-ms 20
"""

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

POS_RGB = (1.00, 0.27, 0.23)
NEG_RGB = (0.16, 0.55, 1.00)
# Habitat patches the far plane to 0.0 m, so "hit nothing" (open sky, a missing
# ceiling, past zfar) is indistinguishable from a 0 m reading. Paint it yellow
# rather than black so it cannot be mistaken for near geometry.
INVALID_RGB = (255, 214, 0)


def event_rgb(x, y, p, width: int, height: int) -> np.ndarray:
    """Splat one time bin of events: positive red, negative blue, both magenta.

    Deliberately binary -- no count accumulation, percentile scaling or gamma, so
    two clips can be compared by how much of the frame lights up at all.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if len(x) == 0:
        return frame

    xi = x.astype(np.int64).clip(0, width - 1)
    yi = y.astype(np.int64).clip(0, height - 1)
    pos = p > 0
    frame[yi[pos], xi[pos], 0] = 255
    frame[yi[~pos], xi[~pos], 2] = 255
    return frame


def depth_rgb(depth: np.ndarray) -> np.ndarray:
    """Grey depth with invalid (0 m) pixels flagged in yellow."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid = depth > 0
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [1, 99])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-3), 0, 1)
        grey = (norm * 255).astype(np.uint8)
        out[valid] = np.stack([grey] * 3, axis=-1)[valid]
    out[~valid] = INVALID_RGB
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensor", default="event_camera_1")
    ap.add_argument("--color", default="color_camera_1")
    ap.add_argument("--depth", default="depth_camera_1")
    ap.add_argument("--bin-ms", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    with h5py.File(args.h5_path, "r") as f:
        ev = f[args.sensor]
        width = int(ev.attrs["width"])
        height = int(ev.attrs["height"])
        x = ev["x"][:]
        y = ev["y"][:]
        t = ev["t"][:]
        p = ev["p"][:]
        rgb = f[args.color]["data"][:] if args.color in f else None
        rgb_t = f[args.color]["sim_time"][:] if args.color in f else None
        dep = f[args.depth]["data"][:] if args.depth in f else None
        dep_t = f[args.depth]["sim_time"][:] if args.depth in f else None
        invalid_frac = float((dep <= 0).mean()) if dep is not None else float("nan")
    print(f"invalid depth pixels: {invalid_frac:.1%}")

    # Event timestamps are microseconds; bin them into fixed windows.
    t = t.astype(np.float64)
    t0, t1 = t[0], t[-1]
    bin_us = args.bin_ms * 1000.0
    edges = np.arange(t0, t1, bin_us)
    starts = np.searchsorted(t, edges)
    ends = np.searchsorted(t, edges + bin_us)

    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)
    try:
        for i, (s, e) in enumerate(zip(starts, ends)):
            ev_img = event_rgb(x[s:e], y[s:e], p[s:e], width, height)
            # nearest colour/depth frame in time (both run slower than events)
            when = (edges[i] + bin_us / 2 - t0) / 1e6
            panels = []
            if rgb is not None:
                j = int(np.argmin(np.abs(rgb_t - rgb_t[0] - when)))
                panels.append(np.asarray(rgb[j])[..., :3])
            if dep is not None:
                k = int(np.argmin(np.abs(dep_t - dep_t[0] - when)))
                panels.append(depth_rgb(dep[k]))
            panels.append(ev_img)
            writer.append_data(np.concatenate(panels, axis=1))
    finally:
        writer.close()
    print(
        f"wrote {args.out} ({len(starts)} frames, {width}x{height}) from {Path(args.h5_path).name}"
    )


main()
