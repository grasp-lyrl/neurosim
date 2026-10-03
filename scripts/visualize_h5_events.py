"""Play a neurosim H5 recording as rgb | depth | events, live or into an mp4.

python scripts/visualize_h5_events.py outputs/flight.h5 [--out outputs/flight.mp4]
"""

import argparse
import itertools
from collections.abc import Iterator

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from build_ms_to_idx import write_ms_to_idx

INVALID_RGB = (255, 214, 0)

Frames = Iterator[tuple[int, np.ndarray]]  # (window start ms, panels side by side)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("h5_path")
    parser.add_argument("--sensor", default="event_camera_1")
    parser.add_argument("--color", default="color_camera_1")
    parser.add_argument("--depth", default="depth_camera_1")
    parser.add_argument("--bin-ms", type=int, default=20)
    parser.add_argument(
        "--speed", type=float, default=1.0, help="playback speed, 1 = sim time"
    )
    parser.add_argument("--out", help="mp4 to write instead of showing a window")
    return parser.parse_args()


def ensure_ms_to_idx(h5_path: str, sensor: str) -> None:
    with h5py.File(h5_path, "r") as f:
        if "ms_to_idx" in f[sensor]:
            return
    with h5py.File(h5_path, "a") as f:
        write_ms_to_idx(f[sensor])


def event_rgb(
    x: np.ndarray, y: np.ndarray, p: np.ndarray, width: int, height: int
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    pos = p > 0
    frame[y[pos], x[pos], 0] = 255
    frame[y[~pos], x[~pos], 2] = 255
    return frame


def depth_rgb(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    lo, hi = np.percentile(depth[valid], [1, 99]) if valid.any() else (0.0, 1.0)
    grey = (np.clip((depth - lo) / max(hi - lo, 1e-3), 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(grey[..., None], 3, axis=2)
    rgb[~valid] = INVALID_RGB
    return rgb


def render(f: h5py.File, sensor: str, color: str, depth: str, bin_ms: int) -> Frames:
    events = f[sensor]
    x, y, p = events["x"], events["y"], events["p"]
    width, height = events.attrs["width"], events.attrs["height"]
    bounds = np.append(events["ms_to_idx"][:], len(x))
    cameras = [
        (f[name]["data"], f[name]["sim_time"][:], draw)
        for name, draw in ((color, np.asarray), (depth, depth_rgb))
        if name in f
    ]
    for ms in range(0, len(bounds) - 1, bin_ms):
        i0, i1 = bounds[ms], bounds[min(ms + bin_ms, len(bounds) - 1)]
        mid = (ms + bin_ms / 2) / 1000
        panels = [
            draw(data[np.abs(times - mid).argmin()]) for data, times, draw in cameras
        ]
        panels.append(event_rgb(x[i0:i1], y[i0:i1], p[i0:i1], width, height))
        yield ms, np.concatenate(panels, axis=1)


def show(frames: Frames, pause: float) -> None:
    ms, frame = next(frames)
    height, width = frame.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 120, height / 120), layout="tight")
    ax.axis("off")
    image = ax.imshow(frame, interpolation="nearest")
    for ms, frame in itertools.chain([(ms, frame)], frames):
        if not plt.fignum_exists(fig.number):
            return
        image.set_data(frame)
        ax.set_title(f"{ms} ms")
        plt.pause(pause)
    plt.show()


def write_video(frames: Frames, out: str, fps: float) -> None:
    with imageio.get_writer(out, fps=fps, macro_block_size=1) as writer:
        for _, frame in frames:
            writer.append_data(frame)


def main() -> None:
    args = parse_args()
    ensure_ms_to_idx(args.h5_path, args.sensor)
    with h5py.File(args.h5_path, "r") as f:
        frames = render(f, args.sensor, args.color, args.depth, args.bin_ms)
        if args.out:
            write_video(frames, args.out, fps=args.speed * 1000 / args.bin_ms)
            print(f"wrote {args.out}")
        else:
            show(frames, pause=args.bin_ms / 1000 / args.speed)


if __name__ == "__main__":
    main()
