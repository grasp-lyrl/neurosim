"""Visualize event-camera data from a Neurosim H5 log.

The script loads an event group (x, y, t, p), bins events into fixed-duration
frames (default: 20 ms), and renders polarity frames where positive events are
red and negative events are blue.
"""

import argparse
from dataclasses import dataclass

import h5py
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class EventGroup:
    name: str
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize H5 event data as polarity frames."
    )
    parser.add_argument("h5_path", help="Path to Neurosim H5 log file")
    parser.add_argument(
        "--sensor",
        default="event_camera_1",
        help="Event sensor group name (default: event_camera_1).",
    )
    parser.add_argument(
        "--bin-ms",
        type=int,
        default=20,
        help="Frame bin duration in milliseconds (default: 20).",
    )
    parser.add_argument(
        "--chunk-events",
        type=int,
        default=2_000_000,
        help="How many events to read per chunk (default: 2000000).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of frames to display (default: 300). Use -1 for all.",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for display timing (default: 1.0).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without interactive display (useful for testing).",
    )
    parser.add_argument(
        "--save-first-frame",
        default=None,
        help="Optional path to save the first rendered frame image.",
    )
    return parser.parse_args()


def add_events_to_frame(
    frame: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
) -> None:
    if x.size == 0:
        return

    xv = x.astype(np.int64, copy=False)
    yv = y.astype(np.int64, copy=False)
    pv = p.astype(np.int8, copy=False)
    pos_mask = pv > 0
    neg_mask = ~pos_mask

    # Binary polarity frame: red for positive events, blue for negative events.
    if np.any(pos_mask):
        frame[yv[pos_mask], xv[pos_mask], 0] = 255
    if np.any(neg_mask):
        frame[yv[neg_mask], xv[neg_mask], 2] = 255


def main() -> None:
    args = parse_args()

    if args.bin_ms <= 0:
        raise ValueError("--bin-ms must be > 0")
    if args.chunk_events <= 0:
        raise ValueError("--chunk-events must be > 0")
    if args.playback_speed <= 0:
        raise ValueError("--playback-speed must be > 0")

    with h5py.File(args.h5_path, "r") as f:
        if args.sensor not in f:
            raise RuntimeError(f"Sensor group '{args.sensor}' not found in H5 file.")

        grp = f[args.sensor]
        if not isinstance(grp, h5py.Group):
            raise RuntimeError(f"'{args.sensor}' exists but is not an HDF5 group.")
        if not {"x", "y", "t", "p"}.issubset(set(grp.keys())):
            raise RuntimeError(
                f"Group '{args.sensor}' does not contain expected datasets x, y, t, p."
            )

        group_info = EventGroup(
            name=args.sensor,
            width=int(grp.attrs.get("width", int(np.max(grp["x"]) + 1))),
            height=int(grp.attrs.get("height", int(np.max(grp["y"]) + 1))),
        )
        x_ds = grp["x"]
        y_ds = grp["y"]
        p_ds = grp["p"]
        if "ms_to_idx" not in grp:
            raise RuntimeError(
                "Missing ms_to_idx lookup. Run: "
                "python scripts/build_ms_to_idx.py <h5_path> --sensor <group_name>"
            )
        ms_to_idx_ds = grp["ms_to_idx"]

        bin_width_ms = args.bin_ms

        print(f"Using sensor: {group_info.name}")
        print(f"Resolution: {group_info.width}x{group_info.height}")
        print(f"Bin size: {args.bin_ms:.3f} ms")
        print(f"Total events: {x_ds.shape[0]}")
        print(f"ms_to_idx length: {ms_to_idx_ds.shape[0]}")

        if not args.headless:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
            image_artist = ax.imshow(
                np.zeros((group_info.height, group_info.width, 3), dtype=np.uint8),
                interpolation="nearest",
            )
            ax.set_title("Event polarity frame")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.tight_layout()

        frame = np.zeros((group_info.height, group_info.width, 3), dtype=np.uint8)
        total_events = x_ds.shape[0]
        frame_idx = 0
        frame_start_ms = 0
        saved_first = False

        def ms_lookup(q_ms: int) -> int:
            if q_ms < 0:
                return 0
            if q_ms >= ms_to_idx_ds.shape[0]:
                return total_events
            return int(ms_to_idx_ds[q_ms])

        while True:
            start_idx = ms_lookup(frame_start_ms)
            end_idx = ms_lookup(frame_start_ms + bin_width_ms)
            if start_idx >= total_events and end_idx >= total_events:
                break

            if end_idx > start_idx:
                add_events_to_frame(
                    frame,
                    np.asarray(x_ds[start_idx:end_idx]),
                    np.asarray(y_ds[start_idx:end_idx]),
                    np.asarray(p_ds[start_idx:end_idx]),
                )

            rgb = frame
            if args.save_first_frame and not saved_first:
                plt.imsave(args.save_first_frame, rgb)
                saved_first = True

            if not args.headless:
                image_artist.set_data(rgb)
                ax.set_title(
                    f"{group_info.name} | frame={frame_idx} | "
                    f"t=[{frame_start_ms}, {frame_start_ms + bin_width_ms}) ms"
                )
                fig.canvas.draw_idle()
                plt.pause((args.bin_ms / 1000.0) / args.playback_speed)

            frame_idx += 1
            if args.max_frames >= 0 and frame_idx >= args.max_frames:
                print(f"Reached max frames: {args.max_frames}")
                if not args.headless:
                    plt.ioff()
                    plt.show()
                return

            frame.fill(0)
            frame_start_ms += bin_width_ms
            if frame_idx > 0 and frame_idx % 50 == 0:
                print(f"Rendered {frame_idx} frames...")

        print(f"Completed. Total rendered frames: {frame_idx}")
        if not args.headless:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()
