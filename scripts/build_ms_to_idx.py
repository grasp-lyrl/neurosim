"""Build a per-millisecond event index lookup for a Neurosim H5 event group.

Creates a dataset named "ms_to_idx" inside the selected event group where:
  ms_to_idx[m] = first event index i such that t[i] >= m * 1000 (us)

So ms_to_idx[10] gives the first event index at or after 10 ms.
"""

import argparse

import h5py
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add ms_to_idx lookup dataset to an event camera group in H5."
    )
    parser.add_argument("h5_path", help="Path to Neurosim H5 file")
    parser.add_argument(
        "--sensor",
        default="event_camera_1",
        help="Event camera group name (default: event_camera_1)",
    )
    parser.add_argument(
        "--chunk-events",
        type=int,
        default=5_000_000,
        help="How many timestamps to process per chunk (default: 5000000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ms_to_idx dataset if present.",
    )
    return parser.parse_args()


def build_ms_to_idx(t_ds: h5py.Dataset, chunk_events: int) -> np.ndarray:
    total_events = int(t_ds.shape[0])
    if total_events == 0:
        # Keep minimal valid lookup where ms_to_idx[0] == 0.
        return np.array([0], dtype=np.int64)

    t0 = int(t_ds[0])
    t_last = int(t_ds[-1])
    max_ms = int(t_last // 1000)

    ms_to_idx = np.empty(max_ms + 1, dtype=np.int64)

    next_ms = 0
    while next_ms <= max_ms and next_ms * 1000 <= t0:
        ms_to_idx[next_ms] = 0
        next_ms += 1

    start = 0
    with tqdm(total=total_events, desc="Building ms_to_idx", unit="ev") as pbar:
        while start < total_events and next_ms <= max_ms:
            end = min(start + chunk_events, total_events)
            t_chunk = np.asarray(t_ds[start:end])
            chunk_last_t = int(t_chunk[-1])

            while next_ms <= max_ms and next_ms * 1000 <= chunk_last_t:
                target_t = next_ms * 1000
                local_idx = int(np.searchsorted(t_chunk, target_t, side="left"))
                ms_to_idx[next_ms] = start + local_idx
                next_ms += 1

            pbar.update(end - start)
            start = end

    if next_ms <= max_ms:
        ms_to_idx[next_ms:] = total_events

    return ms_to_idx


def main() -> None:
    args = parse_args()

    if args.chunk_events <= 0:
        raise ValueError("--chunk-events must be > 0")

    with h5py.File(args.h5_path, "a") as f:
        if args.sensor not in f:
            raise RuntimeError(f"Sensor group '{args.sensor}' not found in H5 file.")

        grp = f[args.sensor]
        if not isinstance(grp, h5py.Group):
            raise RuntimeError(f"'{args.sensor}' exists but is not an HDF5 group.")
        if not {"x", "y", "t", "p"}.issubset(set(grp.keys())):
            raise RuntimeError(
                f"Group '{args.sensor}' does not contain expected datasets x, y, t, p."
            )

        if "ms_to_idx" in grp:
            if not args.overwrite:
                raise RuntimeError(
                    "ms_to_idx already exists. Re-run with --overwrite to rebuild."
                )
            del grp["ms_to_idx"]

        t_ds = grp["t"]
        ms_to_idx = build_ms_to_idx(t_ds, args.chunk_events)

        grp.create_dataset(
            "ms_to_idx",
            data=ms_to_idx,
            dtype=np.int64,
            compression="lzf",
            chunks=(min(1_000_000, ms_to_idx.shape[0]),),
        )

        print(f"Wrote {args.sensor}/ms_to_idx with length {ms_to_idx.shape[0]}")


if __name__ == "__main__":
    main()
