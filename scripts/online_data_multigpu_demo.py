"""Multi-GPU OnlineDataLoader smoke/verification demo.

Builds an OnlineDataLoader from a config (e.g. configs/online_data_hm3d_2gpu.yaml),
pulls N batches, and prints an elaborate per-batch + per-producer + aggregate report
so you can verify at a glance that, at scale:
  * producers are spread across the requested GPUs and all contribute,
  * scenes/sensors are domain-randomized (batches mix many scenes),
  * batch shapes / event counts / metadata look sane,
  * throughput (samples/s, events/s) is reasonable.

Run (in the neurosim conda env, from the repo root)::

    python scripts/online_data_multigpu_demo.py --conf configs/online_data_hm3d_2gpu.yaml --batches 20

Producers use `spawn`, so this lives behind a __main__ guard.
"""

import time
import argparse
import itertools
from collections import Counter
from pathlib import Path

import yaml

from neurosim.online_data import OnlineDataLoader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-GPU OnlineDataLoader demo/verify.")
    p.add_argument("--conf", required=True, help="OnlineDataLoader YAML config")
    p.add_argument("--batches", type=int, default=20, help="How many batches to pull")
    p.add_argument(
        "--num-producers", type=int, default=None, help="Override num_producers"
    )
    p.add_argument(
        "--gpu-ids", type=int, nargs="+", default=None, help="Override gpu_ids"
    )
    p.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    p.add_argument(
        "--sim-time", type=float, default=None, help="Override episode sim_time (s)"
    )
    return p.parse_args()


def _scene_name(path: str) -> str:
    return Path(path).stem.split(".")[0]  # ".../kfPV7w3FaU5.basis.glb" -> "kfPV7w3FaU5"


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.conf))
    if args.sim_time is not None:
        cfg.setdefault("simulator", {})["sim_time"] = args.sim_time

    overrides = {}
    if args.num_producers is not None:
        overrides["num_producers"] = args.num_producers
    if args.gpu_ids is not None:
        overrides["gpu_ids"] = args.gpu_ids
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    loader = OnlineDataLoader.from_config(cfg, **overrides)

    # Map spec_id -> gpu_id so we can attribute each batch row to a GPU.
    spec_gpu = {s.spec_id: s.gpu_id for s in loader._specs}
    od = cfg["online_data"]
    event_uuid = od["roles"]["stream"][0]
    depth_uuid = od["roles"]["anchor"][0]
    n_scenes_pool = "(from scenes_glob; see log above)"

    print("=" * 78)
    print("OnlineDataLoader multi-GPU demo")
    print(f"  config            : {args.conf}")
    print(f"  producers         : {len(loader._specs)}  (spec_id -> gpu: {spec_gpu})")
    print(f"  gpus in use       : {sorted(set(spec_gpu.values()))}")
    print(f"  batch_size        : {loader.batch_size}")
    print(f"  anchor / stream   : {depth_uuid} / {event_uuid}")
    print(f"  scene pool        : {n_scenes_pool}")
    print(f"  pulling batches   : {args.batches}")
    print("=" * 78, flush=True)

    # Aggregates
    scenes_seen: Counter = Counter()
    specs_seen: Counter = Counter()
    gpus_seen: Counter = Counter()
    episodes_seen: set = set()
    total_samples = 0
    total_events = 0
    first_batch_t = None
    last = time.monotonic()
    t0 = time.monotonic()

    try:
        for i, batch in enumerate(itertools.islice(loader, args.batches)):
            now = time.monotonic()
            dt = now - last
            last = now
            if first_batch_t is None:
                first_batch_t = (
                    now  # exclude warm-up (first scene load) from throughput
                )

            depth = batch[depth_uuid]
            counts, events = batch[event_uuid]
            m = batch.meta
            B = depth.shape[0]

            scenes = [_scene_name(s) for s in m.scene]
            spec_ids = m.spec_id.tolist()
            gpus = [spec_gpu.get(s, -1) for s in spec_ids]

            scenes_seen.update(scenes)
            specs_seen.update(spec_ids)
            gpus_seen.update(gpus)
            episodes_seen.update(m.episode_id.tolist())
            total_samples += B
            total_events += int(events.shape[0])

            uniq_scenes = sorted(set(scenes))
            print(
                f"batch {i:>3d} | +{dt:4.1f}s | depth {tuple(depth.shape)} {depth.dtype} | "
                f"events {events.shape[0]:>9d} (avg {events.shape[0] // B:>7d}/sample) | "
                f"gpus {dict(Counter(gpus))} | specs {sorted(set(spec_ids))} | "
                f"{len(uniq_scenes)} scene(s): {uniq_scenes[:4]}{'…' if len(uniq_scenes) > 4 else ''} | "
                f"steps[{int(m.step_idx.min())}..{int(m.step_idx.max())}] "
                f"first/last={int(m.is_first.sum())}/{int(m.is_last.sum())}",
                flush=True,
            )
    finally:
        loader.close()

    elapsed = time.monotonic() - t0
    warm_elapsed = (time.monotonic() - first_batch_t) if first_batch_t else 0.0
    print("=" * 78)
    print("SUMMARY")
    print(
        f"  batches pulled        : {min(args.batches, total_samples // max(1, loader.batch_size))}"
    )
    print(f"  total samples         : {total_samples}")
    print(f"  total events          : {total_events:,}")
    print(
        f"  wall time             : {elapsed:.1f}s (excl warm-up: {warm_elapsed:.1f}s)"
    )
    if warm_elapsed > 0:
        print(
            f"  throughput            : {total_samples / max(warm_elapsed, 1e-9):.1f} samples/s, "
            f"{total_events / max(warm_elapsed, 1e-9):,.0f} events/s"
        )
    print(
        f"  distinct scenes seen  : {len(scenes_seen)}  (top: {scenes_seen.most_common(5)})"
    )
    print(f"  producers represented : {sorted(specs_seen)} of {sorted(spec_gpu)}")
    print(f"  per-GPU sample counts : {dict(sorted(gpus_seen.items()))}")
    print(f"  distinct episodes     : {len(episodes_seen)}")

    # Sanity checks (verification aids)
    gpus_requested = sorted(set(spec_gpu.values()))
    checks = [
        ("all producers contributed", set(specs_seen) == set(spec_gpu)),
        ("both/all GPUs produced", set(gpus_seen) >= set(gpus_requested)),
        ("scene diversity (>1 scene)", len(scenes_seen) > 1),
        (
            "got all requested batches",
            total_samples >= args.batches * loader.batch_size,
        ),
    ]
    print("-" * 78)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'WARN'}] {name}")
    print("=" * 78, flush=True)


if __name__ == "__main__":  # producers use spawn
    main()
