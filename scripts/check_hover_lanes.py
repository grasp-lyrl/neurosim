"""Show what hover trajectories and the lane batcher actually produce.

Two views, both text so it works over ssh:

* the commanded speed of one hover trajectory, straight off the navmesh (CPU only);
* per-lane event counts from a live ``OnlineDataLoader`` running ``batcher: lane``,
  where a hover shows up as a run of empty windows.

    python scripts/check_hover_lanes.py --conf configs/online_data_hm3d_2gpu.yaml \
        --hovers 2 --hover-s 2.0 --sim-time 12 --lanes 2 --batches 400
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import yaml

from neurosim.core.trajectory.habitat_trajs import generate_interesting_traj
from neurosim.online_data import OnlineDataLoader

BLOCKS = " ▁▂▃▄▅▆▇█"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--conf", required=True, help="OnlineDataLoader YAML config")
    p.add_argument("--hovers", type=int, default=2, help="pauses between flight legs")
    p.add_argument("--hover-s", type=float, default=2.0, help="seconds held per pause")
    p.add_argument("--sim-time", type=float, default=12.0, help="episode length (s)")
    p.add_argument("--lanes", type=int, default=2, help="producers, one per batch row")
    p.add_argument("--batches", type=int, default=400, help="0 skips the loader run")
    p.add_argument(
        "--seed", type=int, default=0, help="trajectory seed for the profile"
    )
    p.add_argument("--width", type=int, default=100, help="sparkline columns")
    return p.parse_args()


def sparkline(values: np.ndarray, width: int) -> str:
    """Bucket `values` into `width` columns; empty buckets render as a blank."""
    if not len(values):
        return ""
    buckets = np.array_split(values, min(width, len(values)))
    means = np.array([b.mean() for b in buckets])
    peak = means.max()
    if peak <= 0:
        return " " * len(means)
    levels = np.ceil(means / peak * (len(BLOCKS) - 1)).astype(int)
    return "".join(BLOCKS[i] for i in levels)


def show_trajectory(args: argparse.Namespace, cfg: dict) -> None:
    """Commanded speed of one hover trajectory, from a navmesh with no renderer."""
    import habitat_sim

    # The config's scene is usually blank and bootstrapped from the DR glob per episode.
    scene = cfg.get("visual_backend", {}).get("scene") or ""
    if not scene:
        pattern = cfg["online_data"].get("randomization", {}).get("scenes_glob", "")
        scene = next(iter(sorted(glob.glob(pattern))), "")
    navmesh = Path(scene.replace(".glb", ".navmesh"))
    if not scene or not navmesh.exists():
        print(
            f"no navmesh beside {scene or 'any configured scene'}; skipping profile\n"
        )
        return

    pathfinder = habitat_sim.PathFinder()
    pathfinder.load_nav_mesh(str(navmesh))

    traj_cfg = cfg.get("trajectory", {})
    traj = generate_interesting_traj(
        pathfinder,
        seed=args.seed,
        target_length=traj_cfg.get("target_length", 30.0),
        min_waypoint_distance=traj_cfg.get("min_waypoint_distance", 0.5),
        max_waypoints=traj_cfg.get("max_waypoints", 100),
        v_avg=traj_cfg.get("v_avg", 1.0),
        episode_duration=args.sim_time,
        hovers=args.hovers,
        hover_s=args.hover_s,
    )

    duration = float(traj.t_keyframes[-1])
    ts = np.arange(0.0, duration, 0.02)
    speed = np.array([np.linalg.norm(traj.update(t)["x_dot"]) for t in ts])
    still = speed < 1e-6

    print(f"trajectory  {navmesh.parent.name}  seed {args.seed}")
    print(
        f"  {getattr(traj, 'segments', [traj]).__len__()} flight legs, "
        f"{args.hovers} hovers of {args.hover_s}s, duration {duration:.1f}s"
    )
    print(f"  speed |{sparkline(speed, args.width)}|  peak {speed.max():.2f} m/s")
    print(
        f"  still |{sparkline(still.astype(float), args.width)}|  {still.mean():.0%} of the episode\n"
    )


def show_lanes(args: argparse.Namespace, cfg: dict) -> None:
    """Per-lane event counts from a live loader, one batch row per producer."""
    loader = OnlineDataLoader.from_config(
        cfg,
        batch_size=args.lanes,
        num_producers=args.lanes,
        gpu_ids=[0],
        batcher="lane",
    )
    event_uuid = cfg["online_data"]["roles"]["stream"][0]

    counts = np.zeros((args.batches, args.lanes), dtype=np.int64)
    workers = np.zeros((args.batches, args.lanes), dtype=np.int64)
    episodes = np.zeros((args.batches, args.lanes), dtype=np.int64)
    try:
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            counts[i] = batch[event_uuid][0]
            workers[i] = batch.meta.worker_id
            episodes[i] = batch.meta.episode_id
            if i % 50 == 0:
                print(f"  ... {i}/{args.batches} batches", end="\r", flush=True)
    finally:
        loader.close()

    print(f"\nlane batcher  {args.batches} batches x {args.lanes} lanes")
    for lane in range(args.lanes):
        c = counts[:, lane]
        quiet = c < max(np.median(c) / 20, 1)
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, quiet, 0])))[::2]
        longest = int(runs.max()) if len(runs) else 0
        print(
            f"  lane {lane}  worker {set(workers[:, lane])}  "
            f"episodes {len(set(episodes[:, lane]))}"
        )
        print(f"    events |{sparkline(c.astype(float), args.width)}|")
        print(
            f"    min {c.min():,}  p5 {int(np.percentile(c, 5)):,}  "
            f"median {int(np.median(c)):,}  max {c.max():,}"
        )
        print(
            f"    quiet (< median/20) {quiet.sum()}/{len(c)}, longest run "
            f"{longest} batches = {longest / 50:.1f}s at 50 Hz"
        )

    same = (workers == workers[0]).all()
    print(f"\n  every row stayed on its own producer: {same}")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.conf))
    cfg.setdefault("simulator", {})["sim_time"] = args.sim_time
    cfg.setdefault("trajectory", {}).update(hovers=args.hovers, hover_s=args.hover_s)

    show_trajectory(args, cfg)
    if args.batches:
        show_lanes(args, cfg)


if __name__ == "__main__":
    main()
