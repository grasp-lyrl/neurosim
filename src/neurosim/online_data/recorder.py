"""Offline H5 dataset recorder: domain-randomized episodes → H5 files.

The on-disk counterpart to :class:`~neurosim.online_data.loader.OnlineDataLoader`.
Instead of streaming time-aligned batches into a training loop, each worker runs a
:class:`~neurosim.sims.synchronous_simulator.randomized_simulator.RandomizedSimulator`
and writes **one canonical H5 file per episode** via the simulator's existing
``run(log_h5=...)`` path (the same layout ``test_sim.py`` produces). No bus, batcher,
assembler, or schema — just the simulator + ``H5Logger``.

Two functions:

* :func:`record_episodes` — one GPU, a loop over episode indices. Module-level and
  picklable, so it doubles as the ``spawn`` target.
* :func:`record_dataset` — fan ``num_episodes`` across ``num_workers`` processes on
  ``gpu_ids`` (cycled via ``explicit_gpu_map``), each a worker with its own
  randomized simulator and a distinct seed (``base_seed + i``), the same diversity
  convention as the producer pool.
"""

import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import yaml

from neurosim.online_data.sim_worker import build_randomized_sim

logger = logging.getLogger(__name__)


def record_episodes(
    base_settings: dict,
    *,
    out_dir,
    episodes,
    gpu_id: int = 0,
    seed: int = 0,
    randomization: dict | None = None,
    write_meta: bool = True,
) -> None:
    """Record one canonical H5 file per episode on a single GPU.

    Args:
        base_settings: Simulator settings dict (deep-copied per worker; never mutated).
        out_dir: Output directory; files are ``episode_{ep:06d}.h5`` (+ ``.meta.yaml``).
        episodes: Iterable of **global** episode indices (used as the filenames).
        gpu_id: GPU for this worker's visual backend.
        seed: Worker seed. Scene/sensor sampling uses ``default_rng(seed)``; the
            per-episode trajectory seed is derived deterministically inside
            ``RandomizedSimulator.randomize``.
        randomization: ``domain_randomization`` dict (scenes/sensors/trajectory/
            ``resample_every``). ``None`` => no randomization (fixed scene).
        write_meta: Write ``episode_{ep:06d}.meta.yaml`` with the sampled settings
            (provenance) right after ``randomize()``.

    Module-level + picklable args => used directly as the ``spawn`` target.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rsim = build_randomized_sim(base_settings, randomization, gpu_id, seed)
    scene_rng = np.random.default_rng(seed)
    try:
        for ep in episodes:
            rsim.randomize(scene_rng)
            stem = out_dir / f"episode_{int(ep):06d}"
            if write_meta:
                with open(f"{stem}.meta.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(rsim.last_sampled_settings, f, sort_keys=False)
            rsim.run(log_h5=f"{stem}.h5")
            logger.info(
                "episode %d -> %s.h5 (gpu=%d seed=%d)", ep, stem.name, gpu_id, seed
            )
    finally:
        rsim.close()


def _plan_shards(
    num_episodes: int, num_workers: int, gpu_ids: list[int] | None, base_seed: int
) -> list[dict]:
    """Split ``num_episodes`` into per-worker assignments (pure; no side effects).

    Returns one dict per worker: ``{"episodes": [...], "gpu_id": int, "seed": int}``.
    Episode indices are contiguous and partition ``range(num_episodes)`` exactly;
    empty shards (when ``num_workers > num_episodes``) are dropped; GPUs are cycled
    via ``explicit_gpu_map`` and seeds are ``base_seed + i``.
    """
    from neurosim.core.utils import explicit_gpu_map

    shards = [s for s in np.array_split(np.arange(num_episodes), num_workers) if len(s)]
    gpus = explicit_gpu_map(len(shards), gpu_ids)
    return [
        {"episodes": shard.tolist(), "gpu_id": gpus[i], "seed": base_seed + i}
        for i, shard in enumerate(shards)
    ]


def _write_dataset_setup(
    out_dir: Path,
    base_settings: dict,
    randomization: dict | None,
    num_episodes: int,
    num_workers: int,
    gpus: list[int],
    base_seed: int,
) -> None:
    """Dump a run-level ``dataset_setup.yaml`` for reproducibility/auditing."""
    payload = {
        "num_episodes": int(num_episodes),
        "num_workers": int(num_workers),
        "gpus": [int(g) for g in gpus],
        "base_seed": int(base_seed),
        "randomization": randomization,
        "scene": base_settings.get("visual_backend", {}).get("scene"),
        "sim_time": base_settings.get("simulator", {}).get("sim_time"),
    }
    with open(out_dir / "dataset_setup.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def record_dataset(
    base_settings: dict,
    *,
    out_dir,
    num_episodes: int,
    num_workers: int = 1,
    gpu_ids: list[int] | None = None,
    base_seed: int = 0,
    randomization: dict | None = None,
    mp_context: str = "spawn",
) -> None:
    """Fan ``num_episodes`` across ``num_workers`` processes over ``gpu_ids``.

    Episode indices are sharded contiguously (so filenames are globally unique).
    Worker ``i`` gets ``gpu = explicit_gpu_map(...)[i]`` and ``seed = base_seed + i``
    => independent domain-randomization streams. ``num_workers=1`` runs in-process
    (easy debugging); otherwise workers are spawned (CUDA/Habitat must not fork) and
    joined, and a non-zero exit code raises.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = _plan_shards(num_episodes, num_workers, gpu_ids, base_seed)
    _write_dataset_setup(
        out_dir,
        base_settings,
        randomization,
        num_episodes,
        len(plan),
        [w["gpu_id"] for w in plan],
        base_seed,
    )

    if len(plan) == 1:  # in-process: easy debugging, no spawn overhead
        record_episodes(
            base_settings,
            out_dir=out_dir,
            randomization=randomization,
            **plan[0],
        )
        return

    ctx = mp.get_context(mp_context)
    procs = [
        ctx.Process(
            target=record_episodes,
            kwargs=dict(
                base_settings=base_settings,
                out_dir=str(out_dir),
                randomization=randomization,
                **worker,
            ),
        )
        for worker in plan
    ]
    for p in procs:
        p.start()
    logger.info(
        "started %d recorder worker(s): pids=%s", len(procs), [p.pid for p in procs]
    )
    for p in procs:
        p.join()
    failed = [i for i, p in enumerate(procs) if p.exitcode]
    if failed:
        raise RuntimeError(f"recorder worker(s) failed (exit!=0): {failed}")
    logger.info("dataset complete: %d episodes -> %s", num_episodes, out_dir)
