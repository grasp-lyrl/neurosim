# `neurosim.online_data` — streaming data + offline recording

Two ways to feed supervised training (e.g. monocular depth from events) with
domain-randomized simulator data:

1. **`OnlineDataLoader`** — a torch-`DataLoader`-like stream. Many GPU-parallel
   simulators run in the background, each randomizing its scene/sensors/trajectory,
   and push **time-aligned batches** straight into your training loop. No disk.
2. **The recorder** (`record_dataset`) — the same simulators, but each episode is
   written to **one canonical H5 file** on disk for later/offline training.

Both share the simulator + domain-randomization machinery; they differ only in where
the data goes (RAM stream vs disk).

---

## Part 1 — Online streaming (`OnlineDataLoader`)

The easiest way: put **everything** (sim settings, sensor roles, scenes, domain
randomization, loader knobs) in one YAML and call `OnlineDataLoader.from_config`. A
ready-to-run example config lives at
[`configs/online_data_example.yaml`](../../../configs/online_data_example.yaml)
(edit the scene `.glb` paths to ones you have).

```python
from neurosim.online_data import OnlineDataLoader


def main():
    with OnlineDataLoader.from_config("configs/online_data_example.yaml") as loader:
        for i, batch in enumerate(loader):  # yields forever; bound it
            if i >= 5:
              break
            counts, events = batch["event_camera_1"]  # events (N,4)=[x,y,t_rel,p]; counts (B,)
            depth = batch["depth_camera_1"]           # (B, H, W) float32
            scenes = batch.meta.scene                 # per-row (list of len B)
            print(f"batch {i}: depth={tuple(depth.shape)} events={tuple(events.shape)} "
                  f"counts.sum={int(counts.sum())}")
            # ... your train step here


if __name__ == "__main__":   # required: producers use `spawn`
    main()
```

Expected output (event counts vary with scene/trajectory):

```
batch 0: depth=(4, 480, 640) events=(2874558, 4) counts.sum=2874558
batch 1: depth=(4, 480, 640) events=(6271405, 4) counts.sum=6271405
batch 2: depth=(4, 480, 640) events=(8652082, 4) counts.sum=8652082
```

The config has a `simulator`/`visual_backend`/... block (the sim settings) and an
`online_data` block (roles + DR + loader knobs):

```yaml
online_data:
  batch_size: 4
  num_producers: 2          # simulators running concurrently
  gpu_ids: [0]              # cycled across producers
  base_seed: 0
  roles:                    # <- the schema, in YAML (no Python needed)
    anchor: [depth_camera_1]   # tick defines the sample boundary
    stream: [event_camera_1]   # accumulated within each anchor window
    # latest: [imu_1]          # most-recent value held at the anchor (optional)
  randomization:
    resample_every: 10        # reload scene every N episodes
    scenes:  [{name: a, path: data/hm3d/.../a.basis.glb}, ...]
    sensors: {event_camera_1: {contrast_threshold_pos: {range: [0.1, 0.3]}}}
    trajectory: {v_avg: {range: [0.8, 1.5]}}
# base_settings can also live in a *separate* file:
#   online_data: { base_settings: configs/my-settings.yaml, ... }
```

<details>
<summary>Prefer building it in Python instead? (full control, no config file)</summary>

```python
import yaml, itertools
from neurosim.online_data import OnlineDataLoader, SampleSchema


def main():
    settings = yaml.safe_load(open("configs/hm3d-00000-settings.yaml"))
    settings["simulator"]["sim_time"] = 1.0
    schema = SampleSchema.from_sensor_configs(
        {u: settings["visual_backend"]["sensors"][u]
         for u in ("depth_camera_1", "event_camera_1")},
        anchor=["depth_camera_1"], stream=["event_camera_1"],
    )
    loader = OnlineDataLoader(
        schema, batch_size=4, base_settings=settings,
        randomization={"resample_every": 10, "scenes": [
            {"name": "a", "path": "data/hm3d/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb"}]},
        num_producers=2, gpu_ids=[0], base_seed=0,
    )
    try:
        for batch in itertools.islice(loader, 3):
            ...  # batch["depth_camera_1"], batch["event_camera_1"], batch.meta
    finally:
        loader.close()


if __name__ == "__main__":
    main()
```

</details>

### Terminology

| Term | Meaning |
|---|---|
| **Producer** | One background process running one simulator on one GPU. `num_producers` of them push to a shared queue → consecutive batch rows come from different sims (diverse batches). |
| **Sample** | One time-aligned row: all delivered sensors for one anchor tick. The atomic unit on the wire (`TimeAlignedSample`). |
| **Batch** | `batch_size` samples stacked. A `dict[uuid -> payload]` plus `batch.meta`. |
| **Anchor** (role) | The sensor whose tick *defines* a sample boundary — usually the depth frame. Exactly drives the sample rate. Must not be a streaming-kind sensor. |
| **Stream** (role) | A sensor accumulated **between** anchor ticks into a variable-length packet (e.g. events in the window `(t_prev, t_anchor]`). |
| **Latest** (role) | A sensor whose most-recent value is just held and attached at the anchor tick (e.g. an IMU vector). |
| **Kind** | *What* a sensor is, drives batching: `frame` (dense `H×W`), `event_stream` (`{x,y,t,p}` packet), `vector` (`(D,)`), `vector_stream` (`(k,D)`). Inferred from the sensor `type`; override with `kind_overrides`. |
| **`spec_id` / `worker_id`** | Identify the producer that made a sample (in `batch.meta`). |
| **Episode** | One trajectory flight from sim reset to `sim_time`. Each episode re-randomizes (see below). |
| **`resample_every`** | Episodes between expensive scene/sensor reloads. The trajectory is re-sampled **every** episode regardless. |

**Roles vs kinds are orthogonal.** Role = *how it's assembled into a sample*
(anchor/stream/latest); kind = *what it is / how it's batched* (frame/event_stream/…).
A depth camera is `kind=frame, role=anchor`; an event camera is
`kind=event_stream, role=stream`.

### What the loader hands you

- **Frame sensors** → `(B, H, W[, C])` stacked array.
- **Event sensors** → tuple `(counts, events)`:
  - `counts`: `(B,)` int — events per sample (so you can split the flat array).
  - `events`: `(N_total, 4)` float32, columns `[x, y, t_anchor - t, p]` — **raw** pixel
    coords, anchor-relative microsecond time, polarity. Normalization (`/W`, `/H`,
    `/window`) is left to your training loop (do it on-GPU). See `shift_events`.
- **`batch.meta`** (`BatchMeta`) — per-row arrays: `scene`, `seed`, `spec_id`,
  `worker_id`, `episode_id`, `step_idx`, `t_us`, `window_us`, `is_first`, `is_last`.

### `OnlineDataLoader` config (the knobs that matter)

```python
OnlineDataLoader(
    schema, batch_size,
    base_settings=<settings dict>,       # the sim settings (sensors, dynamics, trajectory)
    randomization=<DR dict>,             # see "Domain randomization" below
    num_producers=8,                     # how many sims run concurrently
    gpu_ids=[0, 1],                      # GPUs cycled across producers ([0,1] + 8 -> 4/GPU)
    base_seed=0,                         # producer i gets seed = base_seed + i
    bus_maxsize=256,                     # backpressure bound (queue capacity)
    ring_caps={"event_camera_1": 5_000_000},  # optional per-stream packet cap
    get_timeout=1.0,                     # consumer poll / producer-death check
    log_dir="outputs/.../logs",          # optional: per-producer logs + run_setup.yaml
)
```

Tips:
- **Placement:** `gpu_ids` is *cycled*. `[0]` puts everything on gpu0; `[0,1,2,3]` with
  `num_producers=4` is one sim per GPU; co-locate the trainer on a separate GPU.
- **Diversity** is automatic: same settings + DR, distinct seed per producer.
- The loader iterates **forever**; bound with `itertools.islice(loader, n_batches)`.
- Always `loader.close()` (or use it as a context manager) to stop producers.

### Building the schema for *any* set of sensors

`SampleSchema.from_sensor_configs(sensor_configs, *, anchor, stream, latest, deliver,
kind_overrides, extras)`:

- `sensor_configs`: `uuid -> config` (typically a subset of
  `settings["visual_backend"]["sensors"]`). Frame sensors must carry `height`/`width`.
- Assign each delivered UUID exactly one role via `anchor` / `stream` / `latest`.
- `deliver` defaults to anchor+stream+latest; set it to expose a subset.
- It is **generic over UUIDs** — add a color camera, a second event camera, an IMU, etc.
  by listing them with the right role. Nothing is hard-coded to "depth"/"events".

---

## Part 2 — Offline recording (`record_dataset`)

Same simulators, but each episode → one canonical H5 file (the layout
`SynchronousSimulator.run(log_h5=...)` / `test_sim.py` produce). Use this to build a
fixed dataset you can re-train on, inspect, or ship.

### Run it

```bash
# in the neurosim conda env
python scripts/generate_dataset.py --conf scripts/configs/generate_dataset_hm3d_sanity.yml
```

Or programmatically (same `__main__`-guard rule — `num_workers > 1` spawns workers):

```python
from neurosim.online_data import record_dataset


def main():
    record_dataset(base_settings_dict, out_dir="data/datasets/ds_v1",
                   num_episodes=500, num_workers=8, gpu_ids=[0, 1], base_seed=0,
                   randomization={"resample_every": 20, "scenes": [...], "sensors": {...}})


if __name__ == "__main__":   # required when num_workers > 1 (workers use `spawn`)
    main()
```

### Generation config (YAML for `scripts/generate_dataset.py`)

```yaml
base_settings: configs/hm3d-00000-settings.yaml   # sim settings (sensors/dynamics/trajectory)
out_dir: data/datasets/hm3d_v1
num_episodes: 500
num_workers: 8           # simulators running concurrently
gpu_ids: [0, 1]          # cycled across workers
base_seed: 0
sim_time: 5.0            # optional override of simulator.sim_time (seconds/episode)
domain_randomization:    # optional; same schema as the loader's `randomization`
  resample_every: 20
  scenes:
    - {name: kfPV7w3FaU5, path: data/hm3d/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb}
    - {name: UVdNNRcVyV1, path: data/hm3d/00001-UVdNNRcVyV1/UVdNNRcVyV1.basis.glb}
  sensors:
    event_camera_1:
      contrast_threshold_pos: {range: [0.1, 0.3]}
  trajectory:
    v_avg: {range: [0.8, 1.5]}
```

### On-disk layout

```
<out_dir>/
  dataset_setup.yaml            # run-level recipe: DR ranges, counts, seeds, gpu map
  episode_000000.h5             # canonical H5: per-sensor groups + state + sim_time/step
  episode_000000.meta.yaml      # the *realized* settings for this episode (provenance)
  episode_000001.h5
  episode_000001.meta.yaml
  ...
```

- Each `.h5` has one group per sensor UUID (`depth_camera_1/data` stacked frames;
  `event_camera_1/{x,y,t,p}` flat-concatenated), plus `state` and `sim_time`/`sim_step`.
- `dataset_setup.yaml` records the DR **recipe** (the ranges/choices you asked for).
- `episode_NNNNNN.meta.yaml` records the DR **realization** actually used for that
  episode — chosen scene, sampled sensor params, sampled `v_avg`, trajectory seed.

### Parallelism knobs

`num_workers` = how many sims run at once; `gpu_ids` = which GPUs (cycled).
`num_workers=1` runs in-process (easy debugging); otherwise workers are spawned and
joined. Episodes are sharded contiguously across workers, so filenames are globally
unique and worker `i` uses `seed = base_seed + i`.

### Viewing recorded events

The visualizer needs a per-millisecond index first:

```bash
python scripts/build_ms_to_idx.py <out_dir>/episode_000000.h5 --sensor event_camera_1
python scripts/visualize_h5_events.py <out_dir>/episode_000000.h5 --sensor event_camera_1
```

---

## Domain randomization (shared by both paths)

The `randomization` / `domain_randomization` dict (`RandomizedSimulator` owns it):

```yaml
resample_every: 20          # reload scene + sensors every N episodes (scene reload is slow)
scenes:                     # pool; one sampled per resample (uniform). Omit -> fixed base scene.
  - {name: a, path: .../a.basis.glb}
  - {name: b, path: .../b.basis.glb}
sensors:                    # per-UUID param overrides, sampled each resample
  event_camera_1:
    contrast_threshold_pos: {range: [0.1, 0.3]}   # uniform in [lo, hi]
    hfov:                   {choices: [70, 90, 120]}  # pick one
trajectory:                 # re-sampled EVERY episode (cheap; rebuilt in-place)
  v_avg: {range: [0.8, 1.5]}
```

- **`{range: [lo, hi]}`** → uniform float; **`{choices: [...]}`** → uniform pick;
  a plain value → fixed override.
- **Cadence:** scene + sensors change every `resample_every` episodes (expensive);
  the **trajectory is re-seeded every episode** so each clip flies a new path. The
  per-episode trajectory seed is derived deterministically from `(seed, episode)`.
- **Determinism / provenance:** runs are reproducible for a fixed
  `(base_seed, num_workers, gpu layout)`. For the recorder, the exact realization per
  file is always captured in `episode_NNNNNN.meta.yaml`. For the loader, per-row
  provenance is in `batch.meta` (`scene`, `seed`, `spec_id`, …).

---

## Which should I use?

| | `OnlineDataLoader` | `record_dataset` |
|---|---|---|
| Output | in-RAM batches | H5 files on disk |
| Re-use across runs | regenerated each run | fixed, reloadable |
| Epochs over fixed data | no (infinite fresh stream) | yes (read files) |
| Best for | large-scale on-the-fly training | reproducible datasets, inspection, sharing |

> Reading recorded H5 back into training samples (windowing/anchoring on disk — the
> offline analogue of the loader's assembler) is a planned `H5EpisodeDataset` reader;
> until then the H5 files are consumable directly with `h5py`.
