# Monocular Depth Training (events → depth)

Train the F3 + DepthAnythingV2 depth model on events+depth streamed live from
neurosim. Simulators run in their own processes; the model trains on one GPU.

## 1. Write the config

Everything lives in one YAML. Start from
[configs/depth_training_config.yml](configs/depth_training_config.yml) — model and
training knobs are already set; you mainly edit the `data:` block.

```yaml
train: { batch: 8, mini_batch: 8 }   # mini_batch == loader batch size

data:
  trainer_gpu: 1                      # GPU the model trains on
  # color_sensor: color_camera_1      # optional, for viz logging
  # sim_time: 5.0                     # optional override of episode length
  # event_time_window_us: ...         # defaults to model frame-T (ms) × 1000

  online_data:                        # same block OnlineDataLoader.from_config reads
    base_settings: configs/apartment_1-settings.yaml   # sensors, dynamics, trajectory
    num_producers: 8                  # parallel simulators
    gpu_ids: [0]                      # GPU(s) for the sims (cycled across producers)
    base_seed: 0                      # producer i gets seed base_seed + i
    bus_maxsize: 256                  # backpressure bound (samples)
    roles:
      anchor: [depth_camera_1]        # depth tick = one sample
      stream: [event_camera_1]        # events accumulate within each depth window
    randomization:
      resample_every: 20              # reload scene+sensors every N episodes
      # scenes:  [{name: ..., path: ...}]    # DR scene pool (omit -> fixed base scene)
      # sensors: {event_camera_1: {contrast_threshold_pos: {range: [0.1, 0.3]}}}
      # trajectory: {v_avg: {range: [0.8, 1.5]}}
```

**GPU layout.** Two GPUs: keep `gpu_ids: [0]`, `trainer_gpu: 1` (sims on gpu0, model
on gpu1). One GPU only: set both to `0` and drop `num_producers` to 2–4 so the sims
leave VRAM for the model.

**Diversity** is automatic: producers share settings but get distinct seeds, fly a
fresh trajectory every episode, and reload scene+sensors every `resample_every`
episodes. Add `scenes` / `sensors` / `trajectory` under `randomization` for more
variety — same grammar as everywhere in `online_data`
(see [online_data/README.md](../../src/neurosim/online_data/README.md)).

> ⚠️ **Match the event window to the depth rate.** Each sample's events are
> everything since the previous depth tick, normalized by `event_time_window_us`
> (default = model frame-T ms × 1000, e.g. 20 ms for a `...x20` model). Set
> `simulator.sensor_rates.depth_camera_1` in the **settings** file to `1000 / T_ms`
> (50 Hz for a 20 ms model) so each sample carries ~one window of events.

## 2. Verify the data path (no model needed)

f3 is imported lazily, so this works before f3 is installed:

```bash
conda run -n neurosim python applications/f3_training/train_depth.py \
    --conf applications/f3_training/configs/depth_training_config.yml \
    --smoke-data --smoke-batches 3
```

Expect `batch 0: depth=(8, 480, 640) events=(N, 4) ... spec_ids=[0, 1, ...]`.

## 3. Train

```bash
conda run -n neurosim python applications/f3_training/train_depth.py \
    --conf applications/f3_training/configs/depth_training_config.yml \
    --name my_run --amp            # add --wandb if installed
```

Useful flags: `--amp` (bf16), `--batches-per-epoch N`, `--retrain-f3` (unfreeze the
backbone), `--init path.pth` (warm-start), `--compile`.

**Outputs:** `outputs/monoculardepth/<name>/` — `models/{last,best}.pth`,
`training.log`, `config.yaml`, prediction/event visualizations, per-producer logs
under `logs/`. Resume is automatic if `last.pth` exists. Stop with `Ctrl-C`; the
loader tears down all simulator processes cleanly.
