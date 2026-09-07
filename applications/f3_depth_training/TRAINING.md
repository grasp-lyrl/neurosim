# Monocular Depth Training (events → depth)

Train F3 + DepthAnythingV2 on events+depth streamed live from neurosim. Simulators
run in their own processes; the model trains on one GPU.

The model lives in [nets/](nets/): F3 and DepthAnythingV2 are in-house, so there is no
`f3` package to install. Two weight files go under `models/` (gitignored): the released F3
backbone and the DAv2 ViT-B checkpoint, both named in the config below.

## 1. Write the config

Everything lives in one YAML — start from
[configs/depth_training_config.yml](configs/depth_training_config.yml) and edit the
`data:` block. `mini_batch` is the loader batch size, `batch` the effective one.

```yaml
data:
  trainer_gpu: 1                      # GPU the model trains on
  min_events_per_sample: 10000        # drop sparser samples before batching

  online_data:                        # OnlineDataLoader.from_config reads this block
    base_settings: configs/online_data_hm3d_2gpu.yaml
    num_producers: 8
    gpu_ids: [0]                      # sims, cycled across producers
    bus_maxsize: 2048
    prefetch: 2                       # batches built ahead on a thread (0 = inline)
    roles:
      anchor: [depth_camera_1]        # depth tick = one sample
      stream: [event_camera_1]        # events accumulate within each depth window
    randomization:
      resample_every: 20
      scenes_glob: data/hm3d/*/*.basis.glb
```

Two GPUs: `gpu_ids: [0]`, `trainer_gpu: 1`. One GPU: both `0`, `num_producers` 2–4.
Full DR grammar in [online_data/README.md](../../src/neurosim/online_data/README.md).

> ⚠️ **Match the event window to the depth rate.** Each sample's events are everything
> since the previous depth tick, normalized by `event_time_window_us` (default = model
> frame-T ms × 1000). Set `simulator.sensor_rates.depth_camera_1` to `1000 / T_ms`
> (50 Hz for a 20 ms model).

## 2. Train

```bash
nohup conda run --no-capture-output -n neurosim python -u \
    -m applications.f3_depth_training.train_depth_nonrec \
    --conf applications/f3_depth_training/configs/depth_training_config.yml \
    --name my_run --batches-per-epoch 2048 --retrain-f3 --amp --wandb \
    > /tmp/my_run.log 2>&1 &
```

`--batches-per-epoch` counts **mini-batches**: 2048 with `mini_batch: 8` is 16,384
samples and `2048 / (batch // mini_batch)` optimizer steps per epoch.

Flags: `--retrain-f3` (unfreeze the backbone), `--amp` (bf16; ~1.4x on the model step,
SSI loss stays fp32), `--init path.pth` (warm-start), `--wandb`. `--compile` (the DAv2
decoder) is off by default and best left there; `eventff` is compiled regardless.

Run from the repo root: the entry points import their siblings (`nets/`, `utils/`), so they
go through `-m`, not a file path.

**Outputs:** `outputs/monoculardepth/<name>/` — `models/{last,best}.pth`, `training.log`,
`config.yaml`, visualizations, per-producer logs under `logs/`. Resume is automatic if
`last.pth` exists.

## 3. Replay a checkpoint over recorded events

```bash
python -m applications.f3_depth_training.replay_depth_h5 \
    --run outputs/monoculardepth/my_run --h5 data/falcon_indoor_flight_3.h5 \
    --video /tmp/replay.mp4
```
