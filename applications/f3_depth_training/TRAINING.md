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

## 3. Metric depth

The default trains relative disparity: the SSI loss forgives any per-frame scale and shift,
so consecutive predictions drift against each other. `--metric` trains depth in metres
instead: a sigmoid head capped at `max_depth`, the SiLog loss, and metrics scored as
predicted (the `*_aligned` keys show what an affine fit would still recover).

```bash
python -m applications.f3_depth_training.train_depth_nonrec \
    --conf applications/f3_depth_training/configs/depth_training_config_voltmeter_metric.yml \
    --name my_metric_run --metric --amp --wandb \
    --init outputs/monoculardepth/my_run/models/best_d1.pth
```

**The canonical camera.** The network never sees intrinsics, so with the FOV randomized
the metric depth of a given input is ambiguous by the focal-length ratio and the head can
only learn the average. The target is therefore `depth * focal_canonical / focal`, after
Metric3D: the head predicts what a camera of `focal_canonical` would see, and inference
scales back by the real focal length. `focal_canonical: 686` is the 50 deg deployment
camera at 640 px wide, so deployment needs no rescale at all; M3ED at f 1033 scales by
1.51. `head_max_depth` is the sigmoid ceiling in canonical metres and must exceed
`max_depth * focal_canonical / f(widest hfov)`.

The per-episode hfov reaches the trainer through `SampleMeta.hfov`; M3ED's focal length is
a constant in `data/m3ed.py`. `--init` from a relative run warm-starts everything but the
emit conv, which is reset because the head differs. The eval and replay scripts read the
head and `focal_canonical` from the run, so they need no flag; `replay_depth_h5` takes
`--focal` for the sensor it is replaying.

Both losses report their two terms, `train/data_term` and `train/grad_term`, per step to
wandb and per epoch to the log, since the gradient term's natural scale differs between
normalized disparity and log depth.

## 4. Replay a checkpoint over recorded events

```bash
python -m applications.f3_depth_training.replay_depth_h5 \
    --run outputs/monoculardepth/my_run --h5 data/falcon_indoor_flight_3.h5 \
    --video /tmp/replay.mp4
```
