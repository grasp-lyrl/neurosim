# Event-camera dynamic-obstacle dodging

Reproduction guide for the event-based avoidance pipeline on the telegraphed
single-obstacle task. `HANDOFF.md` in this directory carries the current
experimental state, measured results, and the ranked next steps; this file
covers only *how to re-run things*.

## Running anything

Every command runs inside the existing container:

```bash
docker exec -w /home/odexter/neurosim \
  -e PYTHONPATH=/home/odexter/neurosim/src:/home/odexter/neurosim/applications/rl \
  neurosim-noros /opt/conda/envs/neurosim/bin/python <script> <args>
```

Add `-e CUDA_VISIBLE_DEVICES=<n>` to pin a job to one GPU; the config chain
asks for `cuda:0`, so this remaps cleanly and is how jobs are run in parallel.

High-resolution Habitat jobs sometimes exit `139` during teardown *after* JSON
and video files have flushed. Check artifact completeness before calling a run
failed.

## Pipeline shape

Four stages, trained separately, each frozen before the next:

```
events (2x480x640 time surface, 10 ms, history 4 -> 8 channels)
  -> train_event_obstacle_localizer.py   SpatialEventLocalizer   (per-frame heatmap + presence)
  -> train_event_obstacle_tracker.py     TemporalHeatmapTracker  (GRU over frozen coarse maps)
  -> train_event_tracker_action_head.py  EventTrackerActionHead  (GRU: track features + ego state -> action)
  -> event_pulse_residual.py             finite avoidance pulse layered on the nominal controller
```

`evaluate_event_geometry_controller.py` holds `EventObstacleTracker`, the
one-step causal inference wrapper that binds the spatial and temporal stages.
Import it rather than re-implementing the glue.

Deployed cost is ~742k parameters and ~1.31 GMACs per frame (see the Orin NX
section of `HANDOFF.md`); inference compute is not the constraint.

## Config chain

Configs inherit through `experiment_config:` and override narrowly. The two
that matter:

- `configs/velocity_dodge_telegraph_v17_no_dither_event_pulse_residual_ppo.yaml`
  is the primary environment. **v17 sets `telegraph_dither_amplitude_m: 0.0`**;
  the older v16 localization chain inherits `0.12` at 2 Hz.
- `configs/velocity_dodge_telegraph_v17_no_dither_event_pulse_residual_ppo_renew5_eval.yaml`
  adds selective pulse renewal and is the collection/evaluation entry point.
  Its `..._no_obstacles_eval.yaml` sibling disables dynamic obstacles for
  empty-scene negative controls.

Configs whose header says REJECTED or diagnostic are retained deliberately as
provenance for the rejected-experiment list in `HANDOFF.md`. Do not delete them
to tidy the directory.

**Never mix v16 positives with v17 negatives.** The dither is present in
exactly one class and absent at deployment, so a mixed corpus makes the dither
the label. This confound was found on 2026-09-04 and invalidated the earlier
detector corpus.

## Reproducing the detector corpus (2026-09-04, current best)

Roughly 375 MB and ~25 s per episode; 96 episodes across six V100s takes about
35 minutes and 40 GB.

Positives, under the v17 no-dither protocol:

```bash
applications/rl/evaluate_velocity_dodge_oracle.py \
  --experiment-config applications/rl/configs/velocity_dodge_telegraph_v17_no_dither_event_pulse_residual_ppo_renew5_eval.yaml \
  --planner sampling_mpc --episodes 16 --seed0 61001 \
  --dataset  <out>/obstacle_raw_v17/obs_a.h5 \
  --output   <out>/obstacle_raw_v17/obs_a.json \
  --include-all-episodes --include-plan-failures \
  --record-privileged --record-obstacle-image-label
```

Empty-scene negatives use the `..._no_obstacles_eval.yaml` config with the same
flags. Add `--rollout-policy <action_head>.pt --beta 0.0` to fly the negatives
with a learned policy, which captures the closed-loop false-dodge ego-motion
distribution; omit it for expert-flown negatives.

Seeds used: positives 61001+, expert negatives 57201+, policy negatives 59201+.
**Seeds 57101+ and 59101+ are deliberately held out** as the evaluation set and
must stay out of training.

## Training the detector

```bash
applications/rl/train_event_obstacle_localizer.py \
  --dataset <all positive and negative .h5> \
  --output <out>/perception_v17/spatial_localizer.json \
  --epochs 30 --batch-size 16 --samples-per-epoch 8000 \
  --num-workers 12 --device cuda:0
```

```bash
applications/rl/train_event_obstacle_tracker.py \
  --dataset <same list> \
  --spatial-checkpoint <out>/perception_v17/spatial_localizer.best.pt \
  --output <out>/perception_v17/temporal_tracker.json \
  --epochs 20 --device cuda:0
```

`--output` names the JSON history; the checkpoint is written alongside as
`<stem>.best.pt`. Both scripts hold out 20% of *episodes* (`--holdout-seed`
7373), so validation never shares an episode with training.

Raise `--num-workers` well above its default of 2. Frames are ~4.9 MB each and
the job is disk-bound: at the default the GPU sits near 15% utilisation, and 12
workers cut epoch time from ~13 to ~5.5 minutes.

## Evaluating false commitments on empty scenes

This is the gating measurement, and it must run on the held-out seeds. It
executes the tracker online and stores ~140 floats per frame instead of raw
events, so it is cheap:

```bash
applications/rl/collect_event_tracker_oracle_features.py \
  --experiment-config applications/rl/configs/velocity_dodge_telegraph_v17_no_dither_event_pulse_residual_ppo_renew5_no_obstacles_eval.yaml \
  --spatial-checkpoint  <spatial>.best.pt \
  --temporal-checkpoint <temporal>.best.pt \
  --output <out>/nominal_57101.pt --summary <out>/nominal_57101.json \
  --attempts 8 --seed0 57101 --zero-action-labels --device cuda:0 --sim-gpu-id 0
```

Repeat with `--seed0 59101` plus `--rollout-checkpoint <action_head>.pt` for the
policy-flown arm.

In each stored `tracker_input` row: index `128` is `probability`, `133` is
`inbound_probability`, `134:137` is `v_body`, `140:143` is `omega_body`. Score a
commitment as a run of >= 2 consecutive frames passing both the presence and
inbound gates, counted per episode so runs never span an episode boundary.
`HANDOFF.md` holds the before/after tables in this exact form.

## Other entry points

These have no importers because they are CLI tools, not libraries:

- `evaluate_event_tracker_action_policy.py` - BC action-head evaluation with
  tracker/inbound threshold flags.
- `evaluate_event_pulse_residual_policy.py` - renewal-aware pulse evaluation;
  produced the pulse-duration/renewal sweep in `HANDOFF.md`.
- `collect_event_tracker_dagger_features.py` - compact DAgger collection on
  states an event policy actually visits.
- `probe_telegraph_event_cue.py` - measures a dither's causal event signature
  against a matched zero-dither rollout. This is the tool that would have
  caught the v16/v17 dither confound earlier.
- `evaluate_velocity_dodge_oracle.py` - privileged oracle rollouts and the raw
  labelled HDF5 collector used above.

## Tests

```bash
docker exec -w /home/odexter/neurosim \
  -e PYTHONPATH=/home/odexter/neurosim/src:/home/odexter/neurosim/applications/rl \
  neurosim-noros /opt/conda/envs/neurosim/bin/python -m pytest -q \
  tests/test_event_obstacle_localizer.py \
  tests/test_event_tracker_action_head.py \
  tests/test_event_pulse_residual.py \
  tests/test_event_geometry_controller.py \
  tests/test_event_geometry_residual.py \
  tests/test_velocity_dodge_oracle.py \
  tests/test_velocity_dodge_gru_bc.py
```
