# velocity_dodge handoff

## Current state — 2026-08-29 (supersedes everything below)

Everything under "Current continuation — 2026-08-11" and later in this file is
HISTORICAL. It describes a `gated_cascaded_velocity` workflow that has since
been reverted (see v20 config comments) and a control path that was found to
be defective (commit 5225691). Read this section only.

### Continuation — 2026-08-31

The privileged-teacher v2 run finished its full 4M-step budget. It failed the
pre-registered gate: mean success was 20.3% over 122 evaluations, the final
ten averaged 19.0%, and the final eval was 20.0%. Its single 40.0% eval at
2.326M did not persist. Final return was -46.9. It beat neither v2 blind gate
(-28.2 return and 32.5% success), confirming the original three-concurrent
task remains unlearnable even from its lossless privileged observation.

The active continuation is the higher-speed single-obstacle task:
`velocity_dodge_teacher_v4_fast.yaml`. It uses one concurrent obstacle,
10 m/s linear obstacles (8 m/s parabola), 5.0 m/s^2 offset acceleration, and
2.5/2.5/1.5 m/s rate limits. Its 40-episode-per-arm gate completed:

| arm | return | success |
|---|---:|---:|
| control | -78.1 | 15.0% |
| const+0.40 | -103.9 | 15.0% |
| const+0.80 | -131.8 | 17.5% |
| weave0.4@0.5Hz | -76.9 | 20.0% |
| **weave0.8@0.5Hz** | **-73.1** | **30.0%** |
| **oracle** | **+10.1** | **62.5%** |

No arm had a tracking-failure or out-of-bounds termination, so the raised
authority is stable. The learned-policy gate is **beat -73.1 return AND 30.0%
success**. The oracle margin (+83.2 return, +32.5 success points) authorises a
privileged-policy training run on v4.

That run is now active: run name `privileged_teacher_v4_fast`, Python PID
`171097` in `neurosim-noros`, started 2026-09-01 00:43 container time. The
4M-step budget writes to `outputs/rl/privileged_teacher_v4_fast/` and
`outputs/rl/train_privileged_teacher_v4_fast.log`. Startup and the first PPO
update completed normally (4,096 steps, 57 fps); all 16 training workers are
on GPU 1 and the learner is on GPU 0. Do not launch a duplicate copy. As with
v2, do not interpret the curve before 2M steps, and require both gate metrics.

#### Oracle follow-up — 2026-09-01

BC remains on hold. Three same-seed diagnostics and two 40-episode oracle
sweeps found no oracle edit that improves both return and success.

- Tracking is effectively exact: in the worst of 20 episodes, command was
  0.362 m, achieved deviation 0.361 m, maximum tracking error 0.009 m, and
  maximum tilt 18.9 degrees. The raised v4 authority is not the bottleneck.
- Timing is late: median effective commit lead is 0.37 s against 0.53 s
  required, 81% of 26 encounters commit late, zero direction flips, and only
  0.28 m median is achieved by closest approach against a 0.35 m first
  candidate. But peak advances of 0.15/0.25/0.35 s scored respectively
  `+8.5/70.0%`, `+8.6/65.0%`, and `+8.2/60.0%`, each with four collisions.
  The original remains better on return (`+10.1`) and collisions (three).
- Feasibility is also real: 108 plans made / 306 failed (73.9%). Two of three
  collision obstacles had no plan. On failed attempts candidate rejections
  were 47.9% speed, 31.3% static clearance, and 20.8% moving-obstacle
  clearance. This is a mixed constraint set, not one missing magnitude cap.
- Adding 0.25/0.30/0.40 m candidates to the original menu produced +9.0
  return / 62.5% success with four collisions and increased runtime from about
  16 to 23 minutes. It also failed the both-metrics rule.

Therefore keep the original oracle unchanged and continue the active
privileged PPO v4 run. Its early evaluations through ~200k steps remain in
the expected noisy band and are not evidence either way; the pre-registered
minimum for interpretation is 2M. If privileged PPO fails, the next teacher
experiment is privileged BC on successful oracle episodes followed by PPO
fine-tuning—not event BC from this 62.5% oracle and not event PPO from scratch.

#### Receding-horizon oracle replacement — 2026-09-01

The conclusion immediately above is superseded for teacher selection. The
quintic oracle remains unchanged as a reproducible baseline, but a new
sampling-MPC privileged oracle now decisively beats it.

Implementation:

- `src/neurosim/rl/receding_horizon_dodge_expert.py` is a vectorised,
  multimodal cross-entropy shooting MPC. It replans every 0.10 s over a 1.50 s
  horizon, warm-starts the previous solution, and optimises continuous
  normalised velocity commands rather than selecting a fixed bump.
- Its rollout model includes actual offset/relative velocity, the task's
  command filter, `delta_velocity_limits_mps`, bounded return-to-path term,
  predicted obstacle acceleration, and the nominal MinSnap path.
- Collision constraints use finite, extremely expensive slack. There is
  therefore always a least-dangerous action; an infeasible encounter no
  longer becomes the old oracle's zero-action / failed-plan label.
- Dynamic scoring is vectorised. The best 24 trajectories are checked against
  Habitat's 26-ray static mesh clearance test and world bounds. This makes the
  implementation suitable for offline teacher generation, but not yet a
  hard-real-time flight planner.
- `evaluate_velocity_dodge_oracle.py --planner sampling_mpc` selects it. The
  default is still `quintic`, so every historical command remains unchanged.

Registered same-seed gate, v4 seeds 9001--9040:

| planner | return | success | obstacle collisions | failed plans |
|---|---:|---:|---:|---:|
| old quintic | +10.1 | 62.5% (25/40) | 3 | 306/414 calls |
| **sampling MPC** | **+26.1** | **92.5% (37/40)** | **2** | **0** |

The MPC had 38 ordinary timeouts, no tracking failures, and no bounds exits.
The three misses were two obstacle collisions (seeds 9008 and 9016) and one
timeout below the 0.10 m required clearance (9022, 0.043 m). Both collisions
were explicitly predicted infeasible in the final replans: their planned
clearance fell to +0.019 and -0.011 m with 0.131 and 0.161 m safety slack.
This is the intended fallback semantics, not a falsely certified safe plan.

Disjoint confirmation, seeds 9101--9120: **+35.0 return, 95.0% success
(19/20), zero collisions, zero failed plans**. The one miss was another
ordinary timeout with 0.038 m clearance. Combined: 56/60 = 93.3% success,
+29.1 return, two collisions, and zero failed plans. Artifacts are
`outputs/rl/mpc_oracle_gate_{a,b,c,d}.{json,log}` and
`outputs/rl/mpc_oracle_holdout_{a,b}.{json,log}`.

One control-path mismatch matters for interpreting why this worked. v4 uses
`residual_control.mode: velocity_command`. That path does **not** consume
`offset_accel_limit_mps2` or `offset_rate_limits_mps`; because v4 does not set
`delta_velocity_limits_mps`, the effective residual limits are still the task
defaults, 0.6/0.6/0.4 m/s. The old expert nevertheless derived a 2.5/1.5 m/s
planning envelope from `offset_rate_limits_mps`, then its action was clipped
against 0.6/0.6/0.4 at execution. That planner/controller mismatch is a
direct mechanism for the measured late commits. Sampling MPC models the live
velocity-command path instead. The task and active PPO run were not changed,
so the oracle comparison remains fair.

Decision: the sampling MPC is now the privileged teacher and the quintic
planner is only a historical baseline. It clears the proposed >80% acceptance
target on both the registered and held-out seeds, so successful MPC episodes
are suitable for privileged BC followed by PPO fine-tuning. Continue the
already-running privileged PPO v4 experiment unchanged as the independent RL
comparison; its 2M-step interpretation rule still applies.

---

## 0. READ FIRST: `outputs/` was deleted

> **RESOLVED 2026-08-29 (next session).** Nothing deleted it. The project was
> moved to a different host, and `outputs/` -- along with `data/` -- is
> gitignored, so neither travelled. The disk figures below are the old host's.
> The practical consequence is unchanged: these artifacts must be regenerated.
> See section 1 for the current host.

At the end of this session `/home/odexter/neurosim/outputs/` no longer exists.
Disk went 790G used / 79G avail -> 669G used / 200G avail, i.e. ~121 GB freed.
I did not delete it and cannot say what did.

**Lost, and expensive to regenerate:**

| artifact | cost to rebuild | needed by |
|---|---|---|
| `outputs/rl/pretrained_encoder_effnet_final.pt` | ~4 h (512k env-steps) | the BC plan in section 6 |
| all training logs `train_v16..v22_effnet.log` | the runs themselves | nothing — numbers preserved below |
| all checkpoints + `vecnormalize.pkl` | days of GPU | nothing critical |
| `outputs/rl/bc/*.h5` (~99 GB) | hours each | nothing — all stale, predate the fixes |
| gate results, `ev_ceiling.log` | ~1 h | nothing — numbers preserved below |

**Survived** (in the git working tree, ALL UNCOMMITTED on branch
`dynamic-task-diagnosis`): every config `v14..v22`, every code change, every
probe script. Crucially, **every important measurement was written into the
config comments as it was made**, so the findings survive the loss of the
logs. That was not luck; keep doing it.

**First action for whoever picks this up: `git add -A && git commit`.** There
are 82 uncommitted paths including all of this session's work. One more
cleanup event and the configs go too.

> Done in commit 5d5107c. The tree is clean; there is nothing left to rescue.

---

## 1. Environment setup (read before running anything)

> **The project moved hosts on 2026-08-29.** The machine described in the rest
> of this file (2x RTX A5000, workspace on `/home`) is gone. Current host:

```
container:  neurosim-noros   (image neurosim:noros)
python:     /opt/conda/envs/neurosim/bin/python
workspace:  /pool/odexter/neurosim   (ZFS, 19 TB free)
            bind-mounted INTO the container at /home/odexter/neurosim
GPUs:       8x Tesla V100-SXM2-32GB  (sm_70 -- NOT the A5000's sm_86)
host user:  odexter = uid 2000023    (uid 1000 here is a different user, fclad)
```

`/home/odexter/neurosim` still exists **on the host** as a 143 MB stale copy of
the tree. It is not what the container sees. Do not edit it; the live tree is
`/pool/odexter/neurosim`. This dual meaning of `/home/odexter/neurosim` --
stale copy on the host, live workspace inside the container -- is the single
most confusing thing about this setup.

This also resolves the `outputs/` mystery in section 0: nothing deleted it, the
work simply moved to a host that never had it. `data/` was lost the same way
and had to be re-fetched.

### Restoring the environment: run the script

Everything below is now automated. After any container recreation:

```bash
docker exec -e HOST_UID=2000023 -e HOST_GID=2000023 \
  -w /home/odexter/neurosim neurosim-noros bash docker/setup_container_env.sh
```

It installs torchvision/SB3/wandb, rebuilds both CUDA extensions **for the
locally detected GPU arch**, re-fetches the Habitat scenes, and verifies the
result. Read the traps below anyway -- they explain why each step is there.

### The traps that will cost you an hour each

**1. `PYTHONPATH` is mandatory.** Without it the container resolves `neurosim`
to a stale copy at `/opt/neurosim` and you get
`ImportError: cannot import name 'deep_update' from 'neurosim.core.utils.utils_gen'`.

```bash
docker exec -w /home/odexter/neurosim <cid> bash -c \
  "PYTHONPATH=/home/odexter/neurosim/src /opt/conda/envs/neurosim/bin/python -u <script>"
```

**2. Container deps are ephemeral.** `pip install`s and the CUDA extension
builds are NOT in the image and are lost whenever the container is recreated.
Verified working set:

```
python 3.10.14   torch 2.9.1+cu128   torchvision 0.24.1+cu128
stable_baselines3 2.9.0   sb3_contrib 2.9.0   h5py 3.14.0   wandb
```

`torchvision` **must** match torch. 0.28.0 against torch 2.9.1 fails at import
with `RuntimeError: operator torchvision::nms does not exist`. Pin 0.24.1.
EfficientNet-B0 comes from torchvision, so nothing in `sb3_features.py` with
`event_backbone: efficientnet_b0` runs without it.

`train_sb3.py` imports `wandb` unconditionally at module scope, so **every**
script that reaches it -- including `return_baseline.py` and the oracle
evaluator, via `load_experiment_config` -- fails without wandb installed, even
with `--no-wandb`.

**2b. The image's `neurosim_cu_esim` is too old, and the failure does not say
so.** The image ships v0.1, whose `EventSimulator` has no `mode` kwarg and no
`DVSVoltmeterSimulator`. Every config with `backend: cuda` -- which is all of
them -- dies at env construction with:

```
TypeError: EventSimulator.__init__() got an unexpected keyword argument 'mode'
```

That reads like a config error. It is a stale dependency. Reinstall from source
(`https://github.com/grasp-lyrl/neurosim_cu_esim`, public); the setup script
clones it to `deps/neurosim_cu_esim` and builds it. **Build for the local GPU
arch** -- pass `TORCH_CUDA_ARCH_LIST`; a wheel built for the old sm_86 host is
wrong on these V100s.

Note `src/neurosim/core/event_sim/cu_rpg_vid2e_esim` is a *different*, unused
backend (`esim_torch`/`esim_cuda`). Building it does not satisfy `backend: cuda`.

**2c. `data/` is gitignored and does not travel.** The Habitat scene mesh
`data/scene_datasets/habitat-test-scenes/skokloster-castle.glb` is a required
input that no commit contains. Re-fetch with
`python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes`.
It writes an **absolute** symlink pointing at the in-container path, which is
broken when read from the host; replace it with a relative one.

**3. Files written by the container are root-owned.** `rm -rf outputs/...`
from the host fails with `Permission denied`. Delete from inside instead:
`docker exec -w /home/odexter/neurosim <cid> rm -rf <path>`. To hand files back,
`chown -R 2000023:2000023` -- **not** the `1000:1000` this file used to say.
uid 1000 was odexter on the old host; here it is an unrelated user, and
chowning to it silently gives your files away.

**3b. `pkill -f <config-name>` silently fails to kill the run.** The pattern
appears in the command line of the very `bash -c` that runs pkill, so pkill
matches its own parent shell and kills it mid-sequence; the training process
can survive. This is nasty because the relaunch that follows appears to
succeed while the OLD process is still running and still writing the log --
a config change then looks applied when it is not, and the result is
attributed to the wrong configuration.

Symptom: the process's `etime` is much larger than the time since restart.
**Always check `etime` after a restart.** Kill by PID instead:

```bash
docker exec <cid> bash -c "ps -eo pid,cmd | grep '[t]rain_sb3.py'"
docker exec <cid> bash -c "kill -TERM <pid>; sleep 5; kill -KILL <pid>"
```

Note the `[t]` bracket trick, which stops grep matching its own command.
Kill leftover `multiprocessing.spawn` workers too, or they hold GPU memory.

**4. Long runs need `docker exec -d` + redirect.** Foreground execs die with
the shell. Pattern used throughout:

```bash
docker exec -d -w /home/odexter/neurosim <cid> bash -c \
  "PYTHONPATH=/home/odexter/neurosim/src /opt/conda/envs/neurosim/bin/python -u \
   applications/rl/train_sb3.py --experiment-config <cfg> \
   --pretrained-encoder <ckpt> --no-wandb --run-name <name> \
   > outputs/rl/train_<name>.log 2>&1"
```

GPUs: 8x Tesla V100-SXM2-32GB. Configs put the learner on GPU 0
(`device: cuda:0`) and env render workers on GPU 1 (`envs_on_gpu0: 0`), which
still works but now leaves six idle GPUs -- there is far more headroom for
parallel gate/baseline runs than any config in this repo assumes. V100s are
sm_70: no bf16, and slower per-GPU than the A5000s for the CNN work, so
**timings quoted in this file (e.g. "~4 h" for the encoder) are from the old
host and have not been re-measured here.**

---

## 2. The task

Quadrotor flies a nominal minimum-snap trajectory through a Habitat scene
(`skokloster-castle`) while spheres are thrown at it. The policy commands a
3-D offset from the nominal path. It perceives obstacles **only** through an
event camera (`obs_mode: combined` = event tensor + own state); the critic
additionally gets a 9-D privileged channel (asymmetric actor-critic).

Key numbers, all verified this session:

- obstacles: r = 0.15 m spheres, `kinematic_speed_mps: 5.0`, spawn 5.5-7.5 m out,
  lead-aimed with 0.25 m scatter + 0.3 m noise, up to 3 concurrent
- vehicle: `agent_radius` 0.15 m -> **contact at 0.30 m centre-to-centre**
- policy rate 20 Hz (`control_rate: 100`, `policy_decimation: 5`)
- episodes ~157 steps; ~2.45 genuine encounters each
- threat window `threat_time_horizon_s: 1.5`

### The units trap that burned two hours

`clearance` in the code is **SURFACE-TO-SURFACE**, not centre-to-centre:

```python
# src/neurosim/rl/env_reactive_dodge.py, _obstacle_relative_states
"clearance": float(np.linalg.norm(rel_pos) - combined_radius)  # 0.15 + 0.15
```

The 0.30 m contact distance is already subtracted, so **`clearance == 0` IS
contact**. I initially read `dodge_clearance_m: 0.02` as "scoring collisions as
clears" — wrong; it meant "missed by >= 2 cm", a genuine but razor-thin miss.

Two thresholds, easily confused:

- `clearance_threshold_m: 1.5` — the **encounter denominator gate** (did this
  obstacle count as a threat at all?) and the shaping-potential horizon
- `dodge_clearance_m` — the **actual clear threshold**

Also: **`authority_used_in_threat` is a misleading metric.** Its denominator is
`max(offset_max_m) = 3.0 m`, which needs ~2.5 s to reach, against a 1.5 s
window. It cannot exceed ~0.24 even for a perfect policy. I repeatedly read
0.12-0.15 as "the policy is barely trying" when it was running at roughly half
of the achievable maximum. Normalise by reachable displacement, not by
`offset_max_m`.

---

## 3. The validated config: `velocity_dodge_dynamic_v20_effnet.yaml`

This is the baseline to build on. It is the only config this session that
**passed the gates**. Its changes from v19:

| parameter | old | new | why |
|---|---|---|---|
| `dodge_clearance_m` | 0.02 | **0.10** | 0.02 sat just under the non-collision ceiling; set from measured oracle distribution |
| `offset_rate_limits_mps` | [0.8,0.8,0.5] | **[1.5,1.5,1.0]** | flip data was stale (see below) |
| `offset_command_tau_s` | 0.2 | **0.1** | lag was 23% of the threat window |
| `action_filter_tau_s` | 0.15 | **0.05** | same |
| `offset_accel_limit_mps2` | 1.5 | 1.5 | user-specified constraint, untouched |

Reachable lateral displacement went 0.707 m -> **1.275 m** in the 1.5 s window.

### Stale-comment trap

The `offset_rate_limits_mps` block still contains long measurements about the
vehicle "flipping past 109 deg on 4/4 seeds" at 1.20 m/s. **Those are from the
OLD control path** (`cmd_ctbr` + a residual mode shifting the POSITION
reference). Commit 5225691 moved to `cmd_vel` + `velocity_command`, where
`F_des = m*(-k_v*(v - cmd_v) + g)` has no position term, and measured
`out_of_bounds` 3-7 -> **0**. The flip mechanism is gone. The user spotted
this; I had initially refused to raise the limit on the strength of the stale
comment. **Re-read comments against `git log -S` before trusting them.**

### Gate results (v20, n=40, `dodge_clearance_m: 0.10`)

| arm | return | success |
|---|---|---|
| control (do-nothing) | -124.3 | 12.5% |
| const+0.40 | -143.0 | 15.0% |
| const+0.80 | -186.1 | 15.0% |
| **weave0.4@0.5Hz** | **-79.3** | 17.5% |
| **weave0.8@0.5Hz** | -99.5 | **32.5%** |
| **privileged oracle** | **-10.6** | **52.5%** |

Oracle beats the best blind arm on **both** metrics (+68.7 return, +20 pts
success) — wider than the v11 gate that authorised earlier training.

**To claim perception is used, a policy must beat -79.3 return AND 32.5%
success** — each blind arm on its own strongest metric. Nothing this session
did. Re-run with:

```bash
python return_baseline.py applications/rl/configs/velocity_dodge_dynamic_v20_effnet.yaml 40
```

Standing rule (from prior sessions, still correct): **run oracle feasibility,
the do-nothing control floor, and BOTH blind baselines before training any
dodge config.** Any change to `dodge_clearance_m` invalidates all of them.

---

## 4. What was tried and ruled out (v16 -> v22)

All flat or regressing. None beat the blind baseline.

| run | change | result |
|---|---|---|
| v16 | `n_steps: 128`, EfficientNet-B0 | flat -165 / 0.41 for 294k steps, EV ~0 |
| v17 | `n_steps: 256` (rollout > episode > gamma horizon) | EV -0.14 -> +0.05, still flat 620k steps |
| v18 | `sde_sample_freq` 20->8, `w_encounter_clear` 8->30 | `encounter_clear_rate` **0.696 in both v17 and v18**, identical to 3 digits |
| v19 | freeze encoder for PPO | `distance_r2` 0.136 -> 0.30, behaviour unchanged/slightly worse |
| v20 | task fix (section 3) | best -88.3 / 0.340, then declined |
| v21 | v20 + freeze | -97.9 / 0.320 — worse than v20 |
| v22 | EV-gated actor release | **EV never reached 0.08**; actor never released; declined anyway |

### The two experiments that actually decide things

**v18 falsified the incentive hypothesis.** Quadrupling the payout for clearing
an obstacle (8 -> 30) changed `encounter_clear_rate` by nothing at all. A
policy that will not dodge harder for 4x the reward is not trading off
incentives.

**v19/v21 falsified the perception hypothesis, twice.** Freezing preserved the
representation 2-5x better and produced no behavioural gain, on both the broken
and the corrected task.

**v22 identified the real wall.** With release gated on measured
`explained_variance >= 0.08` for 3 consecutive updates:

```
EV mean -0.039, max +0.190, only 3/24 updates >= 0.08, longest streak 1
```

The critic cannot reach a usable explained_variance, so PPO never had a
trustworthy advantage signal. This is consistent across v16-v22.

### The EV ceiling probe (`applications/rl/ev_ceiling_probe.py`)

150 episodes, held out by whole episodes, best causal predictor of
discounted return-to-go:

| predictor | EV |
|---|---|
| constant (reference) | 0.000 |
| PPO critic as trained | **0.030** |
| privileged(9) ridge | 0.042 |
| **state+priv ridge (causal ceiling)** | **0.137** |
| state+priv+HINDSIGHT | 0.523 |

Between-episode share of return variance: 45%.

So the critic underperforms its own ceiling by 4.5x, AND the ceiling is low.
**Known defect in this probe:** the MLP rows returned -1.26 and -0.67, i.e.
worse than a constant — that is overfitting (400 epochs, 256 wide, ~14k
training rows), not a finding. Only the ridge numbers are usable. Redo the MLP
with regularisation before anyone leans on the 0.523.

### Still unexplained — do not paper over this

v20, v21 AND v22 all improve to ~35k steps then decline. I attributed this to
PPO's actor updates; **v22 declined with the actor frozen the entire run**
(-96.8 -> -134 reward, 0.310 -> 0.240 success), so that explanation is wrong.
Remaining suspects: the aux task drifting the encoder (which shifts a frozen
actor's inputs), or `VecNormalize` statistics drift. Not diagnosed.

---

## 5. Code changes made this session (all uncommitted)

**`applications/rl/train_sb3.py`**

- **Memory-leak fix in `PreTanhPenalty._hook`.** `output.register_hook(lambda g: g + scale * output.detach())` created an uncollectable cycle
  (output -> hooks -> closure -> output) holding multi-GB graphs; it leaked
  exactly +4215 MiB per rollout once the actor unfroze. Fixed by binding the
  detached value as a default argument. **Do not reintroduce a closure over
  `output` in an autograd hook.**
- **`DistanceAuxCallback` now trains BOTH towers.** It previously built an
  optimizer over `pi_features_extractor` only, so with
  `share_features_extractor=False` the critic's CNN sat at random init forever.
- **`ActorFreezeWarmupCallback` is now EV-gated.** It always printed
  "actor released after N updates (explained_variance now usable)" but never
  read explained_variance — a bare counter making a false claim. New config
  keys: `actor_release_explained_variance`, `actor_release_patience`,
  `actor_freeze_max_updates`. The max-updates fallback prints explicitly
  *without* the "usable" claim.
- `CudaMemoryProbe` bracketing rollout/train phases; `--pretrained-encoder`.

**`src/neurosim/rl/sb3_features.py`** — `event_backbone` selector on
`_EventBackbone`: `small` (~26k params), `large` (~250k), `efficientnet_b0`
(ImageNet-pretrained trunk, stem re-convolved for 2-channel input, classifier
stripped, SpatialSoftmax head to preserve position).

**New scripts:** `ev_ceiling_probe.py`, `pretrain_encoder.py`,
`pretrain_encoder_oracle.py`, `pretrain_encoder_oracle_vec.py`.

**Abandoned:** RecurrentPPO / `AsymmetricRecurrentActorCriticPolicy`. sb3-contrib's
RecurrentPPO retains ~282 MiB per update, permanently, batch-independent
(verified identical at 8 and 4 envs, so fewer envs leaks *faster*). Nothing in
configuration survives it over millions of steps.

---

## 6. Literature review and the recommended plan

Reviewed at the user's request because "it is hard to believe nobody has
tackled this". They were right, and the finding redirects the project.

**Essentially nobody trains dynamic dodging end-to-end with RL from raw
vision. Every working system decouples perception from control.**

| system | perception | control | result |
|---|---|---|---|
| [EVDodgeNet](https://ar5iv.labs.arxiv.org/html/1906.02919) | 3 nets: deblur, ego-motion homography, segmentation+flow | geometric planner | 76-86% |
| [Falanga, *Sci. Robotics* 2020](https://www.science.org/doi/10.1126/scirobotics.aaz9712) | ego-motion compensation + clustering | model-based, 3.5 ms | obstacles to 10 m/s |
| [Threat-Aware RGB-D](https://arxiv.org/pdf/2511.22847) | detection + trajectory prediction | optimisation planner | real throws |
| [Flying in Highly Dynamic Env.](https://arxiv.org/pdf/2503.14352) | **pre-computed 36x36 obstacle map** | PPO | 100% @ 10 dyn obstacles |
| [Approximate IL for Event Flight](https://arxiv.org/html/2603.07578v2) | EfficientNet-B0 + GRU, **aux supervision only** | **BC/DAgger from PPO teacher** | 100% sim, 9.8 m/s real |

Even the PPO one is not fed raw sensor data — perception is solved outside the
learning loop.

### We were misreading our own cited reference

The `freeze_extractor_for_ppo` comment claims arXiv 2603.07578 has "the encoder
trained by auxiliary supervision and the decoder discarded afterwards". Checked
against the paper:

- it is **imitation learning, not RL** — a PPO teacher on **privileged state**,
  then a student cloned via BC + DAgger
- the encoder gets **auxiliary supervision only, never a policy gradient**
- the **decoder is NOT discarded**, it is fine-tuned online; the **encoder** is
  what stays frozen — we had it backwards
- their aux target is a **1-D angular distance map** (LiDAR-like), not our 2-D
  occupancy grid
- their encoder has **two GRU layers** after stages 3 and 4
- their ablation: BC 80%, **DAgger 30%**, their method 100%

This session was spent trying to make PPO's reward train an event encoder —
the one thing none of these papers attempt. The EV ceiling measurement explains
why it cannot work here.

### Recommended plan

1. **Regenerate the pretrained encoder** (lost with `outputs/`). ~4 h.
   `pretrain_encoder_oracle_vec.py` is the fast path — oracle-driven, 8 envs,
   verified byte-identical to the single-env path by `verify_oracle_wrapper.py`
   (`action diff max=0.000000`, `position diff max=0.000000`). It reached
   R^2 0.64-0.71 at 512k env-steps. The small CNN plateaus at 0.10-0.15 no
   matter how much data, so **encoder capacity is the lever** — keep B0.
2. **Collect oracle demonstrations** on v20. ~2.5 h for 400 episodes, ~27 GB.
   `evaluate_velocity_dodge_oracle.py --dataset outputs/rl/bc/oracle_v20.h5`.
   At 52.5% oracle success expect ~210 usable episodes / ~31k frames.
   **Watch disk**: these are ~67 MB/episode.
3. **Train BC**, not DAgger. `train_velocity_dodge_bc.py` already has
   `--distance-aux-weight` (the aux supervision) and
   `--zero-event-counterfactuals` (the does-it-actually-use-events test) built
   in. DAgger scored *worse* than BC in the reference's own ablation, so it is
   not the cheap first move despite being more sophisticated.
4. Only if plain BC underperforms: add the GRU layers (available again — no
   RecurrentPPO, so no leak) and try the 1-D angular distance-map aux target.

---

## 7. Crucial next steps, in order

> **SUPERSEDED 2026-08-29 (next session) — see 7a below.** Steps 1 and 3 are
> done. Steps 2 and 4 are on hold: the reference paper was re-read at source
> and it implies a precondition this project has never tested, which both of
> those steps depend on. The original list is kept for the record.

1. ~~**`git add -A && git commit`.**~~ Done, commit `5d5107c`.
2. **Recreate `outputs/rl/` and regenerate the pretrained encoder** (step 1
   above). Nothing in the BC plan runs without it. — **ON HOLD, see 7a.**
3. ~~**Re-run the v20 gates.**~~ **Done — all six arms reproduce EXACTLY**
   (`outputs/rl/gate_v20_rerun.log`), on a different host, a rebuilt CUDA
   event simulator and a re-downloaded scene mesh:

   | arm | section 3 | re-run |
   |---|---|---|
   | control | -124.3 / 12.5% | **-124.3 / 12.5%** |
   | const+0.40 | -143.0 / 15.0% | **-143.0 / 15.0%** |
   | const+0.80 | -186.1 / 15.0% | **-186.1 / 15.0%** |
   | weave0.4@0.5Hz | -79.3 / 17.5% | **-79.3 / 17.5%** |
   | weave0.8@0.5Hz | -99.5 / 32.5% | **-99.5 / 32.5%** |
   | oracle | -10.6 / 52.5% | **-10.6 / 52.5%** |

   Section 3's numbers are trustworthy and the gate to beat stands unchanged:
   **-79.3 return AND 32.5% success**. The environment is deterministic
   across hosts given the seed, which also means any future disagreement with
   these numbers is a real change, not drift.
4. Then the BC pipeline (steps 2-4 above). — **ON HOLD, see 7a.**

---

## 7a. The precondition, and why steps 2/4 are on hold

`https://arxiv.org/html/2603.07578v2` was re-read at source rather than from
section 6's summary, because that summary had already been wrong once. Two
numbers it omits change the plan:

```
teacher = PPO on PRIVILEGED STATE, success 1.00
student = BC 0.80 | DAgger 0.30 | BC+DAgger 1.00 | approximate-IL 1.00
```

Two consequences.

**1. Our teacher is too weak for the pipeline to produce a readable result.**
Every system in section 6 has a control module that solves the task from
clean obstacle state *before* perception is attached. Ours is a hand-designed
planner at 0.525. A BC student retaining 80% of it lands near **0.42 against
a 0.325 blind baseline** — inside the noise band, for reasons that have
nothing to do with perception. The 4 h encoder job and the 2.5 h demo
collection are both oracle-driven, so both inherit this. Running them as
written cannot produce a measurable perception result whatever the outcome.

**2. Plain BC caps at 0.80 of the teacher.** Section 6's "train BC, not
DAgger" is right about DAgger alone (0.30) but wrong about the ceiling.
Reaching 1.00 needed the online phase, and the paper's trick there is an
**approximate student** — a state-based MLP trained to match the event
student's features and actions — so the online phase never renders events.
That is a **28x** speedup, 52.44 h -> 1.86 h. Habitat event rendering is this
project's slowest loop (~7 env-steps/sec single-env), so this is worth
adopting regardless of everything else.

### The test now running

`velocity_dodge_privileged_teacher_v1.yaml` — v20 with the task, rewards,
obstacles and control untouched, and only the actor's *view* changed. It
asks: **can any policy solve this task given perfect obstacle knowledge?**

Observation is **54-dim**, feedforward `MlpPolicy` (verified by loading the
running checkpoint: `Box(54,)`, first layer `in_features=54`):

```
18  proprioception  v_body, v_ref_body, pos_err_body, offset_body,
                    omega_body, prev_action
36  privileged      3 slots x [valid, rel_pos(3), rel_vel(3),
                               rel_accel(3), clearance, tca]
```

Three slots because `max_concurrent: 3`, so the channel is lossless. The
old single slot hid two obstacles AND silently switched identity whenever
`obstacle_threat_priority` reordered.

**Acceleration is required for sufficiency, not a nicety.** One of the three
obstacle templates is `kinematic_parabola` at 3.0 m/s^2, curving ~2.16 m over
a 1.2 s flight against a 0.30 m contact radius. Position and velocity at one
instant cannot separate a line from a parabola, so without it a feedforward
policy provably cannot infer which it faces — the channel would be
"privileged" but not Markov, and a FAIL could not be attributed to the task
rather than to the encoding.

### Why the reference's distance map does NOT transfer

Checked at source, not from a summary: arXiv 2603.07578's obstacles are
**static trees**. Its teacher is a feedforward MLP reading a 10-bin angular
distance map (11.25 deg each, 120 deg FOV) at a **single instant**, and its
two GRU layers sit **only in the student's event encoder**.

A snapshot distance map is sufficient there because every bit of relative
motion comes from ego-motion, which the teacher already observes. Ours move
at 5 m/s, so that map has no representation for closing speed at all —
adopting it would strictly REMOVE information and make the task unsolvable
feedforward. Their GRUs compensate for a poor sensor, not for a moving world.

The lesson generalises: copy a reference's *structure* (decouple perception
from control; teacher on privileged state; student cloned with aux
supervision), not its *state encoding*, which is a function of its task.

The GRU is still likely relevant later — on the **student**, where a single
event frame genuinely cannot carry depth or closing speed.

- **reaches ~1.0** — the task is sound. It becomes the BC teacher in place of
  the geometric planner, and the planner's specific failure stops mattering.
  Then proceed to BC + the approximate-student online phase.
- **plateaus ~0.5** — the finding is about the **task**, not perception, and
  every event-based result from v16 on was measuring a ceiling rather than a
  perception gap. That would be the most important thing this project could
  learn, and it would explain v16-v22 without appeal to encoders at all.

Beware: `velocity_dodge_privileged_baseline.yaml` is **not** this test
despite its name — it has `privileged_actor: false`, `obs_mode: combined`.
The name is a fossil and section 5's description of it is stale. This is the
first privileged-**actor** run on the task as corrected by commit 5225691.

### The EV ceiling, re-measured on the best possible information (2026-08-30)

`ev_ceiling_probe.py` against the privileged teacher's own 54-dim state --
three obstacle slots with position, velocity AND acceleration, i.e. a
lossless Markov view. 150 episodes, held out by whole episodes:

| predictor | EV(test) |
|---|---|
| constant (reference) | 0.000 |
| PPO critic as trained | **-0.168** |
| state+priv(54) ridge | 0.012 |
| state+priv(54) MLP (regularised) | **0.062** |
| state+priv+HINDSIGHT MLP | **0.819** |

between-episode share of variance: **69.2%**

**EV ~ 0 is CORRECT BEHAVIOUR, not a broken critic.** The return is highly
predictable (0.819) but only from FUTURE facts -- `steps_remaining` and
`crashed` -- that no causal predictor can have. From the present state, with
ground truth about every obstacle, a regularised nonlinear fit reaches 0.062.

This measurement contains no perception, no encoder, no expert. So the
v16-v22 story has an explanation that never mentions any of them: PPO's
advantage is `return - V(s)`, and if V can explain only ~6% of return
variance then the policy gradient is dominated by episode-level noise --
actions are credited for outcomes largely fixed by the episode's draw.

Caveats, stated so nobody over-reads it: the checkpoint is early and weak
(~15-20% success) and a stronger policy would vary less in episode length --
though the oracle itself only reaches 52.5%, so length variance is large
under every policy available here. 30 test episodes is a modest sample.

**Note this supersedes the old 0.137 / 45% figures**, which were measured on
the 9-dim single-obstacle channel AND with the defective unregularised MLP
(that version returned -1.26 and -0.67, worse than a constant). The probe's
MLP is now dropout + weight decay, early-stopped on a validation split held
out by episode.

If the decomposition shows `steps_remaining` alone carries the hindsight EV,
the return is largely a survival-time proxy and the culprit is early
termination rather than the reward's obstacle terms -- note
`crash_penalty_per_remaining_step: 0.6` charges forfeited steps explicitly,
injecting episode length straight into the return.

### THE FIX, and its verification (2026-08-30)

The ceiling above was traced to the **discount horizon**, not the reward's
obstacle terms. `gamma: 0.994` gives 1/(1-gamma) = **167 steps**, longer than
the ~150-step episode, so the critic was asked to predict survival across
obstacles that had not spawned yet (`spawn_interval_s: 0.30`, continuous
arrival). That is irreducible noise, not a modelling failure.

`velocity_dodge_privileged_teacher_v2.yaml` changes exactly two things:

| | v1 | v2 | why |
|---|---|---|---|
| `gamma` | 0.994 | **0.97** | horizon 167 -> 33 steps = 1.65 s, matched to `threat_time_horizon_s: 1.5`. The expert's whole plan (0.55 s rise + 1.0 s return) fits inside it. |
| `crash_penalty_per_remaining_step` | 0.6 | **0.0** | literally proportional to remaining length; charged ~79 on top of the flat 150 at a typical collision |

**Verified BEFORE committing GPU time** (both probes, same checkpoint, ~15 min):

| measurement | v1 | v2 |
|---|---|---|
| state+priv(54) ridge | 0.012 / -0.015 | **0.184** |
| state+priv(54) MLP | 0.062 / -0.033 | **0.231** |
| between-episode variance share | 79.4% | **49.0%** |
| corr(return, length) | +0.958 | +0.897 |
| `r_encounter_clear` share of return variance | 17.3% | **25.8%** |

~~The causal ceiling went from indistinguishable-from-zero to **0.231**~~

> **RETRACTED, same day.** Re-running the identical probe and config against
> the v2 policy's own checkpoint gave ridge **-0.021**, MLP **0.011**. Across
> four runs the state-ceiling estimate reads -0.033, 0.062, 0.231, 0.011 --
> a spread of ~0.26 on a quantity that was quoted to three decimals off a
> 30-episode test set. Either the ceiling depends on which policy generates
> the rollouts, or the 0.231 was noise; a 400-episode reproduction is running
> to separate those.
>
> **What survives the retraction, because it is stable across every run:**
> `steps_remaining` alone explains 0.77-0.92 of return-to-go variance in
> BOTH discount settings, so return is a survival-time proxy regardless of
> gamma. And the between-episode share of variance dropped from 69-79% at
> gamma 0.994 to 46-49% at 0.97, consistently. The gamma change did something
> real to the variance structure. It has NOT been shown to raise what a
> causal predictor can reach.
>
> Method note for whoever reads this next: a 30-episode held-out set cannot
> resolve an EV difference of 0.1. Quote this probe with a repeat, or do not
> quote it to more than one significant figure.

Attribution matters here: the crash-penalty change moved the undiscounted
correlation only modestly (0.958 -> 0.897); **gamma is what lifted the
critic's actual target**. Do not credit the wrong knob.

**v1 control result**: 1M steps, 30 evals, overall mean 18.25%, last 8
20.6% -- below the 32.5% blind arm, on a lossless Markov obstacle channel.
Exactly what a ~0 ceiling predicts.

**v2's own curve, so far (500k / 2M, i.e. 25% of budget): mean 14.2% over
15 evals, NOT ahead of v1's 17.2% over its first 15.** A rise to 25% at
229k was flagged in-session as "the first monotonic-looking trend" and
should not have been -- it was followed by a drop to 5% and a return to the
same noisy 5-25% scatter v1 showed throughout. This is the identical mistake
v14 made and that this file already warns about (13.2/9.0/23.0/26.6% by
quarter, every early plateau/trend call wrong). Do not read this curve again
before 2M, per the pre-registered criteria above.

**v2's gate, re-measured** (`outputs/rl/gate_v2_rerun.log`, 40 episodes):

| arm | v1 return | v2 return | success (identical) |
|---|---|---|---|
| control | -124.3 | -59.1 | 12.5% |
| const+0.40 | -143.0 | -78.7 | 15.0% |
| const+0.80 | -186.1 | -119.6 | 15.0% |
| **weave0.4@0.5Hz** | -79.3 | **-28.2** | 17.5% |
| **weave0.8@0.5Hz** | -99.5 | -40.2 | **32.5%** |
| **oracle** | -10.6 | **+19.0** | **52.5%** |

**GATE FOR v2: beat -28.2 return AND 32.5% success.**

Every success rate AND every termination count is identical to v1, so the
reward edit rescaled scoring without perturbing a single episode outcome --
which is why the pre-registered SUCCESS thresholds carry over unchanged. The
oracle is now the only arm with positive return; its margin over the best
blind arm narrowed from +68.7 to +47.2 return while holding +20 points of
success.

### Why the oracle is weak (measured, so it need not be re-litigated)

`oracle_magnitude_probe.py` + `oracle_commit_probe.py` on v20:

| quantity | measured | verdict |
|---|---|---|
| smallest displacement that clears | median 0.28 m, **max 0.35 m** | magnitude is NOT the limit |
| encounters needing > 1.00 m (menu cap) | **0%** | widening the menu is pointless |
| commit lead time | 0.85 s vs 0.64 s required | commits in time |
| encounters committing late | **0%** | not a timing failure |
| direction flips | **0%** | no thrash |
| **achieved by closest approach** | **0.23 m** vs 0.28 m needed | **undershoots** |

It commits in time, picks a side, holds it — and still arrives 0.05 m short.
(Commit-probe sample is only 4 encounters — directional, not settled.)

`oracle_prediction_probe.py` (40 episodes, reproducing the oracle at 14/40
collisions) then supplied the mechanism, and it is **two** failures of
roughly equal weight, not one:

| quantity | measured |
|---|---|
| plans made / **failed** | 120 / **666 → 84.7% failure** |
| `peak_offset_m` | median **0.35 m** (the smallest candidate), 1/68 near the cap |
| `rise_time_s` | median **1.07 s** — against 0.85 s of commit lead |
| predicted min clearance | +0.298 m |
| **achieved** min clearance | **+0.202 m** |
| predicted clear, made contact | 7/68 (10%) |
| collisions | **7 with a plan, 7 with no plan** |

1. **Feasibility.** 85% of planning attempts find no admissible curve, and
   half the collisions are on obstacles that never got a plan. This does
   *not* contradict `oracle_magnitude_probe.py` finding every encounter
   clearable by a 0.28 m constant displacement: that is pure geometry on the
   nominal path, while the planner must fit a dynamically feasible quintic
   *from its current offset and velocity* clearing every obstacle by
   `safety_margin_m` inside the speed envelope. Usually no such curve exists.

2. **Timing.** `rise_time_s` 1.07 s exceeds the 0.85 s commit lead, so the
   bump **peaks after closest approach** — the vehicle is partway up when the
   obstacle arrives. That is the 0.35 m commanded / 0.20 m achieved
   shortfall, and why 10% of encounters it predicted clear end in contact.
   `minimum_rise_time_s` is only a 0.55 floor; the realised 1.07 comes from
   the magnitude/speed schedule, so lowering the floor will not fix it.

Both are structural to **one-shot open-loop planning** — a single curve per
encounter, predicted against an idealised model, with no feedback on realised
displacement. Neither is a constant worth tuning. A closed-loop policy
trained on the actual dynamics has neither problem by construction, which is
what the precondition test above is for.

Also measured: the task is **not degenerate** — mean escape fraction 0.380,
0/16 degenerate encounters, so a blind guess clears 38%, consistent with the
32.5% blind arm. And `escape_set_probe.py`'s "12% impossible" is an artifact
of its single cross-track axis; sweeping the full sphere clears all 16.

### Considerations / open questions

- **The oracle is weak and is the ceiling.** It collides in 13/40 episodes
  (32.5%) and uses only ~23% of now-available authority
  (`peak_cross_track` 0.295 m median vs 1.275 m reachable). Its planner targets
  minimum-sufficient dodges. **Cloning a weak expert caps the student at a weak
  policy** — fixing the expert's planner is probably higher-value than any
  student-side tuning, and should likely happen before step 2.

  > **Correction (next session).** The claim that the planner "has NOT been
  > rescaled since the rate limits were raised" is wrong, and it is the same
  > read-the-comment-not-the-code mistake this file warns about at the end.
  > `TrajectoryExpertConfig` *was* rescaled: `max_horizontal_speed_mps: 1.5`
  > and `max_vertical_speed_mps: 1.0`, with a comment saying they are matched
  > to `offset_rate_limits_mps`. What was left behind is the *magnitude* menu,
  > `candidate_offsets_m = (0.35, 0.50, 0.65, 0.80, 1.00)`, which still caps at
  > 1.00 m against 1.275 m now reachable — and since candidate cost goes as
  > `magnitude**2`, the planner takes the smallest sufficient dodge regardless.
  > Whether that cap actually binds is a measurement, not an argument:
  > `oracle_magnitude_probe.py` reports the smallest displacement that clears
  > each encounter, over a full sphere of directions.

  > Also note the encoder plan inherits this. `pretrain_encoder_oracle_vec.py`
  > drives data collection with `oracle_action`, so the ~4 h encoder job is
  > sampled from the expert too — not just the BC demos in step 2. Fixing the
  > expert after regenerating the encoder means regenerating it again.
- **Apply the right feasibility framework.** My kinematic sketch was ad-hoc;
  [Falanga RA-L 2019](https://rpg.ifi.uzh.ch/docs/RAL19_Falanga.pdf) derives
  `tau_max = (R - r_obs - d_safe) / v_max` for tolerable perception latency.
  Use it to check the 1.5 s window against 5 m/s obstacles properly.
- **Return vs success are different objectives** and diverge here — `weave0.4`
  is the best blind arm on return while `weave0.8` is best on success. PPO
  optimises return. Scoring on success alone produced a full day of wrong
  "task is degenerate" conclusions in an earlier session. Always report both.
- **`w_along_track` guards a real exploit.** It prices braking-as-evasion (a
  ballistic obstacle aimed for a fixed arrival time simply misses if you lag).
  Do not gate it during threat windows; the intended dodge is cross-track,
  which it does not charge.
- **The encounter denominator is endogenous.** An obstacle counts only if it
  came within `clearance_threshold_m`, a quantity the policy controls, so
  dodging well *deletes* obstacles from the denominator. Watch
  `encounters_total` (steady ~2.45) whenever the clear bonus is raised.

### Method notes for whoever continues

Three of my wrong conclusions this session came from the same mistake:
**reading a parameter's meaning off its name or its comment instead of tracing
the computation.** `clearance` units, `authority_used_in_threat`'s denominator,
and the stale flip measurements were all avoidable by reading the code and
`git log -S` first. Trace before concluding.

And write measurements into the config comments as you make them. That habit is
the only reason this session's findings survived `outputs/` being deleted.

---

# ARCHIVE — historical notes below (superseded, see section 0 above)


## Current continuation — 2026-08-11

The older notes below are historical. The project has since moved from the
privileged PPO diagnostic to a collision-checked local-trajectory expert and
event-policy imitation workflow.

Active workflow (container `neurosim-noros`):

- The 80-episode separated multi-encounter expert collection finished at
  97.5% success. Its cleanly closed streaming dataset is
  `outputs/rl/datasets/trajectory_expert_multi_train80_v10.h5`: 77 admitted
  episodes, 57,827 samples, 34,849 active-dodge samples. Two controller
  tracking failures and one successful episode containing failed planning
  calls were excluded.
- Guard process: PID `3622665`, running
  `applications/rl/run_trajectory_expert_multi_bc_v10.sh 3620415`.
- The guard passed HDF5 verification and finished 20/20 clean no-obstacle
  timeouts (8,000 samples, zero expert plans). BC is now training for 20 epochs
  on 105,976 virtual samples: 57,827 expert base samples, 8,000 genuine quiet
  samples, 5,300 onset duplicates, and 34,849 zero-event counterfactuals. It
  will then evaluate normal, zero-event, and no-obstacle conditions. Do not
  launch a duplicate copy.
- Every container Python command needs
  `PYTHONPATH=/home/odexter/neurosim/src`.

The deployed control mode is `gated_cascaded_velocity`. The actor predicts a
gate plus body-frame velocity correction. The controller manually computes
nominal return velocity from position error; while the gate is open, the
nominal restoring term is suppressed so it does not fight the local avoidance
trajectory. The expert plans smooth quintic departure/return offsets, checks
the full curve against the static Habitat mesh, and swept-validates dynamic
clearance. Its velocity label includes feedback from actual displacement to the
local plan (not to the conflicting nominal path), so DAgger labels remain
corrective on policy-visited states. Newly actionable overlapping obstacles
trigger continuous replanning from actual offset and velocity.

Dataset admission is strict: ordinary expert data includes only successful
episodes with zero failed planning calls. Quiet timeouts are admitted only by
the explicit `--include-all-episodes` quiet collection. HDF5 event frames are
FP16 and read lazily; a live-data GPU-forward smoke test passed.

The event actor receives four-frame event history and the following 19-D state:
body velocity, body-frame reference velocity, body-frame position error,
body-frame current offset, body angular rate, and previous gated action.
Obstacle geometry remains critic-only. Presence-strength and high-resolution
event features are enabled. BC uses balanced threat/quiet sampling, a gated
loss, onset duplication, genuine quiet trajectories, and zero-event
counterfactuals.

Do not accept normal success alone. After v10 finishes, require evidence of:

1. strong held-out normal success;
2. a substantial same-seed degradation with events zeroed;
3. all no-obstacle runs timing out normally with near-zero gate/action and low
   tracking error;
4. useful threat activation and direction agreement in oracle diagnostics.

If perception/quiet gates fail, iterate with targeted imitation or DAgger.
Use PPO only after the actor demonstrates event-conditioned supervised skill.
`collect_velocity_dodge_dagger.py` now resets the trajectory expert on every
episode and supports streaming `.h5` output; prefer that over NPZ for event
rollouts, and do not retain episodes with expert planning failures.
If gates pass, next run the intentional overlap config and apartment-only
held-out config, then broaden with the two-scene config:

- `velocity_dodge_trajectory_expert_multi_overlap.yaml`
- `velocity_dodge_trajectory_expert_multi_separated_apartment.yaml`
- `velocity_dodge_trajectory_expert_multi_separated_two_scene.yaml`

The worktree is intentionally dirty and uncommitted. Preserve unrelated user
changes and use `apply_patch` for edits.

Status as of 2026-08-10, ~01:20 container time. **No training is running.**
The last run (`velocity_dodge_ctbr_v5`) was deliberately killed after a
config/code change and a v6 was *not* launched, per explicit instruction —
resume by launching it yourself (command below) once you've read this.

## Environment gotcha (read this first)

`docker exec` into the `neurosim-noros` container resolves the `neurosim`
package to a **stale, image-baked copy at `/opt/neurosim/src`** (dated
March), not the bind-mounted source tree you'll be editing at
`/home/odexter/neurosim/src`. `pip show neurosim` confirms the editable
install points at `/opt/neurosim`. Every `docker exec` that runs Python
**must** set `PYTHONPATH=/home/odexter/neurosim/src` or it will silently
run old code — this cost a full training run earlier in this session before
being caught (an `ImportError` on a function that didn't exist yet in the
stale copy is what surfaced it). Standalone diagnostic scripts that do
`sys.path.insert(0, '/home/odexter/neurosim/src')` at the top are immune to
this regardless of `docker exec` flags.

```bash
docker exec -w /home/odexter/neurosim -e PYTHONPATH=/home/odexter/neurosim/src \
  neurosim-noros /opt/conda/envs/neurosim/bin/python -m pytest tests/ -q \
  --ignore=tests/test_asynchronous_simulator_cortex.py
```

The `--ignore` is for a pre-existing, unrelated `ModuleNotFoundError: cortex`
at collection time — not something this session touched.

## What this is

`velocity_dodge` is a reactive obstacle-dodge RL task: a quadrotor tracks a
random MinSnap trajectory through a Habitat scene while spheres are thrown
at it (solved to guarantee interception of the *nominal* path). The policy
outputs a body-frame velocity correction, integrated into a leaky-integrator
offset on top of the SE3-controller reference (see
`src/neurosim/rl/README.md` for the full policy I/O / control-mapping
writeup — that doc is accurate and didn't need changes today).

The active config, `applications/rl/configs/velocity_dodge_privileged_baseline.yaml`,
is a **diagnostic baseline**: `privileged_actor: true` hands the actor
ground-truth nearest-obstacle geometry directly (not derived from the event
camera), `obs_mode: state` (no events at all). Purpose, per the file's own
header: if this learns to dodge and a later event-based policy doesn't,
perception is the bottleneck, not the reward/intercept-design/control-
mapping. **This is intentional and not a bug** — someone will ask "why
aren't there events in the video" and the answer is this.

## What today's session did

Resumed a from-scratch rebuild of the `velocity_dodge` env (kept from an
earlier session), then chased down **six real, confirmed bugs** surfaced by
watching rendered eval videos and cross-checking against direct
measurements. All fixes are uncommitted (see `git status` / `git diff` —
nothing has been committed this session). In order found:

1. **Tip-over investigation (dead end, correctly abandoned).** Long chase
   of the SE3 controller intermittently flipping the vehicle past 90°. Root
   cause was never nailed down as a *bug* — measured as a rare (1–3/20),
   latent defect present in the **unmodified baseline** commit (`dd84df7`)
   too, i.e. pre-existing in stock `SE3Control` + `MinSnap`, not introduced
   by this work. Given up on fixing it and reverted several exploratory
   workarounds (a tilt guard, a `cmd_vel`-only control mode, an explicit
   position-cascade controller) that were built while chasing it — none of
   those are in the current diff. If you want to pursue this further, it's
   an orthogonal, standalone repro: stock `SE3Control` tracking a
   `MinSnap` path, no RL, no obstacles.

2. **Collision detection scored against a phantom point 1 m above the
   drone.** `DynamicObstacleManager` carried `agent_height` (default 1.0 m)
   inherited from Habitat's walking-agent convention (agent position =
   feet, sensors mount at `position + height`). This drone's "agent"
   position is already its true body center (set directly from
   `state["x"]`, confirmed via `BaseNeurosimRLEnv._on_episode_reset`), but
   the collision check (`has_agent_collision` in
   `src/neurosim/core/visual_backend/dynamic_obstacles.py`) subtracted
   `agent_height` from every obstacle's altitude before comparing, while
   the throw-aim solver (`_build_intercept_schedule` in
   `src/neurosim/rl/env_reactive_dodge.py`) added it back when choosing
   where to aim. Both were self-consistent with each other (which is why
   it went unnoticed) — throws converged on a point 1 m above the true
   body, and the collision check fired on that same phantom point.
   Measured directly: one collision fired at a raw obstacle distance of
   1.05 m, scored as 0.26 m. Fixed at four call sites: the collision check,
   the throw-aim solver, and two threat-metric functions
   (`_obstacle_relative_states`, `_nominal_counterfactual_collision`) that
   fed the same bug into reward shaping and the privileged-critic
   observation. Two tests in `tests/test_safety_checker.py` had the same
   offset baked into their fixtures and needed updating.

3. **Camera mount, same root cause, different call site.**
   `_create_camera_spec` in `src/neurosim/core/visual_backend/habitat_wrapper.py`
   *also* added `agent_height` to every color/depth sensor's local mount
   offset — including the event camera's internal color sensor and any RGB
   pane that mirrors it for video (`record_eval_videos.py` deliberately
   mirrors the event camera's pose). Measured directly against Habitat's
   own `agent.get_state().sensor_states`: Y-gap was exactly 1.000 m before
   the fix, 0.000 m after.

4. **Malformed MinSnap trajectories crashing mid-training.** MinSnap can
   return an object whose `t_keyframes` implies more segments than its
   polynomial arrays hold (degenerate waypoint geometry drops segments
   internally); it clips query time to `t_keyframes[-1]` rather than what
   the arrays actually cover, so the mismatch survives a successful build
   and only surfaces as an `IndexError` from whichever caller later happens
   to sample the bad tail — in practice, intercept-schedule solving at
   episode reset, which kills a vec-env worker and the whole run. Fixed by
   sampling the trajectory end-to-end at build time
   (`_validate_trajectory` in `src/neurosim/core/trajectory/habitat_trajs.py`),
   inside the existing resample-on-failure retry loop. Also handles the
   `traj.null` case (fewer than 2 distinct waypoints survive MinSnap's own
   dedup) cleanly instead of an `AttributeError`.

5. **Eval videos ran at 2x speed.** `record_eval_videos.py` computed
   playback fps as `world_rate / steps_per_action`, omitting
   `policy_decimation` (2 in this config) — one video frame is written per
   *policy* decision, which advances `policy_decimation * steps_per_action`
   world steps, not just `steps_per_action`. Fixed; confirmed the log line
   now reports 50 fps (the true decision rate) instead of 100.

6. **Every episode started at floor level, and flew too close to
   furniture.** Two compounding issues:
   - `sample_habitat_start` (base env, `src/neurosim/core/visual_backend/safety.py`)
     samples directly off the navmesh with zero height awareness, so the
     trajectory's `start` point (and therefore the whole episode's
     beginning) was always at floor height regardless of any height
     margin. Fixed in `env_reactive_dodge.py::_lift_off_floor`, which now
     delegates to the same margin-aware sampler used for every other
     waypoint (a first attempt that tried to preserve the exact start
     (x, z) and only adjust height failed silently on 2 of 3 test
     episodes, because that specific column wasn't navigable at any safe
     height — a doorway, a low overhang; fixed by resampling (x, z, y)
     together instead of trying to salvage one fixed column).
   - Even after that fix, the drone was flying at ~0.26–0.86 m above
     floor — within furniture height (a couch, per direct user
     observation of the rendered video). Root cause: the height-margin
     logic (`sample_random_navigable_point_with_height` in
     `habitat_trajs.py`) measured altitude against the scene's **global**
     AABB bounds, not the **local** floor height at each sampled point —
     wrong in general (floor height varies across a scene) and specifically
     wrong for "clear the furniture," since furniture sits on the local
     floor. Reworked to sample altitude relative to the local floor
     (`get_random_navigable_point()` already returns local floor height
     before its Y gets overwritten). Along the way, discovered
     `pathfinder.is_navigable(pt, max_y_delta=0.5)` has a **hard default
     0.5 m vertical snap tolerance** — it cannot validate a point more than
     0.5 m from the navmesh surface at all; 500 probed points all showed
     "navigable height" of exactly 0.500 m regardless of the scene's real
     layout, which is what made requiring 1.0 m altitude fail outright
     (100% "no navigable point found" on retry) rather than just
     succeeding less often. The re-validation at elevated height is
     therefore close to meaningless beyond that band and was gated off for
     offsets past ~0.45 m (see the docstring in `habitat_trajs.py` for the
     full reasoning). **Important caveat to carry forward:** there is no
     navmesh-based way to formally guarantee an elevated flight path is
     furniture-free — Habitat's navmesh here only encodes 2D floor
     walkability with that ~0.5 m vertical tolerance, not full 3D
     obstacle volumes. Flying at `min_altitude_m: 1.0` clears *most*
     furniture (verified: MinSnap's smoothed curve deviates only 0.10–0.42 m
     from its own already-validated waypoints, so it's not a
     corner-cutting problem) but isn't a proof of zero collision against
     tall furniture (bookshelves, cabinets).

Every fix above was verified with the full test suite (214 passed,
`--ignore=tests/test_asynchronous_simulator_cortex.py` for the pre-existing
unrelated failure) and, where relevant, a direct numeric re-measurement
(not just "tests pass") — e.g. re-running the exact collision-distance
probe before/after, re-measuring the camera sensor world position via
Habitat's own `sensor_states`, re-sweeping 20 episodes' altitude profiles.

## Verification, post all six fixes (numeric, not yet re-confirmed on video)

20-episode altitude sweep after fix #6 landed (script logic below, not
saved to the repo — recreate from this if the scratchpad script isn't
available to you, it won't be across sessions):

```python
# For each of 20 episodes: sample the nominal trajectory across its
# duration, convert to Habitat frame, compare against navmesh median.
```

Result: all 20 episodes flew between roughly 1.0–2.0 m above the scene's
median navmesh height (was 0.26–0.86 m before fix #6, i.e. squarely in
furniture range). No trajectory-build failures across the sweep.

**Not yet done: a fresh rendered video confirming this visually.** The
numeric measurement above is solid, but nobody has actually watched a video
with `min_altitude_m: 1.0` in effect. Do this before trusting it further —
see To-do below.

## Artifact (video comparisons)

<https://claude.ai/code/artifact/a4332b54-5ff2-4185-8272-881d7c6c553f> —
before/after comparisons for bugs #2, #3, #5 (collision height, camera
mount, video speed). **Does not yet include a comparison for bug #6**
(furniture altitude) — the numeric fix landed but a video was never
generated or published for it (training was killed for the config change
before that step, then this handoff was requested instead). That's the
first thing to do on resume.

## Current config state

`applications/rl/configs/velocity_dodge_privileged_baseline.yaml`, key
values as of now:

```yaml
env:
  policy_decimation: 2       # 100Hz controller / 50Hz policy
  obs_mode: state            # diagnostic: no events
  task:
    config:
      trajectory:
        v_avg: 0.6            # lowered from 1.0 for easier learning (user request)
        min_altitude_m: 1.0   # NEW semantics: above LOCAL floor, not global AABB
        ceiling_margin_m: 0.3
        lateral_margin_m: 1.0
        bounds_margin_m: 0.3
      residual_control:
        mode: velocity_delta_integrated
      privileged_actor: true
      privileged_critic: false
  dynamics:
    control_abstraction: cmd_ctbr
```

Full suite green against this config at every step above.

## Uncommitted changes

Nothing this session has been committed. `git status` shows 18 modified +
7 untracked files. The untracked files (never existed in `dd84df7`) are the
whole `velocity_dodge` task implementation: `src/neurosim/rl/tasks/velocity_dodge.py`,
`src/neurosim/rl/vehicles/velocity_rotorpy.py` (built earlier, currently
unused since the config runs `cmd_ctbr`, not `cmd_vel` — see the tip-over
section above for why), `applications/rl/record_eval_videos.py`,
`src/neurosim/rl/README.md`, two configs, one test file. Consider
committing once you're confident in the current state — ask first per this
repo's normal git-safety rules.

## Depth image question (asked, not implemented)

User asked whether the privileged actor/critic sees a depth image — no,
the privileged channel (`PRIVILEGED_DIM = 9` in `velocity_dodge.py`) is
purely geometric ground truth for the nearest obstacle (valid flag, rel
pos/vel, clearance, time-to-closest-approach), no pixels at all.
Recommended against adding depth *to the critic specifically* (it would be
strictly worse than exact ground truth the critic already has access to,
since depth still requires inferring geometry from pixels). Depth as an
*actor*-perception modality (alongside/instead of the event camera) is a
reasonable, separate idea if the user wants it — would need a new sensor
type, event-representation plumbing, and feature-extractor changes. Not
started.

## Not every obstacle collides with the nominal path — is that expected?

Raised by the user, not fully chased down before the furniture-altitude
report superseded it. Partial answer: `_build_intercept_schedule`
(`env_reactive_dodge.py`) will skip a scheduled throw entirely if no
unoccluded line-of-sight is found within `max_attempts` (8) — logged as
"no unoccluded throw found; skip this slot" — so *some* miss rate is
by design, not a bug. Whether the skip rate alone fully explains the
observed miss rate, or whether there's a second contributing cause, was not
verified. Worth a quick instrumented check (count skipped vs. scheduled
slots per episode) if it's still a live concern.

## To-do, in the order it makes sense to do it

1. **Generate a fresh nominal (zero-action) video with the current config**
   (`min_altitude_m: 1.0` etc. all in effect) and look at it — confirm the
   furniture-clearance fix visually, not just numerically. Use:
   ```bash
   docker exec -i -w /home/odexter/neurosim -e PYTHONPATH=/home/odexter/neurosim/src \
     neurosim-noros /opt/conda/envs/neurosim/bin/python applications/rl/record_eval_videos.py \
     --rollout-config applications/rl/configs/velocity_dodge_privileged_baseline.yaml \
     --nominal --episodes 4 --seed0 97000 \
     --out-dir outputs/rl/videos/v6_altitude_confirm
   ```
   Update the artifact (republish the same file path used earlier this
   session — ask the user for it or check `Artifact` tool's `list` action
   if you don't have the URL in context) with this as a new section.

2. **Launch training as `velocity_dodge_ctbr_v6`** once step 1 looks right:
   ```bash
   docker exec -d -w /home/odexter/neurosim -e PYTHONPATH=/home/odexter/neurosim/src \
     neurosim-noros bash -lc '
   nohup /opt/conda/envs/neurosim/bin/python applications/rl/train_sb3.py \
     --experiment-config applications/rl/configs/velocity_dodge_privileged_baseline.yaml \
     --run-name velocity_dodge_ctbr_v6 --no-wandb \
     > /home/odexter/neurosim/outputs/rl/logs/velocity_dodge_ctbr_v6.log 2>&1 &
   disown
   '
   ```
   Watch the log for crashes (not just routine metrics — filter tightly,
   routine per-eval log lines are noisy):
   ```bash
   docker exec neurosim-noros tail -f -n0 /home/odexter/neurosim/outputs/rl/logs/velocity_dodge_ctbr_v6.log \
     | grep -E "Traceback|BrokenPipe|CRITICAL|Exception|RuntimeError|Killed|OOM|core dumped" \
     | grep -vE "\[Error\]:\[Scene\]"   # this specific pattern is benign Habitat scene-load noise
   ```

3. Runs `v1` through `v5` in `outputs/rl/velocity_dodge_ctbr_v*/` are all
   superseded (each predates at least one of the six fixes above) — their
   checkpoints aren't meaningful for anything except historical comparison.
   Safe to ignore or clean up.

4. Once `v6` has meaningfully progressed (not just the first checkpoint —
   early checkpoints look identical to random and aren't informative),
   generate a policy video and check whether `success_rate` is actually
   climbing past the ~5-7% noise band seen in `v5` at 16-50k steps before
   it was killed.

5. If/when the privileged-actor diagnostic clearly learns to dodge, that's
   the trigger (per the config's own stated purpose) to build the real
   run: `obs_mode: events` or `combined`, `privileged_actor: false`,
   `privileged_critic: true`. Nothing for that transition has been started.
