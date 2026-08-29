# `velocity_dodge`: policy I/O and control mapping

How the dodging policy sees the world, what it emits, and how that emission
becomes motor commands. The short version:

```
event frame + body-frame state
        |
        v
   policy (PPO)  --->  dv : 3-D body-frame velocity correction, in [-1, 1]
        |
        v
   low-pass filter  ->  leaky integrator  ->  offset e (metres)
        |
        v
   SE3 controller tracks a SHIFTED reference:  x_ref + e,  x_dot_ref + de/dt
        |
        v
   cmd_thrust + cmd_w  ->  vehicle rate loop @ 1 kHz
```

The policy never commands thrust or body rates directly. It nudges the
*reference* that a classical SE3 controller is already tracking, so the
controller keeps doing the stabilising and the policy only decides where the
reference should be.

---

## 1. Policy inputs

### Actor

The actor sees **only** what a real drone could sense. No obstacle ground
truth.

**Event frame** — `(2, 480, 640) float32`. Two channels (negative, positive
polarity) of a time-surface representation, `ts_decay_ms: 10`. The sensor
renders at the world rate (1 kHz) and events accumulate across the 20 world
steps that make up one policy step.

> Resolution matters more than it looks: a 0.2 m obstacle at its 3–5 m spawn
> distance subtends only ~12–20 px at full resolution, and 3–5 px if
> downsampled 4×. `downsample_factor` costs CNN compute and rollout-buffer
> memory only — the sensor always renders full-res, so it does not buy
> simulation speed.

**State vector** — `(18,) float32`, all in the **body frame**:

| slice | meaning | why |
|---|---|---|
| `0:3` | `v_body` — own velocity | |
| `3:6` | `v_ref_body` — reference velocity from the trajectory | what it is supposed to be doing |
| `6:9` | `pos_err_body` — position error vs the nominal path | how far off it already is |
| `9:12` | `offset_body` — the integrator's accumulated offset | genuine hidden state; without it the MDP is not Markov |
| `12:15` | `omega_body` — body angular rate | lets the policy discount rotation-induced optical flow, which dominates the event stream and carries no obstacle information |
| `15:18` | `prev_action` — the **filtered** correction in effect | see the filter note in §3 |

Body frame throughout is deliberate: an egocentric camera means "obstacle on
my left ⇒ dodge right" should be one rule, not a different rule per heading.
`test_state_observation_is_body_frame_and_yaw_invariant` pins this.

### Critic (privileged)

The critic additionally receives `(9,)` describing the **nearest** obstacle,
also body-frame: `[valid, rel_pos(3), rel_vel(3), clearance, time_to_closest_approach]`.

This is asymmetric actor-critic: the value function gets to know "was that
episode doomed?" without solving the perception problem the actor is still
learning, which cuts variance in exactly the states that drive the advantage
signal.

**The privilege is deliberately modest.** Under partial observability the
variance-minimising baseline is `E[V | actor observation]`, *not*
`V(true state)`. A critic that can see distinctions the actor fundamentally
cannot will explain away variance the actor can never act on, which *raises*
advantage variance. So the critic only gets quantities that are in principle
recoverable from the event stream — where the obstacle is and how it moves —
and never things like the future spawn schedule.

Enforcement is in `AsymmetricActorCriticPolicy`: the actor's observation has
the `privileged` key zeroed before its encoder runs. Note SB3's
`get_distribution` bypasses `extract_features`, so that path is overridden
too — otherwise the actor would receive privileged data on every `predict()`,
silently, at rollout and eval time.

### Encoder

`_EventBackbone` is 3 strided convs followed by a **spatial softmax**, which
returns the expected `(x, y)` image coordinate of each channel's activation.

A global average pool — the obvious default — answers "is a feature present"
and discards *where*. For dodging, bearing is the decisive fact: it sets the
sign of the correction. Pooling to a scalar forces location to be encoded in
*which* channels fire rather than where, which is far harder for a randomly
initialised CNN to discover, especially before it has ever produced a
successful dodge.

---

## 2. Policy output

`(3,) float32` in `[-1, 1]`: a **body-frame velocity correction**, scaled by
`delta_velocity_limits_mps`.

That is the entire action space. Yaw is *not* controlled — it tracks the
nominal trajectory tangent on purpose. If yaw followed the corrected
velocity, the policy's own dodge would swing the camera, changing what it
sees, changing the next action: a perception/action feedback loop that makes
credit assignment much harder.

---

## 3. Mapping the correction onto the controller

This is the part worth understanding, because the obvious implementation does
not work.

### Why not just add it to the reference velocity

`rotorpy`'s `SE3Control` computes

```python
F_des = m * (-kp_pos * (x - x_ref) - kd_pos * (v - v_ref) + a_ref + g)
```

Add `dv` to `v_ref` alone and the closed loop becomes
`ë + kd·ė + kp·e = kd·dv`, so the steady-state deviation is

```
e_ss = (kd_pos / kp_pos) * dv     ~= 0.6 * dv  with stock gains
```

The position-error term actively fights the correction. `dv = 0.3` buys
**18 cm** — measured 0.184 m against 0.184 m predicted. No increase in `dv`
escapes this; it is a property of the gain ratio, and larger `dv` destabilises
ordinary flight long before it buys useful clearance.

### What is done instead

The correction is integrated into an offset applied to the reference
**position**:

```python
e ← clip( e + dt * (dv_world - e / offset_tau_s),  ±offset_max_m )

flat_shifted = {
    "x":     x_ref     + e,
    "x_dot": x_dot_ref + (e - e_prev) / dt,
    ...                       # x_ddot, yaw unchanged
}
control = se3.update(t, state, flat_shifted)
```

Now the controller's spring pulls *toward* the shifted target, so achieved
deviation equals commanded offset. Measured **achieved / commanded = 1.012**
in the full nonlinear sim — no fighting.

Implemented by building a modified `flat` dict, **not** by forking rotorpy.

Three details that are load-bearing:

**Exact derivative, not the analytic one.** `x_dot` uses
`(e - e_prev)/dt` — the true derivative of the *clipped* offset. The analytic
`dv - e/tau` leaves a residual velocity feedforward once the offset
saturates, which the controller converts into an extra `(kd/kp)·v_residual`
of deviation, so `offset_max_m` would stop being a real bound. Measured: 2.115 m
of excursion against a 1.5 m cap with the analytic form, exactly 1.500 m with
the exact one.

**The action is low-pass filtered before integration.** The offset averages
zero-mean action noise away, but its *derivative* does not — and that
derivative is what SE3 receives as reference velocity, where `kd_pos`
amplifies it into acceleration. Unfiltered random actions put **7/8** episodes
out of bounds; filtering at `0.15 s` cut that to 3/8 and dropped peak tracking
error from 3.18 m to 0.99 m. The observation reports the *filtered* action as
`prev_action` so the filter state stays observable.

**τ and dv multiply.** Sustainable offset is `offset_tau_s * dv`, which must
clear the combined agent+obstacle radius. τ does **not** set how fast a dodge
starts — for small `t`, `e(t) ≈ dv·t` regardless of τ. τ sets how large a held
correction grows and how fast the drone drifts back when the policy stops
correcting, which is what makes returning to the path structural rather than
something the reward has to enforce.

### Timing

At `control_rate: 50` the policy acts every **0.02 s**; the resulting SE3
output is held for 20 world steps at 1 kHz while the vehicle's rate loop
tracks it. One integrator step happens per policy step, so the offset moves at
most `dt·dv ≈ 9 mm` per decision — a dodge is built over tens of decisions,
not one. `dt/τ = 0.025 ≪ 1`, so the discrete integrator faithfully
approximates the continuous one.

---

## 4. Key parameters

| parameter | meaning |
|---|---|
| `delta_velocity_limits_mps` | scales the action to m/s; sets dodge onset speed |
| `offset_tau_s` | leak constant; sustainable offset is `tau * dv`, and return-to-path time constant |
| `offset_max_m` | hard clip on accumulated offset (set to `tau * dv` so it bounds windup without binding normally) |
| `action_filter_tau_s` | low-pass on the action before integration |
| `w_along_track` | prices dodging by *braking* — see below |

**On braking.** Throws are solved to intersect the nominal trajectory at a
fixed time, so simply being *late* defeats them. That shortcut needs no
perception and risks no wall strike, and intercept geometry cannot defend
against it: a ballistic obstacle is committed to a point-in-time and cannot
re-aim. It therefore has to be priced. `w_along_track` penalises the
along-track component of the offset, and `offset_along_track` /
`offset_cross_track` are logged so collapse to braking is visible rather than
inferred.

---

## 5. Where things live

| file | role |
|---|---|
| `tasks/velocity_dodge.py` | observation, privileged channel, reward, success |
| `env_reactive_dodge.py` | `_control_from_velocity_delta` (the mapping above), intercept scheduler |
| `sb3_features.py` | spatial-softmax encoder, `AsymmetricActorCriticPolicy` |
| `core/visual_backend/dynamic_obstacles.py` | spawning, `reset_episode`, intercept schedule |
| `applications/rl/configs/velocity_dodge_sb3_experiment.yaml` | the tuned values, with rationale in comments |
| `tests/test_velocity_dodge_task.py` | pins the numeric contracts above |
