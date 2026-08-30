"""Body-frame velocity-correction dodge task.

The policy emits a body-frame velocity correction which the env integrates
into an offset on the SE3 controller's *reference position* (see
``ReactiveDodgeEnv._control_from_velocity_delta``). Everything the actor
sees is egocentric, so the mapping from "obstacle on my left" to "dodge
right" is yaw-independent and does not have to be relearned per heading.

Reward shape, and why:

- Tracking error is penalised only **outside a deadband**. A continuous
  penalty prices every dodge in proportion to how long it is held, which
  previously made holding the nominal path and accepting the hit the
  cheaper option. Inside the corridor a dodge is free.
- Obstacle clearance enters as **potential-based shaping**
  (``gamma * phi(s') - phi(s)``), which provably leaves the optimal policy
  unchanged. A raw clearance bonus is reward-hackable: loitering where
  obstacles are not is worth more than flying the route.
- The survival bonus is deliberately small. Because a crash also forfeits
  all remaining survival reward, a large per-step bonus makes an early
  crash cost far more than a late one -- a time-varying deterrent that is
  not what we mean. The terminal penalty carries the deterrent instead.

Success is a single legible condition -- survived, with a real encounter
-- rather than a conjunction of RMSE thresholds that is hard to attribute
when it moves.
"""

from typing import Any

import numpy as np

from neurosim.core.coord_trans import rotate_vector_by_quat

from .base import RewardOutcome, TaskStep
from .reactive_dodge import ReactiveDodgeTask

# [valid, rel_pos_body(3), rel_vel_body(3), clearance, time_to_closest_approach]
PRIVILEGED_DIM = 9
# With privileged_obstacle_acceleration, rel_accel_body(3) is inserted after
# rel_vel_body, giving 12 per obstacle slot.
PRIVILEGED_SLOT_DIM_WITH_ACCEL = 12


class VelocityDodgeTask(ReactiveDodgeTask):
    """Dodge via body-frame velocity corrections to the reference position.

    Inherits ``ReactiveDodgeTask`` for control-path plumbing (residual
    config parsing, ``split_action``, tracking errors, threat metrics).
    Observation, reward, success and the privileged critic channel are all
    overridden here.
    """

    def __init__(
        self,
        *,
        w_pos: float = 0.5,
        w_pos_deadband_m: float = 0.3,
        w_vel: float = 0.25,
        w_correction: float = 0.02,
        w_smooth: float = 0.01,
        w_survival: float = 0.02,
        w_along_track: float = 1.0,
        w_clearance: float = 1.0,
        w_encounter_clear: float = 0.0,
        w_boundary: float = 0.0,
        w_no_threat_offset: float = 0.0,
        lateral_only: bool = False,
        lateral_axis_only: bool = False,
        threat_gated_actions: bool = False,
        crash_penalty_per_remaining_step: float = 0.0,
        clearance_threshold_m: float = 1.5,
        boundary_margin_threshold_m: float = 1.0,
        track_pos_error_cap_m: float = 2.7,
        track_vel_error_cap_mps: float = 3.0,
        gamma_shaping: float = 0.99,
        crash_penalty_scene: float = 60.0,
        crash_penalty_obstacle: float = 60.0,
        dodge_clearance_m: float = 0.25,
        privileged_critic: bool = True,
        privileged_actor: bool = False,
        # How many obstacles the privileged channel describes, threat-sorted
        # and zero-padded. Default 1 preserves the historical single-slot
        # layout that every v16-v22 config and checkpoint assumes.
        #
        # 1 is LOSSY here: max_concurrent is 3, so a single slot hides two
        # obstacles, and worse, the slot silently switches identity whenever
        # the threat sort reorders -- the valid flag distinguishes "some
        # obstacle" from "none", never "a different one". Set 3 to make the
        # channel lossless.
        privileged_obstacle_slots: int = 1,
        # Include each obstacle's acceleration. Default False for backward
        # compatibility, but note the channel is NOT sufficient state without
        # it whenever kinematic_parabola templates are in the mix: position
        # and velocity at one instant cannot separate a line from a parabola,
        # so a feedforward policy cannot infer which it faces.
        privileged_obstacle_acceleration: bool = False,
        **kwargs: Any,
    ):
        kwargs.setdefault("residual_control", {"mode": "velocity_delta_integrated"})
        super().__init__(
            threat_gated_actions=threat_gated_actions,
            w_pos=w_pos,
            w_vel=w_vel,
            w_correction=w_correction,
            w_smooth=w_smooth,
            w_survival=w_survival,
            crash_penalty_scene=crash_penalty_scene,
            crash_penalty_obstacle=crash_penalty_obstacle,
            **kwargs,
        )
        if self.residual_control_mode not in {
            "velocity_command",
            "velocity_delta_integrated",
            "gated_velocity_delta_integrated",
            "desired_body_offset",
            "cascaded_velocity",
            "gated_cascaded_velocity",
        }:
            raise ValueError(
                "velocity_dodge requires residual_control.mode "
                "'velocity_command', 'velocity_delta_integrated', or "
                "'desired_body_offset', 'cascaded_velocity', or "
                "'gated_cascaded_velocity'"
            )

        self.w_pos_deadband_m = float(w_pos_deadband_m)
        self.w_along_track = float(w_along_track)
        self.w_clearance = float(w_clearance)
        self.w_encounter_clear = float(w_encounter_clear)
        self.w_boundary = float(w_boundary)
        self.clearance_threshold_m = float(clearance_threshold_m)
        self.boundary_margin_threshold_m = float(boundary_margin_threshold_m)
        # r_track is otherwise unbounded: a diverging/tipping-over episode
        # (event_actor_v4, ~300-350 steps, pos_error up to ~3.8m, vel_error
        # up to ~7 m/s before out_of_bounds) accumulated -230 to -325 total
        # reward, 2-4x a clean obstacle_collision's -60 to -160 -- making a
        # failed hard dodge that risks divergence look far worse in
        # expectation than a passive collision, which can bias the policy
        # toward under-committing. Capped near tracking_failure_pos_error_m
        # (3.0) so the price per step plateaus once things have clearly gone
        # wrong, instead of compounding for hundreds of steps.
        self.track_pos_error_cap_m = float(track_pos_error_cap_m)
        self.track_vel_error_cap_mps = float(track_vel_error_cap_mps)
        self.crash_penalty_per_remaining_step = float(
            crash_penalty_per_remaining_step
        )
        # Charges displacement off the nominal path only while nothing is
        # inbound. The uniform tracking penalty cannot express this: it
        # prices the excursion a real dodge needs exactly as hard as idle
        # loitering, so every attempt to raise it taxed dodging too. Gating
        # it on threat separates the two -- deviate freely under threat,
        # pay to stay away once the sky is clear.
        self.w_no_threat_offset = float(w_no_threat_offset)
        # Zero the along-track (body forward/back) component of the
        # correction, leaving only lateral and vertical evasion. Braking is
        # otherwise heavily used -- 35.7% of in-threat steps sit at full
        # reverse -- and it is a manoeuvre that trades progress for time
        # rather than actually avoiding anything. Whether removing it costs
        # capability is an empirical question: check the privileged oracle's
        # success with and without before enabling.
        self.lateral_only = bool(lateral_only)
        # Stricter than lateral_only: keeps ONLY the body-y (left/right)
        # axis, zeroing forward/back and up/down. Collapses the correction
        # to a single scalar, which is the smallest action space that can
        # still express a dodge. Implies lateral_only.
        self.lateral_axis_only = bool(lateral_axis_only)
        self.gamma_shaping = float(gamma_shaping)
        self.dodge_clearance_m = float(dodge_clearance_m)
        self.privileged_critic = bool(privileged_critic)
        # Diagnostic only: hands the actor the obstacle geometry it would
        # otherwise have to extract from the event stream. If a privileged
        # actor learns to dodge and the event-based one does not, perception
        # is the bottleneck rather than the reward or the control mapping.
        # Never enable for a deployable policy.
        self.privileged_actor = bool(privileged_actor)
        self.privileged_obstacle_slots = max(int(privileged_obstacle_slots), 1)
        self.privileged_obstacle_acceleration = bool(privileged_obstacle_acceleration)
        self.on_reset()

    # ---- Observation ------------------------------------------------------

    @property
    def privileged_slot_dim(self) -> int:
        """Width of one obstacle slot in the privileged channel."""
        return (
            PRIVILEGED_SLOT_DIM_WITH_ACCEL
            if self.privileged_obstacle_acceleration
            else PRIVILEGED_DIM
        )

    @property
    def privileged_channel_dim(self) -> int:
        """Total privileged width: slots x slot width."""
        return self.privileged_obstacle_slots * self.privileged_slot_dim

    @property
    def state_observation_dim(self) -> int:
        # v_body, v_ref_body, pos_err_body, offset_body, omega_body, prev_action
        base = 3 * 5 + self.action_dim
        return base + (self.privileged_channel_dim if self.privileged_actor else 0)

    def _to_body(self, vector: Any, quat: np.ndarray) -> np.ndarray:
        return rotate_vector_by_quat(vector, quat, inverse=True).astype(np.float32)

    def make_state_observation(
        self,
        *,
        state: dict[str, np.ndarray],
        base_state: np.ndarray,
    ) -> np.ndarray:
        del base_state  # everything the actor sees is egocentric

        quat = np.asarray(state["q"], dtype=np.float64)
        flat = self._context.flat or {}
        v_ref = np.asarray(flat.get("x_dot", np.zeros(3)), dtype=np.float64)
        pos_err = np.asarray(state["x"], dtype=np.float64) - np.asarray(
            flat.get("x", state["x"]), dtype=np.float64
        )
        offset = self._context.offset_cmd
        offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)

        prev_action = self._context.previous_action
        if prev_action is None:
            prev_action = np.zeros(self.action_dim, dtype=np.float32)

        parts = [
            self._to_body(state["v"], quat),
            self._to_body(v_ref, quat),
            self._to_body(pos_err, quat),
            self._to_body(offset, quat),
            # Angular rate is already body-frame; it lets the policy
            # discount rotation-induced optical flow, which otherwise
            # dominates the event stream and carries no obstacle signal.
            np.asarray(state["w"], dtype=np.float32).reshape(3),
            np.asarray(prev_action, dtype=np.float32).reshape(self.action_dim),
        ]
        if self.privileged_actor:
            parts.append(self._nearest_obstacle_features(state))
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    @property
    def privileged_observation_dim(self) -> int:
        return self.privileged_channel_dim if self.privileged_critic else 0

    def _obstacle_slot(
        self, row: dict[str, Any] | None, quat: np.ndarray
    ) -> np.ndarray:
        """One obstacle's body-frame geometry, or zeros for an empty slot."""
        if row is None:
            return np.zeros(self.privileged_slot_dim, dtype=np.float32)

        tca = float(row["time_to_closest_approach"])
        parts = [
            np.asarray([1.0], dtype=np.float32),
            self._to_body(row["rel_pos"], quat),
            self._to_body(row["rel_vel"], quat),
        ]
        if self.privileged_obstacle_acceleration:
            parts.append(self._to_body(row.get("rel_accel", np.zeros(3)), quat))
        parts.append(
            np.asarray(
                [
                    float(row["clearance"]),
                    # tca is non-finite when the constant-velocity solution
                    # has no approach; the horizon stands in for "not closing".
                    # Note this collides with a genuine 1.5 s approach -- the
                    # policy cannot separate the two from this scalar alone,
                    # which is another reason rel_pos/rel_vel/rel_accel are
                    # the load-bearing entries and tca is a convenience.
                    tca if np.isfinite(tca) else self.threat_time_horizon_s,
                ],
                dtype=np.float32,
            )
        )
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def _nearest_obstacle_features(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """Body-frame geometry of the N most imminent obstacles, zero-padded.

        Rows arrive already sorted by ``obstacle_threat_priority``. That sort
        keys on a CONSTANT-VELOCITY ``predicted_clearance``, so it misranks
        ballistic obstacles -- a further reason to carry every obstacle rather
        than trust the sort to surface the right one.
        """
        obstacles = self._context.obstacle_relative_states or []
        quat = np.asarray(state["q"], dtype=np.float64)
        slots = [
            self._obstacle_slot(
                obstacles[i] if i < len(obstacles) else None, quat
            )
            for i in range(self.privileged_obstacle_slots)
        ]
        return np.concatenate(slots, axis=0).astype(np.float32, copy=False)

    def make_privileged_observation(
        self, *, state: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Nearest-obstacle geometry in the body frame, for the critic only.

        Deliberately limited to quantities that are in principle
        recoverable from the event stream (where the obstacle is and how it
        is moving). Feeding the critic genuinely unobservable things -- the
        spawn schedule, future throws -- would let it explain variance the
        actor can never act on, inflating advantage variance rather than
        reducing it.
        """
        return self._nearest_obstacle_features(state)

    # ---- Episode state ----------------------------------------------------

    def on_reset(self) -> None:
        super().on_reset()
        self._previous_potential: float | None = None
        self._previous_boundary_potential: float | None = None
        self._encountered = False
        self._min_clearance = np.inf
        # Per-obstacle closest approach. Episode success is conjunctive over
        # every encounter, so with several throws per episode it reads zero
        # long after per-encounter skill starts improving; this gives a
        # dense signal that moves in the meantime.
        self._encounter_min_clearance: dict[int, float] = {}
        self._encounter_clear_rewarded: set[int] = set()
        # Peak correction while a threat is actually present. The per-step
        # means are diluted by the many quiescent steps in an episode, so
        # they understate how hard the policy dodges; these say whether it
        # is using its authority when it matters.
        # Offset already held when each threat window opens. The dodge's
        # usable authority is offset_max minus this, so a policy that starts
        # every encounter part-way to saturation cannot use its full range --
        # and if it is displaced the *wrong* way it has even less. This is
        # what returning to the nominal path between obstacles buys.
        self._offset_at_onset: list[float] = []
        self._was_threatened = False
        self._peak_cross_in_threat = 0.0
        self._peak_along_in_threat = 0.0

    def _split_offset(self, offset: Any, v_ref: Any) -> tuple[float, float]:
        """Split the commanded offset into along-track and cross-track parts.

        Dodging by *braking* -- lagging along the path so a throw solved for
        a fixed arrival time simply misses -- is the degenerate strategy
        this task is most exposed to. It needs no perception (it works
        whichever side the obstacle comes from), risks no wall strike, and
        the intercept geometry cannot defend against it, because a ballistic
        obstacle is committed to arriving at a point at a time and cannot
        re-aim. So it has to be priced explicitly, and measured.
        """
        offset = np.asarray(offset, dtype=np.float64).reshape(3)
        v_ref = np.asarray(v_ref, dtype=np.float64).reshape(3)
        speed = float(np.linalg.norm(v_ref))
        if speed < 1e-6:
            return 0.0, float(np.linalg.norm(offset))

        tangent = v_ref / speed
        along = float(np.dot(offset, tangent))
        cross = float(np.linalg.norm(offset - along * tangent))
        return along, cross

    def _clearance_potential(self, clearance: float) -> float:
        """Zero beyond the threshold, growing negative as clearance shrinks."""
        if not np.isfinite(clearance) or clearance >= self.clearance_threshold_m:
            return 0.0
        deficit = (self.clearance_threshold_m - clearance) / self.clearance_threshold_m
        return -self.w_clearance * float(np.clip(deficit, 0.0, 2.0) ** 2)

    def _boundary_potential(self, margin: float) -> float:
        """Zero away from the scene bounds, growing negative approaching them.

        Symmetric to ``_clearance_potential``, aimed at the scene boundary
        instead of an obstacle. Without this, obstacles get dense shaped
        feedback the whole way in, but the boundary only has a sparse
        terminal penalty on ``out_of_bounds`` -- exactly the credit-assignment
        gap that let evasive corrections run the vehicle out of the scene
        with no graduated feedback on the way there.
        """
        if not np.isfinite(margin) or margin >= self.boundary_margin_threshold_m:
            return 0.0
        deficit = (self.boundary_margin_threshold_m - margin) / self.boundary_margin_threshold_m
        return -self.w_boundary * float(np.clip(deficit, 0.0, 2.0) ** 2)

    # ---- Reward -----------------------------------------------------------

    def compute_reward(self, step: TaskStep) -> RewardOutcome:
        state = step.state
        pos_err, vel_err, _ = self._tracking_errors(state)
        pos_norm = float(np.linalg.norm(pos_err))
        vel_norm = float(np.linalg.norm(vel_err))

        action = np.asarray(step.action, dtype=np.float32)
        correction_energy = float(np.mean(np.square(action)))
        if step.prev_action is None:
            action_smoothness = 0.0
        else:
            action_smoothness = float(np.linalg.norm(action - step.prev_action))

        clearance = float(self._context.min_obstacle_distance)
        predicted_clearance = float(self._context.predicted_closest_distance)
        if np.isfinite(clearance):
            self._min_clearance = min(self._min_clearance, clearance)

        for row in self._context.obstacle_relative_states or []:
            oid = int(row["object_id"])
            self._encounter_min_clearance[oid] = min(
                self._encounter_min_clearance.get(oid, np.inf),
                float(row["clearance"]),
            )
        if self._context.obstacle_threat or clearance <= self.clearance_threshold_m:
            self._encountered = True

        # Deadbanded tracking: free inside the corridor, penalised outside.
        pos_excess = max(pos_norm - self.w_pos_deadband_m, 0.0)

        # Predicted closest approach changes as soon as an evasive velocity
        # bends the collision course; instantaneous range responds only after
        # the vehicle has moved, making credit assignment unnecessarily late.
        potential = self._clearance_potential(predicted_clearance)
        if self._previous_potential is None:
            shaping = 0.0
        else:
            shaping = self.gamma_shaping * potential - self._previous_potential
        self._previous_potential = potential

        # Same potential-based construction, aimed at the scene boundary.
        # Summing two independent potential-based terms is itself
        # potential-based (potential = sum of the two potentials), so this
        # doesn't disturb the shaping theory's optimal-policy guarantee.
        boundary_margin = float(self._context.bounds_margin_m)
        boundary_potential = self._boundary_potential(boundary_margin)
        if self._previous_boundary_potential is None:
            boundary_shaping = 0.0
        else:
            boundary_shaping = (
                self.gamma_shaping * boundary_potential
                - self._previous_boundary_potential
            )
        self._previous_boundary_potential = boundary_potential
        shaping += boundary_shaping

        offset_now_vec = self._context.offset_cmd
        offset_norm_now = (
            0.0
            if offset_now_vec is None
            else float(np.linalg.norm(np.asarray(offset_now_vec, dtype=np.float64)))
        )
        flat = self._context.flat or {}
        offset_along, offset_cross = self._split_offset(
            self._context.offset_cmd
            if self._context.offset_cmd is not None
            else np.zeros(3),
            flat.get("x_dot", np.zeros(3)),
        )

        # An "encounter" is an obstacle that actually came close enough to
        # matter; it counts as cleared if its closest approach stayed above
        # the dodge threshold.
        encounters = [
            c
            for c in self._encounter_min_clearance.values()
            if c <= self.clearance_threshold_m
        ]
        encounters_total = len(encounters)
        encounters_cleared = sum(1 for c in encounters if c > self.dodge_clearance_m)

        newly_cleared = 0
        for row in self._context.obstacle_relative_states or []:
            oid = int(row["object_id"])
            if oid in self._encounter_clear_rewarded:
                continue
            if "rel_pos" not in row or "rel_vel" not in row:
                continue
            rel_pos = np.asarray(row["rel_pos"], dtype=np.float64)
            rel_vel = np.asarray(row["rel_vel"], dtype=np.float64)
            is_receding = float(np.dot(rel_pos, rel_vel)) >= 0.0
            min_clearance = self._encounter_min_clearance.get(oid, np.inf)
            if is_receding and min_clearance > self.dodge_clearance_m:
                self._encounter_clear_rewarded.add(oid)
                newly_cleared += 1

        threatened = bool(self._context.obstacle_threat)
        if threatened and not self._was_threatened:
            self._offset_at_onset.append(offset_norm_now)
        self._was_threatened = threatened
        if self._context.obstacle_threat:
            self._peak_cross_in_threat = max(
                self._peak_cross_in_threat, abs(offset_cross)
            )
            self._peak_along_in_threat = max(
                self._peak_along_in_threat, abs(offset_along)
            )

        r_survival = self.w_survival
        pos_excess_priced = min(pos_excess, self.track_pos_error_cap_m)
        vel_norm_priced = min(vel_norm, self.track_vel_error_cap_mps)
        r_track = -self.w_pos * pos_excess_priced - self.w_vel * vel_norm_priced
        r_correction = -self.w_correction * correction_energy
        r_smooth = -self.w_smooth * action_smoothness
        # Priced on top of the (direction-agnostic) tracking term, so an
        # along-track metre costs strictly more than a cross-track one.
        r_along_track = -self.w_along_track * abs(offset_along)
        r_encounter_clear = self.w_encounter_clear * newly_cleared
        # Only bites once nothing is inbound, so a dodge is never charged
        # for the displacement it needs -- only for failing to give it back.
        #
        # Priced on pos_excess, the *actual* deviation from the nominal path,
        # not on the commanded offset. Measured on v24 over 20 stochastic
        # episodes, no-threat steps ran pos_error 1.633 m against an
        # offset_cmd of only 0.280 m -- the command is 17% of the deviation,
        # so charging it left 83% of the excursion free. That remainder is
        # not plant lag to be written off: the same policy evaluated
        # deterministically sits 0.025 m off the path, so it is the
        # controller failing to track a reference the sampled actions are
        # shaking, and it moves with the policy.
        #
        # It also happens to be the term the standing-offset exploit lives
        # in. Threat fraction was 51.5% deterministic but 20.8% stochastic:
        # obstacles are aimed once at spawn with no re-solve, so drifting
        # ~1.6 m over their 1-2 s flight makes them miss outright. Wandering
        # off the path does not merely dodge the tracking penalty, it
        # deletes the encounters -- which is what every run from v7 onward
        # converged to.
        offset_now = self._context.offset_cmd
        offset_norm = (
            0.0
            if offset_now is None
            else float(np.linalg.norm(np.asarray(offset_now, dtype=np.float64)))
        )
        # Scaled by how clear the sky is, rather than switched by a binary
        # flag. The gate exists so a dodge is never charged for the
        # displacement it needs -- but at the current obstacle density the
        # threat flag is true on ~80% of steps (measured), so an on/off term
        # is inactive four steps in five and can barely do its job. That is a
        # coverage problem, not an avoidance one: nothing charges for holding
        # an offset, so holding one is free.
        #
        # predicted_clearance grows as the nearest threat recedes, so this
        # pays full price once the closest predicted approach is comfortably
        # outside the threat distance, nothing while one is genuinely close,
        # and ramps between. A dodge in progress still pays ~0.
        # Nothing tracked at all reads as fully clear; otherwise the charge
        # ramps from 0 at contact to full at twice the threat distance. The
        # binary flag is deliberately not used: keeping it as a hard zero
        # would only *reduce* coverage on unflagged steps, which is backwards.
        # Reading proximity directly means a step flagged in-threat by a
        # distant obstacle still pays partial price, which is where the
        # missing coverage lives.
        if not np.isfinite(predicted_clearance) or self.threat_distance_m <= 0.0:
            threat_proximity = 1.0
        else:
            threat_proximity = float(
                np.clip(predicted_clearance / (2.0 * self.threat_distance_m), 0.0, 1.0)
            )
        r_no_threat_offset = (
            -self.w_no_threat_offset
            * threat_proximity
            * min(pos_excess, self.track_pos_error_cap_m)
        )
        reward = (
            r_survival
            + r_track
            + r_correction
            + r_smooth
            + r_no_threat_offset
            + r_along_track
            + shaping
            + r_encounter_clear
        )

        self._sample_count += 1
        self._pos_sq_sum += pos_norm * pos_norm
        self._vel_sq_sum += vel_norm * vel_norm
        self._correction_energy_sum += correction_energy

        if pos_norm > self.tracking_failure_pos_error_m:
            self._tracking_failure_count += 1
        else:
            self._tracking_failure_count = 0

        terms = {
            "pos_error": pos_norm,
            "pos_excess": pos_excess,
            "vel_error": vel_norm,
            "boundary_margin_m": boundary_margin,
            "correction_energy": correction_energy,
            "action_smoothness": action_smoothness,
            # Watch these two: a policy that has collapsed to braking shows
            # |offset_along| >> offset_cross during encounters.
            "offset_along_track": offset_along,
            "offset_cross_track": offset_cross,
            "peak_cross_in_threat": self._peak_cross_in_threat,
            # Mean displacement already held when a threat window opened.
            # Near zero means each dodge starts with full authority.
            "offset_at_threat_onset": (
                float(np.mean(self._offset_at_onset))
                if self._offset_at_onset
                else 0.0
            ),
            "authority_free_at_onset": (
                1.0
                - float(np.mean(self._offset_at_onset))
                / float(np.max(self.offset_max_m))
                if self._offset_at_onset
                else 1.0
            ),
            "peak_along_in_threat": self._peak_along_in_threat,
            # Fraction of the available offset actually used when threatened.
            "authority_used_in_threat": (
                self._peak_cross_in_threat / float(np.max(self.offset_max_m))
                if float(np.max(self.offset_max_m)) > 0
                else 0.0
            ),
            "min_obstacle_distance": clearance,
            "predicted_closest_distance": predicted_clearance,
            "obstacle_threat": float(self._context.obstacle_threat),
            "encountered": float(self._encountered),
            "episode_min_clearance": self._min_clearance,
            "encounters_total": float(encounters_total),
            "encounters_cleared": float(encounters_cleared),
            "encounter_clear_rate": (
                encounters_cleared / encounters_total if encounters_total else 0.0
            ),
            "active_obstacle_count": float(self._context.active_obstacle_count),
            "r_survival": r_survival,
            "r_track": r_track,
            "r_correction": r_correction,
            "r_smooth": r_smooth,
            "r_along_track": r_along_track,
            "r_clearance_shaping": shaping,
            "r_encounter_clear": r_encounter_clear,
            "r_no_threat_offset": r_no_threat_offset,
            "offset_norm": offset_norm,
        }
        return RewardOutcome(reward=reward, terms=terms)

    @property
    def crash_penalty(self) -> float:
        """Base penalty plus a charge for the rest of the forfeited episode.

        A flat terminal penalty only deters crashing while the per-step
        reward stays above ``-penalty / remaining_steps``. Below that,
        ending the episode early is worth more than flying on and the
        policy correctly learns to crash -- which is what both previous
        runs did. v8 (flat 60) was paid +234 to die. v9 (flat 150) still
        was, once exploration noise drove the per-step rate to -0.34/-0.51
        against a threshold of -0.246.

        Scaling the penalty with the time a crash gives up moves that
        threshold to ``-(rate + base/remaining)``, so it no longer erodes
        as episodes shorten -- the failure mode above is self-reinforcing
        precisely because each early death shortens the episodes that set
        the threshold. Note this is a bound, not an identity: the charge
        has to out-price the *whole* per-step deficit, of which the
        forfeited ``w_survival`` is only a part, so
        ``crash_penalty_per_remaining_step`` is set well above
        ``w_survival`` rather than equal to it.
        """
        return super().crash_penalty + (
            self.crash_penalty_per_remaining_step
            * float(self._context.remaining_steps)
        )

    def check_success(self, *, state: dict[str, np.ndarray]) -> bool:
        """Survived an episode in which a real encounter occurred.

        The env only calls this when the episode did not terminate, so
        reaching here already implies no crash and no tracking failure.
        """
        del state
        if self._sample_count == 0:
            return False
        if self.require_obstacle_encounter and not self._encountered:
            return False
        return self._min_clearance > self.dodge_clearance_m
