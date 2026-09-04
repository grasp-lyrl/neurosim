"""Evaluate a privileged scripted oracle on the real velocity-dodge env."""

import argparse
import sys
import copy
import json
from pathlib import Path

from dataclasses import replace

import numpy as np

from neurosim.core.coord_trans import rotate_vector_by_quat
from neurosim.core.trajectory.habitat_trajs import validate_static_points_clearance
from neurosim.rl import env_class_for_task
from neurosim.rl.trajectory_dodge_expert import (
    LocalTrajectoryExpert,
    TrajectoryExpertConfig,
)
from neurosim.rl.receding_horizon_dodge_expert import (
    MovingSpherePrediction,
    RecedingHorizonConfig,
    RecedingHorizonDodgeExpert,
)

from train_sb3 import load_experiment_config


def project_sphere_to_pinhole(
    centre,
    camera_position,
    camera_quaternion_wxyz,
    *,
    width,
    height,
    hfov_deg,
    radius_m,
):
    """Project a Habitat-world sphere into a Habitat pinhole camera.

    Returns ``[visible, u_norm, v_norm, radius_norm, depth_norm]``. Coordinates
    and radius are normalized by image dimensions; depth is normalized by the
    camera far plane by the caller. Habitat cameras look along local ``-Z``
    and expose quaternions in Hamilton ``[w, x, y, z]`` order.
    """
    centre = np.asarray(centre, dtype=np.float64)
    camera_position = np.asarray(camera_position, dtype=np.float64)
    q = np.asarray(camera_quaternion_wxyz, dtype=np.float64)
    if q.shape != (4,) or float(np.linalg.norm(q)) < 1e-12:
        return np.zeros(5, dtype=np.float32)
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    local = rotation.T @ (centre - camera_position)
    depth = -float(local[2])
    if depth <= 1e-6:
        return np.zeros(5, dtype=np.float32)
    width = float(width)
    height = float(height)
    focal = width / (2.0 * np.tan(0.5 * np.deg2rad(float(hfov_deg))))
    u = 0.5 * width + focal * float(local[0]) / depth
    v = 0.5 * height - focal * float(local[1]) / depth
    radius_px = focal * float(radius_m) / depth
    # A heatmap cannot represent a centre outside its finite lattice. Treat
    # partially clipped spheres as absent; camera-FOV spawning keeps the
    # actionable approach inside the image, while departed obstacles should
    # become clean negatives rather than impossible coordinate targets.
    visible = 0.0 <= u < width and 0.0 <= v < height
    return np.asarray(
        [
            float(visible),
            u / width,
            v / height,
            radius_px / max(width, height),
            depth,
        ],
        dtype=np.float32,
    )


def obstacle_image_label(env) -> np.ndarray:
    """Ground-truth image label for the task's most imminent obstacle."""
    backend = env.sim.visual_backend
    manager = getattr(backend, "_dynamic_obstacles", None)
    active = getattr(manager, "_active", {}) if manager is not None else {}
    rows = getattr(env._task._context, "obstacle_relative_states", None) or []
    if not active or not rows:
        return np.zeros(5, dtype=np.float32)

    row = rows[0]
    simulator_id = int(row.get("simulator_object_id", row["object_id"]))
    item = active.get(simulator_id)
    if item is None:
        return np.zeros(5, dtype=np.float32)

    settings = backend.settings
    sensors = settings.get("sensors", {})
    camera_uuid = getattr(manager.cfg, "spawn_camera_uuid", None)
    if camera_uuid is None:
        camera_uuid = next(
            (name for name, sensor in sensors.items() if sensor.get("type") == "event"),
            None,
        )
    camera_cfg = sensors.get(camera_uuid, {})
    camera_state = backend.agent.get_state().sensor_states.get(camera_uuid)
    if camera_state is None:
        return np.zeros(5, dtype=np.float32)
    label = project_sphere_to_pinhole(
        np.asarray(item.obj.translation, dtype=np.float64),
        np.asarray(camera_state.position, dtype=np.float64),
        np.asarray(camera_state.rotation.components, dtype=np.float64),
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        hfov_deg=float(camera_cfg["hfov"]),
        radius_m=float(item.collision_radius),
    )
    if label[0] > 0.5:
        label[4] /= max(float(camera_cfg.get("zfar", 20.0)), 1e-6)
    return label


def effective_expert_action(env, action: np.ndarray) -> np.ndarray:
    """Zero the axes the task suppresses, matching what the controller sees.

    The residual delta is the last three entries for every residual mode,
    gated (4-dim) or not (3-dim), so negative indices cover both layouts.
    """
    effective = np.asarray(action, dtype=np.float32).copy()
    task = env._task
    if getattr(task, "lateral_axis_only", False):
        effective[-3] = 0.0
        effective[-1] = 0.0
    elif getattr(task, "lateral_only", False):
        effective[-3] = 0.0
    return effective


def should_include_episode(row: dict, include_all_episodes: bool) -> bool:
    """Return whether an episode is valid imitation data."""
    return bool(row["success"]) or (
        include_all_episodes and row["termination_reason"] == "timeout"
    )


class HDF5ImitationWriter:
    """Append completed episodes without retaining the full dataset in RAM."""

    def __init__(self, path: Path, experiment_config: str):
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = h5py.File(path, "w")
        self.file.attrs["experiment_config"] = experiment_config
        self.file.attrs["format"] = "neurosim_imitation_v1"
        self.samples = 0
        self.threat_samples = 0

    def _append_array(self, name: str, values: np.ndarray) -> None:
        values = np.asarray(values)
        if name not in self.file:
            # One sample per compressed chunk bounds peak memory and makes
            # random lazy minibatches cheap enough without a giant archive
            # decompression step.
            chunks = (1, *values.shape[1:])
            self.file.create_dataset(
                name,
                shape=(0, *values.shape[1:]),
                maxshape=(None, *values.shape[1:]),
                chunks=chunks,
                dtype=values.dtype,
                compression="lzf",
            )
        dataset = self.file[name]
        start = int(dataset.shape[0])
        dataset.resize(start + len(values), axis=0)
        dataset[start:] = values

    def append_episode(
        self,
        observations: list,
        actions: list[np.ndarray],
        *,
        episode: int,
        seed: int,
    ) -> None:
        if not actions:
            # Threat-only DAgger can legitimately see an episode with no
            # actionable expert labels. It contributes no samples and should
            # not manufacture a malformed (0,) action array.
            return
        if len(observations) != len(actions):
            raise ValueError(
                "episode observation/action length mismatch: "
                f"{len(observations)} != {len(actions)}"
            )
        action_array = np.asarray(actions, dtype=np.float32)
        threat = np.linalg.norm(action_array, axis=1) > 0.1
        if observations and isinstance(observations[0], dict):
            for key in observations[0]:
                dtype = np.float16 if key == "events" else np.float32
                self._append_array(
                    f"observations_{key}",
                    np.asarray([obs[key] for obs in observations], dtype=dtype),
                )
        else:
            self._append_array(
                "observations", np.asarray(observations, dtype=np.float32)
            )
        self._append_array("actions", action_array)
        self._append_array("threat", threat.astype(np.bool_))
        self._append_array(
            "episode_index", np.full(len(actions), episode, dtype=np.int32)
        )
        self._append_array("episode_seed", np.full(len(actions), seed, dtype=np.int64))
        self.samples += len(actions)
        self.threat_samples += int(threat.sum())
        self.file.flush()

    def close(self) -> None:
        self.file.attrs["samples"] = self.samples
        self.file.attrs["threat_samples"] = self.threat_samples
        self.file.close()


def load_rollout_policy(path, cfg, env, device):
    """Rebuild the clone actor and load its weights for DAgger rollouts."""
    import torch  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_velocity_dodge_bc import build_clone_net  # noqa: PLC0415

    payload = torch.load(path, map_location=device, weights_only=False)
    preprocessing = payload.get("preprocessing", {})
    recurrent_type = preprocessing.get("recurrent_type")
    if recurrent_type == "event_tracker_action_gru":
        from train_event_tracker_action_head import EventActionPolicy  # noqa: PLC0415

        model = EventActionPolicy(path, device=device)
        model._dagger_recurrent_type = recurrent_type
        return model
    if recurrent_type == "compact_gru":
        from train_velocity_dodge_compact_gru import (  # noqa: PLC0415
            CommitmentController,
            build_compact_model,
        )

        downsample = int(preprocessing.get("downsample_events", 1))
        model = build_compact_model(
            cfg, env, downsample,
            int(preprocessing.get("hidden_size", 128)),
        )
        model.load_state_dict(payload["model"])
        model.downsample_events = downsample
        model._dagger_hidden = None
        model._dagger_controller = CommitmentController(
            **preprocessing.get("controller", {})
        )
        model._dagger_recurrent_type = recurrent_type
        return model.to(device).eval()

    model, _ = build_clone_net(
        cfg,
        env,
        int(preprocessing.get("downsample_events", 1)),
        bool(preprocessing.get("discrete_actions", False)),
        blank_previous_action=bool(
            preprocessing.get("blank_previous_action", False)
        ),
        events_only_state=bool(preprocessing.get("events_only_state", False)),
        blank_tracking_error=bool(
            preprocessing.get("blank_tracking_error", False)
        ),
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def clone_action(model, obs, device):
    """Deterministic clone action for one observation."""
    import torch  # noqa: PLC0415

    from train_velocity_dodge_bc import pool_events  # noqa: PLC0415

    with torch.no_grad():
        if getattr(model, "_dagger_recurrent_type", None) == "event_tracker_action_gru":
            return model.predict(obs)[0]
        batch = (
            {
                k: torch.from_numpy(
                    (
                        pool_events(np.asarray(v), model.downsample_events)
                        if k == "events"
                        else np.asarray(v)
                    )[None]
                ).to(device)
                for k, v in obs.items()
            }
            if isinstance(obs, dict)
            else torch.from_numpy(np.asarray(obs)[None]).to(device)
        )
        if getattr(model, "_dagger_recurrent_type", None) == "compact_gru":
            output, model._dagger_hidden = model.recurrent_step(
                batch, model._dagger_hidden
            )
            return model._dagger_controller.action(output)
        output = model(batch).cpu().numpy()[0]
        if getattr(model, "discrete_actions", False):
            action = np.zeros(3, dtype=np.float32)
            action[1] = float(int(np.argmax(output)) - 1)
            return action
        return output.astype(np.float32)


def expert_for_task(task) -> LocalTrajectoryExpert:
    """Build an expert that only proposes displacements the task can reach.

    ``candidate_offsets_m`` defaults to (0.35, 0.50, 0.65, 0.80, 1.00), but
    under ``desired_body_offset`` the action is normalised by
    ``offset_max_m`` and clipped to +-1, which caps the lateral axis at 0.56
    m and the vertical at 0.36 m. Three of the five candidates are therefore
    unreachable: the planner validates swept clearance assuming it gets to
    0.80 or 1.00 m, the vehicle stops at the cap, and a plan certified safe
    collides anyway.

    Being conservative is the right failure direction here. A candidate that
    is too short is validated honestly and may simply find no solution --
    which the caller records as an unlabelled frame. A candidate that is too
    long is validated dishonestly and produces a label that crashes.
    """
    base = TrajectoryExpertConfig()
    # Track the task's actual rate limit instead of a hand-set constant.
    # TrajectoryExpertConfig hardcodes 1.5/1.0 with a comment claiming they
    # are "matched to offset_rate_limits_mps", but nothing enforced that, so
    # editing the config left the planner solving against a stale envelope.
    # This constant has bound the expert twice now: at 0.70/0.27 it rejected
    # 96% of its own candidates, and at 1.5/1.0 it still rejected 52.7% --
    # far more than moving obstacles (15.8%) or static geometry (3.6%).
    limits = getattr(task, "offset_rate_limits_mps", None)
    if limits is not None:
        limits = np.asarray(limits, dtype=np.float64)
        base = replace(
            base,
            max_horizontal_speed_mps=float(min(limits[0], limits[1])),
            max_vertical_speed_mps=float(limits[2]),
        )
    if getattr(task, "residual_control_mode", None) != "desired_body_offset":
        # Other modes normalise by a velocity limit, so a displacement
        # candidate is not comparable and the defaults stand.
        return LocalTrajectoryExpert(base)
    if not getattr(task, "lateral_axis_only", False):
        # Only narrowed where the reachable magnitude is unambiguous. With
        # the vertical axis in play the cap depends on which direction the
        # planner picks -- and make_plan takes one candidate list for all of
        # them -- so a single conservative number would collapse the set to
        # (0.35, 0.36) and weaken an oracle that measures 55% as it stands.
        # Per-direction reach would need a make_plan API change.
        return LocalTrajectoryExpert(base)

    reach = float(np.asarray(task.offset_max_m, dtype=np.float64)[1])
    candidates = tuple(m for m in base.candidate_offsets_m if m <= reach + 1e-9)
    if not candidates or candidates[-1] < reach - 1e-9:
        candidates = candidates + (reach,)
    return LocalTrajectoryExpert(replace(base, candidate_offsets_m=candidates))


def expert_directions(env) -> list[np.ndarray] | None:
    """Restrict the planner to axes the task will actually apply.

    ``candidate_directions`` offers +-lateral and +-vertical (plus a
    relative-velocity diagonal). Under ``lateral_axis_only`` the task zeroes
    body x and z *after* the plan is chosen, so a vertical plan is deleted
    in full: the expert believes it is dodging while the controller receives
    nothing. Measured 0/10 successes that way -- 6 obstacle collisions, 2
    tracking failures, 1 out-of-bounds -- against 55% with the vertical axis
    available. Planning only in directions the action space can express
    makes the expert solve the problem it is actually scored on, and lets it
    correctly report "no safe dodge" when no lateral one exists instead of
    emitting a phantom label.

    Returning ``None`` keeps the planner's own default set, since
    ``make_plan`` treats a falsy ``directions`` as "choose for me".
    """
    task = env._task
    if not getattr(task, "lateral_axis_only", False):
        return None
    horizontal = np.asarray(
        env._nominal_flat()["x_dot"], dtype=np.float64
    ).copy()
    horizontal[2] = 0.0
    if np.linalg.norm(horizontal) < 1e-6:
        horizontal = np.array([1.0, 0.0, 0.0])
    horizontal /= np.linalg.norm(horizontal)
    lateral = np.cross(np.array([0.0, 0.0, 1.0]), horizontal)
    lateral /= np.linalg.norm(lateral)
    return [lateral, -lateral]


class GlobalStaticPlan:
    """One offset profile for the whole episode, clearing every obstacle.

    Reactive local patching only exists because throws do not exist until
    they are spawned. Pre-placed static obstacles are all known at reset, so
    a single profile can be solved once and tracked -- strictly better than
    re-deciding per encounter, and it cannot reverse direction mid-dodge
    because there is nothing left to re-decide.

    Construction is greedy over obstacles in time order but verified against
    the whole set: each obstacle that the current profile fails to clear adds
    a smooth bump at its closest-approach time, pushed along the lateral
    direction that best separates it from the path.
    """

    def __init__(self, bumps, offset_max):
        self._bumps = bumps
        self._offset_max = np.asarray(offset_max, dtype=np.float64)

    def offset_at(self, time: float) -> np.ndarray:
        total = np.zeros(3)
        for centre, width, vector in self._bumps:
            u = (float(time) - centre) / width
            if abs(u) >= 1.0:
                continue
            # Raised cosine: smooth, compactly supported, peak 1 at u = 0.
            total = total + vector * 0.5 * (1.0 + np.cos(np.pi * u))
        return np.clip(total, -self._offset_max, self._offset_max)


def plan_global_static(env, margin_m: float = 0.35, bump_half_width_s: float = 2.5):
    """Solve one offset profile clearing all pre-placed static obstacles.

    ``bump_half_width_s`` is sized against travel, not reaction time. The
    nominal speed is ~0.6 m/s, so a 0.8 s half-width held the offset over
    only +-0.48 m of path -- shorter than the stretch during which an
    obstacle sitting *on* the path is within its own radius. The taper
    collapsed while the drone was still alongside it, leaving the minimum gap
    negative however tall the bump was: both bumps saturated at the 1.2 m cap
    and still measured gaps of -0.27 to +0.10 m.
    """
    manager = getattr(env.sim.visual_backend, "_dynamic_obstacles", None)
    active = getattr(manager, "_active", {}) if manager else {}
    if not active:
        return None

    to_dyn = env.sim.coord_trans.pos_transform_inv
    agent_radius = float(getattr(manager, "_agent_radius", 0.0))
    offset_max = np.asarray(env._task.offset_max_m, dtype=np.float64)
    end = float(env._nominal_trajectory.t_keyframes[-1])
    times = np.arange(0.0, end, 0.05)

    flats = [env._nominal_trajectory.update(float(t)) for t in times]
    path = np.asarray([f["x"] for f in flats], dtype=np.float64)
    vels = np.asarray([f["x_dot"] for f in flats], dtype=np.float64)

    lateral_axis_only = bool(getattr(env._task, "lateral_axis_only", False))

    trajectory_cfg = dict(getattr(env._task, "trajectory_config", {}) or {})
    static_clearance = float(trajectory_cfg.get("static_clearance_m", 0.20))
    ignored = set(getattr(manager, "_active", {}) or {})

    def mesh_clear(points) -> bool:
        """Does this offset path stay clear of the scene mesh?

        The reactive planner validated every candidate this way; the global
        solver did not, so nothing stopped it from steering the vehicle into
        a wall while dodging an obstacle.
        """
        try:
            validate_static_points_clearance(
                env.sim.visual_backend._sim,
                env.sim.coord_trans.transform_batch(np.asarray(points)),
                static_clearance,
                ignored_object_ids=ignored,
            )
        except ValueError:
            return False
        return True

    obstacles = []
    for item in active.values():
        centre = to_dyn @ np.asarray(item.obj.translation, dtype=np.float64)
        obstacles.append((centre, agent_radius + float(item.collision_radius)))

    # One bump per obstacle, placed at its closest approach to the nominal
    # path; iterations refine magnitudes rather than adding more bumps.
    # Adding a bump per iteration instead produced 19 bumps for 5 obstacles
    # -- each correction disturbed its neighbours -- and the overlapping sum
    # saturated offset_max everywhere, commanding a full lateral jump at t=0.
    # Width scaled to station spacing. A fixed 2.5 s worked when the profile
    # had to hold one offset through a long slow passage, but a slalom needs
    # to swing from one lane to the opposite one between stations -- adjacent
    # opposite-sign bumps of that width overlap and cancel, so neither lane is
    # reached (oracle fell to 17.5%). Half the spacing lets each bump peak
    # before its neighbour pulls the other way.
    peak_times = []
    for centre, radius in obstacles:
        gaps_probe = np.linalg.norm(path - centre, axis=1) - radius
        peak_times.append(float(times[int(np.argmin(gaps_probe))]))
    if len(peak_times) > 1:
        spacing = float(np.median(np.diff(np.sort(peak_times))))
        if spacing > 1e-3:
            bump_half_width_s = float(np.clip(0.55 * spacing, 0.5, bump_half_width_s))

    seeds = []
    for centre, radius in obstacles:
        gaps = np.linalg.norm(path - centre, axis=1) - radius
        index = int(np.argmin(gaps))
        speed_dir = vels[index]
        norm = float(np.linalg.norm(speed_dir))
        speed_dir = speed_dir / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
        away = path[index] - centre
        away = away - speed_dir * float(np.dot(away, speed_dir))
        if lateral_axis_only:
            # Restrict the push to the horizontal axis perpendicular to
            # travel. split_action zeroes body x and z under this setting, so
            # any vertical component of the plan is deleted before it reaches
            # the controller -- the plan would look valid and the vehicle
            # would execute a fraction of it.
            lateral_axis = np.cross(np.array([0.0, 0.0, 1.0]), speed_dir)
            norm = float(np.linalg.norm(lateral_axis))
            if norm < 1e-6:
                lateral_axis, norm = np.array([0.0, 1.0, 0.0]), 1.0
            lateral_axis = lateral_axis / norm
            projection = float(np.dot(away, lateral_axis))
            # Push along whichever sign separates path from obstacle; if the
            # obstacle is dead centre, either sign works.
            preferred = 1.0 if projection >= 0.0 else -1.0
            # Prefer the side that separates from the obstacle, but only if
            # it stays clear of the mesh; otherwise take the other side.
            away = lateral_axis * preferred
            probe_span = np.abs(times - float(times[index])) <= bump_half_width_s
            trial = path.copy()
            trial[probe_span] = trial[probe_span] + away * 0.9
            if not mesh_clear(trial[probe_span]):
                flipped = trial.copy()
                flipped[probe_span] = path[probe_span] - away * 0.9
                if mesh_clear(flipped[probe_span]):
                    away = -away
            norm = 1.0
        else:
            norm = float(np.linalg.norm(away))
            if norm < 1e-6:
                away = np.cross(speed_dir, np.array([0.0, 0.0, 1.0]))
                norm = float(np.linalg.norm(away))
                if norm < 1e-6:
                    away, norm = np.array([0.0, 1.0, 0.0]), 1.0
        seeds.append([float(times[index]), away / norm, 0.0, centre, radius])

    for _ in range(25):
        bumps = [
            (t, bump_half_width_s, direction * magnitude)
            for t, direction, magnitude, _, _ in seeds
            if magnitude > 0.0
        ]
        plan = GlobalStaticPlan(bumps, offset_max)
        offsets = np.asarray([plan.offset_at(float(t)) for t in times])
        worst = 0.0
        for seed in seeds:
            _, _, _, centre, radius = seed
            gap = float(np.min(np.linalg.norm(path + offsets - centre, axis=1)) - radius)
            deficit = margin_m - gap
            worst = max(worst, deficit)
            if deficit > 0.0:
                seed[2] = min(float(seed[2] + deficit * 1.1), float(np.max(offset_max)))
        if worst <= 0.0:
            break

    def build(scale: float) -> GlobalStaticPlan:
        return GlobalStaticPlan(
            [
                (t, bump_half_width_s, direction * magnitude * scale)
                for t, direction, magnitude, _, _ in seeds
                if magnitude > 0.0
            ],
            offset_max,
        )

    # Final mesh check: shrink the profile rather than fly into a wall. A
    # smaller offset may fail to clear an obstacle, which is a worse outcome
    # for that encounter but not a scene collision.
    # Shrink only the bumps that actually graze the mesh, rather than
    # collapsing the whole profile. Scaling everything to 0.0 on one bad lane
    # left the plan doing nothing at all -- planned gaps identical to nominal.
    plan = build(1.0)
    offsets = np.asarray([plan.offset_at(float(t)) for t in times])
    if not mesh_clear(path + offsets):
        for seed in seeds:
            if seed[2] <= 0.0:
                continue
            span = np.abs(times - seed[0]) <= bump_half_width_s
            trial = path.copy()
            local = np.asarray([plan.offset_at(float(t)) for t in times])
            trial = trial + local
            if mesh_clear(trial[span]):
                continue
            for scale in (0.75, 0.5, 0.25, 0.0):
                seed[2] = seed[2] * scale if scale else 0.0
                probe = build(1.0)
                local = np.asarray([probe.offset_at(float(t)) for t in times])
                if mesh_clear((path + local)[span]):
                    break
        plan = build(1.0)
    bumps = plan._bumps
    return plan if bumps else None


def nominal_position_for_plan(env):
    """Nominal path position at a time, for re-validating a cached plan."""

    def _at(time):
        return np.asarray(
            env._nominal_trajectory.update(float(time))["x"], dtype=np.float64
        )

    return _at


def mpc_expert_for_env(env) -> RecedingHorizonDodgeExpert:
    """Build sampling MPC at the environment's actual policy period."""
    policy_dt = float(
        env.policy_decimation * env.steps_per_action / env.sim.config.world_rate
    )
    planner_overrides = dict(
        (getattr(env._task, "trajectory_config", {}) or {}).get(
            "sampling_mpc", {}
        )
    )
    configured_horizon = float(planner_overrides.pop("horizon_s", 1.5))
    configured_replan_interval = float(
        planner_overrides.pop("replan_interval_s", 0.10)
    )
    # Keep the configured threat horizon while making every shooting
    # node coincide with an action the environment can actually change.
    horizon = max(configured_horizon, policy_dt * 8.0)
    steps = max(8, int(round(horizon / policy_dt)))
    horizon = steps * policy_dt
    config = replace(
        RecedingHorizonConfig(),
        horizon_s=horizon,
        step_dt_s=policy_dt,
        replan_interval_s=max(2.0 * policy_dt, configured_replan_interval),
        **planner_overrides,
    )
    return RecedingHorizonDodgeExpert(config)


def _mpc_max_target_velocity_gap(env, expert) -> float:
    """Largest raw target gap consistent with the MPC acceleration limit."""
    acceleration_limit = float(
        expert.config.max_residual_command_acceleration_mps2
    )
    if acceleration_limit <= 0.0:
        return np.inf
    control_dt = float(env.steps_per_action / env.sim.config.world_rate)
    tau = float(env._task.action_filter_tau_s)
    filter_alpha = 1.0 if tau <= control_dt else control_dt / tau
    return acceleration_limit * control_dt / filter_alpha


def _limit_mpc_action_acceleration(env, expert, action: np.ndarray) -> np.ndarray:
    """Project an oracle target onto the physical residual-acceleration ball."""
    limited = np.asarray(action, dtype=np.float32).copy()
    filtered = np.asarray(
        getattr(env, "_filtered_action", np.zeros(3)), dtype=np.float64
    )
    limits = np.asarray(env._task.delta_velocity_limits_mps, dtype=np.float64)
    gap = (np.asarray(limited[:3], dtype=np.float64) - filtered) * limits
    norm = float(np.linalg.norm(gap))
    maximum = _mpc_max_target_velocity_gap(env, expert)
    if norm > maximum:
        limited[:3] = (
            filtered + (np.asarray(limited[:3]) - filtered) * (maximum / norm)
        ).astype(np.float32)
    return limited


def _mpc_static_checker(env, active_object_ids):
    """Return the same mesh/bounds validator used by the quintic oracle."""
    trajectory_cfg = dict(getattr(env._task, "trajectory_config", {}) or {})
    static_clearance = float(trajectory_cfg.get("static_clearance_m", 0.20))
    bounds_margin = float(
        getattr(env._task, "boundary_margin_threshold_m", 0.0) or 0.0
    ) * float(getattr(env._task, "expert_bounds_margin_scale", 1.75))

    def static_path_is_clear(dynamics_points):
        habitat_points = env.sim.coord_trans.transform_batch(dynamics_points)
        try:
            validate_static_points_clearance(
                env.sim.visual_backend._sim,
                habitat_points,
                static_clearance,
                ignored_object_ids=set(active_object_ids),
            )
        except ValueError:
            return False
        for point in habitat_points:
            if not env.sim.safety.is_in_bounds(point):
                return False
            if bounds_margin > 0.0 and env.sim.safety.bounds_margin(point) < bounds_margin:
                return False
        return True

    return static_path_is_clear


def _mpc_visibility_gate_open(env, expert, state, active, now: float) -> bool:
    """Release MPC only once an active obstacle is within camera-scale range.

    The gate is opt-in through ``trajectory.visibility_release_distance_m``.
    It tracks immutable encounter IDs rather than simulator object IDs, which
    Habitat may recycle after despawn. Clearing the cached plan while closed
    guarantees that the first released action is replanned from the actually
    undodged vehicle state instead of following a trajectory computed early.
    """
    trajectory = dict(getattr(env._task, "trajectory_config", {}) or {})
    release_distance = float(
        trajectory.get("visibility_release_distance_m", 0.0) or 0.0
    )
    if release_distance <= 0.0:
        return True

    released = set(getattr(env, "_mpc_visibility_released_ids", set()))
    position = np.asarray(state["x"], dtype=np.float64)
    to_dynamics = env.sim.coord_trans.pos_transform_inv
    active_ids = set()
    newly_released = []
    for _simulator_object_id, item in active.items():
        encounter_id = int(getattr(item, "encounter_id", item.object_id))
        active_ids.add(encounter_id)
        obstacle_position = (
            to_dynamics @ np.asarray(item.obj.translation, dtype=np.float64)
        )
        distance = float(np.linalg.norm(obstacle_position - position))
        if distance <= release_distance and encounter_id not in released:
            released.add(encounter_id)
            newly_released.append((encounter_id, distance))

    env._mpc_visibility_released_ids = released
    if newly_released:
        # Force should_replan() on this same policy tick. In the usual
        # single-obstacle case no plan exists yet, but this also handles a
        # later encounter arriving while an earlier cached plan is present.
        expert.plan = None
        events = list(getattr(env, "_mpc_visibility_release_events", []))
        events.extend(
            {
                "encounter_id": encounter_id,
                "time_s": float(now),
                "center_distance_m": distance,
            }
            for encounter_id, distance in newly_released
        )
        env._mpc_visibility_release_events = events

    is_open = bool(active_ids & released)
    if not is_open:
        expert.plan = None
    return is_open


def mpc_oracle_action(env) -> np.ndarray:
    """Replan a privileged velocity-command trajectory in closed loop."""
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    env._trajectory_expert_frame_unlabelled = False
    if env._task.residual_control_mode != "velocity_command":
        raise ValueError(
            "sampling MPC currently models residual_control.mode=velocity_command; "
            f"got {env._task.residual_control_mode!r}"
        )

    expert = getattr(env, "_receding_horizon_dodge_expert", None)
    if expert is None:
        expert = mpc_expert_for_env(env)
        env._receding_horizon_dodge_expert = expert

    state = env.sim.dynamics.state
    now = float(env.sim.time)
    flat = env._nominal_flat()
    nominal_now = np.asarray(flat["x"], dtype=np.float64)
    nominal_velocity = np.asarray(flat["x_dot"], dtype=np.float64)
    offset = np.asarray(state["x"], dtype=np.float64) - nominal_now
    relative_velocity = np.asarray(state["v"], dtype=np.float64) - nominal_velocity

    cfg = expert.config
    times = cfg.step_dt_s * (np.arange(int(round(cfg.horizon_s / cfg.step_dt_s))) + 1)
    nominal_positions = np.asarray(
        [
            env._nominal_trajectory.update(now + float(delta))["x"]
            for delta in times
        ],
        dtype=np.float64,
    )
    active = env.active_obstacles()
    if not _mpc_visibility_gate_open(env, expert, state, active, now):
        return _limit_mpc_action_acceleration(env, expert, action)
    manager = env.sim.visual_backend._dynamic_obstacles
    to_dynamics = env.sim.coord_trans.pos_transform_inv
    predictions = []
    exact_tcas = []
    for _simulator_object_id, item in active.items():
        position = to_dynamics @ np.asarray(item.obj.translation, dtype=np.float64)
        velocity = to_dynamics @ env.obstacle_velocity(item, now)
        acceleration = to_dynamics @ env.obstacle_acceleration(item)
        swept = (
            position[None, :]
            + times[:, None] * velocity[None, :]
            + 0.5 * times[:, None] ** 2 * acceleration[None, :]
        )
        combined_radius = float(manager._agent_radius) + float(item.collision_radius)
        clearance = np.linalg.norm(nominal_positions - swept, axis=1) - combined_radius
        closest_index = int(np.argmin(clearance))
        # Exclude harmless bystanders, but use the correct acceleration model
        # rather than the task's historical constant-velocity threat sort.
        if float(clearance[closest_index]) > float(env._task.threat_distance_m):
            continue
        predictions.append(
            MovingSpherePrediction(
                object_id=int(getattr(item, "encounter_id", item.object_id)),
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                combined_radius=combined_radius,
            )
        )
        exact_tcas.append(float(times[closest_index]))

    if not predictions:
        # velocity_command already includes a bounded return-to-path term.
        # Continuing stochastic optimisation after the obstacle has passed
        # only fights that deterministic recovery and can preserve a small
        # standing offset. A zero policy command is the exact nominal-recovery
        # action for this mode.
        expert.plan = None
        return _limit_mpc_action_acceleration(env, expert, action)

    if expert.should_replan(now):
        quaternion = np.asarray(state["q"], dtype=np.float64)
        body_to_world = np.column_stack(
            [
                rotate_vector_by_quat(axis, quaternion)
                for axis in np.eye(3, dtype=np.float64)
            ]
        )
        mask = np.ones(3, dtype=np.float64)
        if getattr(env._task, "lateral_axis_only", False):
            mask[[0, 2]] = 0.0
        elif getattr(env._task, "lateral_only", False):
            mask[0] = 0.0

        control_dt = float(env.steps_per_action / env.sim.config.world_rate)
        per_control_alpha = min(control_dt / env._task.action_filter_tau_s, 1.0)
        action_filter_alpha = 1.0 - (1.0 - per_control_alpha) ** int(
            env.policy_decimation
        )
        plan = expert.make_plan(
            now=now,
            time_to_closest_approach=(min(exact_tcas) if exact_tcas else None),
            initial_offset=offset,
            initial_relative_velocity=relative_velocity,
            initial_filtered_action=np.asarray(
                getattr(env, "_filtered_action", np.zeros(3)), dtype=np.float64
            ),
            body_to_world=body_to_world,
            delta_velocity_limits=np.asarray(
                env._task.delta_velocity_limits_mps, dtype=np.float64
            ),
            action_filter_alpha=action_filter_alpha,
            max_target_delta_velocity_mps=_mpc_max_target_velocity_gap(
                env, expert
            ),
            return_gain_hz=float(env._task.return_gain_hz),
            max_return_speed_mps=float(env._task.max_return_speed_mps),
            nominal_positions=nominal_positions,
            nominal_velocity=nominal_velocity,
            obstacles=predictions,
            action_mask=mask,
            static_path_is_clear=_mpc_static_checker(env, active),
        )
        env._trajectory_expert_plans = getattr(env, "_trajectory_expert_plans", 0) + 1
        if plan.safety_slack > 1e-6:
            env._mpc_slack_replans = getattr(env, "_mpc_slack_replans", 0) + 1
        diagnostics = getattr(env, "_trajectory_expert_plan_diagnostics", [])
        diagnostics.append(
            {
                "planner": "sampling_mpc",
                "object_ids": list(plan.object_ids),
                "predicted_min_clearance_m": plan.predicted_min_clearance,
                "safety_slack_m": plan.safety_slack,
                "peak_offset_m": float(np.max(np.linalg.norm(plan.offsets, axis=1))),
                "objective": plan.objective,
                "static_clear": plan.static_clear,
            }
        )
        env._trajectory_expert_plan_diagnostics = diagnostics

    if expert.plan is None:
        # Defensive only: make_plan always returns a finite-slack trajectory.
        env._trajectory_expert_failed_plans = (
            getattr(env, "_trajectory_expert_failed_plans", 0) + 1
        )
        return _limit_mpc_action_acceleration(env, expert, action)
    action[:3] = np.clip(expert.plan.action_at(now), -1.0, 1.0).astype(np.float32)
    return _limit_mpc_action_acceleration(env, expert, action)


def oracle_action(env) -> np.ndarray:
    """Track a cached, collision-checked local avoidance trajectory."""
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    env._trajectory_expert_frame_unlabelled = False
    expert = getattr(env, "_trajectory_dodge_expert", None)
    if expert is None:
        expert = expert_for_task(env._task)
        env._trajectory_dodge_expert = expert

    state = env.sim.dynamics.state
    now = float(env.sim.time)

    # Gate the global static planner on the task actually having pre-placed
    # static obstacles. It treats every active obstacle as stationary, so on
    # the dynamic task it would fly a profile solved against where the throws
    # happened to be at the first call -- for the whole episode. Today that
    # never fires only because no throw is airborne on step 1 (verified 0/10
    # episodes), which is an accident of spawn timing, not a guarantee: a
    # pre-placed obstacle or a zero spawn delay would silently turn the
    # reactive oracle into a static one, and it would read as a mysteriously
    # bad planner rather than a wrong code path. Both tasks share
    # mode: desired_body_offset, so nothing else separates them.
    backend = getattr(getattr(env, "sim", None), "visual_backend", None)
    manager = getattr(backend, "_dynamic_obstacles", None)
    has_static = int(getattr(getattr(manager, "cfg", None), "static_obstacle_count", 0) or 0) > 0
    if has_static and getattr(env._task, "residual_control_mode", None) == "desired_body_offset":
        global_plan = getattr(env, "_global_static_plan", "unset")
        if global_plan == "unset":
            global_plan = plan_global_static(env)
            env._global_static_plan = global_plan
        if global_plan is not None:
            body = rotate_vector_by_quat(
                global_plan.offset_at(now),
                np.asarray(state["q"], dtype=np.float64),
                inverse=True,
            )
            action[:3] = np.clip(body / env._task.offset_max_m, -1.0, 1.0)
            return action
    plan = expert.plan
    if plan is not None and now >= plan.end_time:
        expert.plan = None
        plan = None

    rows = env._obstacle_relative_states(state)
    actionable = [
        row
        for row in rows
        if float(row["predicted_clearance"]) <= env._task.threat_distance_m
        and 0.0
        < float(row["time_to_closest_approach"])
        <= env._task.threat_time_horizon_s
    ]
    if plan is not None:
        new_threats = [
            row
            for row in actionable
            if int(row["object_id"]) not in plan.validated_object_ids
        ]
        if new_threats:
            # A newly actionable obstacle does not by itself invalidate the
            # plan -- with static obstacles nothing about the world changed,
            # only which obstacles happened to cross a proximity threshold.
            # Discarding the plan there made the vehicle reverse direction
            # mid-dodge (visible on video as going one way and then abruptly
            # the other), because a fresh search can flip on a marginally
            # different geometry. Keep a plan that still clears everything.
            active_now = env.active_obstacles()
            to_dyn = env.sim.coord_trans.pos_transform_inv
            manager_now = env.sim.visual_backend._dynamic_obstacles
            obstacle_states = [
                (
                    to_dyn @ np.asarray(other.obj.translation, dtype=np.float64),
                    to_dyn @ env.obstacle_velocity(other, now),
                    to_dyn @ env.obstacle_acceleration(other),
                    float(manager_now._agent_radius) + float(other.collision_radius),
                )
                for other in active_now.values()
            ]
            if expert.plan_clears(
                plan, now, nominal_position_for_plan(env), obstacle_states
            ):
                new_threats = []
        if not new_threats:
            nominal_position_now = np.asarray(
                env._nominal_flat()["x"], dtype=np.float64
            )
            actual_displacement = (
                np.asarray(state["x"], dtype=np.float64) - nominal_position_now
            )
            residual_world = expert.world_velocity_residual(
                now,
                env._task.outer_position_gain_hz,
                gate=1.0,
                actual_displacement=actual_displacement,
            )
            if env._task.residual_control_mode == "desired_body_offset":
                # This mode's action IS the target displacement, not a
                # velocity, so feeding it the velocity residual would be a
                # units error that silently understates the expert. The plan
                # already carries the displacement it wants to be at.
                residual_body = rotate_vector_by_quat(
                    np.asarray(plan.evaluate(now)[0], dtype=np.float64),
                    np.asarray(state["q"], dtype=np.float64),
                    inverse=True,
                )
                normalized = np.clip(
                    residual_body / env._task.offset_max_m, -1.0, 1.0
                )
            else:
                residual_body = rotate_vector_by_quat(
                    residual_world, np.asarray(state["q"], dtype=np.float64), inverse=True
                )
                normalized = np.clip(
                    residual_body / env._task.delta_velocity_limits_mps, -1.0, 1.0
                )
            # This oracle was written against the gated 4-dim action space
            # (gate + 3-axis residual), but several modes have no gate
            # channel and the action IS the 3-axis residual. Key the layout
            # off the action width rather than a hand-maintained mode list:
            # the list omitted velocity_command, whose 3-wide action made
            # `action[1:4] = normalized` raise a broadcast error, and any
            # future ungated mode would have hit the same edge.
            if action.shape[0] >= 4:
                action[0] = 1.0
                action[1:4] = normalized
            else:
                action[:3] = normalized
            return action
        # A newly spawned/actionable obstacle invalidates the cached
        # single-encounter plan. Replan continuously from the displacement
        # and relative velocity the vehicle has actually reached.
        expert.plan = None

    if not actionable:
        return action

    threat = actionable[0]
    object_id = int(threat["object_id"])
    simulator_object_id = int(threat.get("simulator_object_id", object_id))
    tca = float(threat["time_to_closest_approach"])

    flat = env._nominal_flat()
    nominal_position_now = np.asarray(flat["x"], dtype=np.float64)
    nominal_velocity_now = np.asarray(flat["x_dot"], dtype=np.float64)
    active = env.active_obstacles()
    item = active.get(simulator_object_id)
    if item is None:
        return action
    to_dynamics = env.sim.coord_trans.pos_transform_inv
    obstacle_position = to_dynamics @ np.asarray(item.obj.translation, dtype=np.float64)
    obstacle_velocity = to_dynamics @ env.obstacle_velocity(item, now)
    obstacle_acceleration = to_dynamics @ env.obstacle_acceleration(item)
    manager = env.sim.visual_backend._dynamic_obstacles
    combined_radius = float(manager._agent_radius) + float(item.collision_radius)
    trajectory_cfg = dict(getattr(env._task, "trajectory_config", {}) or {})
    static_clearance = float(trajectory_cfg.get("static_clearance_m", 0.20))

    def nominal_position(time):
        return np.asarray(env._nominal_trajectory.update(float(time))["x"], dtype=np.float64)

    # Planned paths need more clearance than the termination threshold,
    # because the vehicle overshoots the reference while tracking it. Raising
    # the expert's speed envelope from 0.70 to 1.5 m/s made dodges twice as
    # fast and pushed out_of_bounds from 2/30 episodes to 10/30 -- the
    # avoidance itself improved (encounter_clear_rate 0.589 -> 0.872,
    # collisions 17 -> 7) but the excursions gave the gain straight back.
    bounds_margin = float(
        getattr(env._task, "boundary_margin_threshold_m", 0.0) or 0.0
    ) * float(
        getattr(env._task, "expert_bounds_margin_scale", 1.75)
    )

    def static_path_is_clear(dynamics_points):
        habitat_points = env.sim.coord_trans.transform_batch(dynamics_points)
        try:
            validate_static_points_clearance(
                env.sim.visual_backend._sim,
                habitat_points,
                static_clearance,
                ignored_object_ids=set(active),
            )
        except ValueError:
            return False
        # Mesh clearance is not the same as staying inside the flyable
        # volume, and the planner was only checking the former: measured on
        # 20 seeds, 5 oracle episodes ended out_of_bounds, and on four of
        # them the *nominal* path completes all 751 steps with obstacles
        # disabled. The expert was avoiding obstacles by flying out of the
        # world box -- a quarter of the ceiling lost to a planner blind spot
        # rather than to hard avoidance.
        for point in habitat_points:
            if not env.sim.safety.is_in_bounds(point):
                return False
            if bounds_margin > 0.0 and env.sim.safety.bounds_margin(point) < bounds_margin:
                return False
        return True

    plan = expert.make_plan(
        object_id=object_id,
        now=now,
        time_to_closest_approach=tca,
        obstacle_position=obstacle_position,
        obstacle_velocity=obstacle_velocity,
        obstacle_acceleration=obstacle_acceleration,
        combined_radius=combined_radius,
        nominal_position=nominal_position,
        nominal_velocity=nominal_velocity_now,
        static_path_is_clear=static_path_is_clear,
        directions=expert_directions(env),
        additional_obstacles=[
            (
                int(other_id),
                to_dynamics
                @ np.asarray(other.obj.translation, dtype=np.float64),
                to_dynamics @ env.obstacle_velocity(other, now),
                to_dynamics @ env.obstacle_acceleration(other),
                float(manager._agent_radius) + float(other.collision_radius),
            )
            for other_id, other in active.items()
            if int(other_id) != object_id
        ],
        start_offset=np.asarray(state["x"], dtype=np.float64)
        - nominal_position_now,
        start_velocity=np.asarray(state["v"], dtype=np.float64)
        - nominal_velocity_now,
    )
    if plan is None:
        env._trajectory_expert_failed_plans = (
            getattr(env, "_trajectory_expert_failed_plans", 0) + 1
        )
        # This frame has no usable expert label: the zero action below means
        # "found no safe dodge", not "nothing to dodge". Recorded so the
        # collector can drop the frame rather than teaching the policy to
        # sit still while an obstacle closes.
        env._trajectory_expert_frame_unlabelled = True
        return action
    env._trajectory_expert_plans = getattr(env, "_trajectory_expert_plans", 0) + 1
    diagnostics = getattr(env, "_trajectory_expert_plan_diagnostics", [])
    sample_times = np.linspace(plan.start_time, plan.end_time, 101)
    peak_speed = max(
        float(np.linalg.norm(plan.evaluate(float(time))[1])) for time in sample_times
    )
    diagnostics.append(
        {
            "object_id": plan.object_id,
            "peak_offset_m": float(np.linalg.norm(plan.peak_offset)),
            "predicted_min_clearance_m": plan.predicted_min_clearance,
            "rise_time_s": plan.peak_time - plan.start_time,
            "peak_planned_residual_mps": peak_speed,
        }
    )
    env._trajectory_expert_plan_diagnostics = diagnostics
    return oracle_action(env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default="applications/rl/configs/velocity_dodge_privileged_easy_offset.yaml",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--planner",
        choices=("quintic", "sampling_mpc"),
        default="quintic",
        help=(
            "Privileged planner to execute. The default preserves historical "
            "oracle results; sampling_mpc continuously replans velocity commands."
        ),
    )
    parser.add_argument(
        "--rollout-policy",
        default=None,
        help=(
            "Clone .pt to fly during collection (DAgger). The oracle still "
            "labels every frame; only who holds the stick changes."
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help=(
            "Probability of executing the expert's action instead of the "
            "clone's. 1.0 reproduces plain BC collection, 0.0 is pure "
            "on-policy DAgger; decay it across iterations."
        ),
    )
    parser.add_argument("--seed0", type=int, default=130000)
    parser.add_argument(
        "--visibility-release-distance-m",
        type=float,
        default=None,
        help=(
            "Optional center-distance gate for sampling MPC. The expert holds "
            "zero and replans only once an obstacle reaches this range."
        ),
    )
    parser.add_argument(
        "--output", default="outputs/rl/velocity_dodge_oracle_easy_summary.json"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional .h5 (streaming, recommended) or NPZ imitation dataset",
    )
    parser.add_argument(
        "--include-all-episodes",
        action="store_true",
        help="Include unsuccessful episodes in the optional dataset",
    )
    parser.add_argument(
        "--include-plan-failures",
        action="store_true",
        help="Keep episodes containing frames where the expert found no feasible plan",
    )
    parser.add_argument(
        "--record-privileged",
        action="store_true",
        help=(
            "Add the teacher's privileged geometry to each dict-observation "
            "row. This is supervision only; the rollout policy never sees it."
        ),
    )
    parser.add_argument(
        "--record-obstacle-image-label",
        action="store_true",
        help=(
            "Store presence, normalized image centre/radius, and normalized "
            "depth for the most imminent obstacle. Labels are never exposed "
            "to the rollout policy."
        ),
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.experiment_config)
    env_cfg = copy.deepcopy(cfg["env"])
    if args.visibility_release_distance_m is not None:
        trajectory = env_cfg["task"]["config"].setdefault("trajectory", {})
        trajectory["visibility_release_distance_m"] = float(
            args.visibility_release_distance_m
        )
    env_cfg["enable_visualization"] = False
    env_cls = env_class_for_task(env_cfg["task"]["name"])
    env = env_cls(env_config=env_cfg, train=False)
    rng = np.random.default_rng(args.seed0)
    rollout_model = None
    if args.rollout_policy:
        import torch  # noqa: PLC0415

        device = torch.device(str(cfg["ppo"].get("device", "cuda:0")))
        rollout_model = load_rollout_policy(
            args.rollout_policy, cfg, env, device
        )
        print(f"DAgger: flying {args.rollout_policy} at beta={args.beta}")
    else:
        device = None
    results = []
    successful_observations = []
    successful_actions = []
    dataset_path = Path(args.dataset) if args.dataset else None
    hdf5_writer = None
    if dataset_path is not None and dataset_path.suffix.lower() in {".h5", ".hdf5"}:
        hdf5_writer = HDF5ImitationWriter(dataset_path, args.experiment_config)
    try:
        for episode in range(args.episodes):
            seed = args.seed0 + episode
            obs, _ = env.reset(seed=seed)
            if rollout_model is not None:
                if getattr(
                    rollout_model, "_dagger_recurrent_type", None
                ) == "compact_gru":
                    rollout_model._dagger_hidden = None
                    rollout_model._dagger_controller.reset()
                elif hasattr(rollout_model, "reset"):
                    rollout_model.reset()
            env._trajectory_dodge_expert = expert_for_task(env._task)
            env._trajectory_expert_plans = 0
            env._trajectory_expert_failed_plans = 0
            env._trajectory_expert_plan_diagnostics = []
            env._mpc_slack_replans = 0
            env._mpc_visibility_released_ids = set()
            env._mpc_visibility_release_events = []
            if args.planner == "sampling_mpc":
                env._receding_horizon_dodge_expert = mpc_expert_for_env(env)
            episode_observations = []
            episode_actions = []
            total_reward = 0.0
            peak_cross = 0.0
            policy_dt = float(
                env.policy_decimation
                * env.steps_per_action
                / env.sim.config.world_rate
            )
            previous_velocity = np.asarray(
                env.sim.dynamics.state["v"], dtype=np.float64
            ).copy()
            previous_delta_velocity = np.asarray(
                getattr(env, "_last_delta_velocity", np.zeros(3)), dtype=np.float64
            ).copy()
            previous_acceleration = None
            previous_command_acceleration = None
            acceleration_norms = []
            jerk_norms = []
            command_acceleration_norms = []
            command_jerk_norms = []
            while True:
                action = (
                    mpc_oracle_action(env)
                    if args.planner == "sampling_mpc"
                    else oracle_action(env)
                )
                # DAgger: label with the expert, but let the clone drive.
                # Plain BC only ever sees states the expert visits, so the
                # learner has no example of recovering from its own errors
                # -- measured as 0% success with 7/20 tracking failures,
                # the vehicle compounding small mistakes until pos_error
                # passed 3.0 m. Relabelling states the *clone* reaches is
                # what supplies those corrections.
                executed = action
                if rollout_model is not None and rng.random() >= args.beta:
                    executed = clone_action(rollout_model, obs, device)
                # Frames where the expert found no feasible dodge carry a
                # zero action that means "gave up", not "nothing to do".
                # Recording them teaches the policy to sit still while an
                # obstacle closes. Skip only the frame; the episode still
                # steps normally and its remaining frames are kept -- one
                # with 147 such frames out of ~560 is mostly good data, and
                # the per-episode filter threw all of it away (yield 0/8).
                record_frame = not bool(
                    getattr(env, "_trajectory_expert_frame_unlabelled", False)
                )
                if record_frame and isinstance(obs, dict):
                    recorded_observation = {
                            key: np.asarray(
                                value,
                                dtype=(
                                    np.float16
                                    if hdf5_writer is not None and key == "events"
                                    else np.float32
                                ),
                            ).copy()
                            for key, value in obs.items()
                        }
                    if args.record_privileged:
                        recorded_observation["privileged"] = np.asarray(
                            env._task.make_privileged_observation(
                                state=env.sim.dynamics.state
                            ),
                            dtype=np.float32,
                        ).copy()
                    if args.record_obstacle_image_label:
                        recorded_observation["obstacle_image"] = (
                            obstacle_image_label(env)
                        )
                    episode_observations.append(recorded_observation)
                elif record_frame:
                    episode_observations.append(
                        np.asarray(obs, dtype=np.float32).copy()
                    )
                if record_frame:
                    # Label with what the env will actually apply, not what
                    # the expert asked for. Under lateral_only /
                    # lateral_axis_only the task zeroes the restricted axes
                    # before they reach the controller, so recording the raw
                    # 3-vector would train the clone to reproduce components
                    # that do nothing -- diluting the direction loss across
                    # dimensions the action space cannot express.
                    episode_actions.append(effective_expert_action(env, action))
                obs, reward, terminated, truncated, info = env.step(executed)
                velocity = np.asarray(
                    env.sim.dynamics.state["v"], dtype=np.float64
                ).copy()
                acceleration = (velocity - previous_velocity) / policy_dt
                acceleration_norms.append(float(np.linalg.norm(acceleration)))
                if previous_acceleration is not None:
                    jerk_norms.append(
                        float(np.linalg.norm(acceleration - previous_acceleration))
                        / policy_dt
                    )
                previous_velocity = velocity
                previous_acceleration = acceleration

                delta_velocity = np.asarray(
                    getattr(env, "_last_delta_velocity", np.zeros(3)),
                    dtype=np.float64,
                ).copy()
                command_acceleration = (
                    delta_velocity - previous_delta_velocity
                ) / policy_dt
                command_acceleration_norms.append(
                    float(np.linalg.norm(command_acceleration))
                )
                if previous_command_acceleration is not None:
                    command_jerk_norms.append(
                        float(
                            np.linalg.norm(
                                command_acceleration - previous_command_acceleration
                            )
                        )
                        / policy_dt
                    )
                previous_delta_velocity = delta_velocity
                previous_command_acceleration = command_acceleration
                total_reward += float(reward)
                terms = info.get("reward_terms", {}) or {}
                peak_cross = max(peak_cross, float(terms.get("offset_cross_track", 0.0)))
                if terminated or truncated:
                    row = {
                        "episode": episode,
                        "seed": seed,
                        "success": bool(info.get("is_success", False)),
                        "termination_reason": info.get("termination_reason", "timeout"),
                        "reward": total_reward,
                        "min_clearance": float(
                            terms.get("episode_min_clearance", np.inf)
                        ),
                        "peak_cross_track": peak_cross,
                        "peak_acceleration_mps2": float(max(acceleration_norms)),
                        "p95_acceleration_mps2": float(
                            np.percentile(acceleration_norms, 95)
                        ),
                        "rms_acceleration_mps2": float(
                            np.sqrt(np.mean(np.square(acceleration_norms)))
                        ),
                        "peak_jerk_mps3": float(max(jerk_norms, default=0.0)),
                        "peak_residual_command_acceleration_mps2": float(
                            max(command_acceleration_norms)
                        ),
                        "p95_residual_command_acceleration_mps2": float(
                            np.percentile(command_acceleration_norms, 95)
                        ),
                        "peak_residual_command_jerk_mps3": float(
                            max(command_jerk_norms, default=0.0)
                        ),
                        "expert_plans": env._trajectory_expert_plans,
                        "expert_failed_plans": env._trajectory_expert_failed_plans,
                        "mpc_slack_replans": env._mpc_slack_replans,
                        "visibility_release_events": list(
                            env._mpc_visibility_release_events
                        ),
                        "expert_plan_diagnostics": env._trajectory_expert_plan_diagnostics,
                    }
                    results.append(row)
                    # ``--include-all-episodes`` is primarily used to collect
                    # quiet no-obstacle rollouts, whose success flag is false
                    # because no encounter occurred. Do not turn controller
                    # runaways or invalid trajectories into zero-action expert
                    # labels; only ordinary timeout episodes are valid data.
                    # DAgger keeps every episode regardless of outcome. The
                    # whole point is to label states the clone reaches, and
                    # the crashes are the most informative of those -- the
                    # frames where the expert says "dodge" and the clone did
                    # not are exactly the corrections plain BC never sees.
                    # Individual frames are still dropped when the expert has
                    # no feasible plan, so no zero-action labels sneak in.
                    include_episode = args.rollout_policy is not None or (
                        should_include_episode(row, args.include_all_episodes)
                        and (
                            args.include_plan_failures
                            or int(row["expert_failed_plans"]) == 0
                        )
                    )
                    if include_episode:
                        if hdf5_writer is not None:
                            hdf5_writer.append_episode(
                                episode_observations,
                                episode_actions,
                                episode=episode,
                                seed=seed,
                            )
                        else:
                            successful_observations.extend(episode_observations)
                            successful_actions.extend(episode_actions)
                    print(row, flush=True)
                    break
    finally:
        env.close()
        if hdf5_writer is not None:
            hdf5_writer.close()

    payload = {
        "experiment_config": args.experiment_config,
        "planner": args.planner,
        "episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    if args.dataset and hdf5_writer is None:
        dataset = Path(args.dataset)
        dataset.parent.mkdir(parents=True, exist_ok=True)
        actions = np.asarray(successful_actions, dtype=np.float32)
        threat = np.linalg.norm(actions, axis=1) > 0.1
        if successful_observations and isinstance(successful_observations[0], dict):
            dataset_payload = {
                f"observations_{key}": np.asarray(
                    [obs[key] for obs in successful_observations],
                    # Half precision is ample for bounded time surfaces and
                    # keeps event-imitation datasets tractable.
                    dtype=np.float16 if key == "events" else np.float32,
                )
                for key in successful_observations[0]
            }
            dataset_payload.update(actions=actions, threat=threat)
            np.savez_compressed(dataset, **dataset_payload)
        else:
            observations = np.asarray(successful_observations, dtype=np.float32)
            np.savez_compressed(
                dataset,
                observations=observations,
                actions=actions,
                threat=threat,
            )
        print(
            f"dataset={dataset} samples={len(actions)} threat={int(threat.sum())} "
            f"non_threat={int((~threat).sum())}"
        )
    elif hdf5_writer is not None:
        print(
            f"dataset={dataset_path} samples={hdf5_writer.samples} "
            f"threat={hdf5_writer.threat_samples} "
            f"non_threat={hdf5_writer.samples - hdf5_writer.threat_samples}"
        )
    print(f"success_rate={payload['success_rate']:.1%}; summary={output}")


if __name__ == "__main__":
    main()
