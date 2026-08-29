"""Dynamic obstacle spawning and lifecycle management for visual backends.

This module is intentionally backend-agnostic and interacts with Habitat-sim
through manager/module objects injected by the wrapper.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def solve_spawn_retarget(
    entry: dict[str, Any],
    drone_position: np.ndarray,
    drone_velocity: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a scheduled projectile against current constant-velocity motion."""
    target_velocity = (
        np.asarray(drone_velocity, dtype=np.float64)
        if drone_velocity is not None
        else np.asarray(entry.get("target_velocity", np.zeros(3)), dtype=np.float64)
    )
    speed = float(entry["speed_mps"])
    lead = float(entry["lead_distance_m"])
    approach = np.asarray(entry["approach"], dtype=np.float64)
    aim_offset = np.asarray(entry.get("aim_offset", np.zeros(3)), dtype=np.float64)
    flight_time = lead / speed
    target = np.asarray(drone_position, dtype=np.float64) + target_velocity * flight_time
    return target + approach * lead + aim_offset, -approach * speed


@dataclass(slots=True)
class DynamicObstacleTemplate:
    """Template configuration for one obstacle type."""

    handle: str
    motion_mode: str = "dynamic_throw"
    scale: tuple[float, float, float] | None = None
    mass: float | None = None
    ttl_s: float = 3.0
    kinematic_speed_mps: float = 5.0
    parabola_gravity_mps2: float = 9.81
    # Explicit collision radius (m). When None it is derived from the
    # object's AABB bounding sphere, which overestimates a rounded object
    # by up to sqrt(3) -- a 0.12 cube reads 0.173. Set this when the
    # effective obstacle size matters to the task (it sets the clearance
    # a dodge has to achieve) rather than tuning `scale` to hit it.
    collision_radius: float | None = None


@dataclass(slots=True)
class DynamicObstaclesConfig:
    """Runtime configuration for dynamic obstacle spawning.

    ``dataset_config_file``: optional path to a Habitat scene-dataset config
    (e.g. ``data/objects/ycb/ycb.scene_dataset_config.json``).  When set the
    manager loads that dataset into the ``MetadataMediator`` before resolving
    template handles, enabling real-world objects such as YCB bananas,
    cracker boxes, soup cans, etc. in addition to the built-in Habitat
    primitives (cube, sphere, capsule …).

    Download YCB assets with:
        python -m habitat_sim.utils.datasets_download --uids ycb --data-path data/

    Then reference objects by their short YCB name, e.g. ``"011_banana"`` or
    ``"003_cracker_box"`` – the manager resolves them via substring matching.

    ``azimuth_range_deg`` is relative to the drone's heading, not an absolute
    world-space angle. ``0`` means straight ahead, positive angles move around
    the drone's right-hand side in the XZ plane, and values near ``±180`` place
    the obstacle behind the drone. The default range keeps spawns in a forward
    cone instead of anywhere around the full circle.
    """

    enabled: bool = False
    spawn_interval_s: float = 2.0
    # Jitter applied to the spawn interval, as a fraction of it. Firing on a
    # fixed period is exploitable: at a hard 2.5 s cadence a trained policy's
    # speed varied 78% of its mean with phase in the spawn cycle, while its
    # action magnitude varied only 2% -- it had learned the rhythm and was
    # timing its behaviour to the clock rather than to anything it saw.
    # Nothing in the observation encodes time, so that structure can only
    # come from the period itself. Sampling each wait uniformly in
    # [(1-j), (1+j)] * interval keeps the mean rate while making the phase
    # carry no information.
    spawn_interval_jitter: float = 0.0
    max_concurrent: int = 4
    throw_speed_range_mps: tuple[float, float] = (4.0, 9.0)
    angular_speed_range_radps: tuple[float, float] = (0.0, 2.0)
    azimuth_range_deg: tuple[float, float] = (-45.0, 45.0)
    radial_distance_range_m: tuple[float, float] = (2.5, 5.0)
    relative_height_range_m: tuple[float, float] = (-0.5, 1.5)
    aim_noise_std_m: float = 0.15
    # Fraction of throws that solve for where the drone *will* be rather
    # than where it is at spawn time. Aiming purely at the current position
    # makes any sustained motion evasive on its own -- over a 1.4-2.0 s
    # flight a drone moving ~1.1 m/s displaces several times the 0.3 m hit
    # radius -- so a policy can survive by wandering without ever
    # perceiving anything. Mixing the two per throw keeps that from
    # collapsing into a single exploitable behaviour in the other
    # direction: a pure lead solve is defeated by holding still, and a pure
    # current-position aim is defeated by moving at all.
    lead_aim_probability: float = 0.0
    # Radius of a uniform random scatter applied to the throw target, on top
    # of aim_noise_std_m. At 0 every throw is aimed at the drone, which is
    # what makes position itself a strategy: throws are solved once at spawn
    # and never re-aimed, so drifting ~1.6 m over their 1-2 s flight makes
    # them miss outright (measured threat fraction 51.5% deterministic vs
    # 20.8% stochastic). A policy can then survive by wandering, with no
    # perception at all.
    #
    # Scattered throws remove that class of strategy -- if a throw is not
    # aimed at you, moving does not make it miss -- and they change the
    # perception problem too. An aimed throw is on near-constant bearing:
    # it barely moves across the image plane and only looms, which is the
    # weakest possible signal for a sensor that fires on change (measured at
    # 0.5% of event mass at the range a dodge must commit). Traffic that
    # crosses the field of view carries real angular rate, which is what an
    # event camera actually sees.
    aim_scatter_m: float = 0.0
    # Seconds before predicted impact at which a throw stops correcting and
    # goes ballistic. Above 0 this enables mid-flight re-aiming.
    #
    # Aiming once at spawn couples threat to evadability in the wrong way:
    # a wide scatter means throws miss anyway (measured control success
    # 57-79% at every density tested, so passivity already wins), while a
    # tight scatter means escaping needs only to out-move the scatter radius
    # -- and measured drift under stochastic actions is ~1.6 m against a
    # 1.13 m mean scatter, so wandering already beats it. Tightening makes
    # that worse, not better.
    #
    # Re-aiming breaks the coupling: throws follow the drone, so drifting
    # stops paying, while the commit time keeps a late decisive dodge
    # effective. That puts the required reaction inside the window where the
    # event signal is strongest (SNR 0.014-0.020 at 0.25-0.5 s to impact,
    # against 0.008 further out).
    # Obstacles placed on the nominal path at episode reset. Non-zero also
    # disables the mid-episode spawner, so the field is a fixed property of
    # the episode rather than something spawned in response to the drone.
    static_obstacle_count: int = 0
    # Lateral offset for alternating slalom lanes; 0 places every obstacle on
    # the path, which a single constant offset defeats.
    slalom_offset_m: float = 0.0
    reaim_commit_time_s: float = 0.0
    reaim_max_turn_rate_radps: float = 3.0
    # The organic spawner has no scene-mesh awareness otherwise: a candidate
    # position is just a geometric offset from the drone, so it can land
    # behind a wall or in an adjacent room, and the obstacle then flies
    # straight through that wall to reach its throw target. Each spawn
    # attempt is rejected and resampled if a raycast finds scene geometry
    # between the candidate and the drone.
    max_spawn_attempts: int = 12
    spawn_los_slack_m: float = 0.1
    seed: int | None = None
    dataset_config_file: str | None = None
    templates: list[DynamicObstacleTemplate] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DynamicObstaclesConfig":
        if not data:
            return cls(enabled=False)

        templates_cfg = data.get("templates", [])
        templates: list[DynamicObstacleTemplate] = []
        for t in templates_cfg:
            if "handle" not in t:
                raise ValueError("Each dynamic obstacle template must define 'handle'")
            scale = t.get("scale")
            templates.append(
                DynamicObstacleTemplate(
                    handle=t["handle"],
                    motion_mode=t.get("motion_mode", "dynamic_throw"),
                    scale=tuple(scale) if scale is not None else None,
                    mass=t.get("mass"),
                    ttl_s=float(t.get("ttl_s", 3.0)),
                    kinematic_speed_mps=float(t.get("kinematic_speed_mps", 5.0)),
                    parabola_gravity_mps2=float(t.get("parabola_gravity_mps2", 9.81)),
                    collision_radius=(
                        None
                        if t.get("collision_radius") is None
                        else float(t["collision_radius"])
                    ),
                )
            )

        return cls(
            enabled=bool(data.get("enabled", False)),
            spawn_interval_s=float(data.get("spawn_interval_s", 2.0)),
            spawn_interval_jitter=float(data.get("spawn_interval_jitter", 0.0)),
            max_concurrent=int(data.get("max_concurrent", 4)),
            throw_speed_range_mps=tuple(data.get("throw_speed_range_mps", [4.0, 9.0])),
            angular_speed_range_radps=tuple(
                data.get("angular_speed_range_radps", [0.0, 2.0])
            ),
            azimuth_range_deg=tuple(data.get("azimuth_range_deg", [-45.0, 45.0])),
            radial_distance_range_m=tuple(
                data.get("radial_distance_range_m", [2.5, 5.0])
            ),
            relative_height_range_m=tuple(
                data.get("relative_height_range_m", [-0.5, 1.5])
            ),
            aim_noise_std_m=float(data.get("aim_noise_std_m", 0.15)),
            lead_aim_probability=float(data.get("lead_aim_probability", 0.0)),
            aim_scatter_m=float(data.get("aim_scatter_m", 0.0)),
            static_obstacle_count=int(data.get("static_obstacle_count", 0)),
            slalom_offset_m=float(data.get("slalom_offset_m", 0.0)),
            reaim_commit_time_s=float(data.get("reaim_commit_time_s", 0.0)),
            reaim_max_turn_rate_radps=float(
                data.get("reaim_max_turn_rate_radps", 3.0)
            ),
            max_spawn_attempts=int(data.get("max_spawn_attempts", 12)),
            spawn_los_slack_m=float(data.get("spawn_los_slack_m", 0.1)),
            seed=data.get("seed"),
            dataset_config_file=data.get("dataset_config_file"),
            templates=templates,
        )


@dataclass(slots=True)
class ActiveObstacle:
    """Tracks one spawned obstacle."""

    object_id: int
    obj: Any
    born_time: float
    motion_mode: str
    ttl_s: float
    spawn_position: np.ndarray
    velocity: np.ndarray
    gravity_mps2: float
    collision_radius: float
    # Offset from the drone this throw was aimed at, preserved across
    # re-aims so a throw keeps its own miss distance instead of homing in.
    aim_offset: np.ndarray | None = None


class DynamicObstacleManager:
    """Manages spawn/update/despawn of dynamic obstacles."""

    def __init__(
        self,
        cfg: DynamicObstaclesConfig,
        sim: Any,
        hsim_module: Any,
        agent_dimensions: tuple[float, float] = (0.0, 0.0),
    ):
        self.cfg = cfg
        self._sim = sim
        self._hsim = hsim_module
        self._agent_height = float(agent_dimensions[0])
        self._agent_radius = float(agent_dimensions[1])
        self._rigid_obj_mgr = self._sim.get_rigid_object_manager()
        self._obj_attr_mgr = self._sim.get_object_template_manager()

        self._rng = np.random.default_rng(cfg.seed)
        self._last_spawn_time = -np.inf
        self._active: dict[int, ActiveObstacle] = {}

        self._registered_template_handles: list[str] = []
        self._resolved_templates: list[DynamicObstacleTemplate] = []
        self._intercept_schedule: list[dict[str, Any]] = []
        self._intercept_spawned = 0
        self._previous_drone_position: np.ndarray | None = None
        self._previous_drone_time: float | None = None
        # Wait until the next spawn; resampled after every spawn so the
        # cadence cannot be used as a clock (see spawn_interval_jitter).
        self._next_spawn_wait: float = float(cfg.spawn_interval_s)
        # When set, spawn/target geometry is computed against this position
        # instead of the live drone position -- see set_aim_override.
        self._aim_override: np.ndarray | None = None

        if self.cfg.enabled:
            self._load_dataset_config()
            self._prepare_templates()
            if not self._resolved_templates:
                logger.warning(
                    "dynamic_obstacles enabled but no valid templates found; disabling"
                )
                self.cfg.enabled = False

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def needs_physics_step(self) -> bool:
        """Whether the world should step Habitat physics for obstacle simulation."""
        if not self.cfg.enabled:
            return False
        if self._active:
            return True
        return (
            np.isfinite(self._last_spawn_time) and self.cfg.spawn_interval_s > 0.0
        ) or (len(self._resolved_templates) > 0)

    def set_intercept_schedule(self, entries: list[dict[str, Any]] | None) -> None:
        """Install scheduled throws, optionally retargeted at spawn time.

        Each entry needs ``spawn_time``, ``spawn_position`` and ``velocity``
        (Habitat frame), and optionally ``template_index``. Entries are
        Entries with ``retarget_at_spawn`` carry an approach, speed and lead
        distance rather than an immutable projectile trajectory. At launch we
        solve against the drone's current constant-velocity prediction. This
        prevents an offset selected before the obstacle appears from making
        every later throw miss an obsolete nominal path.
        """
        self._intercept_schedule = list(entries or [])
        self._intercept_spawned = 0

    def reset_episode(self, seed: int | None = None) -> None:
        """Clear obstacle state at an episode boundary.

        Both timers here are absolute ``sim_time`` values while the episode
        clock restarts at 0, so without this the spawn test
        ``sim_time - last_spawn >= interval`` stays false for a whole
        episode after the first, and ``sim_time - born >= ttl`` never
        expires anything. The observable effect is that obstacles appear in
        episode 0 and never again -- which silently makes any
        encounter-gated success metric unreachable.

        Reseeding also stops spawn timing from depending on how many
        episodes happened to run before in the same process, which
        otherwise makes evaluations irreproducible.
        """
        for object_id in list(self._active):
            self._remove_obstacle(object_id)
        self._last_spawn_time = -np.inf
        self._next_spawn_wait = float(self.cfg.spawn_interval_s)
        self._intercept_schedule = []
        self._intercept_spawned = 0
        self._previous_drone_position = None
        self._previous_drone_time = None
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

    def _sample_spawn_wait(self) -> float:
        """Wait before the next spawn, jittered so the cadence is not a clock."""
        j = float(np.clip(self.cfg.spawn_interval_jitter, 0.0, 1.0))
        if j <= 0.0:
            return float(self.cfg.spawn_interval_s)
        return float(
            self.cfg.spawn_interval_s * self._rng.uniform(1.0 - j, 1.0 + j)
        )

    def set_aim_override(self, position: np.ndarray) -> None:
        """Spawn/aim against ``position`` instead of the live drone position.

        For evaluation only: the organic spawner throws at wherever the
        drone actually is, so with it enabled two different checkpoints
        given "the same seed" still see different obstacle encounters --
        their trajectories diverge, so the throws aimed at their live
        positions diverge too. Pointing this at the seed-determined nominal
        trajectory instead makes encounters reproducible across checkpoints
        for a fair, paired comparison. Orientation (which obstacles count as
        "in front") still tracks the live camera, so this closes the
        dominant source of drift, not all of it.
        """
        self._aim_override = np.asarray(position, dtype=np.float64)

    def clear_aim_override(self) -> None:
        self._aim_override = None

    def step(
        self,
        sim_time: float,
        drone_position: np.ndarray,
        drone_quaternion: np.ndarray,
        camera_quaternion: np.ndarray | None = None,
    ) -> None:
        if not self.cfg.enabled or self.cfg.static_obstacle_count > 0:
            # Pre-placed fields are fixed at reset; nothing spawns in flight.
            return

        if self._aim_override is not None:
            drone_position = self._aim_override
        drone_position = np.asarray(drone_position, dtype=np.float64)
        drone_velocity = None
        if self._previous_drone_position is not None and self._previous_drone_time is not None:
            elapsed = float(sim_time) - self._previous_drone_time
            if elapsed > 1e-9:
                drone_velocity = (drone_position - self._previous_drone_position) / elapsed

        if self._intercept_schedule:
            self._spawn_due_intercepts(sim_time, drone_position, drone_velocity)
        elif (
            sim_time - self._last_spawn_time >= self._next_spawn_wait
            and len(self._active) < self.cfg.max_concurrent
        ):
            self._spawn_one(
                sim_time,
                drone_position,
                drone_quaternion,
                drone_velocity,
                camera_quaternion,
            )
            self._last_spawn_time = sim_time
            self._next_spawn_wait = self._sample_spawn_wait()

        # Order matters: _reaim re-bases spawn_position/born_time to *now*, so
        # running it first makes _update_kinematic compute t = 0 and pin every
        # throw at its current position -- obstacles frozen in mid-air, which
        # measured as 0.0 encounters per episode. Advancing first, then
        # re-basing, leaves the next tick a full dt to integrate over.
        self._update_kinematic(sim_time)
        self._reaim(sim_time, drone_position, drone_velocity)
        self._despawn_expired(sim_time)
        self._previous_drone_position = drone_position.copy()
        self._previous_drone_time = float(sim_time)

    def _spawn_due_intercepts(
        self,
        sim_time: float,
        drone_position: np.ndarray,
        drone_velocity: np.ndarray | None,
    ) -> None:
        """Spawn scheduled throws whose launch time has arrived."""
        while self._intercept_spawned < len(self._intercept_schedule):
            entry = self._intercept_schedule[self._intercept_spawned]
            if sim_time < float(entry["spawn_time"]):
                break
            self._intercept_spawned += 1
            if len(self._active) >= self.cfg.max_concurrent:
                # Dropping is preferable to deferring: a delayed throw no
                # longer intersects the path it was solved against.
                logger.debug("Skipping intercept: max_concurrent reached")
                continue
            template = self._resolved_templates[
                int(entry.get("template_index", 0)) % len(self._resolved_templates)
            ]
            spawn_position = np.asarray(entry["spawn_position"], dtype=np.float64)
            velocity = np.asarray(entry["velocity"], dtype=np.float64)
            if bool(entry.get("retarget_at_spawn", False)):
                spawn_position, velocity = solve_spawn_retarget(
                    entry, drone_position, drone_velocity
                )
            self._instantiate(
                template=template,
                spawn_position=np.asarray(spawn_position, dtype=np.float32),
                linear_velocity=np.asarray(velocity, dtype=np.float32),
                sim_time=sim_time,
            )
            self._last_spawn_time = sim_time

    def has_agent_collision(self, agent_position: np.ndarray) -> bool:
        """Check sphere-sphere collision between the agent and any active obstacle.

        ``agent_position`` is the drone's true body-center position (see
        ``BaseNeurosimRLEnv._on_episode_reset``, which sets the Habitat
        agent's pose directly from ``state["x"]`` with no offset), and
        ``item.obj.translation`` is the obstacle's true rendered center --
        both already comparable with no height adjustment. An earlier
        version subtracted ``_agent_height`` here, a leftover from Habitat's
        walking-agent convention (position = feet, sensors mounted at
        position + height) that does not apply to a drone whose "agent"
        position already is its body center. That made the check fire
        against obstacles up to ``_agent_height`` away from the drone's
        actual (and rendered) position -- confirmed empirically: a raw
        obstacle distance of 1.05 m was scored as a 0.26 m "collision".
        """
        for item in self._active.values():
            obstacle_pos = np.asarray(item.obj.translation, dtype=np.float32)
            dist = float(np.linalg.norm(agent_position - obstacle_pos))
            if dist <= self._agent_radius + item.collision_radius:
                return True
        return False

    def cleanup(self) -> None:
        """Remove active obstacles and temp template registrations."""
        for object_id in list(self._active):
            self._remove_obstacle(object_id)

        if hasattr(self._obj_attr_mgr, "remove_template_by_handle"):
            for handle in self._registered_template_handles:
                try:
                    self._obj_attr_mgr.remove_template_by_handle(handle)
                except Exception:
                    logger.debug("Failed to remove temporary template '%s'", handle)

        self._registered_template_handles.clear()
        self._resolved_templates.clear()

    def _load_dataset_config(self) -> None:
        """Load an external Habitat scene-dataset config into the MetadataMediator.

        This makes objects from datasets like YCB (bananas, cracker boxes, cans …)
        available to the object template manager so their handles can be resolved
        by ``_prepare_templates``.

        Set ``dataset_config_file`` in the dynamic-obstacles config block, e.g.::

            dataset_config_file: "data/objects/ycb/ycb.scene_dataset_config.json"

        Download YCB assets once with:
            python -m habitat_sim.utils.datasets_download --uids ycb --data-path data/
        """
        if not self.cfg.dataset_config_file:
            return
        try:
            self._sim.metadata_mediator.active_dataset = self.cfg.dataset_config_file
            # active_dataset replaces the underlying C++ managers; refresh the
            # Python references so _prepare_templates sees the new handles.
            self._obj_attr_mgr = self._sim.get_object_template_manager()
            self._rigid_obj_mgr = self._sim.get_rigid_object_manager()
            logger.info(
                "Loaded dynamic-obstacle dataset config: %s",
                self.cfg.dataset_config_file,
            )
        except Exception:
            logger.warning(
                "Failed to load dataset config '%s' for dynamic obstacles; "
                "only built-in primitive handles will be available.",
                self.cfg.dataset_config_file,
                exc_info=True,
            )

    def _prepare_templates(self) -> None:
        available_handles = []
        if hasattr(self._obj_attr_mgr, "get_template_handles"):
            available_handles = list(self._obj_attr_mgr.get_template_handles())

        for idx, template in enumerate(self.cfg.templates):
            resolved_handle = self._resolve_template_handle(template.handle)
            if resolved_handle is None:
                logger.warning(
                    "Could not resolve dynamic obstacle template handle '%s'. "
                    "Available examples: %s",
                    template.handle,
                    available_handles[:20],
                )
                continue

            configured_handle = resolved_handle
            if template.scale is not None or template.mass is not None:
                configured_handle = self._register_configured_template(
                    resolved_handle, template, idx
                )
                if configured_handle is None:
                    continue

            self._resolved_templates.append(
                DynamicObstacleTemplate(
                    handle=configured_handle,
                    motion_mode=template.motion_mode,
                    scale=template.scale,
                    mass=template.mass,
                    ttl_s=template.ttl_s,
                    kinematic_speed_mps=template.kinematic_speed_mps,
                    parabola_gravity_mps2=template.parabola_gravity_mps2,
                    collision_radius=template.collision_radius,
                )
            )
            logger.info(
                "Dynamic obstacle template enabled: cfg='%s' resolved='%s' mode='%s'",
                template.handle,
                configured_handle,
                template.motion_mode,
            )

    def _resolve_template_handle(self, query: str) -> str | None:
        # Try exact handle first.
        all_handles = []
        if hasattr(self._obj_attr_mgr, "get_template_handles"):
            all_handles = self._obj_attr_mgr.get_template_handles()
            if query in all_handles:
                return query
            query_lower = query.lower()
            for handle in all_handles:
                if handle.lower() == query_lower:
                    return handle
            # Many Habitat templates use handle substrings; choose first stable match.
            try:
                matches = self._obj_attr_mgr.get_template_handles(query)
                if matches:
                    return matches[0]
            except Exception:
                pass

            for handle in all_handles:
                if query_lower in handle.lower():
                    return handle

        return None

    def _register_configured_template(
        self,
        source_handle: str,
        template: DynamicObstacleTemplate,
        idx: int,
    ) -> str | None:
        if not hasattr(self._obj_attr_mgr, "get_template_by_handle") or not hasattr(
            self._obj_attr_mgr, "register_template"
        ):
            logger.warning(
                "Object template manager does not support cloning templates; "
                "using original handle '%s'",
                source_handle,
            )
            return source_handle

        try:
            tpl = self._obj_attr_mgr.get_template_by_handle(source_handle)
            if template.scale is not None:
                tpl.scale = np.array(template.scale, dtype=np.float32)
            if template.mass is not None:
                tpl.mass = float(template.mass)

            new_handle = f"ns_dynobs_{idx}_{source_handle.replace('/', '_')}"
            self._obj_attr_mgr.register_template(tpl, new_handle)
            self._registered_template_handles.append(new_handle)
            return new_handle
        except Exception:
            logger.exception(
                "Failed to clone template '%s' for dynamic obstacle", source_handle
            )
            return None

    def _spawn_one(
        self,
        sim_time: float,
        drone_position: np.ndarray,
        drone_quaternion: np.ndarray,
        drone_velocity: np.ndarray | None = None,
        camera_quaternion: np.ndarray | None = None,
    ) -> None:
        if not self._resolved_templates:
            return

        template = self._resolved_templates[
            int(self._rng.integers(0, len(self._resolved_templates)))
        ]

        if template.motion_mode == "static":
            # Placed on the path ahead rather than thrown, so the drone flies
            # into it unless it moves. A static obstacle is also the strongest
            # perception case available: a collision-course throw sits on
            # near-constant bearing and only looms (measured at 0.5% of event
            # mass at commit range), while something the drone flies *past*
            # sweeps across the image plane with real angular rate, which is
            # what an event camera actually responds to.
            spawn_position = self._sample_static_position(
                drone_position, drone_velocity
            )
        else:
            spawn_position = self._sample_spawn_position(
                drone_position, drone_quaternion, drone_velocity, camera_quaternion
            )
        if spawn_position is None:
            return

        obj = self._rigid_obj_mgr.add_object_by_template_handle(template.handle)
        if obj is None or getattr(obj, "object_id", -1) < 0:
            logger.warning(
                "Failed to spawn dynamic obstacle for template '%s'", template.handle
            )
            return
        if template.motion_mode == "static":
            self._instantiate(
                template=template,
                spawn_position=np.asarray(spawn_position, dtype=np.float32),
                linear_velocity=np.zeros(3, dtype=np.float32),
                sim_time=sim_time,
                obj=obj,
                aim_offset=None,
            )
            return

        target = np.asarray(drone_position, dtype=np.float32).copy()
        # Half (by config) of the throws lead the drone instead of aiming
        # where it currently is, so neither holding still nor simply
        # drifting is a blanket answer -- see lead_aim_probability.
        if (
            drone_velocity is not None
            and self.cfg.lead_aim_probability > 0.0
            and self._rng.random() < self.cfg.lead_aim_probability
        ):
            if template.motion_mode == "dynamic_throw":
                speed = float(np.mean(self.cfg.throw_speed_range_mps))
            else:
                speed = float(template.kinematic_speed_mps)
            lead_t = self._intercept_time(
                spawn_position, drone_position, drone_velocity, speed
            )
            if lead_t is not None:
                # Never extrapolate past the obstacle's own lifetime: a
                # far-future intercept is aiming at a prediction the
                # constant-velocity assumption no longer supports.
                lead_t = min(lead_t, float(template.ttl_s))
                target = (
                    np.asarray(drone_position, dtype=np.float64)
                    + np.asarray(drone_velocity, dtype=np.float64) * lead_t
                ).astype(np.float32)
        target += self._rng.normal(0.0, self.cfg.aim_noise_std_m, size=3).astype(
            np.float32
        )
        if self.cfg.aim_scatter_m > 0.0:
            # Uniform inside a sphere, so the scatter is isotropic rather than
            # concentrated on the shell a normalised direction would give.
            direction = self._rng.normal(size=3)
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            radius = self.cfg.aim_scatter_m * float(self._rng.random()) ** (1.0 / 3.0)
            target = target + (direction * radius).astype(np.float32)

        gravity = float(template.parabola_gravity_mps2)
        if template.motion_mode == "dynamic_throw":
            linear_velocity = self._compute_ballistic_velocity(
                spawn_position,
                target,
                gravity_mps2=gravity,
                speed_range_mps=self.cfg.throw_speed_range_mps,
            )
        elif template.motion_mode == "kinematic_parabola":
            linear_velocity = self._compute_ballistic_velocity(
                spawn_position,
                target,
                gravity_mps2=gravity,
                speed_range_mps=(
                    float(template.kinematic_speed_mps),
                    float(template.kinematic_speed_mps),
                ),
            )
        else:
            delta = target - spawn_position
            norm = float(np.linalg.norm(delta))
            if norm < 1e-6:
                direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                direction = delta / norm
            linear_velocity = direction * float(template.kinematic_speed_mps)

        angular_velocity = self._rng.uniform(
            self.cfg.angular_speed_range_radps[0],
            self.cfg.angular_speed_range_radps[1],
            size=3,
        ).astype(np.float32)

        self._instantiate(
            aim_offset=(
                np.asarray(target, dtype=np.float64)
                - np.asarray(drone_position, dtype=np.float64)
            ),
            template=template,
            spawn_position=spawn_position,
            linear_velocity=linear_velocity,
            sim_time=sim_time,
            obj=obj,
            angular_velocity=angular_velocity,
        )

    def _instantiate(
        self,
        *,
        template: DynamicObstacleTemplate,
        spawn_position: np.ndarray,
        linear_velocity: np.ndarray,
        sim_time: float,
        obj: Any = None,
        angular_velocity: np.ndarray | None = None,
        aim_offset: np.ndarray | None = None,
    ) -> None:
        """Place a configured obstacle and register it as active."""
        if obj is None:
            obj = self._rigid_obj_mgr.add_object_by_template_handle(template.handle)
            if obj is None or getattr(obj, "object_id", -1) < 0:
                logger.warning(
                    "Failed to spawn dynamic obstacle for template '%s'",
                    template.handle,
                )
                return
        if angular_velocity is None:
            angular_velocity = self._rng.uniform(
                self.cfg.angular_speed_range_radps[0],
                self.cfg.angular_speed_range_radps[1],
                size=3,
            ).astype(np.float32)

        spawn_position = np.asarray(spawn_position, dtype=np.float32)
        linear_velocity = np.asarray(linear_velocity, dtype=np.float32)
        obj.translation = spawn_position

        if template.collision_radius is not None:
            collision_radius = float(template.collision_radius)
        else:
            # Fall back to the object's AABB bounding sphere.
            bb = obj.root_scene_node.cumulative_bb
            half_extents = (
                np.array(bb.max, dtype=np.float32) - np.array(bb.min, dtype=np.float32)
            ) / 2.0
            collision_radius = float(np.linalg.norm(half_extents))

        motion_mode = template.motion_mode
        if motion_mode in {"kinematic_line", "kinematic_parabola", "static"}:
            obj.motion_type = self._hsim.physics.MotionType.KINEMATIC
            obj.linear_velocity = np.zeros(3, dtype=np.float32)
            obj.angular_velocity = np.zeros(3, dtype=np.float32)
        else:
            if motion_mode != "dynamic_throw":
                logger.warning(
                    "Unknown motion_mode '%s'; defaulting to dynamic_throw",
                    motion_mode,
                )
                motion_mode = "dynamic_throw"
            obj.motion_type = self._hsim.physics.MotionType.DYNAMIC
            obj.linear_velocity = linear_velocity
            obj.angular_velocity = angular_velocity

        self._active[obj.object_id] = ActiveObstacle(
            object_id=obj.object_id,
            obj=obj,
            born_time=sim_time,
            motion_mode=motion_mode,
            ttl_s=template.ttl_s,
            spawn_position=spawn_position,
            velocity=linear_velocity,
            gravity_mps2=float(template.parabola_gravity_mps2),
            collision_radius=collision_radius,
            # Where this throw was aimed relative to the drone, so re-aiming
            # preserves its own miss distance instead of homing in.
            aim_offset=aim_offset,
        )

    def _compute_ballistic_velocity(
        self,
        spawn_position: np.ndarray,
        target_position: np.ndarray,
        gravity_mps2: float,
        speed_range_mps: tuple[float, float],
    ) -> np.ndarray:
        """Compute an initial velocity that arcs toward the target.

        Uses a gravity-compensated ballistic launch in Habitat coordinates
        (Y-up, gravity along negative Y).
        """
        delta = np.asarray(target_position - spawn_position, dtype=np.float32)
        horizontal = np.asarray([delta[0], 0.0, delta[2]], dtype=np.float32)
        horizontal_dist = float(np.linalg.norm(horizontal))

        speed = float(self._rng.uniform(speed_range_mps[0], speed_range_mps[1]))
        speed = max(speed, 1e-3)

        # Choose flight time from horizontal motion so vertical term can compensate gravity.
        flight_time = max(horizontal_dist / speed, 0.25)

        vx = float(delta[0] / flight_time)
        vz = float(delta[2] / flight_time)
        vy = float((delta[1] + 0.5 * gravity_mps2 * (flight_time**2)) / flight_time)

        return np.array([vx, vy, vz], dtype=np.float32)

    @staticmethod
    def _yaw_from_quaternion(q: np.ndarray) -> float:
        """Extract yaw (rotation about Y-up axis) from a Habitat quaternion.

        Habitat uses Hamilton convention [w, x, y, z] with Y-up.
        The agent forward direction is -Z in local frame, so the yaw in
        the XZ plane is atan2(forward_x, -forward_z) after rotating -Z
        by the quaternion.
        """
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # Forward (-Z) rotated by quaternion, projected to XZ plane:
        fwd_x = -2.0 * (x * z + w * y)
        fwd_z = -1.0 + 2.0 * (x * x + y * y)
        return float(np.arctan2(fwd_z, fwd_x))

    @staticmethod
    def _intercept_time(
        spawn_position: np.ndarray,
        drone_position: np.ndarray,
        drone_velocity: np.ndarray,
        speed_mps: float,
    ) -> float | None:
        """Time for a projectile at ``speed_mps`` to meet a drifting drone.

        Solves ``|R + V t| = s t`` for the earliest positive ``t``, where
        ``R`` is the drone's offset from the spawn point and ``V`` its
        velocity. Returns ``None`` when the quadratic has no positive root,
        which is the genuinely uncatchable case (drone outrunning the
        throw); the caller then falls back to aiming at the current
        position rather than inventing a target behind itself.
        """
        R = np.asarray(drone_position, dtype=np.float64) - np.asarray(
            spawn_position, dtype=np.float64
        )
        V = np.asarray(drone_velocity, dtype=np.float64)
        s = float(speed_mps)

        a = float(V @ V) - s * s
        b = 2.0 * float(R @ V)
        c = float(R @ R)

        if abs(a) < 1e-9:
            if abs(b) < 1e-9:
                return None
            t = -c / b
            return t if t > 1e-6 else None

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        root = float(np.sqrt(disc))
        candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
        positive = [t for t in candidates if t > 1e-6]
        return min(positive) if positive else None

    def _has_line_of_sight(self, origin: np.ndarray, target: np.ndarray) -> bool:
        """Whether scene geometry blocks the straight path origin -> target."""
        import habitat_sim

        delta = np.asarray(target, dtype=np.float64) - np.asarray(
            origin, dtype=np.float64
        )
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return True

        ray = habitat_sim.geo.Ray(
            np.asarray(origin, dtype=np.float32),
            (delta / distance).astype(np.float32),
        )
        try:
            hits = self._sim.cast_ray(ray, max_distance=distance)
        except Exception:  # raycasting unavailable -> do not block spawning
            logger.debug("cast_ray unavailable; skipping spawn line-of-sight check")
            return True

        if not hits.has_hits():
            return True
        return float(hits.hits[0].ray_distance) >= distance - self.cfg.spawn_los_slack_m

    def place_static_obstacles(self, positions, sim_time: float = 0.0) -> int:
        """Place stationary obstacles at exact world points, once per episode.

        Pre-placement rather than spawning: the manager only receives the
        drone's position and velocity, so a mid-episode "static" spawn had to
        extrapolate the current heading -- and the MinSnap path curves away
        from that line. The miss rate tracked the lead distance exactly (at
        7 m the drone passed no closer than 1.27 m from a 0.45 m hit radius),
        so obstacles ended up beside a path the drone never flew.

        The environment knows the trajectory at reset and can sample points
        directly on it, which makes a passive drone collide by construction
        instead of by luck, and lets a privileged planner solve the whole
        episode at once rather than patching reactively at 3 m of notice.
        """
        if not self.cfg.enabled or not self._resolved_templates:
            return 0
        placed = 0
        for point in positions:
            template = self._resolved_templates[
                int(self._rng.integers(0, len(self._resolved_templates)))
            ]
            obj = self._rigid_obj_mgr.add_object_by_template_handle(template.handle)
            if obj is None or getattr(obj, "object_id", -1) < 0:
                logger.warning("Failed to pre-place obstacle '%s'", template.handle)
                continue
            self._instantiate(
                template=template,
                spawn_position=np.asarray(point, dtype=np.float32),
                linear_velocity=np.zeros(3, dtype=np.float32),
                sim_time=float(sim_time),
                obj=obj,
                aim_offset=None,
            )
            placed += 1
        return placed

    def _sample_static_position(
        self, drone_position: np.ndarray, drone_velocity: np.ndarray | None
    ) -> np.ndarray | None:
        """A point on the path ahead, offset laterally by the aim scatter.

        Placement follows the current heading rather than the planned path:
        the manager does not see the trajectory, and over the few metres of
        lead distance the path is locally straight enough for the drone to
        arrive. Line of sight is still required, so obstacles do not land
        behind walls.
        """
        position = np.asarray(drone_position, dtype=np.float64)
        heading = None
        if drone_velocity is not None:
            speed = float(np.linalg.norm(drone_velocity))
            if speed > 1e-6:
                heading = np.asarray(drone_velocity, dtype=np.float64) / speed
        if heading is None:
            return None

        lo, hi = self.cfg.radial_distance_range_m
        for _ in range(max(int(self.cfg.max_spawn_attempts), 1)):
            distance = float(self._rng.uniform(lo, hi))
            lateral = self._rng.normal(size=3)
            lateral -= heading * float(np.dot(lateral, heading))
            norm = float(np.linalg.norm(lateral))
            if norm > 1e-9 and self.cfg.aim_scatter_m > 0.0:
                lateral = lateral / norm * float(
                    self._rng.uniform(0.0, self.cfg.aim_scatter_m)
                )
            else:
                lateral = np.zeros(3)
            candidate = position + heading * distance + lateral
            if self._has_line_of_sight(candidate, position):
                return candidate
        return None

    def _sample_spawn_position(
        self,
        drone_position: np.ndarray,
        drone_quaternion: np.ndarray,
        drone_velocity: np.ndarray | None = None,
        camera_quaternion: np.ndarray | None = None,
    ) -> np.ndarray | None:
        drone_position = np.asarray(drone_position, dtype=np.float32)

        # Reference direction for "in front", in priority order:
        #
        # 1. The camera's own rendered orientation (Habitat's
        #    sensor_states[...].rotation), when available -- ground truth
        #    for what the camera can actually see, with no re-derivation of
        #    the sensor's local mount rotation by hand.
        # 2. Velocity heading, as a fallback for callers that don't thread
        #    the camera orientation through (e.g. tests constructing this
        #    manager directly).
        # 3. Raw body-orientation yaw, only when the vehicle is essentially
        #    stationary and heading is undefined.
        #
        # Velocity heading alone was tried first and measured wrong: the
        # vehicle's actual orientation does not track its instantaneous
        # velocity direction closely (divergence of 30-100+ degrees
        # observed), so an obstacle "in front" of the velocity vector can
        # still spawn well outside what the camera renders. Confirmed by
        # inspecting actual video frames -- an obstacle flagged as an
        # active threat at 6 m, 3.6 m, and 0.6 m was never visible in the
        # RGB pane at any of those distances under the velocity-only
        # version of this fix.
        if camera_quaternion is not None:
            drone_yaw = self._yaw_from_quaternion(camera_quaternion)
        else:
            speed_xz = (
                float(np.hypot(drone_velocity[0], drone_velocity[2]))
                if drone_velocity is not None
                else 0.0
            )
            if speed_xz > 0.1:
                drone_yaw = float(np.arctan2(drone_velocity[2], drone_velocity[0]))
            else:
                drone_yaw = self._yaw_from_quaternion(drone_quaternion)

        # Azimuth is sampled relative to the drone's heading of travel.
        # 0 deg is straight ahead; 90 deg is to the right; 180 deg is behind.
        # Each candidate is rejected and resampled if scene geometry (a
        # wall, a piece of furniture) blocks the straight path back to the
        # drone -- otherwise the obstacle spawns behind it and flies
        # straight through on its way to the throw target.
        for _ in range(max(1, self.cfg.max_spawn_attempts)):
            azimuth = drone_yaw + np.deg2rad(
                self._rng.uniform(
                    self.cfg.azimuth_range_deg[0], self.cfg.azimuth_range_deg[1]
                )
            )
            radius = self._rng.uniform(
                self.cfg.radial_distance_range_m[0],
                self.cfg.radial_distance_range_m[1],
            )
            rel_height = self._rng.uniform(
                self.cfg.relative_height_range_m[0],
                self.cfg.relative_height_range_m[1],
            )

            offset = np.array(
                [
                    radius * np.cos(azimuth),
                    rel_height,
                    radius * np.sin(azimuth),
                ],
                dtype=np.float32,
            )
            candidate = drone_position + offset
            if self._has_line_of_sight(candidate, drone_position):
                return candidate

        logger.debug(
            "No line-of-sight obstacle spawn position found after %d attempts",
            self.cfg.max_spawn_attempts,
        )
        return None

    def _reaim(
        self,
        sim_time: float,
        drone_position: np.ndarray,
        drone_velocity: np.ndarray | None = None,
    ) -> None:
        """Steer in-flight throws back onto their intended aim point.

        Each throw keeps the offset it was originally aimed with, so
        re-aiming preserves its individual miss distance rather than turning
        every obstacle into a homing missile. Steering is rate-limited and
        stops ``reaim_commit_time_s`` before predicted arrival, after which
        the throw is ballistic and a late dodge works.
        """
        commit = float(self.cfg.reaim_commit_time_s)
        if commit <= 0.0:
            return
        max_turn = float(self.cfg.reaim_max_turn_rate_radps)
        for item in self._active.values():
            if item.motion_mode not in {"kinematic_line", "kinematic_parabola"}:
                continue
            speed = float(np.linalg.norm(item.velocity))
            if speed < 1e-6:
                continue
            position = np.asarray(item.obj.translation, dtype=np.float64)
            target = np.asarray(drone_position, dtype=np.float64)
            if item.aim_offset is not None:
                target = target + item.aim_offset
            # Lead the drone rather than chasing where it currently is. Once
            # the throw commits it is ballistic for reaim_commit_time_s, and
            # a throw aimed at a stale point lets the drone drift off it for
            # free -- which is a large part of why a passive drone was still
            # surviving 57.5% of episodes against throws that track it.
            to_target = target - position
            range_m = float(np.linalg.norm(to_target))
            if drone_velocity is not None and range_m > 1e-6:
                lead_t = self._intercept_time(
                    position, target, np.asarray(drone_velocity, dtype=np.float64), speed
                )
                if lead_t is not None:
                    lead_t = min(float(lead_t), float(item.ttl_s))
                    to_target = (
                        target
                        + np.asarray(drone_velocity, dtype=np.float64) * lead_t
                        - position
                    )
                    range_m = float(np.linalg.norm(to_target))
            if range_m < 1e-6 or range_m / speed <= commit:
                continue  # committed: ballistic from here
            desired = to_target / range_m
            current = item.velocity / speed
            previous_time = self._previous_drone_time
            if previous_time is None:
                continue
            # Explicit None check: a previous time of exactly 0.0 is falsy, so
            # `or sim_time` made dt zero and silently skipped every correction.
            dt = max(float(sim_time) - float(previous_time), 0.0)
            if dt <= 0.0:
                continue
            cos_a = float(np.clip(np.dot(current, desired), -1.0, 1.0))
            angle = float(np.arccos(cos_a))
            if angle < 1e-9:
                continue
            new_dir = desired if angle <= max_turn * dt else (
                current
                + (desired - current) * (max_turn * dt / angle)
            )
            new_dir = new_dir / max(float(np.linalg.norm(new_dir)), 1e-9)
            # Re-base the trajectory here so the displacement integration
            # downstream stays consistent with the new heading.
            item.velocity = new_dir * speed
            item.spawn_position = position
            item.born_time = float(sim_time)

    def _update_kinematic(self, sim_time: float) -> None:
        for item in self._active.values():
            if item.motion_mode not in {"kinematic_line", "kinematic_parabola"}:
                continue

            t = max(0.0, sim_time - item.born_time)
            displacement = item.velocity * t
            if item.motion_mode == "kinematic_parabola":
                displacement[1] -= 0.5 * item.gravity_mps2 * (t**2)

            item.obj.translation = item.spawn_position + displacement

    def _despawn_expired(self, sim_time: float) -> None:
        expired: list[int] = []
        for object_id, item in self._active.items():
            if sim_time - item.born_time >= item.ttl_s:
                expired.append(object_id)

        for object_id in expired:
            self._remove_obstacle(object_id)

    def _remove_obstacle(self, object_id: int) -> None:
        self._active.pop(object_id, None)
        try:
            self._rigid_obj_mgr.remove_object_by_id(object_id)
        except Exception:
            logger.debug("Failed to remove obstacle object_id=%s", object_id)
