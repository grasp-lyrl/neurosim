"""Dynamic obstacle spawning and lifecycle management for visual backends.

This module is intentionally backend-agnostic and interacts with Habitat-sim
through manager/module objects injected by the wrapper.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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


@dataclass(slots=True)
class DynamicObstaclesConfig:
    """Runtime configuration for dynamic obstacle spawning."""

    enabled: bool = False
    spawn_interval_s: float = 2.0
    max_concurrent: int = 4
    throw_speed_range_mps: tuple[float, float] = (4.0, 9.0)
    angular_speed_range_radps: tuple[float, float] = (0.0, 2.0)
    azimuth_range_deg: tuple[float, float] = (0.0, 360.0)
    radial_distance_range_m: tuple[float, float] = (2.5, 5.0)
    relative_height_range_m: tuple[float, float] = (-0.5, 1.5)
    aim_noise_std_m: float = 0.15
    seed: int | None = None
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
                )
            )

        return cls(
            enabled=bool(data.get("enabled", False)),
            spawn_interval_s=float(data.get("spawn_interval_s", 2.0)),
            max_concurrent=int(data.get("max_concurrent", 4)),
            throw_speed_range_mps=tuple(data.get("throw_speed_range_mps", [4.0, 9.0])),
            angular_speed_range_radps=tuple(
                data.get("angular_speed_range_radps", [0.0, 2.0])
            ),
            azimuth_range_deg=tuple(data.get("azimuth_range_deg", [0.0, 360.0])),
            radial_distance_range_m=tuple(
                data.get("radial_distance_range_m", [2.5, 5.0])
            ),
            relative_height_range_m=tuple(
                data.get("relative_height_range_m", [-0.5, 1.5])
            ),
            aim_noise_std_m=float(data.get("aim_noise_std_m", 0.15)),
            seed=data.get("seed"),
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


class DynamicObstacleManager:
    """Manages spawn/update/despawn of dynamic obstacles."""

    def __init__(
        self,
        cfg: DynamicObstaclesConfig,
        sim: Any,
        hsim_module: Any,
    ):
        self.cfg = cfg
        self._sim = sim
        self._hsim = hsim_module
        self._rigid_obj_mgr = self._sim.get_rigid_object_manager()
        self._obj_attr_mgr = self._sim.get_object_template_manager()

        self._rng = np.random.default_rng(cfg.seed)
        self._last_spawn_time = -np.inf
        self._active: dict[int, ActiveObstacle] = {}

        self._registered_template_handles: list[str] = []
        self._resolved_templates: list[DynamicObstacleTemplate] = []

        if self.cfg.enabled:
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

    def step(
        self,
        sim_time: float,
        simsteps: int,
        drone_position: np.ndarray,
        dt: float,
    ) -> None:
        del simsteps, dt
        if not self.cfg.enabled:
            return

        if (
            sim_time - self._last_spawn_time >= self.cfg.spawn_interval_s
            and len(self._active) < self.cfg.max_concurrent
        ):
            self._spawn_one(sim_time, drone_position)
            self._last_spawn_time = sim_time

        self._update_kinematic(sim_time)
        self._despawn_expired(sim_time)

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

    def get_state(self) -> list[dict[str, Any]]:
        """Return a lightweight snapshot of active obstacle states."""
        state = []
        for item in self._active.values():
            state.append(
                {
                    "object_id": item.object_id,
                    "motion_mode": item.motion_mode,
                    "born_time": item.born_time,
                    "ttl_s": item.ttl_s,
                    "position": np.asarray(item.obj.translation, dtype=np.float32),
                    "velocity": item.velocity.copy(),
                }
            )
        return state

    def has_drone_collision(self) -> bool:
        """Placeholder collision API for future RL integration."""
        return False

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

    def _spawn_one(self, sim_time: float, drone_position: np.ndarray) -> None:
        if not self._resolved_templates:
            return

        template = self._resolved_templates[
            int(self._rng.integers(0, len(self._resolved_templates)))
        ]
        obj = self._rigid_obj_mgr.add_object_by_template_handle(template.handle)

        if obj is None or getattr(obj, "object_id", -1) < 0:
            logger.warning(
                "Failed to spawn dynamic obstacle for template '%s'", template.handle
            )
            return

        spawn_position = self._sample_spawn_position(drone_position)
        target = np.asarray(drone_position, dtype=np.float32).copy()
        target += self._rng.normal(0.0, self.cfg.aim_noise_std_m, size=3).astype(
            np.float32
        )

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

        obj.translation = spawn_position.astype(np.float32)

        motion_mode = template.motion_mode
        if motion_mode == "dynamic_throw":
            obj.motion_type = self._hsim.physics.MotionType.DYNAMIC
            obj.linear_velocity = linear_velocity
            obj.angular_velocity = angular_velocity
        elif motion_mode in {"kinematic_line", "kinematic_parabola"}:
            obj.motion_type = self._hsim.physics.MotionType.KINEMATIC
            obj.linear_velocity = np.zeros(3, dtype=np.float32)
            obj.angular_velocity = np.zeros(3, dtype=np.float32)
        else:
            logger.warning(
                "Unknown motion_mode '%s'; defaulting to dynamic_throw", motion_mode
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
            spawn_position=spawn_position.astype(np.float32),
            velocity=linear_velocity.astype(np.float32),
            gravity_mps2=gravity,
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

    def _sample_spawn_position(self, drone_position: np.ndarray) -> np.ndarray:
        drone_position = np.asarray(drone_position, dtype=np.float32)
        azimuth = np.deg2rad(
            self._rng.uniform(
                self.cfg.azimuth_range_deg[0], self.cfg.azimuth_range_deg[1]
            )
        )
        radius = self._rng.uniform(
            self.cfg.radial_distance_range_m[0], self.cfg.radial_distance_range_m[1]
        )
        rel_height = self._rng.uniform(
            self.cfg.relative_height_range_m[0], self.cfg.relative_height_range_m[1]
        )

        offset = np.array(
            [
                radius * np.cos(azimuth),
                rel_height,
                radius * np.sin(azimuth),
            ],
            dtype=np.float32,
        )
        return drone_position + offset

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
