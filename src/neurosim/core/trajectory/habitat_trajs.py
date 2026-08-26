"""NavMesh path generation utilities.

It provides:
- A baseline shortest-path query wrapper
- A non-shortest, quadrotor-friendly "interesting" path generator

The interesting path is generated via sampling random navigable points in the scene
and then creating shortest paths between them and adding desired yaws.
"""

import math
import random
import logging
import numpy as np
import magnum as mn

from rotorpy.trajectories.minsnap import MinSnap

logger = logging.getLogger(__name__)


def find_shortest_path_points(pathfinder, start: np.ndarray, goal: np.ndarray):
    """Find shortest path points on a Habitat-Sim PathFinder.

    Returns (found, geodesic_distance, points).
    """
    import habitat_sim  # local import to avoid hard dependency at module import time

    sp = habitat_sim.ShortestPath()
    sp.requested_start = start
    sp.requested_end = goal
    found = pathfinder.find_path(sp)
    pts = np.array([np.array(p) for p in sp.points], dtype=np.float32)
    return found, float(sp.geodesic_distance), pts


def sample_minsnap_trajectory(
    traj, dt: float = 1.0 / 30.0
) -> tuple[np.ndarray, list[mn.Quaternion]]:
    """Sample a MinSnap trajectory at fixed time intervals."""
    # MinSnap.t_keyframes is the arrival times at waypoints.
    t_total = traj.t_keyframes[-1]
    # Ensure we don't include t_total or go beyond it due to float precision
    ts = np.arange(0, t_total - 1e-6, dt)

    points = []
    quats = []

    # Habitat gravity (Y up)
    g = np.array([0, -9.81, 0])

    for t in ts:
        flat = traj.update(t)
        pos = flat["x"]
        acc = flat["x_ddot"]
        yaw = flat["yaw"]

        points.append(pos)

        # Compute orientation from flatness
        # Thrust vector aligns with body up (Y in Habitat)
        # We assume the drone's "Up" axis aligns with the thrust vector (acc - g).
        t_vec = acc - g
        if np.linalg.norm(t_vec) < 1e-6:
            t_vec = np.array([0, 1, 0])

        y_cam = t_vec / np.linalg.norm(t_vec)

        # Desired forward direction based on yaw
        # Assuming yaw=0 means looking along +Z (Habitat convention with atan2)
        # forward_desired = [sin(yaw), 0, cos(yaw)]
        # This corresponds to rotation around Y.
        forward_desired = np.array([math.sin(yaw), 0, math.cos(yaw)])

        # X_cam (Right) = cross(forward_desired, Y_cam)
        # This ensures X is perpendicular to Up and roughly Right.
        x_cam = np.cross(forward_desired, y_cam)
        if np.linalg.norm(x_cam) < 1e-6:
            x_cam = np.array([1, 0, 0])
        x_cam /= np.linalg.norm(x_cam)

        # Z_cam (Backward) = cross(X_cam, Y_cam)
        # Camera looks -Z, so -Z should be Forward.
        z_cam = np.cross(x_cam, y_cam)

        # R = [x_cam, y_cam, z_cam]
        R = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Convert to quaternion
        q = mn.Quaternion.from_matrix(mn.Matrix3(R))
        quats.append(q)

    return np.array(points), quats


def sample_random_navigable_point_with_height(
    pathfinder,
    max_retries: int = 100,
    min_altitude_m: float = 0.0,
    max_altitude_m: float | None = None,
    ceiling_margin_m: float = 0.3,
    lateral_margin_m: float = 0.0,
) -> np.ndarray | None:
    """Sample a random navigable point with random height and check navigability.

    Altitude is measured above the *local* floor height at the sampled
    (x, z) -- ``get_random_navigable_point()`` returns a point already on
    the navmesh surface there, which is exactly that local floor height,
    before its Y gets overwritten below. Measuring against the scene's
    global AABB instead (an earlier version of this function did) is wrong
    whenever floor height varies across the scene, and more importantly
    doesn't answer the question that actually matters for a flying vehicle:
    furniture sits on the local floor, so clearing it means flying some
    minimum height above *that* floor, not above whatever the lowest point
    in the whole scene happens to be.

    Args:
        pathfinder: Habitat pathfinder instance
        max_retries: Number of retries before giving up
        min_altitude_m: Minimum height above the local floor at the sampled
            (x, z). Clearing typical furniture (a couch, a table) needs
            roughly 1 m. Habitat's ``is_navigable`` cannot itself confirm
            that height is clear: it has a hard default vertical snap
            tolerance of 0.5 m (``max_y_delta``, unexposed here), beyond
            which it isn't testing 3D headroom at all -- confirmed
            empirically, 500 probed points all topped out at exactly
            0.500 m of "navigable" height regardless of the scene's actual
            layout. There is no navmesh-based way to verify an elevated
            point is furniture-free beyond that band, so this only
            guarantees the (x, z) is over real floor (which
            ``get_random_navigable_point`` already established) and a
            sensible global ceiling clearance -- not a collision guarantee
            against furniture at the sampled height.
        max_altitude_m: Maximum height above the local floor, or ``None``
            for "up to `ceiling_margin_m` below the scene's global ceiling".
        ceiling_margin_m: Keep this far below the scene's highest point,
            regardless of ``max_altitude_m``.
        lateral_margin_m: Keep this far from the scene's horizontal extremes,
            so a point sampled hard against a wall still has room to dodge.

    Returns:
        A navigable point with random height, or None if unable to find one
    """
    bounds = pathfinder.get_bounds()
    min_x, _, min_z = bounds[0]
    max_x, global_max_y, max_z = bounds[1]
    ceiling_y = global_max_y - ceiling_margin_m

    for _ in range(max_retries):
        # Sample x, z from navigable mesh; the returned y is the local
        # floor height at this (x, z).
        point = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if point[0] < min_x + lateral_margin_m or point[0] > max_x - lateral_margin_m:
            continue
        if point[2] < min_z + lateral_margin_m or point[2] > max_z - lateral_margin_m:
            continue

        floor_y = float(point[1])
        lo = floor_y + min_altitude_m
        hi = ceiling_y if max_altitude_m is None else min(ceiling_y, floor_y + max_altitude_m)
        if hi <= lo:
            continue  # no usable band above this particular floor point

        point[1] = random.uniform(lo, hi)

        # is_navigable's default max_y_delta is 0.5 m; beyond that offset
        # from the navmesh surface it just returns False without having
        # tested anything meaningful, so at typical flight altitudes this
        # check would silently reject every candidate. Only apply it when
        # the sampled height is within that band, where it still catches a
        # genuinely bad (x, z) (e.g. hard against a wall).
        if abs(point[1] - floor_y) > 0.45 or pathfinder.is_navigable(point):
            return point

    return None


def densify_path(points: np.ndarray, max_dist: float = 2.0) -> np.ndarray:
    """
    Insert intermediate points if segments are too long.
    This helps MinSnap to have enough constraints.
    """
    if len(points) < 2:
        return points

    new_points = [points[0]]
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dist = np.linalg.norm(p2 - p1)
        if dist > max_dist:
            num_segments = int(np.ceil(dist / max_dist))
            for j in range(1, num_segments):
                alpha = j / num_segments
                new_points.append(p1 + alpha * (p2 - p1))
        new_points.append(p2)
    return np.array(new_points)


def calculate_smooth_yaw(
    points: np.ndarray,
    lookahead_dist: float = 1.0,
    frame: str = "habitat",
) -> np.ndarray:
    """Calculate smooth yaw angles using a lookahead point.

    The yaw convention depends on which frame ``points`` lives in:

    - ``"habitat"`` (Y up): the horizontal plane is XZ, so yaw comes from
      ``atan2(-dx, dz)``.
    - ``"dynamics"`` (rotorpy, Z up): the horizontal plane is XY, so yaw is
      ``atan2(dy, dx)``. ``SE3Control`` consumes this convention directly --
      it builds ``c1_des = [cos(yaw), sin(yaw), 0]`` and aligns body-x with
      it.

    Calling this with the wrong frame silently reads the wrong axis pair:
    on a path already mapped into the dynamics frame, the habitat formula
    treats altitude (``dz`` there is up) as a horizontal component, so the
    heading tracks climb/descent instead of the direction of travel.

    Args:
        points: Array of shape (N, 3) containing path points
        lookahead_dist: Distance to look ahead for calculating yaw
        frame: Frame ``points`` are expressed in; ``"habitat"`` or ``"dynamics"``

    Returns:
        Array of shape (N,) containing yaw angles in radians
    """
    if frame not in {"habitat", "dynamics"}:
        raise ValueError(f"frame must be 'habitat' or 'dynamics', got {frame!r}")
    n_pts = len(points)
    yaw_angles = np.zeros(n_pts)

    for i in range(n_pts):
        # Find a point lookahead_dist away
        current_p = points[i]
        target_p = points[-1]  # Default to last point

        # Search forward for a point at least lookahead_dist away
        for j in range(i + 1, n_pts):
            dist = np.linalg.norm(points[j] - current_p)
            if dist >= lookahead_dist:
                target_p = points[j]
                break

        direction = target_p - current_p
        if np.linalg.norm(direction) < 1e-3:
            # If we are at the end or stuck, use the previous yaw
            if i > 0:
                yaw_angles[i] = yaw_angles[i - 1]
            else:
                yaw_angles[i] = 0.0
        elif frame == "dynamics":
            yaw_angles[i] = np.arctan2(direction[1], direction[0])
        else:
            yaw_angles[i] = np.arctan2(-direction[0], direction[2])

    return np.unwrap(yaw_angles)


def _ray_is_clear(
    sim,
    origin: np.ndarray,
    direction: np.ndarray,
    distance: float,
    ignored_object_ids: set[int] | None = None,
) -> bool:
    """Return whether a Habitat ray reaches ``distance`` without hitting geometry."""
    import habitat_sim

    ray = habitat_sim.geo.Ray(
        np.asarray(origin, dtype=np.float32),
        np.asarray(direction, dtype=np.float32),
    )
    hits = sim.cast_ray(ray, max_distance=float(distance))
    if not hits.has_hits():
        return True
    ignored = ignored_object_ids or set()
    for hit in hits.hits:
        if int(getattr(hit, "object_id", -1)) in ignored:
            continue
        return float(hit.ray_distance) >= distance - 1e-4
    return True


def _validate_static_clearance(
    traj: MinSnap,
    sim,
    trajectory_to_habitat,
    clearance_m: float,
    sample_spacing_m: float,
) -> None:
    """Reject a nominal trajectory whose swept body intersects scene geometry.

    Habitat's navmesh is a 2-D walking surface and cannot establish free space
    at flight altitude.  This check queries the rendered scene mesh directly.
    It samples the MinSnap curve densely in arc length, checks the centerline
    between samples, and casts a spherical set of clearance probes at every
    sample.  Sampling is capped at half the requested clearance so an obstacle
    cannot sit between two body-clearance probes without also crossing a
    centerline segment or a neighbouring probe volume.
    """
    if clearance_m <= 0.0:
        return

    end = float(traj.t_keyframes[-1])
    # First sample finely in time, then resample by arc length.  v_max is 3 m/s,
    # so 200 Hz bounds the initial chord length to roughly 1.5 cm.
    times = np.linspace(0.0, end, max(2, int(math.ceil(end * 200.0)) + 1))
    dyn_points = np.asarray([traj.update(float(t))["x"] for t in times])
    hab_points = np.asarray(trajectory_to_habitat(dyn_points), dtype=np.float64)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(hab_points, axis=0), axis=1))]
    )
    spacing = min(float(sample_spacing_m), 0.5 * float(clearance_m))
    distances = np.arange(0.0, cumulative[-1], spacing)
    distances = np.append(distances, cumulative[-1])
    points = np.column_stack(
        [np.interp(distances, cumulative, hab_points[:, axis]) for axis in range(3)]
    )

    validate_static_points_clearance(sim, points, clearance_m)


def validate_static_points_clearance(
    sim,
    points: np.ndarray,
    clearance_m: float,
    ignored_object_ids: set[int] | None = None,
) -> None:
    """Validate a sampled Habitat-frame path against the rendered scene mesh."""
    points = np.asarray(points, dtype=np.float64)
    if clearance_m <= 0.0 or len(points) == 0:
        return

    # Axis, edge-diagonal, and corner-diagonal rays cover the body sphere in
    # 26 directions.  This is intentionally conservative and runs only while
    # constructing a trajectory, not in the control loop.
    directions = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                vec = np.array([x, y, z], dtype=np.float64)
                norm = np.linalg.norm(vec)
                if norm > 0.0:
                    directions.append(vec / norm)

    for index, point in enumerate(points):
        for direction in directions:
            if not _ray_is_clear(
                sim, point, direction, clearance_m, ignored_object_ids
            ):
                raise ValueError(
                    f"trajectory lacks {clearance_m:.3f} m static "
                    f"clearance at path sample {index}/{len(points) - 1}, "
                    f"point={point.tolist()}, direction={direction.tolist()}"
                )
        segment = point - points[index - 1] if index else None
        segment_length = float(np.linalg.norm(segment)) if index else 0.0
        if index and segment_length > 1e-8 and not _ray_is_clear(
            sim,
            points[index - 1],
            segment / segment_length,
            segment_length,
            ignored_object_ids,
        ):
            raise ValueError(
                f"nominal trajectory centerline intersects static geometry at "
                f"segment {index - 1}/{len(points) - 2}"
            )


def _validate_trajectory(
    traj: MinSnap,
    samples: int = 12,
    *,
    collision_sim=None,
    trajectory_to_habitat=None,
    static_clearance_m: float = 0.0,
    collision_sample_spacing_m: float = 0.05,
) -> None:
    """Evaluate the trajectory end-to-end so malformed ones fail at build time.

    MinSnap can return an object whose ``t_keyframes`` implies more segments
    than its polynomial arrays actually hold (degenerate waypoint geometry
    makes it drop segments internally), and it clips the query time to
    ``t_keyframes[-1]`` rather than to what the arrays actually cover -- so
    the mismatch survives a successful build and only surfaces as an
    ``IndexError`` from whichever caller later happens to sample the bad
    tail, e.g. intercept-schedule solving at episode reset, which kills the
    worker and the run. Sampling here converts that into an ordinary build
    failure, which the retry above already handles by drawing a different
    path.

    Also covers the ``self.null`` case (fewer than two distinct waypoints
    survived MinSnap's own deduplication -- degenerate geometry, e.g. every
    sampled waypoint collapsing to nearly one point), where ``t_keyframes``
    is never set at all.
    """
    if getattr(traj, "null", False) or not hasattr(traj, "t_keyframes"):
        raise ValueError("MinSnap trajectory is null (fewer than 2 distinct waypoints)")
    end = float(traj.t_keyframes[-1])
    for t in np.linspace(0.0, end, samples):
        traj.update(float(t))
    if static_clearance_m > 0.0:
        if collision_sim is None or trajectory_to_habitat is None:
            raise ValueError(
                "static_clearance_m requires collision_sim and trajectory_to_habitat"
            )
        _validate_static_clearance(
            traj,
            collision_sim,
            trajectory_to_habitat,
            static_clearance_m,
            collision_sample_spacing_m,
        )


def generate_interesting_traj(
    pathfinder,
    seed: int,
    max_path_attempts: int = 6,
    collision_sim=None,
    trajectory_to_habitat=None,
    static_clearance_m: float = 0.0,
    collision_sample_spacing_m: float = 0.05,
    **kwargs,
) -> MinSnap:
    """Build a trajectory, resampling the path if MinSnap cannot solve it.

    The underlying QPs are not always feasible for a given set of waypoints:
    cvxopt returns ``None`` for the yaw coefficients (surfacing as a
    ``TypeError`` inside MinSnap), and near-collinear waypoints can make the
    position problem rank-deficient. Both depend only on the sampled
    geometry, so a fresh sample is a valid fix -- and without it a single
    infeasible sample aborts the episode, and with it the training run.

    The offset is a prime so successive attempts decorrelate rather than
    walking through neighbouring seeds.

    An explicit ``start`` in ``kwargs`` is honoured only on the first
    attempt. Callers can pin the start point (e.g. to the vehicle's actual
    spawn), but a caller-supplied start was not sampled with this
    function's own margins -- if it happens to sit hard against a wall,
    every waypoint set built from it fails the bounds check identically,
    and reusing it across all retries would exhaust them on one bad point
    rather than ever trying a different one.
    """
    last_exc: Exception | None = None
    for attempt in range(max_path_attempts):
        attempt_kwargs = dict(kwargs)
        if attempt > 0:
            attempt_kwargs.pop("start", None)
        try:
            traj = _generate_interesting_traj_once(
                pathfinder=pathfinder,
                seed=seed + attempt * 7919,
                **attempt_kwargs,
            )
            _validate_trajectory(
                traj,
                collision_sim=collision_sim,
                trajectory_to_habitat=trajectory_to_habitat,
                static_clearance_m=static_clearance_m,
                collision_sample_spacing_m=collision_sample_spacing_m,
            )
            return traj
        except Exception as exc:  # noqa: BLE001 - solver failures are varied
            last_exc = exc
            logger.warning(
                "MinSnap build failed (attempt %d/%d): %s; resampling path",
                attempt + 1,
                max_path_attempts,
                exc,
            )
    raise RuntimeError(
        f"Could not build a MinSnap trajectory in {max_path_attempts} attempts"
    ) from last_exc


def _generate_interesting_traj_once(
    pathfinder,
    seed: int,
    target_length: float = 30.0,
    min_waypoint_distance: float = 2.0,
    max_waypoints: int = 100,
    v_avg: float = 1.0,
    start: np.ndarray | None = None,
    max_tries_per_waypoint: int = 100,
    coord_transform=None,
    min_altitude_m: float = 0.0,
    max_altitude_m: float | None = None,
    ceiling_margin_m: float = 0.3,
    lateral_margin_m: float = 0.0,
    bounds_margin_m: float = 0.0,
) -> MinSnap:
    """Generate a longer trajectory by sampling distant waypoints and connecting them.

    Algorithm:
    1. Start at a random navigable point (or provided `start`).
    2. Sample waypoints that are at least `min_waypoint_distance` away from each other.
    3. Connect consecutive waypoints with shortest paths.
    4. Add random heights to waypoints while ensuring navigability.
    5. Smooth the combined path using MinSnap.

    Args:
        pathfinder: Habitat pathfinder instance
        seed: Random seed for reproducibility
        target_length: Minimum total path length to achieve
        min_waypoint_distance: Minimum distance between sampled waypoints
        max_waypoints: Maximum number of waypoints to sample
        v_avg: Average velocity for MinSnap trajectory
        start: Starting point (if None, a random navigable point is used)
        max_tries_per_waypoint: Maximum tries per waypoint sampling
        coord_transform: Optional coordinate transform function to apply to path points.
                         Useful to convert from visual sim to dynamics coordinate system.
        min_altitude_m: Minimum height above the *local* floor for every waypoint
                         (see `sample_random_navigable_point_with_height`). The
                         navmesh's own walkability check is a much weaker
                         guarantee here -- it's generated for a much taller
                         walking-agent collision cylinder than the actual
                         vehicle, and was observed to pass entire trajectories
                         through space a drone would fly through furniture to
                         reach. ~1 m clears most household furniture.
        max_altitude_m: Maximum height above the local floor, or `None` for
                         "up to `ceiling_margin_m` below the scene's ceiling".
        ceiling_margin_m: Keep waypoints this far below the scene's ceiling.
        lateral_margin_m: Keep waypoints this far from the scene's horizontal extremes.
        bounds_margin_m: Reject (and let the caller resample) any path that comes
                         within this distance of the scene bounds. Waypoint-level
                         margins alone don't cover this: the flown path comes from
                         `find_shortest_path_points`, whose navmesh points can hug
                         walls even when the sampled waypoints themselves did not.

    Returns:
        MinSnap trajectory object
    """
    pathfinder.seed(seed)
    random.seed(seed)

    max_tries = max_tries_per_waypoint * max_waypoints

    if start is None:
        start = sample_random_navigable_point_with_height(
            pathfinder,
            min_altitude_m=min_altitude_m,
            max_altitude_m=max_altitude_m,
            ceiling_margin_m=ceiling_margin_m,
            lateral_margin_m=lateral_margin_m,
        )
        if start is None:
            raise RuntimeError(
                "Unable to sample initial navigable point."
                "Are you just unlucky? Or is the navmesh broken?"
            )

    # Initialize path points with start and sample waypoints
    full_points_list = [start.reshape(1, 3)]
    current_pos = start
    total_length = 0.0
    num_tries = 0
    num_waypoints = 1

    while (
        total_length < target_length
        and num_waypoints < max_waypoints
        and num_tries < max_tries
    ):
        num_tries += 1

        # Sample a random navigable point with random height within bounds
        candidate = sample_random_navigable_point_with_height(
            pathfinder,
            min_altitude_m=min_altitude_m,
            max_altitude_m=max_altitude_m,
            ceiling_margin_m=ceiling_margin_m,
            lateral_margin_m=lateral_margin_m,
        )

        # ensure candidate is valid and sufficiently far
        if (
            candidate is None
            or np.linalg.norm(candidate - current_pos) < min_waypoint_distance
        ):
            continue

        # Try to find a path to this candidate
        found, path_len, seg = find_shortest_path_points(
            pathfinder, current_pos, candidate
        )
        if found:
            if len(seg) > 1:
                # Densify the segment to ensure MinSnap has enough constraints
                seg = densify_path(seg, max_dist=1.5)
                # `find_shortest_path_points` snaps every intermediate point
                # to the navmesh, i.e. floor level, discarding whatever
                # height `current_pos`/`candidate` were actually sampled at.
                # Interpolating between the two endpoints' own heights keeps
                # the flown path at the height that was sampled (and
                # margin-checked), instead of diving to the floor and back
                # between every waypoint pair.
                seg = np.asarray(seg, dtype=np.float32).copy()
                arc = np.concatenate(
                    [[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
                )
                frac = arc / arc[-1] if arc[-1] > 1e-6 else np.zeros_like(arc)
                seg[:, 1] = current_pos[1] + frac * (candidate[1] - current_pos[1])
                # Skip the first point to avoid duplicates
                full_points_list.append(seg[1:])
            current_pos = candidate
            total_length += path_len
            num_waypoints += 1

    logger.info(
        f"Sampled {num_waypoints} waypoints with total length {total_length:.2f}"
    )

    full_path = np.concatenate(full_points_list, axis=0)

    logger.info(f"Generated raw path with {len(full_path)} points.")

    # Remove consecutive duplicates to avoid zero-duration segments in MinSnap
    if len(full_path) > 1:
        full_path_dedup = [full_path[0]]
        for i in range(1, len(full_path)):
            if np.linalg.norm(full_path[i] - full_path[i - 1]) > 1e-3:
                full_path_dedup.append(full_path[i])
        full_path = np.array(full_path_dedup)

    # Reject paths that run too close to the scene bounds. Waypoint-level
    # margins don't cover this on their own: the flown path comes from
    # `find_shortest_path_points`, whose navmesh points hug walls. A drone
    # flying within `bounds_margin_m` of the boundary cannot sidestep an
    # obstacle without leaving the scene AABB. Raising here lets the
    # existing resample-retry draw a path with room to manoeuvre.
    if bounds_margin_m > 0.0:
        lo, hi = pathfinder.get_bounds()
        lo = np.asarray(lo, dtype=np.float64) + bounds_margin_m
        hi = np.asarray(hi, dtype=np.float64) - bounds_margin_m
        if np.any(full_path < lo) or np.any(full_path > hi):
            raise ValueError(
                f"path passes within {bounds_margin_m} m of the scene bounds; "
                "no room to dodge"
            )

    if coord_transform is not None:
        full_path = coord_transform(full_path)
        logger.info("Applied inverse coordinate transform to path")

    # Calculate desired yaw angles for waypoints to look ahead. The frame
    # must match whichever space `full_path` is now in: once `coord_transform`
    # has mapped it to dynamics (Z up), the habitat (Y up) formula would read
    # altitude as a horizontal component and yield a heading tied to
    # climb/descent rather than the direction of travel.
    yaw_frame = "dynamics" if coord_transform is not None else "habitat"
    yaw_angles = calculate_smooth_yaw(
        full_path, lookahead_dist=2.0, frame=yaw_frame
    )

    traj = MinSnap(
        points=full_path,
        yaw_angles=yaw_angles,
        yaw_rate_max=2 * np.pi,
        poly_degree=7,
        yaw_poly_degree=7,
        v_max=3.0,
        v_avg=v_avg,
        v_start=np.zeros(3),
        v_end=np.zeros(3),
        verbose=False,
    )

    return traj


def _sample_freespace_points(
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    rng: np.random.Generator,
    target_length: float,
    min_waypoint_distance: float,
    max_waypoints: int,
    start: np.ndarray | None,
) -> np.ndarray:
    """Sample a waypoint chain inside an axis-aligned box.

    The navmesh-based sampler cannot be used in open space: it draws points
    from a 2-D walking surface, so there is no navmesh above a roof and none
    at all in an empty stage. Here the flight volume is stated explicitly and
    waypoints are drawn from it directly.

    Successive waypoints are kept at least ``min_waypoint_distance`` apart so
    MinSnap does not receive near-coincident points (which makes its QP
    rank-deficient), and the chain stops once the accumulated length reaches
    ``target_length``.
    """
    box_lo = np.asarray(box_lo, dtype=np.float64).reshape(3)
    box_hi = np.asarray(box_hi, dtype=np.float64).reshape(3)
    if np.any(box_hi <= box_lo):
        raise ValueError(f"empty flight box: lo={box_lo} hi={box_hi}")

    if start is None:
        current = rng.uniform(box_lo, box_hi)
    else:
        current = np.clip(np.asarray(start, dtype=np.float64).reshape(3), box_lo, box_hi)

    points = [current.copy()]
    total = 0.0
    for _ in range(max_waypoints):
        if total >= target_length:
            break
        for _ in range(200):
            candidate = rng.uniform(box_lo, box_hi)
            step = float(np.linalg.norm(candidate - current))
            if step >= min_waypoint_distance:
                break
        else:
            # The box is too small to place another waypoint at the required
            # separation; return what we have rather than emitting a
            # degenerate pair.
            break
        points.append(candidate.copy())
        total += step
        current = candidate

    if len(points) < 2:
        raise RuntimeError(
            "free-space sampler produced fewer than 2 waypoints; "
            f"box={box_lo}..{box_hi} min_waypoint_distance={min_waypoint_distance}"
        )
    return np.asarray(points, dtype=np.float64)


def generate_freespace_traj(
    seed: int,
    box_lo,
    box_hi,
    target_length: float = 20.0,
    min_waypoint_distance: float = 2.0,
    max_waypoints: int = 100,
    v_avg: float = 1.0,
    start: np.ndarray | None = None,
    coord_transform=None,
    max_path_attempts: int = 6,
    collision_sim=None,
    trajectory_to_habitat=None,
    static_clearance_m: float = 0.0,
    collision_sample_spacing_m: float = 0.05,
    densify_max_dist: float = 2.0,
    **_ignored,
) -> MinSnap:
    """MinSnap trajectory through an explicit free-space box, no navmesh.

    Mirrors ``generate_interesting_traj``'s retry contract: MinSnap's QPs are
    not always feasible for a given waypoint set, and since feasibility
    depends only on the sampled geometry, a fresh sample is a valid fix.

    ``collision_sim`` is still honoured. In a genuinely empty stage no ray
    hits anything and validation passes trivially, but when this is used to
    fly above real geometry the clearance check is what keeps the path off
    the roof.
    """
    last_exc: Exception | None = None
    for attempt in range(max_path_attempts):
        try:
            rng = np.random.default_rng(seed + attempt * 7919)
            points = _sample_freespace_points(
                box_lo=box_lo,
                box_hi=box_hi,
                rng=rng,
                target_length=target_length,
                min_waypoint_distance=min_waypoint_distance,
                max_waypoints=max_waypoints,
                # Only the first attempt honours a caller-supplied start, for
                # the same reason as generate_interesting_traj: a pinned start
                # that fails would fail identically on every retry.
                start=start if attempt == 0 else None,
            )
            dense = densify_path(points, max_dist=densify_max_dist)
            # Transform first, then compute yaw in whichever frame the points
            # now live in -- the same order generate_interesting_traj uses.
            # Computing yaw in habitat space and transforming afterwards only
            # coincides with this when the transform is exactly the axis
            # permutation that maps atan2(-dx, dz) onto atan2(dy, dx), which
            # is not a property to depend on silently.
            if coord_transform is not None:
                dense = coord_transform(dense)
            yaw_frame = "dynamics" if coord_transform is not None else "habitat"
            yaw = calculate_smooth_yaw(dense, lookahead_dist=2.0, frame=yaw_frame)
            traj = MinSnap(
                points=np.asarray(dense),
                yaw_angles=np.asarray(yaw),
                yaw_rate_max=2 * np.pi,
                poly_degree=7,
                yaw_poly_degree=7,
                v_max=3.0,
                v_avg=v_avg,
                v_start=np.zeros(3),
                v_end=np.zeros(3),
                verbose=False,
            )
            _validate_trajectory(
                traj,
                collision_sim=collision_sim,
                trajectory_to_habitat=trajectory_to_habitat,
                static_clearance_m=static_clearance_m,
                collision_sample_spacing_m=collision_sample_spacing_m,
            )
            return traj
        except Exception as exc:  # noqa: BLE001 - solver failures are varied
            last_exc = exc
            logger.warning(
                "free-space MinSnap build failed (attempt %d/%d): %s; resampling",
                attempt + 1,
                max_path_attempts,
                exc,
            )
    raise RuntimeError(
        f"Could not build a free-space MinSnap trajectory in {max_path_attempts} attempts"
    ) from last_exc
