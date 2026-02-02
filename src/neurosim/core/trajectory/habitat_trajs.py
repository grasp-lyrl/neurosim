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
) -> np.ndarray | None:
    """Sample a random navigable point with random height and check navigability.

    Args:
        pathfinder: Habitat pathfinder instance
        max_retries: Number of retries before giving up

    Returns:
        A navigable point with random height, or None if unable to find one
    """
    # Get height bounds from pathfinder
    bounds = pathfinder.get_bounds()
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]

    for _ in range(max_retries):
        # Sample x, z from navigable mesh
        point = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if point[0] < min_x or point[0] > max_x:
            continue
        if point[2] < min_z or point[2] > max_z:
            continue

        # Sample random height within specified range
        point[1] = random.uniform(min_y, max_y)

        # Check if navigable
        if pathfinder.is_navigable(point):
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


def calculate_smooth_yaw(points: np.ndarray, lookahead_dist: float = 1.0) -> np.ndarray:
    """Calculate smooth yaw angles using a lookahead point.

    Args:
        points: Array of shape (N, 3) containing path points
        lookahead_dist: Distance to look ahead for calculating yaw

    Returns:
        Array of shape (N,) containing yaw angles in radians
    """
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
        else:
            # Habitat convention: yaw=0 is +Z, so atan2(x, z)
            yaw_angles[i] = np.arctan2(-direction[0], direction[2])

    return np.unwrap(yaw_angles)


def generate_interesting_traj(
    pathfinder,
    seed: int,
    target_length: float = 30.0,
    min_waypoint_distance: float = 2.0,
    max_waypoints: int = 100,
    v_avg: float = 1.0,
    start: np.ndarray | None = None,
    max_tries_per_waypoint: int = 100,
    coord_transform=None,
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

    Returns:
        MinSnap trajectory object
    """
    pathfinder.seed(seed)
    random.seed(seed)

    max_tries = max_tries_per_waypoint * max_waypoints

    if start is None:
        start = sample_random_navigable_point_with_height(pathfinder)
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
        candidate = sample_random_navigable_point_with_height(pathfinder)

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

    if coord_transform is not None:
        full_path = coord_transform(full_path)
        logger.info("Applied inverse coordinate transform to path")

    # Calculate desired yaw angles for waypoints to look ahead
    yaw_angles = calculate_smooth_yaw(full_path, lookahead_dist=2.0)

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
