"""NavMesh path generation utilities.

This module is intended to sit "inside" the navmesh logic of neurosim.
It provides:
- A baseline shortest-path query wrapper
- A non-shortest, quadrotor-friendly "interesting" path generator

The interesting path is generated via a random walk on the NavMesh (RRT*-style expansion)
followed by physics-inspired orientation smoothing (yaw lookahead, banking roll, pitch).
"""

import math
import random
from dataclasses import dataclass
import numpy as np
import magnum as mn

try:
    from rotorpy.trajectories.minsnap import MinSnap
except ImportError:
    MinSnap = None


@dataclass(frozen=True)
class NavmeshPath:
    points: np.ndarray  # (N,3)
    orientations: list[mn.Quaternion]  # length N


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
        # Thrust vector aligns with body up (Y in Habitat?)
        # We assume the drone's "Up" axis aligns with the thrust vector (acc - g).
        t_vec = acc - g
        if np.linalg.norm(t_vec) < 1e-6:
            t_vec = np.array([0, 1, 0])

        y_cam = t_vec / np.linalg.norm(t_vec)

        # Desired forward direction based on yaw
        # Assuming yaw=0 means looking along -Z (Habitat convention)
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


def _generate_random_walk(
    pathfinder,
    start: np.ndarray,
    target_length: float,
    step_radius: float,
) -> np.ndarray:
    points_list = [start.reshape(1, 3)]
    current_pos = start
    current_len = 0.0

    # Safety break
    max_iters = int(target_length * 2 / 0.1)  # heuristic
    iters = 0

    while current_len < target_length and iters < max_iters:
        iters += 1
        # Sample near current position
        next_target = np.array(
            pathfinder.get_random_navigable_point_near(
                circle_center=current_pos, radius=step_radius, max_tries=10
            ),
            dtype=np.float32,
        )

        # If we failed to find a point or it's too close, try again or skip
        dist = np.linalg.norm(next_target - current_pos)
        if dist < 0.5:
            continue

        # Compute shortest path to ensure connectivity
        found, dist_seg, seg_points = find_shortest_path_points(
            pathfinder, current_pos, next_target
        )

        if found and len(seg_points) > 1:
            # Append points (skip first as it duplicates current_pos)
            points_list.append(seg_points[1:])
            current_pos = seg_points[-1]
            current_len += dist_seg
        else:
            pass

    if len(points_list) > 1:
        return np.concatenate(points_list, axis=0)
    else:
        return np.array([start], dtype=np.float32)


def _generate_rrt_path(
    pathfinder,
    start: np.ndarray,
    goal: np.ndarray,
    step_radius: float,
    max_iters: int = 2000,
    goal_bias: float = 0.1,
) -> np.ndarray:
    """Generate a path using RRT on the NavMesh."""
    # Tree structure: list of (point, parent_index)
    tree = [(start, -1)]

    for _ in range(max_iters):
        # 1. Sample
        if random.random() < goal_bias:
            q_rand = goal
        else:
            q_rand = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)

        # 2. Nearest
        dists = np.linalg.norm([n[0] - q_rand for n in tree], axis=1)
        nearest_idx = np.argmin(dists)
        q_near = tree[nearest_idx][0]

        # 3. Steer
        vec = q_rand - q_near
        dist = np.linalg.norm(vec)
        if dist < 1e-3:
            continue

        step = min(dist, step_radius)
        q_new = q_near + (vec / dist) * step
        q_new = np.array(pathfinder.snap_point(q_new), dtype=np.float32)

        if np.linalg.norm(q_new - q_near) < 0.1:
            continue

        # 4. Check connectivity
        found, _, _ = find_shortest_path_points(pathfinder, q_near, q_new)
        if not found:
            continue

        # Add to tree
        new_idx = len(tree)
        tree.append((q_new, nearest_idx))

        # Check goal
        if np.linalg.norm(q_new - goal) < step_radius:
            found_goal, _, _ = find_shortest_path_points(pathfinder, q_new, goal)
            if found_goal:
                tree.append((goal, new_idx))
                # Reconstruct
                waypoints = []
                curr_idx = len(tree) - 1
                while curr_idx != -1:
                    waypoints.append(tree[curr_idx][0])
                    curr_idx = tree[curr_idx][1]
                waypoints = waypoints[::-1]

                # Densify
                full_points = [waypoints[0].reshape(1, 3)]
                for i in range(len(waypoints) - 1):
                    found, _, seg = find_shortest_path_points(
                        pathfinder, waypoints[i], waypoints[i + 1]
                    )
                    if found and len(seg) > 1:
                        full_points.append(seg[1:])
                return np.concatenate(full_points, axis=0)

    # Fallback: shortest path
    found, _, pts = find_shortest_path_points(pathfinder, start, goal)
    if found:
        return pts
    return np.array([start], dtype=np.float32)


def generate_interesting_path(
    pathfinder,
    seed: int,
    target_length: float = 20.0,
    step_radius: float = 0.5,
    start: np.ndarray | None = None,
    goal: np.ndarray | None = None,
    mode: str = "random_walk",
    v_avg: float = 1.0,
) -> NavmeshPath:
    """Generate an "interesting" non-shortest quadrotor path.

    Algorithm:
    1. Start at a random navigable point (or `start`).
    2. If mode is "random_walk":
       Randomly sample a new navigable point within `step_radius` of the current tip.
       Repeat until total path length exceeds `target_length`.
    3. If mode is "rrt":
       Use RRT to find a path from start to goal (or random goal).
    4. Smooth using MinSnap.
    """
    pathfinder.seed(seed)
    random.seed(seed)

    if start is None:
        start = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)

    if mode == "rrt":
        if goal is None:
            goal = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)
        full_path = _generate_rrt_path(pathfinder, start, goal, step_radius)
    else:
        full_path = _generate_random_walk(pathfinder, start, target_length, step_radius)
    
    print(f"Generated raw path with {len(full_path)} points, length approx {target_length:.2f}m")
    
    # Remove consecutive duplicates to avoid zero-duration segments in MinSnap
    if len(full_path) > 1:
        full_path_dedup = [full_path[0]]
        for i in range(1, len(full_path)):
            if np.linalg.norm(full_path[i] - full_path[i - 1]) > 1e-3:
                full_path_dedup.append(full_path[i])
        full_path = np.array(full_path_dedup)

    # Calculate desired yaw angles for waypoints to look ahead
    # We can use the simple lookahead logic on the waypoints
    # to feed "desired yaw" to MinSnap.

    # Simple yaw calculation for waypoints
    n_pts = len(full_path)
    yaw_angles = np.zeros(n_pts)
    for i in range(n_pts - 1):
        tangent = full_path[i + 1] - full_path[i]
        yaw_angles[i] = math.atan2(tangent[0], tangent[2])
    yaw_angles[-1] = yaw_angles[-2] if n_pts > 1 else 0.0

    # Unwrap yaw to prevent jumps
    yaw_angles = np.unwrap(yaw_angles)

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
    )

    points, quats = sample_minsnap_trajectory(traj, dt=1.0 / 30.0)
    return NavmeshPath(points=points, orientations=quats)


def world_points_to_topdown(
    points_xyz: np.ndarray,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    meters_per_pixel: float,
) -> np.ndarray:
    """Convert world (x,y,z) to topdown pixel coords (px, py) using bounds."""
    min_b, _max_b = bounds
    px = (points_xyz[:, 0] - float(min_b[0])) / meters_per_pixel
    py = (points_xyz[:, 2] - float(min_b[2])) / meters_per_pixel
    return np.stack([px, py], axis=1)
