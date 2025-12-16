"""Standalone Habitat-Sim NavMesh path script.

- Samples two navigable points and computes the shortest path (baseline).
- Saves topdown + 3D debug visualizations; can optionally render a short video.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import habitat_sim
import habitat_sim.utils.common as utils

from neurosim.core.trajectory.habitat_trajs import (
    generate_interesting_traj,
    sample_minsnap_trajectory,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_cfg(
    scene_path: str,
    width: int = 256,
    height: int = 256,
    agent_height: float = 1.5,
    agent_radius: float = 0.1,
):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    sensor_specs = []
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [height, width]
    color_sensor_spec.position = [0.0, agent_height, 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.height = agent_height
    agent_cfg.radius = agent_radius

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def world_to_topdown(
    points: list[np.ndarray], pf, meters_per_pixel: float
) -> list[np.ndarray]:
    bounds = pf.get_bounds()
    out: list[np.ndarray] = []
    for p in points:
        px = (p[0] - bounds[0][0]) / meters_per_pixel
        py = (p[2] - bounds[0][2]) / meters_per_pixel
        out.append(np.array([px, py]))
    return out


def save_visualizations(sim, points: list[np.ndarray], out_dir: str, prefix: str):
    ensure_dir(out_dir)
    pf = sim.pathfinder

    meters_per_pixel = 0.05
    height = pf.get_bounds()[0][1]
    td = pf.get_topdown_view(meters_per_pixel, height)
    td_u8 = td.astype(np.uint8)
    colors = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    td_rgb = colors[td_u8]

    pts2d = world_to_topdown(points, pf, meters_per_pixel)

    # Top-down visualization with gradient colors
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(td_rgb)
    ax.axis("off")

    # Use a colormap to color the path by progression
    if len(pts2d) > 1:
        cmap = plt.get_cmap("jet")
        for i in range(len(pts2d) - 1):
            color_idx = i / max(len(pts2d) - 1, 1)
            color = cmap(color_idx)
            ax.plot(
                [pts2d[i][0], pts2d[i + 1][0]],
                [pts2d[i][1], pts2d[i + 1][1]],
                color=color,
                linewidth=2,
                alpha=0.8,
            )

    fig.savefig(
        os.path.join(out_dir, f"{prefix}_topdown.png"), bbox_inches="tight", dpi=150
    )
    plt.close(fig)

    # 3D visualization with gradient colors
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot path segments with color gradient
    if len(points) > 1:
        cmap = plt.get_cmap("jet")
        for i in range(len(points) - 1):
            color_idx = i / max(len(points) - 1, 1)
            color = cmap(color_idx)
            ax.plot(
                [xs[i], xs[i + 1]],
                [zs[i], zs[i + 1]],
                [ys[i], ys[i + 1]],
                color=color,
                linewidth=1.5,
                alpha=0.7,
            )

        # Mark start and end
        ax.scatter(
            [xs[0]], [zs[0]], [ys[0]], color="green", s=100, label="Start", zorder=10
        )
        ax.scatter(
            [xs[-1]], [zs[-1]], [ys[-1]], color="red", s=100, label="End", zorder=10
        )
        ax.legend()

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title("3D Trajectory Path")
    fig.savefig(os.path.join(out_dir, f"{prefix}_3d.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--v_avg", type=float, default=1.0)
    parser.add_argument("--target_length", type=float, default=30.0)
    parser.add_argument("--agent_height", type=float, default=1.0)
    parser.add_argument("--agent_radius", type=float, default=0.35)
    parser.add_argument("--agent_max_climb", type=float, default=1.0)
    parser.add_argument("--agent_max_slope", type=float, default=90.0)
    args = parser.parse_args()

    cfg = make_cfg(
        args.scene,
        agent_height=args.agent_height,
        agent_radius=args.agent_radius,
    )
    sim = habitat_sim.Simulator(cfg)

    # Recompute navmesh with agent-specific parameters
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()

    # Configure agent parameters for better path planning
    navmesh_settings.agent_height = args.agent_height
    navmesh_settings.agent_radius = args.agent_radius

    # Keep other settings at defaults (commented out for reference):
    # navmesh_settings.cell_size = 0.05
    # navmesh_settings.cell_height = 0.2
    navmesh_settings.agent_max_climb = args.agent_max_climb
    navmesh_settings.agent_max_slope = args.agent_max_slope
    # navmesh_settings.filter_low_hanging_obstacles = True
    # navmesh_settings.filter_ledge_spans = True
    # navmesh_settings.filter_walkable_low_height_spans = True
    # navmesh_settings.region_min_size = 20
    # navmesh_settings.region_merge_size = 20
    # navmesh_settings.edge_max_len = 12.0
    # navmesh_settings.edge_max_error = 1.3
    # navmesh_settings.verts_per_poly = 6.0
    # navmesh_settings.detail_sample_dist = 6.0
    # navmesh_settings.detail_sample_max_error = 1.0

    print(
        f"Recomputing navmesh with agent_height={args.agent_height}, agent_radius={args.agent_radius}..."
    )
    navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    if not navmesh_success:
        print("Failed to build the navmesh! Try different parameters.")
        sim.close()
        return

    print("Navmesh recomputed successfully.")

    # Interesting path
    pf = sim.pathfinder
    if not pf.is_loaded:
        print("Pathfinder not loaded.")
        sim.close()
        return

    # Generate interesting path using new API
    traj = generate_interesting_traj(
        pf, seed=args.seed, target_length=args.target_length, v_avg=args.v_avg
    )

    points, orientations = sample_minsnap_trajectory(traj)

    print(f"Interesting path generated; points={len(points)}")
    save_visualizations(sim, list(points), args.out, prefix="interesting")

    if args.save_video:
        try:
            from habitat_sim.utils import viz_utils as vut

            agent = sim.initialize_agent(0)
            agent_state = habitat_sim.AgentState()
            observations = []
            for i, p in enumerate(points):
                agent_state.position = p
                agent_state.rotation = utils.quat_from_magnum(orientations[i])
                agent.set_state(agent_state)
                observations.append(sim.get_sensor_observations())

            vut.make_video(
                observations=observations,
                primary_obs="color_sensor",
                primary_obs_type="color",
                video_file=os.path.join(args.out, "interesting_path"),
                fps=30,
                open_vid=False,
            )
            print("Saved video: interesting_path.mp4")
        except Exception as e:
            print(f"Video rendering unavailable: {e}")

    sim.close()


if __name__ == "__main__":
    main()
