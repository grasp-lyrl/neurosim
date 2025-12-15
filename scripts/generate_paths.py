#!/usr/bin/env python3
"""Standalone Habitat-Sim NavMesh path script.

- Samples two navigable points and computes the shortest path (baseline).
- Generates an "interesting" quadrotor trajectory using RRT* style expansion.
- Saves topdown + 3D debug visualizations; can optionally render a short video.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List

import numpy as np
import magnum as mn


# Add src to path to import neurosim
def get_repo_root() -> str:
    try:
        import git

        repo = git.Repo(".", search_parent_directories=True)
        return repo.working_tree_dir
    except Exception:
        # Fallback: assume we are in scripts/ and root is ../
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


root = get_repo_root()
if os.path.join(root, "src") not in sys.path:
    sys.path.append(os.path.join(root, "src"))

from neurosim.core.navmesh_paths import generate_interesting_path, NavmeshPath


def _try_import_habitat_sim():
    try:
        import habitat_sim
        from habitat_sim.utils import common as utils

        return habitat_sim, utils
    except ImportError:
        candidates = [
            os.path.join(root, "deps/habitat-sim/build/lib.linux-x86_64-cpython-310"),
            os.path.join(root, "deps/habitat-sim/build/lib.linux-x86_64-cpython-311"),
            os.path.join(root, "deps/habitat-sim/build/lib"),
        ]
        for p in candidates:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)
        import habitat_sim
        from habitat_sim.utils import common as utils

        return habitat_sim, utils


habitat_sim, utils = _try_import_habitat_sim()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_cfg(
    scene_path: str,
    scene_dataset: str | None = None,
    width: int = 256,
    height: int = 256,
    agent_height: float = 1.5,
    agent_radius: float = 0.1,
):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene_path
    if scene_dataset:
        sim_cfg.scene_dataset_config_file = scene_dataset
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
    points: List[np.ndarray], pf, meters_per_pixel: float
) -> List[np.ndarray]:
    bounds = pf.get_bounds()
    out: List[np.ndarray] = []
    for p in points:
        px = (p[0] - bounds[0][0]) / meters_per_pixel
        py = (p[2] - bounds[0][2]) / meters_per_pixel
        out.append(np.array([px, py]))
    return out


def save_visualizations(sim, points: List[np.ndarray], out_dir: str, prefix: str):
    ensure_dir(out_dir)
    pf = sim.pathfinder

    meters_per_pixel = 0.05
    height = pf.get_bounds()[0][1]
    td = pf.get_topdown_view(meters_per_pixel, height)
    td_u8 = td.astype(np.uint8)
    colors = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    td_rgb = colors[td_u8]

    try:
        import matplotlib.pyplot as plt

        pts2d = world_to_topdown(points, pf, meters_per_pixel)
        plt.figure(figsize=(10, 8))
        plt.axis("off")
        plt.imshow(td_rgb)
        for p in pts2d:
            plt.plot(p[0], p[1], marker="o", markersize=3, color="red")
        plt.savefig(os.path.join(out_dir, f"{prefix}_topdown.png"), bbox_inches="tight")
        plt.close()

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, zs, ys)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        fig.savefig(os.path.join(out_dir, f"{prefix}_3d.png"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Skipping matplotlib visualizations: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--agent_height", type=float, default=0.2)
    parser.add_argument("--agent_radius", type=float, default=0.2)
    parser.add_argument(
        "--mode",
        type=str,
        default="random_walk",
        choices=["random_walk", "rrt"],
        help="Path generation mode",
    )
    args = parser.parse_args()

    root = get_repo_root()
    data_path = os.path.join(root, "data")
    scene = args.scene or os.path.join(
        data_path, "scene_datasets/habitat-test-scenes/apartment_1.glb"
    )
    out_dir = args.out or os.path.join(root, "examples/tutorials/nav_output")

    cfg = make_cfg(
        scene,
        args.dataset,
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
    navmesh_settings.agent_max_climb = 0.2
    navmesh_settings.agent_max_slope = 90.0
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
    nav_path = generate_interesting_path(
        pf, seed=args.seed, target_length=20.0, step_radius=2.0, mode=args.mode
    )

    print(f"Interesting path generated; points={len(nav_path.points)}")
    save_visualizations(sim, list(nav_path.points), out_dir, prefix="interesting")

    if args.save_video:
        try:
            from habitat_sim.utils import viz_utils as vut

            agent = sim.initialize_agent(0)
            agent_state = habitat_sim.AgentState()
            observations = []
            for i, p in enumerate(nav_path.points):
                agent_state.position = p
                agent_state.rotation = utils.quat_from_magnum(nav_path.orientations[i])
                agent.set_state(agent_state)
                observations.append(sim.get_sensor_observations())

            vut.make_video(
                observations=observations,
                primary_obs="color_sensor",
                primary_obs_type="color",
                video_file=os.path.join(out_dir, "interesting_path"),
                fps=30,
                open_vid=False,
            )
            print("Saved video: interesting_path.mp4")
        except Exception as e:
            print(f"Video rendering unavailable: {e}")

    sim.close()


if __name__ == "__main__":
    main()
