"""Scan ``habitat_random_minsnap`` trajectories for failures and film the worst ones.

The camera pose comes from the trajectory reference, not a closed-loop rollout: the SE3
controller tracks these to ~0.02 m median error, far below the clearance thresholds here.

Example:
    python scripts/check_trajectories.py --config configs/online_data_hm3d_2gpu.yaml \
        --scenes 'data/hm3d/0000*/*.basis.glb' --seeds 10 --record 5
"""

import argparse
import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import habitat_sim
import habitat_sim.utils.common as hutils
import imageio
import magnum as mn
import numpy as np
import yaml

from neurosim.core.trajectory.habitat_trajs import (
    calculate_smooth_yaw,
    generate_interesting_traj,
    sample_minsnap_trajectory,
    sample_waypoint_path,
)

logger = logging.getLogger(__name__)

PROBE_DIRS = (
    mn.Vector3(1, 0, 0),
    mn.Vector3(-1, 0, 0),
    mn.Vector3(0, 1, 0),
    mn.Vector3(0, -1, 0),
    mn.Vector3(0, 0, 1),
    mn.Vector3(0, 0, -1),
)
PROBE_MAX_DIST = 3.0
BURIED_M = 0.15  # below this the camera is inside geometry: the frame is a surface
YAW_RATE_MAX = 2 * np.pi  # what generate_interesting_traj hands MinSnap as yaw_rate_max
SAMPLE_HZ = 20.0
CAMERA_WH = (512, 384)


@dataclass
class SceneCfg:
    """Navmesh, camera mount and trajectory parameters read from a settings YAML."""

    agent_height: float
    agent_radius: float
    agent_max_climb: float
    agent_max_slope: float
    hfov: float
    mount: np.ndarray
    sim_time: float
    target_length: float
    min_waypoint_distance: float
    max_waypoints: int
    v_avg: float


class EpisodeReport(NamedTuple):
    """One (scene, seed) trajectory measured; ``error`` is empty when it built."""

    scene: str
    seed: int
    error: str
    duration: float
    frozen_tail: float
    yaw_rate_peak: float
    n_yaw_over_budget: int
    buried_frac: float
    buried_longest: float
    clearance_median: float
    out_of_bounds_frac: float


def read_scene_cfg(config: Path) -> SceneCfg:
    """Parse the settings YAML into the parameters this diagnostic needs."""
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    vb = settings["visual_backend"]
    traj = settings["trajectory"]
    camera = next(
        cfg for cfg in vb["sensors"].values() if cfg["type"] in ("color", "depth")
    )
    # HabitatWrapper adds agent_height to the configured sensor y, so the camera rides
    # that far above the trajectory point.
    mount = np.asarray(camera["position"], dtype=float).copy()
    mount[1] += vb["agent_height"]
    return SceneCfg(
        agent_height=vb["agent_height"],
        agent_radius=vb["agent_radius"],
        agent_max_climb=vb["agent_max_climb"],
        agent_max_slope=vb["agent_max_slope"],
        hfov=camera["hfov"],
        mount=mount,
        sim_time=settings["simulator"]["sim_time"],
        target_length=traj["target_length"],
        min_waypoint_distance=traj["min_waypoint_distance"],
        max_waypoints=traj["max_waypoints"],
        v_avg=traj["v_avg"],
    )


def open_scene(scene: str, cfg: SceneCfg) -> habitat_sim.Simulator:
    """Habitat simulator with physics (cast_ray needs it) and the config's navmesh."""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene
    sim_cfg.enable_physics = True

    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "color"
    spec.sensor_type = habitat_sim.SensorType.COLOR
    spec.resolution = [CAMERA_WH[1], CAMERA_WH[0]]
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.position = mn.Vector3(*cfg.mount)
    spec.hfov = cfg.hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    navmesh = habitat_sim.NavMeshSettings()
    navmesh.set_defaults()
    navmesh.agent_height = cfg.agent_height
    navmesh.agent_radius = cfg.agent_radius
    navmesh.agent_max_climb = cfg.agent_max_climb
    navmesh.agent_max_slope = cfg.agent_max_slope
    sim.recompute_navmesh(sim.pathfinder, navmesh)
    return sim


def probe_clearance(sim: habitat_sim.Simulator, point: np.ndarray) -> float:
    """Distance to the nearest geometry along any axis, capped at PROBE_MAX_DIST."""
    origin = mn.Vector3(*point)
    hits = [
        sim.cast_ray(habitat_sim.geo.Ray(origin, d), max_distance=PROBE_MAX_DIST)
        for d in PROBE_DIRS
    ]
    dists = [h.hits[0].ray_distance for h in hits if h.has_hits()]
    return min(dists, default=PROBE_MAX_DIST)


def camera_track(points: np.ndarray, quats: list, mount: np.ndarray) -> np.ndarray:
    """World camera position per pose: agent position plus the body-frame mount."""
    local = mn.Vector3(*mount)
    return np.array(
        [
            np.asarray(p) + np.asarray(q.transform_vector(local))
            for p, q in zip(points, quats)
        ]
    )


def longest_run(mask: np.ndarray, dt: float) -> float:
    """Duration of the longest unbroken True run."""
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best * dt


def yaw_rate_demand(path: np.ndarray, v_avg: float) -> np.ndarray:
    """Yaw rate each waypoint transition demands; above YAW_RATE_MAX the yaw QP fails."""
    yaw = calculate_smooth_yaw(path, lookahead_dist=2.0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.abs(np.diff(yaw)) / (seg / v_avg)


def build_trajectory(pathfinder, seed: int, cfg: SceneCfg):
    """The production trajectory for this (scene, seed)."""
    return generate_interesting_traj(
        pathfinder,
        seed=seed,
        target_length=cfg.target_length,
        min_waypoint_distance=cfg.min_waypoint_distance,
        max_waypoints=cfg.max_waypoints,
        v_avg=cfg.v_avg,
    )


def scan_episode(
    sim: habitat_sim.Simulator, scene: str, seed: int, cfg: SceneCfg
) -> EpisodeReport:
    """Measure one trajectory, recording rather than raising when generation fails."""
    blank = EpisodeReport(scene, seed, "", 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)

    # The raw path is sampled first so a MinSnap failure still yields yaw diagnostics;
    # both calls reseed from `seed`, so they see the same waypoints.
    path, _ = sample_waypoint_path(
        sim.pathfinder,
        seed=seed,
        target_length=cfg.target_length,
        min_waypoint_distance=cfg.min_waypoint_distance,
        max_waypoints=cfg.max_waypoints,
    )
    # A single point means no navmesh segment was ever connected: MinSnap degenerates to
    # a constant, so the drone hovers on the spot for the whole episode.
    if len(path) < 2:
        return blank._replace(
            error=f"degenerate path: {len(path)} point, drone cannot move"
        )

    rates = yaw_rate_demand(path, cfg.v_avg)
    blank = blank._replace(
        yaw_rate_peak=float(rates.max()),
        n_yaw_over_budget=int((rates > YAW_RATE_MAX).sum()),
    )

    # Observing solver failure is this script's job, so it is caught rather than raised.
    try:
        traj = build_trajectory(sim.pathfinder, seed, cfg)
    except Exception as exc:  # noqa: BLE001
        return blank._replace(error=f"{type(exc).__name__}: {exc}")

    dt = 1.0 / SAMPLE_HZ
    points, quats = sample_minsnap_trajectory(traj, dt=dt)
    clearances = np.array(
        [probe_clearance(sim, c) for c in camera_track(points, quats, cfg.mount)]
    )
    lo, hi = (np.asarray(b, dtype=float) for b in sim.pathfinder.get_bounds())
    duration = float(traj.t_keyframes[-1])

    return blank._replace(
        duration=duration,
        frozen_tail=max(0.0, cfg.sim_time - duration),
        buried_frac=float(np.mean(clearances < BURIED_M)),
        buried_longest=longest_run(clearances < BURIED_M, dt),
        clearance_median=float(np.median(clearances)),
        out_of_bounds_frac=float(
            np.mean(~np.all((points >= lo) & (points <= hi), axis=1))
        ),
    )


def navmesh_panel(
    sim: habitat_sim.Simulator, points: np.ndarray, clearances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Top-down navmesh with the path coloured by clearance, and its pixel coordinates."""
    meters_per_pixel = 0.05
    pathfinder = sim.pathfinder
    grid = np.asarray(
        pathfinder.get_topdown_view(meters_per_pixel, np.median(points[:, 1]))
    )
    panel = np.where(grid[..., None], 90, 20).astype(np.uint8).repeat(3, axis=2)

    lo = np.asarray(pathfinder.get_bounds()[0], dtype=float)
    pix = np.stack(
        [
            (points[:, 0] - lo[0]) / meters_per_pixel,
            (points[:, 2] - lo[2]) / meters_per_pixel,
        ],
        axis=1,
    ).astype(np.int32)

    for i in range(len(pix) - 1):
        blend = float(np.clip(clearances[i] / (2 * BURIED_M), 0.0, 1.0))
        colour = (
            0,
            int(60 + 195 * blend),
            int(255 * (1.0 - blend)),
        )  # BGR: red = buried
        cv2.line(panel, tuple(pix[i]), tuple(pix[i + 1]), colour, 2, cv2.LINE_AA)
    cv2.circle(panel, tuple(pix[0]), 5, (255, 160, 0), -1)
    cv2.circle(panel, tuple(pix[-1]), 5, (255, 0, 255), -1)

    # Letterbox into a fixed box: scenes differ wildly in aspect, and libx264 rejects
    # an odd-width frame, which a free-running panel width produces.
    width, height = CAMERA_WH
    scale = min(width / panel.shape[1], height / panel.shape[0])
    resized = cv2.resize(
        panel, (round(panel.shape[1] * scale), round(panel.shape[0] * scale))
    )
    box = np.full((height, width, 3), 20, dtype=np.uint8)
    origin = np.array(
        [(width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2]
    )
    box[
        origin[1] : origin[1] + resized.shape[0],
        origin[0] : origin[0] + resized.shape[1],
    ] = resized
    return box, (pix * scale + origin).astype(np.int32)


def render_episode(
    sim: habitat_sim.Simulator, traj, cfg: SceneCfg, out_path: Path
) -> None:
    """Write an mp4 of the camera view with a clearance HUD beside the top-down path."""
    fps = SAMPLE_HZ
    points, quats = sample_minsnap_trajectory(traj, dt=1.0 / fps)
    clearances = np.array(
        [probe_clearance(sim, c) for c in camera_track(points, quats, cfg.mount)]
    )
    speeds = np.linalg.norm(np.gradient(points, 1.0 / fps, axis=0), axis=1)
    panel, pix = navmesh_panel(sim, points, clearances)

    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    width, height = CAMERA_WH

    with imageio.get_writer(str(out_path), fps=int(fps), macro_block_size=1) as writer:
        for i, (position, rotation) in enumerate(zip(points, quats)):
            state.position = position
            state.rotation = hutils.quat_from_magnum(rotation)
            agent.set_state(state)
            rgb = sim.get_sensor_observations()["color"][..., :3]
            frame = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

            buried = clearances[i] < BURIED_M
            if buried:
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)
            hud = [
                f"t {i / fps:5.1f}s",
                f"speed {speeds[i]:4.2f} m/s",
                f"clearance {clearances[i]:4.2f} m",
                "BURIED - camera inside geometry" if buried else "",
            ]
            for row, line in enumerate(hud):
                origin = (10, 24 + 22 * row)
                cv2.putText(
                    frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3
                )
                cv2.putText(
                    frame,
                    line,
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            side = panel.copy()
            cv2.circle(side, tuple(pix[i]), 4, (255, 255, 255), -1)
            composed = np.hstack([frame, side])
            writer.append_data(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB))


def print_table(reports: list[EpisodeReport], cfg: SceneCfg) -> None:
    """One row per episode, worst first, then the counts that matter."""
    header = (
        f"{'scene':<24}{'seed':>5}{'dur':>7}{'frozen':>8}{'yawpk':>7}{'over':>6}"
        f"{'buried%':>9}{'longest':>9}{'oob%':>7}  status"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(reports, key=rank, reverse=True):
        print(
            f"{Path(r.scene).parent.name:<24}{r.seed:>5}{r.duration:>7.1f}"
            f"{r.frozen_tail:>8.1f}{r.yaw_rate_peak:>7.2f}{r.n_yaw_over_budget:>6}"
            f"{100 * r.buried_frac:>9.1f}{r.buried_longest:>9.1f}"
            f"{100 * r.out_of_bounds_frac:>7.1f}  {r.error[:40] or 'ok'}"
        )

    total = len(reports)
    failed = [r for r in reports if r.error]
    print("-" * len(header))
    print(f"generation failed              {len(failed)}/{total}")
    print(
        f"  yaw rate over budget         {sum(1 for r in failed if r.n_yaw_over_budget)}"
    )
    print(
        f"frozen tail > 0.5s             {sum(1 for r in reports if r.frozen_tail > 0.5)}"
        f"/{total}  (sim_time {cfg.sim_time:g}s)"
    )
    print(
        f"camera buried > 1% of episode  {sum(1 for r in reports if r.buried_frac > 0.01)}"
        f"/{total}"
    )
    print(
        f"left scene bounds              {sum(1 for r in reports if r.out_of_bounds_frac)}"
        f"/{total}"
    )


def rank(report: EpisodeReport) -> tuple:
    """Worst first: hard failures, then the longest unbroken buried run."""
    return (bool(report.error), report.buried_longest, report.frozen_tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="settings YAML")
    parser.add_argument("--scenes", type=str, help="glob; default is the config scene")
    parser.add_argument("--seeds", type=int, default=10, help="seeds per scene")
    parser.add_argument("--out", type=Path, default=Path("outputs/traj_check"))
    parser.add_argument(
        "--record", type=int, default=0, help="film the N worst episodes"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = read_scene_cfg(args.config)
    scenes = (
        sorted(glob.glob(args.scenes))
        if args.scenes
        else [yaml.safe_load(args.config.read_text())["visual_backend"]["scene"]]
    )
    if not scenes:
        raise SystemExit(f"no scenes matched {args.scenes!r}")

    print(f"{len(scenes)} scene(s) x {args.seeds} seeds, sim_time {cfg.sim_time:g}s")
    print(
        f"navmesh height={cfg.agent_height} radius={cfg.agent_radius} "
        f"climb={cfg.agent_max_climb} slope={cfg.agent_max_slope}"
    )

    reports = []
    for scene in scenes:
        sim = open_scene(scene, cfg)
        reports += [scan_episode(sim, scene, s, cfg) for s in range(args.seeds)]
        sim.close()

    print_table(reports, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scan.json").write_text(
        json.dumps([r._asdict() for r in reports], indent=2), encoding="utf-8"
    )
    print(f"\nscan written to {args.out / 'scan.json'}")

    # Failed episodes rank first but have no trajectory to film, so they are excluded
    # here rather than silently consuming the --record budget.
    filmable = [r for r in sorted(reports, key=rank, reverse=True) if not r.error]
    worst = filmable[: args.record]
    for scene in {r.scene for r in worst}:
        sim = open_scene(scene, cfg)
        for report in [r for r in worst if r.scene == scene]:
            out_path = args.out / f"{Path(scene).parent.name}_seed{report.seed:03d}.mp4"
            render_episode(
                sim, build_trajectory(sim.pathfinder, report.seed, cfg), cfg, out_path
            )
            print(
                f"  {out_path}  buried {100 * report.buried_frac:.1f}% "
                f"longest {report.buried_longest:.1f}s"
            )
        sim.close()


if __name__ == "__main__":
    main()
