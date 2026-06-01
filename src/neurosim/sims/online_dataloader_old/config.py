"""
Configuration for the online dataloader.

Provides a simple, self-contained configuration format where:
- Total simulations = len(scenes) x len(trajectories) x len(seeds)
- No YAML inheritance - all settings in one file
- Seeds control random trajectory generation
"""

import yaml
import logging
from pathlib import Path
from copy import deepcopy
from typing import Iterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SceneConfig:
    """Configuration for a single scene.

    Attributes:
        name: Identifier for the scene
        scene_path: Path to the scene file (.glb)
        sim_time: Simulation time per trajectory in seconds
    """

    name: str
    scene_path: str
    sim_time: float = 30.0


@dataclass(slots=True)
class TrajectoryConfig:
    """Configuration for trajectory generation.

    Attributes:
        name: Identifier for this trajectory configuration
        model: Trajectory model type (e.g., "habitat_random_minsnap")
        target_length: Target trajectory length in meters
        min_waypoint_distance: Minimum distance between waypoints
        max_waypoints: Maximum number of waypoints
        v_avg: Average velocity in m/s
    """

    name: str
    model: str = "habitat_random_minsnap"
    target_length: float = 30.0
    min_waypoint_distance: float = 2.0
    max_waypoints: int = 100
    v_avg: float = 1.0


@dataclass
class DatasetConfig:
    """Main configuration for the online dataset.

    Total simulations = len(scenes) x len(trajectories) x len(seeds)

    Attributes:
        scenes: List of scene configurations
        trajectories: List of trajectory configurations
        seeds: List of random seeds for trajectory generation
        simulator: Simulator settings (world_rate, control_rate, sensor_rates, etc.)
        visual_backend: Visual backend settings (sensors, gpu_id, etc.)
        dynamics: Dynamics model settings
        controller: Controller settings
        publish_state: Whether to publish vehicle state
        ipc_pub_addr: ZMQ address for publishing data
    """

    scenes: list[SceneConfig] = field(default_factory=list)
    trajectories: list[TrajectoryConfig] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42])

    # Core simulation settings (self-contained, no inheritance)
    simulator: dict = field(default_factory=dict)
    visual_backend: dict = field(default_factory=dict)
    dynamics: dict = field(default_factory=dict)
    controller: dict = field(default_factory=dict)

    # Publishing settings
    publish_state: bool = True
    ipc_pub_addr: str = "ipc:///tmp/neurosim_data_pub"

    def __post_init__(self):
        # Convert scene dicts to SceneConfig if needed
        processed_scenes = []
        for scene in self.scenes:
            if isinstance(scene, dict):
                processed_scenes.append(SceneConfig(**scene))
            elif isinstance(scene, SceneConfig):
                processed_scenes.append(scene)
            else:
                raise TypeError(f"Invalid scene type: {type(scene)}")
        self.scenes = processed_scenes

        # Convert trajectory dicts to TrajectoryConfig if needed
        processed_trajs = []
        for traj in self.trajectories:
            if isinstance(traj, dict):
                processed_trajs.append(TrajectoryConfig(**traj))
            elif isinstance(traj, TrajectoryConfig):
                processed_trajs.append(traj)
            else:
                raise TypeError(f"Invalid trajectory type: {type(traj)}")
        self.trajectories = processed_trajs

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "DatasetConfig":
        """Load configuration from a YAML file.

        The YAML format is self-contained:

        ```yaml
        scenes:
          - name: apartment_1
            scene_path: data/scene_datasets/habitat-test-scenes/apartment_1.glb
            sim_time: 30.0
          - name: skokloster
            scene_path: data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
            sim_time: 45.0

        trajectories:
          - name: slow
            target_length: 30.0
            v_avg: 1.0
          - name: fast
            target_length: 50.0
            v_avg: 1.5

        seeds: [42, 123, 456]

        simulator:
          world_rate: 1000
          control_rate: 100
          sensor_rates: {...}
          viz_rates: {...}
          additional_sensors: {...}

        visual_backend:
          gpu_id: 0
          sensors: {...}
          ...

        dynamics:
          model: rotorpy_multirotor_euler
          vehicle: crazyflie

        controller:
          model: rotorpy_se3
          vehicle: crazyflie
        ```
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def generate_run_configs(
        self, infinite: bool = True
    ) -> Iterator[tuple[dict, dict]]:
        """Generate (settings_dict, metadata) tuples for each simulation run.

        Loops through all configurations:
        Total runs per cycle = len(scenes) x len(trajectories) x len(seeds)

        If infinite=True, continues indefinitely cycling through all runs.

        Yields:
            Tuple of (settings_dict, metadata) where:
            - settings_dict: Complete settings for SynchronousSimulator
            - metadata: Dict with scene name, trajectory name, seed, run index
        """
        run_idx = 0

        while True:
            for scene in self.scenes:
                for traj in self.trajectories:
                    for seed in self.seeds:
                        # Build complete settings dict
                        settings = {
                            "simulator": deepcopy(self.simulator),
                            "visual_backend": deepcopy(self.visual_backend),
                            "dynamics": deepcopy(self.dynamics),
                            "controller": deepcopy(self.controller),
                        }

                        # Set scene-specific settings
                        settings["visual_backend"]["scene"] = scene.scene_path
                        settings["simulator"]["sim_time"] = scene.sim_time

                        # Set trajectory with seed
                        settings["trajectory"] = {
                            "model": traj.model,
                            "seed": seed,
                            "target_length": traj.target_length,
                            "min_waypoint_distance": traj.min_waypoint_distance,
                            "max_waypoints": traj.max_waypoints,
                            "v_avg": traj.v_avg,
                        }

                        # Build metadata
                        metadata = {
                            "scene_name": scene.name,
                            "scene_path": scene.scene_path,
                            "trajectory_name": traj.name,
                            "seed": seed,
                            "run_idx": run_idx,
                            "sim_time": scene.sim_time,
                        }

                        yield settings, metadata
                        run_idx += 1

            if not infinite:
                break

    def get_total_runs(self) -> int:
        """Get total number of simulation runs.

        Total = len(scenes) x len(trajectories) x len(seeds)
        """
        return len(self.scenes) * len(self.trajectories) * len(self.seeds)

    def __repr__(self) -> str:
        return (
            "DatasetConfig("
            f"scenes={len(self.scenes)}, "
            f"trajectories={len(self.trajectories)}, "
            f"seeds={len(self.seeds)}, "
            f"total_runs={self.get_total_runs()})"
        )
