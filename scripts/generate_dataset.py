"""Generate a domain-randomized H5 dataset from the randomized simulator.

Runs ``num_workers`` simulators in parallel (across ``gpu_ids``), each randomizing
its own scene/sensors/trajectory per episode and writing **one canonical H5 file per
episode** (the same layout ``test_sim.py`` produces). This is the offline counterpart
to the ``online_data`` streaming loader — see ``offline-h5-dataset-plan.md``.

Config (YAML) keys::

    base_settings: configs/apartment_1-settings.yaml   # path to sim settings
    out_dir:       data/depth_events_dataset_v1
    num_episodes:  500
    num_workers:   8
    gpu_ids:       [0, 1]          # cycled across workers
    base_seed:     0
    sim_time:      null            # optional override of simulator.sim_time (seconds)
    domain_randomization:          # optional; scenes/sensors/trajectory/resample_every
      resample_every: 20
      scenes:  [{name: apt1, path: data/.../apartment_1.glb}]
      sensors: {event_camera_1: {contrast_threshold_pos: {range: [0.1, 0.3]}}}
      trajectory: {v_avg: {range: [0.8, 1.5]}}

Usage::

    python scripts/generate_dataset.py --conf scripts/configs/generate_dataset_config.yml
"""

import yaml
import logging
import argparse

from neurosim.online_data import record_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a domain-randomized H5 dataset (one file per episode)."
    )
    parser.add_argument("--conf", required=True, help="Dataset generation config YAML")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()

    with open(args.conf, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(cfg["base_settings"], "r", encoding="utf-8") as f:
        base_settings = yaml.safe_load(f)

    sim_time = cfg.get("sim_time")
    num_episodes = int(cfg.get("num_episodes", 1))
    num_workers = int(cfg.get("num_workers", 1))
    out_dir = cfg["out_dir"]

    if sim_time is not None:
        base_settings.setdefault("simulator", {})["sim_time"] = sim_time

    record_dataset(
        base_settings,
        out_dir=out_dir,
        num_episodes=num_episodes,
        num_workers=num_workers,
        gpu_ids=cfg.get("gpu_ids", [0]),
        base_seed=int(cfg.get("base_seed", 0)),
        randomization=cfg.get("domain_randomization"),
    )


if __name__ == "__main__":
    main()
