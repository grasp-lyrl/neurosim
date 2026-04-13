"""On-disk logging helpers for SB3 training runs (layout, workers, rollouts)."""

import yaml
import logging
from typing import Any
from pathlib import Path
from collections import Counter

_TRAIN_LOGGER_NAME = "neurosim.rl.train_disk"


def write_run_setup(log_dir: Path, payload: dict[str, Any]) -> None:
    """Write ``run_setup.yaml`` (GPU layout, env counts, key hyperparameters)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "run_setup.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, default_flow_style=False, sort_keys=False)


def configure_training_disk_logger(log_file: Path) -> logging.Logger:
    """Single process-wide file logger for the training driver (rollouts, etc.)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(_TRAIN_LOGGER_NAME)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    log.addHandler(fh)
    log.propagate = False
    return log


def attach_worker_env_logger(
    log_dir: Path,
    *,
    role: str,
    env_idx: int,
    gpu_id: int,
) -> logging.Logger:
    """File logger for one vec-env worker (SubprocVecEnv child process).

    Logs go to ``log_dir/workers/{role}_env_{idx:03d}.log``.
    """
    workers = log_dir / "workers"
    workers.mkdir(parents=True, exist_ok=True)
    name = f"neurosim.rl.worker.{role}.{env_idx}"
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    path = workers / f"{role}_env_{env_idx:03d}.log"
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    log.addHandler(fh)
    log.propagate = False
    log.info(
        "worker_start role=%s env_idx=%s gpu_id=%s logfile=%s",
        role,
        env_idx,
        gpu_id,
        path,
    )
    return log


def gpu_assignment_summary(assign: list[int]) -> dict[str, Any]:
    counts = dict(sorted(Counter(assign).items()))
    return {"env_to_gpu": assign, "envs_per_gpu": counts}
