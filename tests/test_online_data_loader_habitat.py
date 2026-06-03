"""Habitat end-to-end test for OnlineDataLoader (PR3).

Spawns a real producer process (SimulatorWorker on Habitat) feeding the bus, and
consumes a batch through the loader. Skips when Habitat / the apartment_1 scene
asset is unavailable. Run in the ``neurosim`` conda env.
"""

import copy
import itertools
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("habitat_sim")

from neurosim.online_data import SampleSchema, OnlineDataLoader

_SETTINGS = Path("configs/apartment_1-settings.yaml")
_SCENE = Path("data/scene_datasets/habitat-test-scenes/apartment_1.glb")

if not _SETTINGS.exists() or not _SCENE.exists():
    pytest.skip("apartment_1 settings/scene asset not available", allow_module_level=True)

BATCH = 4


def _short_settings(sim_time: float = 5.0) -> dict:
    with open(_SETTINGS) as f:
        settings = yaml.safe_load(f)
    settings["simulator"]["sim_time"] = sim_time
    return settings


def test_loader_end_to_end():
    settings = _short_settings()
    schema = SampleSchema.from_sensor_configs(
        {
            uuid: cfg
            for uuid, cfg in settings["visual_backend"]["sensors"].items()
            if uuid in ("depth_camera_1", "event_camera_1")
        },
        anchor=["depth_camera_1"],
        stream=["event_camera_1"],
    )
    loader = OnlineDataLoader(
        schema,
        batch_size=BATCH,
        base_settings=copy.deepcopy(settings),
        worker_kwargs={"gpu_id": 0, "seed": 0},
        bus_maxsize=64,
        event_time_window_us=50_000,
        get_timeout=2.0,
    )
    try:
        batches = list(itertools.islice(loader, 1))
    finally:
        loader.close()

    assert len(batches) == 1, "no batch produced (producer may have died)"
    batch = batches[0]

    depth = batch["depth_camera_1"]
    assert depth.shape == (BATCH, 480, 640)

    counts, events = batch["event_camera_1"]
    assert counts.shape == (BATCH,)
    assert events.shape[1] == 4 and events.shape[0] == int(counts.sum())

    m = batch.meta
    assert m.t_us.shape == (BATCH,)
    assert np.all(np.diff(m.t_us) > 0)  # samples in time order within the batch
    assert m.scene[0].endswith("apartment_1.glb")

    assert loader._proc is None  # producer cleanly terminated by close()
