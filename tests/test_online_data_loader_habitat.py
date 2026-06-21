"""Habitat end-to-end test for the multi-producer OnlineDataLoader (PR4).

Spawns TWO real producer processes (SimulatorWorkers on Habitat, distinct seeds)
feeding one bus, and consumes batches through the loader — exercising multi-GPU-
style fan-out (both on gpu0 here), diversity (batches mix ``spec_id``), and clean
teardown. Skips when Habitat / the apartment_1 scene asset is unavailable. Run in
the ``neurosim`` conda env.
"""

import copy
import itertools
from pathlib import Path

import pytest
import yaml

pytest.importorskip("habitat_sim")

from neurosim.online_data import SampleSchema, OnlineDataLoader

_SETTINGS = Path("configs/apartment_1-settings.yaml")
_SCENE = Path("data/scene_datasets/habitat-test-scenes/apartment_1.glb")

if not _SETTINGS.exists() or not _SCENE.exists():
    pytest.skip(
        "apartment_1 settings/scene asset not available", allow_module_level=True
    )

BATCH = 4
N_PRODUCERS = 2


def _short_settings(sim_time: float = 5.0) -> dict:
    with open(_SETTINGS) as f:
        settings = yaml.safe_load(f)
    settings["simulator"]["sim_time"] = sim_time
    return settings


def test_loader_multi_producer_end_to_end():
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
        num_producers=N_PRODUCERS,
        gpu_ids=[0],  # both producers on gpu0
        base_seed=0,
        bus_maxsize=64,
        get_timeout=2.0,
    )
    seen_spec_ids = set()
    batches = []
    try:
        # Consume several batches so both producers are represented.
        for batch in itertools.islice(loader, 4):
            batches.append(batch)
            seen_spec_ids.update(batch.meta.spec_id.tolist())
    finally:
        loader.close()

    assert len(batches) == 4, "too few batches (a producer may have died)"

    batch = batches[0]
    depth = batch["depth_camera_1"]
    assert depth.shape == (BATCH, 480, 640)
    counts, events = batch["event_camera_1"]
    assert counts.shape == (BATCH,)
    assert events.shape[1] == 4 and events.shape[0] == int(counts.sum())

    # Diversity: samples come from both producers (distinct spec_ids).
    assert len(seen_spec_ids) >= 2, f"expected >=2 spec_ids, saw {seen_spec_ids}"

    # Clean teardown: no producer left alive / tracked.
    assert loader._procs == []
