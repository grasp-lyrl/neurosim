"""Contracts for the depth training pipeline: data sources, optimizer, validation routing."""

import ast
import logging
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from applications.f3_depth_training.data import process_batch, usable_samples
from applications.f3_depth_training.data.m3ed import DepthFrame, collate_frames
from applications.f3_depth_training.utils import (
    ScaleAndShiftInvariantLoss,
    build_optimizer,
    build_scheduler,
)

APP = Path(__file__).resolve().parents[1] / "applications" / "f3_depth_training"


@pytest.fixture
def batch_args():
    return types.SimpleNamespace(
        event_sensor="ev",
        depth_sensor="dep",
        color_sensor=None,
        event_norm=(640, 480, 20000),
        max_events=0,
        max_disparity=1000.0,
        min_disparity=0.05,
    )


# ── the simulator source ─────────────────────────────────────────────────────
def test_process_batch_normalizes_events_to_the_model_frame(batch_args):
    events = np.array(
        [[320.0, 240.0, 10000.0, 1.0], [640.0, 480.0, 20000.0, 0.0]], np.float32
    )
    batch = {
        "ev": (np.array([2], np.int32), events),
        "dep": np.full((1, 4, 4), 5.0, np.float32),
    }
    ff_events, counts, _, _ = process_batch(batch, batch_args, torch.device("cpu"))

    assert torch.allclose(ff_events[0, :3], torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(ff_events[1, :3], torch.tensor([1.0, 1.0, 1.0]))
    assert ff_events[0, 3] == 1.0, "polarity must not be normalized"
    assert counts.tolist() == [2]


def test_process_batch_converts_depth_to_inverse_depth(batch_args):
    events = np.zeros((1, 4), np.float32)
    batch = {
        "ev": (np.array([1], np.int32), events),
        "dep": np.full((1, 4, 4), 5.0, np.float32),
    }
    _, _, disparity, _ = process_batch(batch, batch_args, torch.device("cpu"))
    assert torch.allclose(disparity, torch.full((1, 4, 4), 0.2)), "disparity is 1/depth"


def test_the_cap_thins_only_the_samples_over_budget():
    from applications.f3_depth_training.data import cap_events

    events = np.arange(4 * 900, dtype=np.float32).reshape(900, 4)
    counts = np.array([100, 500, 300], np.int32)
    capped, kept = cap_events(events, counts, 200)
    assert kept.tolist() == [100, 200, 200]
    assert len(capped) == 500
    assert kept.dtype == counts.dtype


def test_the_cap_spans_the_window_rather_than_truncating_it():
    """Events arrive oldest-first, so keeping a prefix would shorten the window."""
    from applications.f3_depth_training.data import cap_events

    events = np.zeros((1000, 4), np.float32)
    events[:, 2] = np.linspace(20_000, 0, 1000)  # age, oldest first
    capped, _ = cap_events(events, np.array([1000], np.int32), 100)
    assert capped[0, 2] == pytest.approx(20_000), "oldest event dropped"
    assert capped[-1, 2] == pytest.approx(0), "newest event dropped"


def test_an_unset_or_generous_cap_is_a_no_op():
    from applications.f3_depth_training.data import cap_events

    events = np.arange(4 * 40, dtype=np.float32).reshape(40, 4)
    counts = np.array([10, 30], np.int32)
    for cap in (0, 30, 10_000):
        capped, kept = cap_events(events, counts, cap)
        assert capped is events and kept is counts, f"cap={cap} copied needlessly"


def test_usable_samples_needs_half_the_depth_valid():
    mask = torch.zeros(3, 4, 4, dtype=torch.bool)
    mask[0] = True
    mask[1, :2] = True  # exactly half
    assert usable_samples(mask).tolist() == [True, True, False]


# ── the recorded source ──────────────────────────────────────────────────────
def test_collate_concatenates_ragged_event_windows():
    frames = [
        DepthFrame(
            torch.zeros(5, 4), torch.zeros(8, 8), torch.zeros(8, 8, dtype=torch.bool)
        ),
        DepthFrame(
            torch.zeros(3, 4), torch.zeros(8, 8), torch.zeros(8, 8, dtype=torch.bool)
        ),
    ]
    events, counts, disparity, mask = collate_frames(frames)
    assert events.shape == (8, 4), "windows concatenate, they do not pad"
    assert counts.tolist() == [5, 3]
    assert counts.dtype == torch.int32
    assert disparity.shape == mask.shape == (2, 8, 8)


# ── optimizer and schedule ───────────────────────────────────────────────────
class Named(nn.Module):
    """The three name patterns build_optimizer scales by, plus the undecayed embed."""

    def __init__(self):
        super().__init__()
        self.pretrained = nn.Linear(4, 4)
        self.eventff = nn.Linear(4, 4)
        self.head = nn.Linear(4, 4)
        self.patch_embed = nn.Sequential()
        self.patch_embed.add_module("proj", nn.Conv2d(1, 1, 1))


def test_the_optimizer_scales_the_backbone_and_head_apart():
    optimizer = build_optimizer(Named(), 1e-5)
    assert sorted({g["lr"] for g in optimizer.param_groups}) == [5e-6, 1e-5, 1e-4]


def test_the_patch_embed_and_one_dimensional_tensors_skip_decay():
    model = Named()
    optimizer = build_optimizer(model, 1e-5)
    decayed = {
        id(p) for g in optimizer.param_groups if g["weight_decay"] for p in g["params"]
    }
    for name, param in model.named_parameters():
        expected = param.ndim > 1 and "patch_embed.proj" not in name
        assert (id(param) in decayed) == expected, name


def test_the_schedule_warms_up_holds_then_anneals():
    optimizer = build_optimizer(Named(), 1e-5)
    scheduler = build_scheduler(
        optimizer, epochs=100, warmup_epochs=10, cooldown_epochs=30
    )
    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert lrs[0] < lrs[10], "warmup must ramp"
    assert lrs[10] == pytest.approx(max(lrs)), "the hold sits at the peak"
    assert lrs[10] == pytest.approx(lrs[69]), "the middle is flat"
    assert lrs[70] == pytest.approx(max(lrs)), "the cooldown starts at the peak"
    assert lrs[99] < lrs[85] < lrs[70], "the cooldown anneals toward zero"


def test_no_schedule_phases_leaves_a_constant_rate():
    optimizer = build_optimizer(Named(), 1e-5)
    scheduler = build_scheduler(
        optimizer, epochs=10, warmup_epochs=0, cooldown_epochs=0
    )
    start = scheduler.get_last_lr()[0]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(start)


# ── validation routing ───────────────────────────────────────────────────────
def make_args(validation):
    return types.SimpleNamespace(
        validation=validation,
        device=torch.device("cpu"),
        min_disparity=0.05,
        batches_per_epoch=100,
        log_interval=10,
    )


def validator_for(validation, data_cfg=None):
    pytest.importorskip("wandb", reason="the training entry point imports wandb")
    from applications.f3_depth_training.train_depth_nonrec import make_validator

    return make_validator(
        make_args(validation),
        logging.getLogger("test"),
        None,
        None,
        ScaleAndShiftInvariantLoss(),
        data_cfg or {},
        (640, 480, 20),
    )


def test_the_simulator_is_the_default_source():
    assert callable(validator_for({}))


def test_an_unknown_validation_source_fails_at_startup():
    with pytest.raises(ValueError, match="not 'simulator' or 'm3ed'"):
        validator_for({"source": "nonsense"})


def test_choosing_m3ed_without_sequences_fails_at_startup():
    with pytest.raises(AssertionError, match="data.m3ed is empty"):
        validator_for({"source": "m3ed"})


# ── layering ─────────────────────────────────────────────────────────────────
def test_uniform_blend_weights_every_checkpoint_equally():
    from applications.f3_depth_training.average_checkpoints import blend

    assert blend([None] * 4, "uniform", 0.6) == [0.25] * 4


def test_ema_blend_favours_the_newest_and_sums_to_one():
    from applications.f3_depth_training.average_checkpoints import blend

    weights = blend([None] * 4, "ema", 0.5)
    assert weights == pytest.approx([1 / 15, 2 / 15, 4 / 15, 8 / 15])
    assert sum(weights) == pytest.approx(1.0)


def test_averaging_recovers_the_mean_and_keeps_integer_buffers():
    from applications.f3_depth_training.average_checkpoints import average_weights

    states = [
        {"w": torch.full((2, 2), 1.0), "steps": torch.tensor([1])},
        {"w": torch.full((2, 2), 3.0), "steps": torch.tensor([2])},
    ]
    out = average_weights(states, [0.5, 0.5])
    assert torch.allclose(out["w"], torch.full((2, 2), 2.0))
    assert out["steps"].tolist() == [2], "integer buffers come from the last state"
    assert out["w"].dtype == torch.float32, "dtype survives the float64 accumulation"


def imported_modules(path: Path) -> set[str]:
    """Every module name the file imports, relative ones by their last segment."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[-1])
    return names


def test_the_data_sources_stay_free_of_training_only_imports():
    """wandb is a training dependency; the recorded path has to run without it."""
    for path in [*(APP / "data").glob("*.py"), APP / "utils" / "__init__.py"]:
        reached = imported_modules(path)
        assert "wandb" not in reached, f"{path.name} imports wandb"
        assert "experiment" not in reached, f"{path.name} imports run scaffolding"
