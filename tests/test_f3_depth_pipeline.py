"""Contracts for the depth training pipeline: data sources, optimizer, validation routing."""

import ast
import logging
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from applications.f3_depth_training.data import (
    process_batch,
    usable_sample_filter,
    usable_samples,
)
from applications.f3_depth_training.data.m3ed import DepthFrame, collate_frames
from applications.f3_depth_training.utils import (
    MetricDepth,
    RelativeDepth,
    ScaleAndShiftInvariantLoss,
    build_mode,
    build_optimizer,
    build_scheduler,
    mean_scores,
)

APP = Path(__file__).resolve().parents[1] / "applications" / "f3_depth_training"

RELATIVE = RelativeDepth(0.05, 1000.0, ScaleAndShiftInvariantLoss())
CONF = {
    "loss": "ssimae",
    "alpha": 0.5,
    "scales": 4,
    "lambd": 0.5,
    "min_disparity": 0.05,
    "max_disparity": 1000.0,
    "min_depth": 0.2,
    "max_depth": 20.0,
    "head_max_depth": 26.0,
}
METRIC_CONF = CONF | {"loss": "siloggrad"}


class LoaderBatch(dict):
    """A loader batch: sensor payloads, plus the per-row metadata the loader attaches."""

    def __init__(self, sensors, hfov=90.0):
        super().__init__(sensors)
        rows = len(next(iter(sensors.values()))[0])
        self.meta = types.SimpleNamespace(hfov=np.full(rows, hfov, np.float32))


@pytest.fixture
def batch_args():
    return types.SimpleNamespace(
        event_sensor="ev",
        depth_sensor="dep",
        color_sensor=None,
        event_norm=(640, 480, 20000),
        max_events=0,
    )


# ── the simulator source ─────────────────────────────────────────────────────
def test_process_batch_normalizes_events_to_the_model_frame(batch_args):
    events = np.array(
        [[320.0, 240.0, 10000.0, 1.0], [640.0, 480.0, 20000.0, 0.0]], np.float32
    )
    batch = LoaderBatch(
        {
            "ev": (np.array([2], np.int32), events),
            "dep": np.full((1, 4, 4), 5.0, np.float32),
        }
    )
    ff_events, counts, _, _ = process_batch(batch, batch_args, torch.device("cpu"))

    assert torch.allclose(ff_events[0, :3], torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(ff_events[1, :3], torch.tensor([1.0, 1.0, 1.0]))
    assert ff_events[0, 3] == 1.0, "polarity must not be normalized"
    assert counts.tolist() == [2]


def test_process_batch_keeps_depth_in_metres(batch_args):
    events = np.zeros((1, 4), np.float32)
    depth = np.full((1, 4, 4), 5.0, np.float32)
    depth[0, 0, 0] = 0.0  # an invalid read stays 0 for the mode to mask
    batch = LoaderBatch(
        {"ev": (np.array([1], np.int32), events), "dep": depth}, hfov=90.0
    )
    _, _, out, _ = process_batch(batch, batch_args, torch.device("cpu"))
    assert torch.equal(out, torch.from_numpy(depth)), (
        "the mode converts, not the loader"
    )


def test_the_sample_filter_rejects_mostly_unread_depth():
    keep = usable_sample_filter("dep", "ev", 0.2, min_events=1)
    events = {"x": np.zeros(5)}
    read = types.SimpleNamespace(sensors={"dep": np.full((4, 4), 5.0), "ev": events})
    unread = types.SimpleNamespace(sensors={"dep": np.zeros((4, 4)), "ev": events})
    assert keep(read)
    assert not keep(unread), (
        "0 m reads are the unusable ones; far pixels wait for the batch gate"
    )


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
        DepthFrame(torch.zeros(5, 4), torch.zeros(8, 8)),
        DepthFrame(torch.zeros(3, 4), torch.zeros(8, 8)),
    ]
    events, counts, depth = collate_frames(frames)
    assert events.shape == (8, 4), "windows concatenate, they do not pad"
    assert counts.tolist() == [5, 3]
    assert counts.dtype == torch.int32
    assert depth.shape == (2, 8, 8)


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
        RELATIVE,
        None,
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


def test_a_full_post_warmup_cosine_needs_no_hold():
    """cooldown = epochs - warmup leaves no flat stretch for the metrics to wander in."""
    optimizer = build_optimizer(Named(), 1e-5)
    scheduler = build_scheduler(
        optimizer, epochs=400, warmup_epochs=10, cooldown_epochs=390
    )
    lrs = []
    for _ in range(400):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert lrs[10] == pytest.approx(max(lrs)), "the peak is the end of warmup"
    assert lrs[10] > lrs[100] > lrs[200] > lrs[300] > lrs[399], "monotone decay after"
    assert lrs[399] < 1e-3 * max(lrs), "anneals to ~zero"


# --------------------------------------------------------------------------- #
# The range the loss is allowed to see, in each mode
# --------------------------------------------------------------------------- #
def test_the_relative_target_excludes_both_the_unreadable_and_the_far():
    """The depth clamp is two-sided, so a one-sided mask keeps a whole flat patch."""
    depth = torch.tensor([[0.0, 0.5, 5.0, 20.0, 80.0]])  # 0 m reads are invalid
    disparity, keep = RELATIVE.target(depth)
    assert torch.allclose(disparity[keep], 1.0 / depth[keep]), "disparity is 1/depth"
    assert keep.tolist() == [[False, True, True, False, False]], (
        "0 m must drop as unreadable, and both 20 m and 80 m as beyond the far limit"
    )


def test_the_far_limit_follows_min_disparity():
    """min_disparity is the range knob: 0.05 is 20 m, 0.1 is 10 m."""
    depth = torch.tensor([[5.0, 15.0]])
    for min_disparity, expected in ((0.05, [[True, True]]), (0.1, [[True, False]])):
        mode = RelativeDepth(min_disparity, 1000.0, ScaleAndShiftInvariantLoss())
        assert mode.target(depth)[1].tolist() == expected


def test_the_metric_target_keeps_only_the_supervised_depth_range():
    mode = build_mode(METRIC_CONF, metric=True)
    depth = torch.tensor([[[0.0, 0.1, 0.5, 5.0, 20.0, 80.0]]])
    _, keep = mode.target(depth)
    assert keep.tolist() == [[[False, False, True, True, False, False]]]


def test_the_metric_target_passes_depth_through_unchanged():
    """The head emits metres, so the target is the depth itself over the supervised range."""
    mode = build_mode(METRIC_CONF, metric=True)
    depth = torch.rand(2, 8, 8) * 9 + 1
    target, _ = mode.target(depth)
    assert torch.equal(target, depth)
    assert torch.equal(mode.to_metres(target), depth)


def test_the_head_ceiling_clears_the_supervised_range():
    """A sigmoid cap under max_depth would clip the far field."""
    import yaml

    conf = yaml.safe_load(
        (APP / "configs" / "depth_training_config_voltmeter_metric.yml").read_text()
    )
    mode = build_mode(conf, metric=True)
    assert mode.head_max_depth >= mode.max_depth


def test_the_flag_and_the_loss_have_to_agree():
    assert isinstance(build_mode(CONF, metric=False), RelativeDepth)
    assert isinstance(build_mode(METRIC_CONF, metric=True), MetricDepth)
    with pytest.raises(AssertionError, match="needs --metric"):
        build_mode(METRIC_CONF, metric=False)
    with pytest.raises(AssertionError, match="silog or siloggrad"):
        build_mode(CONF, metric=True)


def test_silog_without_the_gradient_term_is_alpha_zero():
    plain = build_mode(CONF | {"loss": "silog"}, metric=True).loss_fn
    grad = build_mode(METRIC_CONF, metric=True).loss_fn
    assert (plain.alpha, plain.name) == (0.0, "SiLogLoss")
    assert (grad.alpha, grad.name) == (0.5, "SiLogGradLoss")


def test_mean_scores_skips_batches_where_a_metric_was_undefined():
    batches = [{"d1": 50.0, "d1_near": float("nan")}, {"d1": 70.0, "d1_near": 80.0}]
    assert mean_scores(batches) == {"d1": 60.0, "d1_near": 80.0}


def test_usable_samples_honours_the_configured_fraction():
    """The gate is a config knob, not the hardcoded half it used to be."""
    valid = torch.zeros(3, 10, 10, dtype=torch.bool)
    valid[0, :6] = True  # 60% valid
    valid[1, :9] = True  # 90%
    valid[2] = True  # 100%

    assert usable_samples(valid, 0.5).tolist() == [True, True, True]
    assert usable_samples(valid, 0.9).tolist() == [False, True, True]
