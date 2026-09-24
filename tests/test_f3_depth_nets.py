"""Contracts for the in-house F3 + DepthAnythingV2 model code (no f3 dependency)."""

import math
import re
from pathlib import Path

import pytest
import torch
import yaml

from applications.f3_depth_training.nets import (
    EventFFDepthAnythingV2,
    LatentMemory,
    RecurrentEventFFDepthAnythingV2,
    batch_cropper,
    get_resize_shapes,
    load_depth_weights,
    load_f3_weights,
    reset_state,
    warm_start,
)
from applications.f3_depth_training.utils import (
    MetricDepth,
    RelativeDepth,
    ScaleAndShiftInvariantLoss,
    SiLogLoss,
    align_least_squares,
    depth_metrics,
)

APP = Path(__file__).resolve().parents[1] / "applications" / "f3_depth_training"

W, H, CHANNELS = 64, 48, 16
DAV2_SIZE = 70
RELATIVE = RelativeDepth(0.05, 1000.0, ScaleAndShiftInvariantLoss())
METRIC = MetricDepth(0.2, 20.0, 26.0, SiLogLoss())
SIGMOID = {"size": DAV2_SIZE, "encoder": "vits", "head": "sigmoid", "max_depth": 20.0}

# The real backbone at 1/10th the width: same stage structure, both hash levels (one
# direct, one colliding), so shapes and index ranges are exercised at test speed.
F3_CONFIG = {
    "frame_sizes": [W, H, 20],
    "dims": [8, 12, 16],
    "convkernels": [3, 3, 3],
    "convdepths": [1, 1, 1],
    "convbtlncks": [2, 2, 2],
    "convdilations": [1, 1, 1],
    "dskernels": [5, 3, 3],
    "dsstrides": [4, 2, 2],
    "patch_size": 1,
    "use_upsampling": True,
    "upsampling_dims": CHANNELS,
    "multi_hash_encoder": {
        "coarsest_resolution": [4, 4, 1],
        "finest_resolution": [32, 18, 8],
        "levels": 2,
        "feature_size": 2,
        "log2_entries_per_level": 12,
    },
}


def make_events(counts, seed=0):
    """Concatenated [x, y, t, p] events on the pixel grid, plus the counts vector."""
    generator = torch.Generator().manual_seed(seed)
    n = sum(counts)
    return (
        torch.stack(
            [
                torch.randint(0, W, (n,), generator=generator) / W,
                torch.randint(0, H, (n,), generator=generator) / H,
                torch.rand(n, generator=generator),
                torch.randint(0, 2, (n,), generator=generator).float(),
            ],
            dim=1,
        ),
        torch.tensor(counts, dtype=torch.int32),
    )


@pytest.fixture(scope="module")
def f3_config(tmp_path_factory):
    path = tmp_path_factory.mktemp("nets") / "f3_tiny.yml"
    path.write_text(yaml.safe_dump(F3_CONFIG))
    return str(path)


@pytest.fixture(scope="module")
def model(f3_config):
    torch.manual_seed(0)
    return EventFFDepthAnythingV2(f3_config, {"size": DAV2_SIZE, "encoder": "vits"})


@pytest.fixture(scope="module")
def events():
    return make_events([80, 120])


@pytest.fixture(scope="module")
def recurrent(f3_config):
    torch.manual_seed(0)
    return RecurrentEventFFDepthAnythingV2(
        f3_config, {"size": DAV2_SIZE, "encoder": "vits"}
    )


# ── shape and dtype contracts ────────────────────────────────────────────────
def test_feature_field_is_x_major_over_the_sensor(model, events):
    ff_events, counts = events
    field = model.eventff.feature_field(ff_events[:, :3], counts)
    levels = model.eventff.multi_hash_encoder
    assert field.shape == (2, levels.levels * levels.feature_size, W, H)
    assert field.dtype == torch.float32


def test_f3_upsamples_back_to_full_resolution(model, events):
    ff_events, counts = events
    assert model.eventff(ff_events[:, :3], counts).shape == (2, CHANNELS, W, H)


def test_field_is_row_major_for_the_decoder(model, events):
    ff_events, counts = events
    assert model.field(ff_events, counts).shape == (2, CHANNELS, H, W)


def test_a_tick_with_no_events_yields_a_zero_field(model):
    ff_events, counts = make_events([0, 60])
    field = model.eventff.feature_field(ff_events[:, :3], counts)
    assert torch.count_nonzero(field[0]) == 0
    assert torch.count_nonzero(field[1]) > 0


def test_forward_predicts_disparity_over_the_crop(model, events):
    ff_events, counts = events
    cparams = torch.tensor([[0, 0, H, H], [0, W - H, H, W]])
    pred, field = model(ff_events, counts, cparams)
    assert pred.shape == (2, H, H)
    assert field.shape == (2, CHANNELS, H, H)


def test_infer_image_covers_the_whole_frame(model, events):
    ff_events, counts = make_events([90])
    cparams = torch.tensor([0, 0, H, W])
    pred, _ = model.infer_image(ff_events, counts, cparams)
    assert pred.shape == (H, W)


# ── invariants ───────────────────────────────────────────────────────────────
def test_get_resize_shapes_keeps_aspect_and_lands_on_a_patch_multiple():
    fh, fw = get_resize_shapes(480, 640, 308, 14)
    assert (fh, fw) == (308, 420)
    assert fw % 14 == 0
    assert 0 <= fw - 640 / 480 * fh < 14, "rounded up by less than one patch"


def test_get_resize_shapes_leaves_a_square_square():
    assert get_resize_shapes(480, 480, 308, 14) == (308, 308)


def test_batch_cropper_matches_naive_indexing():
    images = torch.randn(3, 2, 16, 20)
    cparams = torch.tensor([[0, 0, 8, 8], [4, 6, 12, 14], [8, 12, 16, 20]])
    naive = torch.stack(
        [images[i, :, y0:y1, x0:x1] for i, (y0, x0, y1, x1) in enumerate(cparams)]
    )
    assert torch.equal(batch_cropper(images, cparams), naive)


def test_ssi_loss_is_invariant_to_an_affine_prediction():
    torch.manual_seed(0)
    pred, target = torch.rand(2, 12, 12), torch.rand(2, 12, 12)
    mask = torch.ones_like(target, dtype=torch.bool)
    loss_fn = ScaleAndShiftInvariantLoss()
    base = loss_fn(pred, target, mask)
    scaled = loss_fn(3.7 * pred - 12.5, target, mask)
    assert torch.allclose(base, scaled, atol=1e-5), (
        f"scale and shift changed the loss: {base.item()} vs {scaled.item()}"
    )


# Never reached by a forward, both kept so the released DAv2 checkpoint loads strict:
# nothing masks tokens here, and the deepest fusion block has no finer input to merge.
UNUSED_DAV2 = {
    "pretrained.mask_token",
    "depth_head.scratch.refinenet4.resConfUnit1.conv1.weight",
    "depth_head.scratch.refinenet4.resConfUnit1.conv1.bias",
    "depth_head.scratch.refinenet4.resConfUnit1.conv2.weight",
    "depth_head.scratch.refinenet4.resConfUnit1.conv2.bias",
}


# ── gradient flow ────────────────────────────────────────────────────────────
def test_a_frozen_eventff_takes_no_gradient(model, events):
    ff_events, counts = events
    cparams = torch.tensor([[0, 0, H, H], [0, W - H, H, W]])
    model.zero_grad(set_to_none=True)
    model(ff_events, counts, cparams)[0].square().mean().backward()

    assert all(p.grad is None for p in model.eventff.parameters())
    starved = {n for n, p in model.dav2.named_parameters() if p.grad is None}
    assert starved == UNUSED_DAV2, f"decoder parameters left untrained: {starved}"
    assert any(
        p.grad.abs().sum() > 0 for p in model.dav2.parameters() if p.grad is not None
    )


def test_retrain_unfreezes_the_backbone(f3_config):
    frozen = EventFFDepthAnythingV2(f3_config, {"size": DAV2_SIZE, "encoder": "vits"})
    assert not any(p.requires_grad for p in frozen.eventff.parameters())
    retrained = EventFFDepthAnythingV2(
        f3_config, {"size": DAV2_SIZE, "encoder": "vits"}, retrain=True
    )
    assert all(p.requires_grad for p in retrained.eventff.parameters())


# ── checkpoint compatibility ─────────────────────────────────────────────────
def test_load_accepts_compiled_keys_and_f3s_dropped_head(model):
    state = {
        k.replace("eventff.", "eventff._orig_mod."): v
        for k, v in model.state_dict().items()
    }
    state["eventff._orig_mod.pred.weight"] = torch.zeros(1)
    load_depth_weights(model, state)


def test_load_rejects_a_key_matching_no_module(model):
    state = dict(model.state_dict())
    state["eventff.mystery"] = torch.zeros(1)
    with pytest.raises(AssertionError, match="match no module"):
        load_depth_weights(model, state)


def test_load_rejects_a_checkpoint_missing_weights(model):
    state = {k: v for k, v in model.state_dict().items() if "depth_head" not in k}
    with pytest.raises(AssertionError, match="no weights for"):
        load_depth_weights(model, state)


# ── import hygiene ───────────────────────────────────────────────────────────
def app_sources(*suffixes):
    return [
        p
        for p in APP.rglob("*")
        if p.suffix in suffixes and "__pycache__" not in p.parts
    ]


def test_the_model_imports_nothing_from_the_training_scripts():
    for path in [
        p for p in app_sources(".py") if p.parent.name in ("nets", "dav2", "utils")
    ]:
        text = path.read_text()
        assert "train_depth" not in text and "replay_depth" not in text, path


def test_the_application_no_longer_depends_on_f3():
    imports = re.compile(r"^\s*(from|import)\s+f3\b", re.M)
    offenders = [
        p
        for p in app_sources(".py", ".yml", ".yaml")
        if imports.search(p.read_text()) or "deps/fast-feature-fields" in p.read_text()
    ]
    assert not offenders, f"still reaching into f3: {offenders}"


# ── LatentMemory (ConvGRU on the stride-16 latent) ───────────────────────────
PASSTHROUGH = torch.sigmoid(torch.tensor(6.0))  # the update gate's zero-init bias

MEMORY_CHANNELS = 8


def latent(batch=2, channels=MEMORY_CHANNELS, w=4, h=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, channels, w, h, generator=generator)


def test_the_first_tick_passes_the_latent_through():
    memory = LatentMemory(MEMORY_CHANNELS)
    x = latent()
    held = memory(x, torch.zeros_like(x))
    assert torch.allclose(held, PASSTHROUGH * x, atol=1e-6), (
        "an untrained memory must start as the non-recurrent model, up to sigmoid(6)"
    )


def test_the_carried_state_changes_the_output():
    memory = LatentMemory(MEMORY_CHANNELS)
    x = latent()
    empty = memory(x, torch.zeros_like(x))
    carried = memory(x, latent(seed=7))
    assert not torch.allclose(empty, carried), "the state did not reach the output"


def test_both_the_gate_and_the_candidate_train_on_the_first_step():
    memory = LatentMemory(MEMORY_CHANNELS)
    x = latent()
    memory(x, torch.zeros_like(x)).square().mean().backward()
    assert memory.gates.weight.grad.abs().sum() > 0
    assert memory.candidate.weight.grad.abs().sum() > 0


def test_a_later_tick_backpropagates_into_an_earlier_one():
    memory = LatentMemory(MEMORY_CHANNELS)
    first = latent()
    first.requires_grad_(True)
    state = memory(first, torch.zeros_like(first))
    memory(latent(seed=2), state).square().mean().backward()
    assert first.grad.abs().sum() > 0, "memory carried no gradient across ticks"


def test_detaching_the_state_truncates_the_gradient():
    memory = LatentMemory(MEMORY_CHANNELS)
    first = latent()
    first.requires_grad_(True)
    state = memory(first, torch.zeros_like(first))
    memory(latent(seed=2), state.detach()).square().mean().backward()
    assert first.grad is None


def test_reset_zeroes_only_the_rows_that_restart():
    state = latent()
    reset = reset_state(state, torch.tensor([True, False]))
    assert torch.count_nonzero(reset[0]) == 0
    assert torch.equal(reset[1], state[1])


def test_the_state_survives_autocast_in_float32():
    memory = LatentMemory(MEMORY_CHANNELS)
    x, state = latent(), latent(seed=3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        held = memory(x, state)
    assert held.dtype is torch.float32, (
        "a state carried over thousands of ticks must not be held at bf16 precision"
    )


# ── RecurrentEventFFDepthAnythingV2 ──────────────────────────────────────────
def test_contract_then_expand_still_equals_encode(model, events):
    """The split that makes room for the memory must not change F3's output."""
    ff_events, counts = events
    field = model.eventff.feature_field(ff_events[:, :3], counts)
    bottleneck, skips = model.eventff.contract(field)
    assert torch.equal(
        model.eventff.expand(bottleneck, skips, field), model.eventff.encode(field)
    )


def test_a_step_returns_disparity_the_field_and_the_next_state(recurrent, events):
    ff_events, counts = events
    cparams = torch.tensor([[0, 0, H, H], [0, W - H, H, W]])
    state = recurrent.initial_state(2, device="cpu")
    pred, field, state = recurrent(ff_events, counts, cparams, state)
    f3 = recurrent.eventff
    assert pred.shape == (2, H, H)
    assert field.shape == (2, CHANNELS, H, H)
    assert state.shape == (
        2,
        f3.latent_channels,
        W // f3.latent_stride,
        H // f3.latent_stride,
    )


def test_a_non_recurrent_checkpoint_warm_starts_it(recurrent, f3_config):
    plain = EventFFDepthAnythingV2(f3_config, {"size": DAV2_SIZE, "encoder": "vits"})
    state = {k: v for k, v in recurrent.state_dict().items() if "memory." not in k}
    with pytest.raises(AssertionError, match="no weights for"):
        load_depth_weights(recurrent, state)
    load_depth_weights(plain, state)
    load_depth_weights(recurrent, state, fresh=("memory.",))


def test_the_first_tick_predicts_what_the_non_recurrent_model_would(recurrent, events):
    plain = EventFFDepthAnythingV2(
        recurrent.eventff_config, {"size": DAV2_SIZE, "encoder": "vits"}
    )
    load_depth_weights(
        plain, {k: v for k, v in recurrent.state_dict().items() if "memory." not in k}
    )
    ff_events, counts = events
    cparams = torch.tensor([[0, 0, H, H], [0, W - H, H, W]])
    with torch.no_grad():
        want = plain(ff_events, counts, cparams)[0]
        got = recurrent(ff_events, counts, cparams, recurrent.initial_state(2, "cpu"))[
            0
        ]
    error = (got - want).abs().max() / want.abs().max()
    assert error < 5e-3, f"warm start is not near the non-recurrent model: {error:.5f}"


def test_the_carried_state_changes_a_later_prediction(recurrent, events):
    ff_events, counts = events
    cparams = torch.tensor([[0, 0, H, H], [0, W - H, H, W]])
    with torch.no_grad():
        empty = recurrent.initial_state(2, device="cpu")
        _, _, carried = recurrent(ff_events, counts, cparams, empty)
        fresh_pred = recurrent(ff_events, counts, cparams, empty)[0]
        carried_pred = recurrent(ff_events, counts, cparams, carried)[0]
    assert not torch.allclose(fresh_pred, carried_pred), "the memory changed nothing"


def test_gradients_reach_the_memory_through_a_frozen_backbone(recurrent, events):
    ff_events, counts = events
    cparams = torch.tensor([[0, 0, H, H], [0, W - H, H, W]])
    recurrent.zero_grad(set_to_none=True)
    state = recurrent.initial_state(2, device="cpu")
    loss = 0.0
    for _ in range(2):
        pred, _, state = recurrent(ff_events, counts, cparams, state)
        loss = loss + pred.square().mean()
    loss.backward()

    assert all(p.grad is None for p in recurrent.eventff.parameters())
    assert all(p.grad.abs().sum() > 0 for p in recurrent.memory.parameters())


def test_load_f3_weights_takes_a_bare_state_dict_with_the_head(model, tmp_path):
    """f3's backbone weights are a plain state dict that still carries `pred.*`."""
    state = dict(model.eventff.state_dict())
    state["pred.weight"] = torch.zeros(1)
    path = tmp_path / "f3.pth"
    torch.save(state, path)
    load_f3_weights(model.eventff, path)


def test_load_f3_weights_rejects_compiled_keys(model, tmp_path):
    """A checkpoint saved from a compiled module prefixes every key, and must not load."""
    path = tmp_path / "f3_compiled.pth"
    torch.save(
        {f"_orig_mod.{k}": v for k, v in model.eventff.state_dict().items()}, path
    )
    with pytest.raises(AssertionError, match="no weights for"):
        load_f3_weights(model.eventff, path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_moving_the_model_carries_the_hash_encoder_buffers(f3_config):
    """`nets/` pins no device, so `.to()` has to carry `resolutions` and `is_ceil` too."""
    model = EventFFDepthAnythingV2(
        f3_config, {"size": DAV2_SIZE, "encoder": "vits"}
    ).cuda()
    encoder = model.eventff.multi_hash_encoder
    assert encoder.resolutions.is_cuda and encoder.is_ceil.is_cuda

    ff_events, counts = make_events([40])
    cparams = torch.tensor([[0, 0, H, H]], device="cuda")
    pred, _ = model(ff_events.cuda(), counts.cuda(), cparams)
    assert pred.is_cuda


# ── exponential head (DA3-style log disparity) ───────────────────────────────
def test_the_exp_head_emits_positive_finite_disparity(f3_config, events):
    model = EventFFDepthAnythingV2(
        f3_config, {"size": DAV2_SIZE, "encoder": "vits", "head": "exp"}
    )
    ff_events, counts = events
    cparams = torch.tensor([[0, 0, H, H], [0, W - H, H, W]])
    pred = model(ff_events, counts, cparams)[0]
    assert torch.isfinite(pred).all(), "exp overflowed at initialisation"
    assert (pred > 0).all(), "disparity must be positive"


def test_the_exp_head_starts_near_one(f3_config, events):
    """z ~ 0 at init, so the first predictions sit around exp(0) and cannot explode."""
    model = EventFFDepthAnythingV2(
        f3_config, {"size": DAV2_SIZE, "encoder": "vits", "head": "exp"}
    )
    ff_events, counts = events
    cparams = torch.tensor([[0, 0, H, H], [0, W - H, H, W]])
    pred = model(ff_events, counts, cparams)[0]
    assert 0.1 < pred.median() < 10, (
        f"median disparity {pred.median():.3g} is off scale"
    )


def test_all_heads_share_a_state_dict(f3_config):
    """Only the emit conv differs, and it is index 2 either way, so weights transfer."""
    relu = EventFFDepthAnythingV2(f3_config, {"size": DAV2_SIZE, "encoder": "vits"})
    exp = EventFFDepthAnythingV2(
        f3_config, {"size": DAV2_SIZE, "encoder": "vits", "head": "exp"}
    )
    sigmoid = EventFFDepthAnythingV2(f3_config, SIGMOID)
    assert set(relu.state_dict()) == set(exp.state_dict()) == set(sigmoid.state_dict())


# ── sigmoid head (DAv2-metric-style depth in metres) ─────────────────────────
def test_the_sigmoid_head_emits_depth_within_max_depth(f3_config, events):
    model = EventFFDepthAnythingV2(f3_config, SIGMOID)
    ff_events, counts = events
    cparams = torch.tensor([[0, 0, H, H], [0, W - H, H, W]])
    pred = model(ff_events, counts, cparams)[0]
    assert (pred > 0).all() and (pred < 20.0).all()
    assert 5 < pred.median() < 15, "a reset emit conv starts mid-range"


def test_warm_start_resets_only_the_emit_conv_across_the_metric_boundary(
    f3_config, tmp_path
):
    exp = EventFFDepthAnythingV2(
        f3_config, {"size": DAV2_SIZE, "encoder": "vits", "head": "exp"}
    )
    exp.save_configs(str(tmp_path))
    torch.save({"model": exp.state_dict()}, tmp_path / "best.pth")

    metric = EventFFDepthAnythingV2(f3_config, SIGMOID)
    assert warm_start(metric, tmp_path / "best.pth")
    emit_weight = "dav2.depth_head.scratch.output_conv2.2.weight"
    for k, v in exp.state_dict().items():
        copied = torch.equal(metric.state_dict()[k], v)
        assert copied == (k != emit_weight), k

    same_head = EventFFDepthAnythingV2(
        f3_config, {"size": DAV2_SIZE, "encoder": "vits", "head": "exp"}
    )
    assert not warm_start(same_head, tmp_path / "best.pth")
    assert torch.equal(
        same_head.state_dict()[emit_weight], exp.state_dict()[emit_weight]
    )


# ── SiLog loss ───────────────────────────────────────────────────────────────
def test_silog_matches_the_f3_reference_formula():
    torch.manual_seed(0)
    pred, target = torch.rand(2, 12, 12) * 9 + 1, torch.rand(2, 12, 12) * 9 + 1
    mask = torch.rand(2, 12, 12) > 0.3
    diff = torch.log(target[mask]) - torch.log(pred[mask] + 1e-6)
    want = torch.sqrt((diff**2).mean() - 0.5 * diff.mean() ** 2)
    assert SiLogLoss(lambd=0.5)(pred, target, mask) == pytest.approx(want.item())


def test_silog_penalises_the_global_scale_the_ssi_loss_forgives():
    """Why the metric mode exists: a 2x depth error is free under SSI."""
    torch.manual_seed(0)
    depth = torch.rand(2, 12, 12) * 9 + 1
    mask = torch.ones_like(depth, dtype=torch.bool)
    silog, ssi = SiLogLoss(lambd=0.5), ScaleAndShiftInvariantLoss()
    assert silog(depth, depth, mask) == pytest.approx(0.0, abs=1e-3)
    assert silog(2 * depth, depth, mask) == pytest.approx(math.log(2) / math.sqrt(2))
    assert ssi(2 / depth, 1 / depth, mask) == pytest.approx(0.0, abs=1e-5)


def test_silog_ignores_invalid_target_pixels():
    torch.manual_seed(0)
    depth = torch.rand(2, 12, 12) * 9 + 1
    mask = torch.rand(2, 12, 12) > 0.3
    target = depth.masked_fill(~mask, 0.0)  # the loader's convention
    loss = SiLogLoss(lambd=0.5, alpha=0.5)(depth, target, mask)
    assert torch.isfinite(loss) and loss == pytest.approx(0.0, abs=1e-3)


def test_the_relu_head_is_still_the_default(f3_config):
    model = EventFFDepthAnythingV2(f3_config, {"size": DAV2_SIZE, "encoder": "vits"})
    assert model.dav2.head == "relu"


@pytest.fixture
def disparity():
    """Two images of disparity in [0.1, 1.55], i.e. depth 0.65 m to 10 m, all valid."""
    torch.manual_seed(0)
    return torch.rand(2, 8, 8) * 1.45 + 0.1, torch.ones(2, 8, 8, dtype=torch.bool)


def test_alignment_recovers_a_known_affine_map(disparity):
    target, mask = disparity
    aligned = align_least_squares((target - 0.3) / 7.0, target, mask)
    assert torch.allclose(aligned, target, atol=1e-5), "the fit did not invert the map"


def test_alignment_fits_each_image_separately(disparity):
    target, mask = disparity
    pred = torch.stack([target[0] * 3 + 1.0, target[1] * 1e5 - 2e5])
    aligned = align_least_squares(pred, target, mask)
    assert torch.allclose(aligned, target, atol=1e-5), (
        "one gauge was fitted for the batch"
    )


def test_a_constant_prediction_gets_the_best_constant_fit(disparity):
    target, mask = disparity
    aligned = align_least_squares(torch.full_like(target, 0.7), target, mask)
    expected = target.mean((1, 2), keepdim=True).expand_as(target)
    assert torch.allclose(aligned, expected, atol=1e-5)


def test_relative_depth_metrics_ignore_the_gauge(disparity):
    """The point of the protocol: an affine map of the prediction changes nothing."""
    target, mask = disparity
    pred = torch.rand_like(target) * 1.45 + 0.1
    plain = RELATIVE.metrics(pred, target, mask)
    walked = RELATIVE.metrics(pred * 1e14 + 5.0, target, mask)
    for k, value in plain.items():
        assert walked[k] == pytest.approx(value, rel=1e-4), f"{k} moved with the gauge"


def test_a_perfect_prediction_scores_perfectly(disparity):
    target, mask = disparity
    results = RELATIVE.metrics(target, target, mask)
    assert results["abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert results["d1"] == pytest.approx(100.0)
    assert results["silog"] == pytest.approx(0.0, abs=1e-3)


def test_metric_names_match_what_each_mode_returns(disparity):
    """`metric_names` seeds the trainer's best-results table; a missing key raises mid-run."""
    target, mask = disparity
    assert set(RELATIVE.metric_names) == set(RELATIVE.metrics(target, target, mask))
    depth = 1.0 / target
    assert set(METRIC.metric_names) == set(METRIC.metrics(depth, depth, mask))


def test_masked_pixels_do_not_reach_the_metrics(disparity):
    target, mask = disparity
    mask = mask.clone()
    mask[:, :, 4:] = False
    corrupted = target.clone()
    corrupted[:, :, 4:] = 1e6
    assert RELATIVE.metrics(corrupted, target, mask) == pytest.approx(
        RELATIVE.metrics(target, target, mask)
    )


def test_metric_metrics_catch_the_scale_error_the_aligned_ones_forgive(disparity):
    target, mask = disparity
    depth = 1.0 / target
    scores = METRIC.metrics(2 * depth, depth, mask)
    assert scores["abs_rel"] == pytest.approx(1.0)
    assert scores["d1"] == pytest.approx(0.0)
    assert scores["abs_rel_aligned"] == pytest.approx(0.0, abs=1e-4)
    assert scores["d1_aligned"] == pytest.approx(100.0)


def test_near_field_metrics_count_only_truth_under_five_metres():
    truth = torch.tensor([[[1.0, 2.0, 8.0, 12.0]]])
    pred = truth.clone()
    pred[..., 2:] *= 2  # only the far pixels are wrong
    mask = torch.ones_like(truth, dtype=torch.bool)
    scores = depth_metrics(pred, truth, mask)
    assert scores["abs_rel"] == pytest.approx(0.5)
    assert scores["abs_rel_near"] == 0.0 and scores["d1_near"] == 100.0


def test_near_field_metrics_are_nan_without_a_near_pixel():
    truth = torch.full((1, 2, 2), 8.0)
    scores = depth_metrics(truth, truth, torch.ones_like(truth, dtype=torch.bool))
    assert scores["d1"] == 100.0
    assert math.isnan(scores["d1_near"]), "undefined, not zero, so the mean skips it"


# --------------------------------------------------------------------------- #
# Bilinear scatter in the feature field
# --------------------------------------------------------------------------- #
def scatter_field(model, coords, bilinear=True):
    """feature_field for one sample of events at `coords` (normalized x, y)."""
    events = torch.zeros(len(coords), 3)
    events[:, 0] = torch.tensor([c[0] for c in coords])
    events[:, 1] = torch.tensor([c[1] for c in coords])
    counts = torch.tensor([len(coords)], dtype=torch.int32)
    was, model.eventff.bilinear_splat = model.eventff.bilinear_splat, bilinear
    try:
        return model.eventff.feature_field(events, counts)
    finally:
        model.eventff.bilinear_splat = was


def test_sensor_grid_events_land_in_one_cell(model):
    """Integer pixels are the case every existing dataset is in: nothing may smear."""
    cells = ((0, 0), (17, 9), (W - 1, H - 1))
    field = scatter_field(model, [(px / W, py / H) for px, py in cells])

    energy = field[0].abs().sum(0)  # [W, H]
    occupied = (energy > 1e-6).nonzero()
    assert len(occupied) == len(cells), f"one cell per event, got {len(occupied)}"
    assert {tuple(c.tolist()) for c in occupied} == set(cells)


def test_half_pixel_event_splits_between_neighbours(model):
    """A coordinate exactly between two cells must share itself evenly."""
    field = scatter_field(model, [((10 + 0.5) / W, 20 / H)])

    energy = field[0].abs().sum(0)
    assert energy[10, 20] > 0 and energy[11, 20] > 0
    assert torch.allclose(energy[10, 20], energy[11, 20], rtol=1e-3)


def test_scatter_conserves_event_weight(model):
    """Splatting redistributes an event, it must not create or destroy any of it."""
    coords = [(0.5, 0.5), (0.2379, 0.7512), (0.9993, 0.0007), (0.0, 1.0)]
    field = scatter_field(model, coords)

    events = torch.zeros(len(coords), 3)
    events[:, 0] = torch.tensor([c[0] for c in coords])
    events[:, 1] = torch.tensor([c[1] for c in coords])
    encoded = model.eventff.multi_hash_encoder(events.unsqueeze(0)).squeeze(0)
    assert torch.allclose(field.sum(), encoded.sum(), rtol=1e-4, atol=1e-6)


def test_fractional_coordinates_leave_no_unreachable_cells(model):
    """The point of the change: a resampled stream must still be able to fill the grid."""
    # A stretch like undistortion: source pixels spread apart, so rounding alone would
    # leave cells between them that no event can ever reach.
    stretched = [
        ((px * 1.7 + 0.35) / W, (py * 1.7 + 0.35) / H)
        for px in range(12)
        for py in range(12)
    ]
    inside = [(x, y) for x, y in stretched if x < 1 and y < 1]
    field = scatter_field(model, inside)

    reached = (field[0].abs().sum(0) > 1e-6).sum().item()
    assert reached > len(inside), (
        f"{len(inside)} events reached only {reached} cells; the splat is not spreading"
    )


def test_the_splat_is_off_by_default(model):
    """Training never sees fractional coordinates, so it must not pay for the splat."""
    assert model.eventff.bilinear_splat is False


def test_rounding_and_splatting_agree_on_the_sensor_grid(model):
    """The two paths must be interchangeable for every stream we train on."""
    coords = [
        (px / W, py / H) for px, py in ((0, 0), (17, 9), (31, 22), (W - 1, H - 1))
    ]
    rounded = scatter_field(model, coords, bilinear=False)
    splatted = scatter_field(model, coords, bilinear=True)
    assert torch.allclose(rounded, splatted, atol=1e-5)


def test_warm_start_treats_a_checkpoint_without_a_config_as_relative(
    f3_config, tmp_path
):
    """f3-era checkpoints have no depth_config.yml beside them and predate metric heads."""
    relu = EventFFDepthAnythingV2(f3_config, {"size": DAV2_SIZE, "encoder": "vits"})
    torch.save({"model": relu.state_dict()}, tmp_path / "f3.pth")
    emit_weight = "dav2.depth_head.scratch.output_conv2.2.weight"

    exp = EventFFDepthAnythingV2(
        f3_config, {"size": DAV2_SIZE, "encoder": "vits", "head": "exp"}
    )
    assert not warm_start(exp, tmp_path / "f3.pth"), "relative to relative loads as is"
    assert torch.equal(exp.state_dict()[emit_weight], relu.state_dict()[emit_weight])
    assert warm_start(EventFFDepthAnythingV2(f3_config, SIGMOID), tmp_path / "f3.pth")


def test_metric_scores_a_constant_overprediction_by_its_ratio(disparity):
    """abs_rel and d1 are ratios, so a uniform 1.2x reads the same at any depth."""
    target, mask = disparity
    truth = 1.0 / target
    near = METRIC.metrics(truth * 1.2, truth, mask)
    far = METRIC.metrics(2.0 * truth * 1.2, 2.0 * truth, mask)
    assert near["d1"] == pytest.approx(far["d1"])
    assert near["abs_rel"] == pytest.approx(far["abs_rel"])
    assert far["rmse"] > near["rmse"], "rmse is in metres, so it doubles with the scene"


# ── age-only hash encoder (`temporal_hash`) ──────────────────────────────────
TEMPORAL_CONFIG = {**F3_CONFIG, "temporal_hash": True}
SIGMOID_FREE = {"size": DAV2_SIZE, "encoder": "vits"}


def f3_config_path(tmp_path):
    """The baseline 3-D hash config, written where a test can point at it."""
    path = tmp_path / "f3_3d.yml"
    path.write_text(yaml.safe_dump(F3_CONFIG))
    return str(path)


@pytest.fixture(scope="module")
def temporal_config(tmp_path_factory):
    path = tmp_path_factory.mktemp("nets") / "f3_tiny_temporal.yml"
    path.write_text(yaml.safe_dump(TEMPORAL_CONFIG))
    return str(path)


@pytest.fixture(scope="module")
def temporal(temporal_config):
    torch.manual_seed(0)
    return EventFFDepthAnythingV2(
        temporal_config, {"size": DAV2_SIZE, "encoder": "vits"}
    )


def test_temporal_hash_emits_the_same_channels_as_the_3d_one(temporal, model):
    """The swap is only a drop-in if `downsample_layers[0]` sees the same width."""
    swapped, original = temporal.eventff, model.eventff
    assert swapped.feature_size == original.feature_size
    encoders = (swapped.multi_hash_encoder, original.multi_hash_encoder)
    assert len({e.levels * e.feature_size for e in encoders}) == 1


def test_temporal_hash_still_upsamples_back_to_full_resolution(temporal, events):
    ff_events, counts = events
    assert temporal.eventff(ff_events[:, :3], counts).shape == (2, CHANNELS, W, H)


def test_temporal_hash_reads_age_and_ignores_position(temporal):
    """The point of the encoder: two events of equal age encode identically anywhere."""
    encoder = temporal.eventff.multi_hash_encoder
    same_age = torch.tensor([[0.1, 0.2, 7 / 20], [0.9, 0.8, 7 / 20]])
    encoded = encoder(same_age.unsqueeze(0)).squeeze(0)
    assert torch.equal(encoded[0], encoded[1])


def test_temporal_hash_distinguishes_ages(temporal):
    encoder = temporal.eventff.multi_hash_encoder
    ages = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 9 / 20]])
    encoded = encoder(ages.unsqueeze(0)).squeeze(0)
    assert not torch.allclose(encoded[0], encoded[1])


def test_the_age_table_is_one_row_per_bucket(temporal):
    """`forward` must index a prebuilt [buckets, L*F] table, not interpolate per event."""
    encoder = temporal.eventff.multi_hash_encoder
    buckets = TEMPORAL_CONFIG["frame_sizes"][2]
    table = encoder.interpolate(encoder.ages)
    assert encoder.buckets == buckets
    assert table.shape == (buckets, encoder.levels * encoder.feature_size)


def test_every_bucket_reads_the_row_the_table_holds(temporal):
    """Quantizing in `forward` has to land on the same row `interpolate` built."""
    encoder = temporal.eventff.multi_hash_encoder
    table = encoder.interpolate(encoder.ages)
    ages = encoder.ages.unsqueeze(-1).expand(-1, 3).clone()
    assert torch.equal(encoder(ages.unsqueeze(0)).squeeze(0), table)


def test_age_past_the_last_bucket_clamps_rather_than_reading_off_the_table(temporal):
    encoder = temporal.eventff.multi_hash_encoder
    beyond = torch.tensor([[0.5, 0.5, 1.0]])
    table = encoder.interpolate(encoder.ages)
    assert torch.equal(encoder(beyond.unsqueeze(0)).squeeze(0)[0], table[-1])


def test_the_age_table_trains(temporal_config, events):
    """`--retrain-f3` is the only way to use this, so gradients must reach the table."""
    torch.manual_seed(0)
    model = EventFFDepthAnythingV2(
        temporal_config, {"size": DAV2_SIZE, "encoder": "vits"}, retrain=True
    )
    ff_events, counts = events
    model.eventff(ff_events[:, :3], counts).sum().backward()
    table = model.eventff.multi_hash_encoder.table
    assert table.grad is not None and table.grad.abs().sum() > 0


def test_the_temporal_frontend_exports(temporal, events):
    """Deployment is AOTI or TRT off `torch.export`, with the event count dynamic."""
    ff_events, counts = events
    exported = torch.export.export(
        temporal.eventff,
        (ff_events[:, :3], counts),
        dynamic_shapes={
            "events": {0: torch.export.Dim("n_events", min=2)},
            "counts": None,
        },
    )
    fewer = make_events([30, 40], seed=1)
    assert torch.allclose(
        exported.module()(fewer[0][:, :3], fewer[1]),
        temporal.eventff(fewer[0][:, :3], fewer[1]),
        atol=1e-5,
    )


def test_a_3d_hash_checkpoint_warm_starts_the_conv_stack(temporal_config, tmp_path):
    """The released weights are the only F3 we have: the convs must still be usable."""
    torch.manual_seed(0)
    released = EventFFDepthAnythingV2(f3_config_path(tmp_path), SIGMOID_FREE)
    state = {k: v for k, v in released.eventff.state_dict().items()}
    path = tmp_path / "f3_3d.pth"
    torch.save(state, path)

    torch.manual_seed(1)
    swapped = EventFFDepthAnythingV2(temporal_config, SIGMOID_FREE)
    before = swapped.eventff.multi_hash_encoder.table.clone()
    swapped.load_eventff_weights(path)

    convs = [k for k in state if not k.startswith("multi_hash_encoder.")]
    assert convs, "nothing outside the encoder to warm-start"
    loaded = swapped.eventff.state_dict()
    assert all(torch.equal(loaded[k], state[k]) for k in convs)
    # The hashmap has nowhere to land, so the age table is still this run's own init.
    assert torch.equal(swapped.eventff.multi_hash_encoder.table, before)


def test_a_3d_hash_checkpoint_is_still_rejected_by_a_3d_model(f3_config, tmp_path):
    """The allowance is for the encoder swap only: a truncated checkpoint must still fail."""
    torch.manual_seed(0)
    model = EventFFDepthAnythingV2(f3_config, SIGMOID_FREE)
    state = model.eventff.state_dict()
    path = tmp_path / "f3_partial.pth"
    torch.save({k: v for k, v in state.items() if "downsample_layers.0" not in k}, path)

    with pytest.raises(AssertionError, match="no weights for"):
        model.load_eventff_weights(path)
