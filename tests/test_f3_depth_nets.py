"""Contracts for the in-house F3 + DepthAnythingV2 model code (no f3 dependency)."""

import re
from pathlib import Path

import pytest
import torch
import yaml

from applications.f3_depth_training.nets import (
    EventFFDepthAnythingV2,
    batch_cropper,
    get_resize_shapes,
    load_depth_weights,
    load_f3_weights,
)
from applications.f3_depth_training.utils import ScaleAndShiftInvariantLoss

APP = Path(__file__).resolve().parents[1] / "applications" / "f3_depth_training"

W, H, CHANNELS = 64, 48, 16
DAV2_SIZE = 70

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
