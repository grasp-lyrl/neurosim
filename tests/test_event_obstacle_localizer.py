import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "applications" / "rl"))

from train_event_obstacle_localizer import (
    SpatialEventLocalizer,
    gaussian_heatmaps,
    localization_loss,
)
from train_event_obstacle_tracker import (
    TemporalHeatmapTracker,
    inbound_targets,
    temporal_loss,
)


def test_spatial_localizer_shapes_and_finite_loss():
    model = SpatialEventLocalizer(in_channels=8, feature_channels=16)
    events = torch.rand(2, 8, 32, 48)
    labels = torch.tensor(
        [[1.0, 0.25, 0.75, 0.01, 0.4], [0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    output = model(events)
    assert output["heatmap_logits"].shape == (2, 16, 24)
    assert output["presence_logit"].shape == (2,)
    assert output["centre"].shape == output["geometry"].shape == (2, 2)
    loss, parts = localization_loss(output, labels, 32, 48)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_gaussian_heatmap_is_zero_for_absent_target():
    labels = torch.tensor(
        [[1.0, 0.5, 0.5, 0.01, 0.3], [0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    heatmap = gaussian_heatmaps(labels, 9, 11)
    assert heatmap.shape == (2, 9, 11)
    assert heatmap[0].max() == 1.0
    assert heatmap[1].count_nonzero() == 0


def test_temporal_tracker_shapes_and_finite_loss():
    model = TemporalHeatmapTracker(hidden_size=32)
    maps = torch.softmax(torch.rand(2, 5, 30, 40).flatten(2), dim=-1).reshape(
        2, 5, 30, 40
    )
    auxiliary = torch.rand(2, 5, 7)
    labels = torch.zeros(2, 5, 5)
    labels[0, :, 0] = 1.0
    labels[0, :, 1:3] = torch.rand(5, 2)
    prediction = model(maps, auxiliary)
    assert prediction["heatmap_logits"].shape == (2, 5, 30, 40)
    assert prediction["presence_logit"].shape == (2, 5)
    assert prediction["inbound_logit"].shape == (2, 5)
    assert prediction["centre"].shape == prediction["geometry"].shape == (2, 5, 2)
    loss, parts = temporal_loss(prediction, labels)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_inbound_target_uses_only_prior_visible_depth():
    labels = torch.zeros(1, 7, 5)
    labels[0, :, 0] = 1.0
    labels[0, :, 4] = torch.tensor([0.40, 0.40, 0.40, 0.40, 0.38, 0.36, 0.34])
    target, valid = inbound_targets(labels, lag=3, dt=0.05)
    assert not bool(valid[0, :3].any())
    assert bool(valid[0, 3:].all())
    assert not bool(target[0, 3])
    assert bool(target[0, 4:].all())


def test_inbound_target_supervises_empty_scenes_as_negative():
    """An empty scene is a definite negative: nothing can be inbound."""
    labels = torch.zeros(1, 7, 5)
    target, valid = inbound_targets(labels, lag=3, dt=0.05)
    # The first ``lag`` frames still have no interval behind them.
    assert not bool(valid[0, :3].any())
    assert bool(valid[0, 3:].all())
    assert not bool(target[0].any())


def test_inbound_target_masks_frames_visible_at_one_end_only():
    """Appearing or disappearing mid-interval carries no closure evidence."""
    labels = torch.zeros(1, 9, 5)
    labels[0, 4:, 0] = 1.0
    labels[0, 4:, 4] = torch.tensor([0.40, 0.35, 0.30, 0.25, 0.20])
    target, valid = inbound_targets(labels, lag=3, dt=0.05)
    # Frames 4-6 pair an absent past with a visible present: ambiguous.
    assert not bool(valid[0, 4:7].any())
    # Frame 3 pairs absent with absent: a definite negative.
    assert bool(valid[0, 3]) and not bool(target[0, 3])
    # Frames 7-8 are the first visible pairs, and they are closing.
    assert bool(valid[0, 7:].all()) and bool(target[0, 7:].all())


def test_presence_loss_balances_classes_against_negative_flooding():
    """Adding empty frames must not drag the presence loss toward "absent"."""
    torch.manual_seed(0)

    def presence_only_loss(labels):
        output = {
            "presence_logit": torch.zeros(labels.shape[:2]),
            "inbound_logit": torch.zeros(labels.shape[:2]),
            "heatmap_logits": torch.zeros(*labels.shape[:2], 30, 40),
            "centre": torch.full((*labels.shape[:2], 2), 0.5),
            "geometry": torch.full((*labels.shape[:2], 2), 0.5),
        }
        return temporal_loss(output, labels)[1]["presence_loss"]

    balanced = torch.zeros(1, 8, 5)
    balanced[0, :4, 0] = 1.0
    balanced[0, :4, 1:] = 0.5
    flooded = torch.zeros(1, 40, 5)
    flooded[0, :4, 0] = 1.0
    flooded[0, :4, 1:] = 0.5
    # Same four positives, ten times the negatives, at a chance-level logit.
    assert torch.isclose(
        presence_only_loss(balanced), presence_only_loss(flooded), atol=1e-6
    )
