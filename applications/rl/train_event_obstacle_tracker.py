"""Train a recurrent obstacle tracker over frozen event-localizer heatmaps.

The spatial encoder is deliberately frozen.  This separates the question
"does temporal integration turn weak 10 ms evidence into a usable track?"
from further changes to the event representation or image encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_event_obstacle_localizer import (
    EventLocalizationDataset,
    SpatialEventLocalizer,
    gaussian_heatmaps,
)


def _selected_episode_rows(dataset):
    """Return dataset rows grouped in their original episode order."""
    episodes = []
    current_key, current_rows = None, []
    for file_index, row in dataset.rows:
        episode = int(dataset.files[file_index]["episode_index"][row])
        key = (file_index, episode)
        if key != current_key:
            if current_rows:
                episodes.append((current_key, current_rows))
            current_key, current_rows = key, []
        current_rows.append(row)
    if current_rows:
        episodes.append((current_key, current_rows))
    return episodes


@torch.no_grad()
def build_heatmap_cache(dataset, spatial_model, device, batch_size=16):
    """Freeze spatial predictions into compact 30x40 probability maps."""
    spatial_model.eval()
    cached = []
    for (file_index, episode), rows in _selected_episode_rows(dataset):
        file = dataset.files[file_index]
        map_chunks, auxiliary_chunks = [], []
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            events = torch.from_numpy(
                np.asarray(file["observations_events"][batch_rows], dtype=np.float32)
            ).to(device)
            output = spatial_model(events)
            map_height, map_width = output["heatmap_logits"].shape[-2:]
            probability = torch.softmax(
                output["heatmap_logits"].flatten(1) / 0.25, dim=-1
            ).reshape(-1, map_height, map_width)
            # Pool to a resolution-independent 30x40 probability map. Scale
            # the adaptive average back to a sum so each map still integrates
            # to one for both 120x160 and 240x320 spatial encoders.
            coarse = F.adaptive_avg_pool2d(
                probability[:, None], (30, 40)
            )[:, 0] * (map_height * map_width / (30 * 40))
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(
                (1, 2)
            ) / np.log(map_height * map_width)
            auxiliary = torch.cat(
                [
                    torch.sigmoid(output["presence_logit"])[:, None],
                    output["centre"],
                    output["geometry"],
                    coarse.amax((1, 2))[:, None],
                    entropy[:, None],
                ],
                dim=-1,
            )
            map_chunks.append(coarse.cpu())
            auxiliary_chunks.append(auxiliary.cpu())
        labels = torch.from_numpy(
            np.asarray(file["observations_obstacle_image"][rows], dtype=np.float32)
        )
        cached.append(
            {
                "key": (int(file_index), int(episode)),
                "maps": torch.cat(map_chunks),
                "auxiliary": torch.cat(auxiliary_chunks),
                "labels": labels,
                "image_shape": tuple(
                    int(value)
                    for value in file["observations_events"].shape[-2:]
                ),
            }
        )
    return cached


class HeatmapSequenceDataset(torch.utils.data.Dataset):
    """Overlapping, episode-bounded windows from a frozen heatmap cache."""

    def __init__(self, episodes, length=32, stride=8, augment_horizontal=False):
        self.episodes = episodes
        self.length = int(length)
        self.augment_horizontal = bool(augment_horizontal)
        self.windows = []
        for episode_index, episode in enumerate(episodes):
            count = len(episode["labels"])
            starts = list(range(0, max(count - self.length + 1, 1), stride))
            last = max(count - self.length, 0)
            if not starts or starts[-1] != last:
                starts.append(last)
            self.windows.extend((episode_index, start) for start in starts)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_index, start = self.windows[index]
        episode = self.episodes[episode_index]
        stop = min(start + self.length, len(episode["labels"]))
        maps = episode["maps"][start:stop].clone()
        auxiliary = episode["auxiliary"][start:stop].clone()
        labels = episode["labels"][start:stop].clone()
        # All collected expert episodes are longer than the training window.
        if len(labels) != self.length:
            raise ValueError("episode shorter than temporal training window")
        if self.augment_horizontal and bool(torch.rand(()) < 0.5):
            maps = maps.flip(-1)
            auxiliary[:, 1] = 1.0 - auxiliary[:, 1]
            positive = labels[:, 0] > 0.5
            labels[positive, 1] = 1.0 - labels[positive, 1]
        return maps, auxiliary, labels


class TemporalHeatmapTracker(nn.Module):
    """GRU that integrates coarse spatial distributions over time."""

    def __init__(self, auxiliary_size=7, hidden_size=128):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 10, 96),
            nn.SiLU(),
        )
        self.auxiliary_encoder = nn.Sequential(
            nn.Linear(auxiliary_size, 32), nn.SiLU()
        )
        self.gru = nn.GRU(128, hidden_size, batch_first=True)
        # Keep launch-phase adaptation isolated from the established
        # presence/centre tracker.  The branch consumes the same causal
        # features but has its own recurrent state and can be fine-tuned
        # without changing the controller's original tracking signal.
        self.inbound_gru = nn.GRU(128, hidden_size, batch_first=True)
        self.heatmap_head = nn.Linear(hidden_size, 30 * 40)
        self.presence_head = nn.Linear(hidden_size, 1)
        # Whether the visible obstacle is already moving toward the camera.
        # This is trained from temporal changes in the projected depth label,
        # not supplied to the model at inference time.  It lets control retain
        # the pre-launch wait while releasing immediately after a late track.
        self.inbound_head = nn.Linear(hidden_size, 1)
        self.geometry_head = nn.Linear(hidden_size, 2)

    def forward(self, maps, auxiliary, hidden=None):
        batch, steps = maps.shape[:2]
        # A uniform distribution has mean 1/1200. Scaling to mean one avoids
        # presenting the map encoder with unnecessarily tiny activations.
        map_features = self.map_encoder((maps * 1200.0).reshape(-1, 1, 30, 40))
        map_features = map_features.reshape(batch, steps, -1)
        auxiliary_features = self.auxiliary_encoder(auxiliary)
        combined = torch.cat([map_features, auxiliary_features], dim=-1)
        if isinstance(hidden, tuple):
            tracker_hidden, inbound_hidden = hidden
        else:
            # Old callers/checkpoints supplied only the tracker state.
            tracker_hidden, inbound_hidden = hidden, None
        recurrent, tracker_hidden = self.gru(combined, tracker_hidden)
        inbound_recurrent, inbound_hidden = self.inbound_gru(
            combined, inbound_hidden
        )
        heatmap_logits = self.heatmap_head(recurrent).reshape(
            batch, steps, 30, 40
        )
        attention = torch.softmax(heatmap_logits.flatten(2) / 0.25, dim=-1)
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, 30, device=maps.device),
            torch.linspace(0.0, 1.0, 40, device=maps.device),
            indexing="ij",
        )
        centre = torch.stack(
            [attention @ xx.flatten(), attention @ yy.flatten()], dim=-1
        )
        return {
            "heatmap_logits": heatmap_logits,
            "presence_logit": self.presence_head(recurrent)[..., 0],
            "inbound_logit": self.inbound_head(inbound_recurrent)[..., 0],
            "centre": centre,
            "geometry": torch.sigmoid(self.geometry_head(recurrent)),
            # Frozen causal representation used by downstream event-policy
            # heads. Exposing it avoids retraining or duplicating perception.
            "features": recurrent,
            "hidden": (tracker_hidden, inbound_hidden),
        }


def inbound_targets(labels, *, lag=3, dt=0.05, speed_threshold_mps=1.5):
    """Derive a causal launch/inbound target from projected depth history.

    The first ``lag`` frames have no target.  Positive range closure above the
    threshold identifies post-launch motion without exposing object state to
    the network.

    A frame with no obstacle visible at either end of the interval is a
    definite negative: nothing can be inbound in a scene containing nothing.
    Supervising those frames is what stops the head from firing on ego-motion.
    Restricting the mask to visible pairs left the head untrained on empty
    scenes, where it went on to fire on 10-15% of frames and released the
    avoidance pulse against background motion.  Frames visible at only one end
    stay masked: an obstacle that appears or disappears mid-interval carries no
    unambiguous closure evidence.
    """
    present = labels[..., 0] > 0.5
    target = torch.zeros_like(labels[..., 0], dtype=torch.bool)
    valid = torch.zeros_like(target)
    if labels.shape[-2] <= lag:
        return target, valid
    closure_mps = (
        labels[..., :-lag, 4] - labels[..., lag:, 4]
    ) * 20.0 / (lag * dt)
    visible_pair = present[..., :-lag] & present[..., lag:]
    empty_pair = ~present[..., :-lag] & ~present[..., lag:]
    valid[..., lag:] = visible_pair | empty_pair
    target[..., lag:] = visible_pair & (closure_mps > speed_threshold_mps)
    return target, valid


def temporal_loss(
    output,
    labels,
    early_weight=1.0,
    early_depth_norm=0.35,
    inbound_loss_weight=0.5,
):
    present = labels[..., 0] > 0.5
    early = present & (labels[..., 4] >= early_depth_norm)
    frame_weight = torch.ones_like(labels[..., 0])
    frame_weight[early] = float(early_weight)
    # Balance the presence classes so that adding empty-scene negatives buys
    # nuisance diversity without moving the operating point.  An unweighted
    # sum lets a negative-heavy corpus collapse recall, which is the failure
    # already recorded for the earlier nominal-negative heads.  Relative
    # emphasis inside the positive class (``early_weight``) is preserved.
    presence_weight = frame_weight.clone()
    absent = ~present
    if bool(present.any()) and bool(absent.any()):
        presence_weight[present] *= 0.5 / presence_weight[present].sum()
        presence_weight[absent] *= 0.5 / presence_weight[absent].sum()
    presence_loss = (
        nn.functional.binary_cross_entropy_with_logits(
            output["presence_logit"], labels[..., 0], reduction="none"
        )
        * presence_weight
    ).sum() / presence_weight.sum()
    inbound_target, inbound_valid = inbound_targets(labels)
    if bool(inbound_valid.any()):
        target_values = inbound_target[inbound_valid].to(
            output["inbound_logit"].dtype
        )
        positive = target_values > 0.5
        negative = ~positive
        inbound_sample_weight = torch.ones_like(target_values)
        if bool(positive.any()) and bool(negative.any()):
            inbound_sample_weight[positive] = 0.5 / positive.sum()
            inbound_sample_weight[negative] = 0.5 / negative.sum()
            inbound_sample_weight *= len(inbound_sample_weight)
        inbound_loss = (
            nn.functional.binary_cross_entropy_with_logits(
                output["inbound_logit"][inbound_valid],
                target_values,
                reduction="none",
            )
            * inbound_sample_weight
        ).mean()
    else:
        inbound_loss = output["inbound_logit"].sum() * 0.0
    flat_labels = labels.reshape(-1, 5)
    target = gaussian_heatmaps(flat_labels, 30, 40, sigma=1.0).reshape(
        *labels.shape[:2], 30, 40
    )
    target = target[present].flatten(1)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    if bool(present.any()):
        spatial_per_frame = -(
            target
            * torch.log_softmax(
                output["heatmap_logits"][present].flatten(1) / 0.25, dim=-1
            )
        ).sum(-1)
        positive_weight = frame_weight[present]
        spatial_loss = (spatial_per_frame * positive_weight).sum() / positive_weight.sum()
        centre_per_frame = nn.functional.smooth_l1_loss(
            output["centre"][present], labels[present, 1:3], reduction="none"
        ).mean(-1)
        centre_loss = (centre_per_frame * positive_weight).sum() / positive_weight.sum()
        geometry_target = torch.stack(
            [
                (labels[..., 3] * 320.0 / 10.0).clamp(0, 1),
                labels[..., 4].clamp(0, 1),
            ],
            dim=-1,
        )
        geometry_per_frame = nn.functional.smooth_l1_loss(
            output["geometry"][present], geometry_target[present], reduction="none"
        ).mean(-1)
        geometry_loss = (geometry_per_frame * positive_weight).sum() / positive_weight.sum()
    else:
        zero = output["centre"].sum() * 0.0
        spatial_loss = centre_loss = geometry_loss = zero
    total = (
        presence_loss
        + float(inbound_loss_weight) * inbound_loss
        + 0.25 * spatial_loss
        + 5.0 * centre_loss
        + geometry_loss
    )
    return total, {
        "presence_loss": presence_loss,
        "inbound_loss": inbound_loss,
        "spatial_loss": spatial_loss,
        "centre_loss": centre_loss,
        "geometry_loss": geometry_loss,
    }


@torch.no_grad()
def validation_metrics(model, episodes, device, thresholds=(0.5, 0.7, 0.9)):
    model.eval()
    records = []
    for episode in episodes:
        maps = episode["maps"][None].to(device)
        auxiliary = episode["auxiliary"][None].to(device)
        output = model(maps, auxiliary)
        records.append(
            {
                "labels": episode["labels"].numpy(),
                "probability": torch.sigmoid(output["presence_logit"])[0]
                .cpu()
                .numpy(),
                "inbound_probability": torch.sigmoid(output["inbound_logit"])[0]
                .cpu()
                .numpy(),
                "centre": output["centre"][0].cpu().numpy(),
                "geometry": output["geometry"][0].cpu().numpy(),
                "image_shape": episode["image_shape"],
            }
        )
    labels = np.concatenate([record["labels"] for record in records])
    probability = np.concatenate([record["probability"] for record in records])
    inbound_probability = np.concatenate(
        [record["inbound_probability"] for record in records]
    )
    centre = np.concatenate([record["centre"] for record in records])
    geometry = np.concatenate([record["geometry"] for record in records])
    present = labels[:, 0] > 0.5
    inbound_rows, inbound_valid_rows = [], []
    for record in records:
        episode_labels = torch.from_numpy(record["labels"])[None]
        target, valid = inbound_targets(episode_labels)
        inbound_rows.append(target[0].numpy())
        inbound_valid_rows.append(valid[0].numpy())
    inbound = np.concatenate(inbound_rows)
    inbound_valid = np.concatenate(inbound_valid_rows)
    heights = np.concatenate(
        [np.full(len(record["labels"]), record["image_shape"][0]) for record in records]
    )
    widths = np.concatenate(
        [np.full(len(record["labels"]), record["image_shape"][1]) for record in records]
    )
    pixel_scale = np.stack([widths, heights], axis=-1)
    centre_error = np.linalg.norm(
        (centre[present] - labels[present, 1:3]) * pixel_scale[present], axis=-1
    )
    radius_error = np.abs(
        geometry[present, 0] * 10 - labels[present, 3] * widths[present]
    )
    depth_error = np.abs((geometry[present, 1] - labels[present, 4]) * 20)
    result = {
        "centre_median_px": float(np.median(centre_error)),
        "centre_p90_px": float(np.percentile(centre_error, 90)),
        "centre_within_8px": float(np.mean(centre_error <= 8)),
        "centre_median_320eq_px": float(
            np.median(centre_error * 320.0 / widths[present])
        ),
        "centre_within_8px_320eq": float(
            np.mean(centre_error * 320.0 / widths[present] <= 8)
        ),
        "radius_mae_px": float(np.mean(radius_error)),
        "depth_mae_m": float(np.mean(depth_error)),
    }
    for threshold in (0.5, 0.9, 0.95, 0.99):
        predicted_inbound = inbound_probability >= threshold
        positive = inbound_valid & inbound
        negative = inbound_valid & ~inbound
        tp = int(np.sum(predicted_inbound & positive))
        fp = int(np.sum(predicted_inbound & negative))
        fn = int(np.sum(~predicted_inbound & positive))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        result[f"inbound_threshold_{threshold:g}"] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "prelaunch_false_rate": fp / max(int(negative.sum()), 1),
        }
    early = present & (labels[:, 4] >= 7.0 / 20.0)
    for threshold in thresholds:
        predicted = probability >= threshold
        tp = int(np.sum(predicted & present))
        fp = int(np.sum(predicted & ~present))
        fn = int(np.sum(~predicted & present))
        tn = int(np.sum(~predicted & ~present))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        detected_error = np.linalg.norm(
            (centre[predicted & present] - labels[predicted & present, 1:3])
            * pixel_scale[predicted & present],
            axis=-1,
        )
        result[f"threshold_{threshold:g}"] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "false_positive_rate": fp / max(fp + tn, 1),
            "detected_centre_median_px": float(np.median(detected_error))
            if len(detected_error)
            else float("nan"),
            "detected_centre_within_8px": float(np.mean(detected_error <= 8))
            if len(detected_error)
            else float("nan"),
            "detected_centre_median_320eq_px": float(
                np.median(
                    detected_error * 320.0 / widths[predicted & present]
                )
            )
            if len(detected_error)
            else float("nan"),
            "detected_centre_within_8px_320eq": float(
                np.mean(
                    detected_error * 320.0 / widths[predicted & present] <= 8
                )
            )
            if len(detected_error)
            else float("nan"),
            "early_recall": float(np.mean(predicted[early]))
            if bool(early.any())
            else float("nan"),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--spatial-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--early-weight",
        type=float,
        default=1.0,
        help="Loss multiplier for visible positive frames at or beyond 7 m.",
    )
    parser.add_argument(
        "--inbound-loss-weight",
        type=float,
        default=0.5,
        help="Multiplier for the supervised launch/inbound classification loss.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=5252)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--holdout-seed", type=int, default=7373)
    parser.add_argument("--initialize-from")
    parser.add_argument(
        "--inbound-head-only",
        action="store_true",
        help=(
            "Freeze the established tracker and train only the isolated "
            "inbound GRU/head branch. "
            "This is the guarded default for the adaptive-release experiment."
        ),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    common = dict(
        paths=args.dataset,
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
    )
    train_frames = EventLocalizationDataset(holdout_side="train", **common)
    test_frames = EventLocalizationDataset(holdout_side="test", **common)
    sample_events, _ = train_frames[0]
    spatial = SpatialEventLocalizer(in_channels=sample_events.shape[0]).to(device)
    spatial_payload = torch.load(
        args.spatial_checkpoint, map_location=device, weights_only=False
    )
    spatial.load_state_dict(spatial_payload["model"])
    print("Building frozen spatial caches...", flush=True)
    train_episodes = build_heatmap_cache(train_frames, spatial, device)
    test_episodes = build_heatmap_cache(test_frames, spatial, device)
    train_frames.close()
    test_frames.close()

    train = HeatmapSequenceDataset(
        train_episodes,
        length=args.sequence_length,
        stride=args.sequence_stride,
        augment_horizontal=True,
    )
    loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=2,
        persistent_workers=True,
    )
    model = TemporalHeatmapTracker().to(device)
    if args.initialize_from:
        initial = torch.load(
            args.initialize_from, map_location=device, weights_only=False
        )
        missing, unexpected = model.load_state_dict(initial["model"], strict=False)
        allowed_missing = {
            "inbound_head.weight",
            "inbound_head.bias",
            *{
                f"inbound_gru.{name}"
                for name in model.inbound_gru.state_dict()
            },
        }
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(
                f"incompatible initialization: missing={missing}, "
                f"unexpected={unexpected}"
            )
        if not any(key.startswith("inbound_gru.") for key in initial["model"]):
            model.inbound_gru.load_state_dict(model.gru.state_dict())
    if args.inbound_head_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("inbound_head.")
                or name.startswith("inbound_gru.")
            )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output_path.with_name(f"{output_path.stem}.best.pt")
    history, best_metric = [], -float("inf")
    print(
        json.dumps(
            {
                "train_episodes": len(train_episodes),
                "test_episodes": len(test_episodes),
                "train_windows": len(train),
                "sequence_length": args.sequence_length,
                "timesurface_decay_ms": 10.0,
                "inbound_head_only": bool(args.inbound_head_only),
            }
        ),
        flush=True,
    )
    for epoch in range(args.epochs):
        model.train()
        sums, batches = {}, 0
        for maps, auxiliary, labels in loader:
            maps = maps.to(device)
            auxiliary = auxiliary.to(device)
            labels = labels.to(device)
            prediction = model(maps, auxiliary)
            loss, parts = temporal_loss(
                prediction,
                labels,
                early_weight=args.early_weight,
                inbound_loss_weight=args.inbound_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in {"loss": loss, **parts}.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach())
            batches += 1
        metrics = validation_metrics(model, test_episodes, device)
        row = {
            "epoch": epoch,
            **{key: value / batches for key, value in sums.items()},
            **metrics,
        }
        threshold = metrics["threshold_0.7"]
        metric = (
            threshold["f1"]
            + threshold["detected_centre_within_8px_320eq"]
            + 0.5 * threshold["early_recall"]
            - 0.01 * threshold["detected_centre_median_320eq_px"]
            - 0.25 * threshold["false_positive_rate"]
            + 0.5 * metrics["inbound_threshold_0.95"]["f1"]
            - 0.25
            * metrics["inbound_threshold_0.95"]["prelaunch_false_rate"]
        )
        if metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_metric": metric,
                    "model_type": "temporal_event_heatmap_tracker",
                    "spatial_checkpoint": args.spatial_checkpoint,
                    "sequence_length": args.sequence_length,
                    "timesurface_decay_ms": 10.0,
                    "inbound_head_only": bool(args.inbound_head_only),
                },
                checkpoint,
            )
        history.append(row)
        print(json.dumps(row), flush=True)
    output_path.write_text(
        json.dumps(
            {
                "history": history,
                "best_validation_metric": best_metric,
                "best_checkpoint": str(checkpoint),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
