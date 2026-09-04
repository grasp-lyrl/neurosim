"""Train a spatial event encoder to detect and localize a dynamic obstacle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class EventLocalizationDataset(torch.utils.data.Dataset):
    """Lazy, episode-split access to labeled native-resolution event frames."""

    def __init__(
        self,
        paths,
        *,
        holdout_fraction=0.2,
        holdout_side="train",
        holdout_seed=7373,
        augment_horizontal=False,
    ):
        import h5py

        self.files = [h5py.File(path, "r") for path in paths]
        self.augment_horizontal = bool(augment_horizontal)
        episode_keys = []
        for file_index, file in enumerate(self.files):
            if "observations_obstacle_image" not in file:
                raise ValueError("dataset lacks observations_obstacle_image")
            episode_keys.extend(
                (file_index, int(episode))
                for episode in np.unique(file["episode_index"][:])
            )
        shuffled = np.random.default_rng(holdout_seed).permutation(
            np.asarray(episode_keys, dtype=np.int64)
        )
        n_test = max(int(round(len(episode_keys) * holdout_fraction)), 1)
        test = {tuple(row) for row in shuffled[:n_test].tolist()}
        selected = test if holdout_side == "test" else set(episode_keys) - test
        self.rows = []
        self.selected_episodes = 0
        for file_index, file in enumerate(self.files):
            episode_index = np.asarray(file["episode_index"][:])
            for episode in np.unique(episode_index):
                if (file_index, int(episode)) not in selected:
                    continue
                self.selected_episodes += 1
                self.rows.extend(
                    (file_index, int(row))
                    for row in np.flatnonzero(episode_index == episode)
                )
        self.present = np.asarray(
            [
                self.files[fi]["observations_obstacle_image"][row, 0] > 0.5
                for fi, row in self.rows
            ],
            dtype=np.bool_,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        fi, row = self.rows[index]
        file = self.files[fi]
        events = np.asarray(file["observations_events"][row], dtype=np.float32)
        label = np.asarray(
            file["observations_obstacle_image"][row], dtype=np.float32
        ).copy()
        if self.augment_horizontal and bool(torch.rand(()) < 0.5):
            events = np.ascontiguousarray(events[..., ::-1])
            if label[0] > 0.5:
                label[1] = 1.0 - label[1]
        return torch.from_numpy(events), torch.from_numpy(label)

    @property
    def balanced_weights(self):
        weights = np.empty(len(self), dtype=np.float64)
        for present, share in ((False, 0.5), (True, 0.5)):
            mask = self.present == present
            if bool(mask.any()):
                weights[mask] = share / int(mask.sum())
        return weights

    def close(self):
        for file in self.files:
            file.close()


class SpatialEventLocalizer(nn.Module):
    """Quarter-resolution heatmap encoder with differentiable coordinates."""

    def __init__(self, in_channels=8, feature_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            # Retain a half-resolution map. At 7 m the sphere radius is only
            # ~2.6 input pixels; a second stride-2 stage reduced it below one
            # cell and validation localization plateaued near 36 px.
            nn.Conv2d(16, feature_channels, 3, stride=1, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
            nn.Conv2d(
                feature_channels, feature_channels, 3, padding=2, dilation=2
            ),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
        )
        self.heatmap_head = nn.Conv2d(feature_channels, 1, 1)
        self.presence_head = nn.Sequential(
            nn.Linear(2 * feature_channels, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.geometry_head = nn.Sequential(
            nn.Linear(feature_channels, 32), nn.SiLU(), nn.Linear(32, 2)
        )

    def forward(self, events):
        features = self.encoder(events)
        heatmap_logits = self.heatmap_head(features)[:, 0]
        batch, height, width = heatmap_logits.shape
        # A low-temperature spatial softmax turns the dense heatmap into an
        # actual point estimate. Temperature 1 left early logits effectively
        # uniform over 4,800 cells and produced ~39 px median errors even as
        # the detection head improved.
        attention = torch.softmax(heatmap_logits.flatten(1) / 0.25, dim=-1)
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=events.device),
            torch.linspace(0.0, 1.0, width, device=events.device),
            indexing="ij",
        )
        centre = torch.stack(
            [attention @ xx.flatten(), attention @ yy.flatten()], dim=-1
        )
        attended = (
            features.flatten(2) * attention.unsqueeze(1)
        ).sum(dim=-1)
        pooled = torch.cat(
            [features.mean((2, 3)), features.amax((2, 3))], dim=-1
        )
        return {
            "heatmap_logits": heatmap_logits,
            "presence_logit": self.presence_head(pooled)[:, 0],
            "centre": centre,
            "geometry": torch.sigmoid(self.geometry_head(attended)),
        }


def gaussian_heatmaps(labels, height, width, sigma=1.5):
    yy, xx = torch.meshgrid(
        torch.arange(height, device=labels.device),
        torch.arange(width, device=labels.device),
        indexing="ij",
    )
    cx = labels[:, 1, None, None] * (width - 1)
    cy = labels[:, 2, None, None] * (height - 1)
    heatmap = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return heatmap * labels[:, 0, None, None]


def localization_loss(output, labels, image_height, image_width):
    present = labels[:, 0] > 0.5
    presence_loss = nn.functional.binary_cross_entropy_with_logits(
        output["presence_logit"], labels[:, 0]
    )
    target_heatmap = gaussian_heatmaps(
        labels, *output["heatmap_logits"].shape[-2:]
    )
    # Treat localization as matching a spatial probability distribution.
    # Pixelwise MSE/BCE collapses to all-background because even a 2.6 px
    # obstacle occupies very few of 4,800 cells. Normalizing the projected
    # Gaussian removes that class imbalance and gives every positive frame
    # one equally weighted localization target.
    if bool(present.any()):
        target_distribution = target_heatmap[present].flatten(1)
        target_distribution = target_distribution / target_distribution.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        heatmap_loss = -(
            target_distribution
            * torch.log_softmax(
                output["heatmap_logits"][present].flatten(1) / 0.25, dim=-1
            )
        ).sum(dim=-1).mean()
    else:
        heatmap_loss = output["heatmap_logits"].sum() * 0.0
    zero = output["centre"].sum() * 0.0
    centre_loss = (
        nn.functional.smooth_l1_loss(output["centre"][present], labels[present, 1:3])
        if bool(present.any()) else zero
    )
    geometry_target = torch.stack(
        [
            (labels[:, 3] * max(image_height, image_width) / 10.0).clamp(0, 1),
            labels[:, 4].clamp(0, 1),
        ],
        dim=-1,
    )
    geometry_loss = (
        nn.functional.smooth_l1_loss(
            output["geometry"][present], geometry_target[present]
        )
        if bool(present.any()) else zero
    )
    total = presence_loss + 0.25 * heatmap_loss + 5.0 * centre_loss + geometry_loss
    return total, {
        "presence_loss": presence_loss,
        "heatmap_loss": heatmap_loss,
        "centre_loss": centre_loss,
        "geometry_loss": geometry_loss,
    }


@torch.no_grad()
def validation_metrics(model, dataset, device, batch_size, num_workers=2):
    # Native-resolution frames are ~4.9 MB each, so validation is disk-bound.
    # Reading it single-threaded left the GPU at 15% utilisation.
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )
    tp = fp = fn = tn = 0
    centre_errors, depth_errors, radius_errors = [], [], []
    sample_width = None
    for events, labels in loader:
        sample_width = int(events.shape[-1])
        events, labels = events.to(device), labels.to(device)
        output = model(events)
        present = labels[:, 0] > 0.5
        predicted = torch.sigmoid(output["presence_logit"]) >= 0.5
        tp += int((predicted & present).sum())
        fp += int((predicted & ~present).sum())
        fn += int((~predicted & present).sum())
        tn += int((~predicted & ~present).sum())
        if bool(present.any()):
            scale = torch.tensor(
                [events.shape[-1], events.shape[-2]], device=device
            )
            centre_errors.extend(
                torch.linalg.norm(
                    (output["centre"][present] - labels[present, 1:3]) * scale,
                    dim=-1,
                ).cpu().tolist()
            )
            radius_errors.extend(
                (
                    output["geometry"][present, 0] * 10.0
                    - labels[present, 3] * max(events.shape[-2:])
                ).abs().cpu().tolist()
            )
            depth_errors.extend(
                ((output["geometry"][present, 1] - labels[present, 4]) * 20.0)
                .abs().cpu().tolist()
            )
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_positive_rate": fp / max(fp + tn, 1),
        "centre_median_px": float(np.median(centre_errors)),
        "centre_p90_px": float(np.percentile(centre_errors, 90)),
        "centre_within_8px": float(np.mean(np.asarray(centre_errors) <= 8.0)),
        # Report an angularly comparable metric across 320- and 640-wide
        # sensors. Eight pixels at width 320 corresponds to sixteen pixels at
        # width 640 under the unchanged 120-degree field of view.
        "centre_median_320eq_px": float(
            np.median(centre_errors) * 320.0 / sample_width
        ),
        "centre_p90_320eq_px": float(
            np.percentile(centre_errors, 90) * 320.0 / sample_width
        ),
        "centre_within_8px_320eq": float(
            np.mean(np.asarray(centre_errors) * 320.0 / sample_width <= 8.0)
        ),
        "radius_mae_px": float(np.mean(radius_errors)),
        "depth_mae_m": float(np.mean(depth_errors)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--holdout-seed", type=int, default=7373)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help=(
            "Dataset reader processes. Native-resolution event frames are "
            "~4.9 MB each, so training is disk-bound; raise this well above "
            "the default on a corpus of any size."
        ),
    )
    parser.add_argument("--initialize-from")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    common = dict(
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
    )
    train = EventLocalizationDataset(
        args.dataset, holdout_side="train", augment_horizontal=True, **common
    )
    test = EventLocalizationDataset(
        args.dataset, holdout_side="test", **common
    )
    sample_events, _ = train[0]
    model = SpatialEventLocalizer(in_channels=sample_events.shape[0]).to(device)
    if args.initialize_from:
        initial = torch.load(
            args.initialize_from, map_location=device, weights_only=False
        )
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(train.balanced_weights, dtype=torch.double),
        min(args.samples_per_epoch, len(train)), replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_name(f"{output.stem}.best.pt")
    history = []
    best_metric = -float("inf")
    try:
        print(json.dumps({
            "train_frames": len(train), "test_frames": len(test),
            "train_episodes": train.selected_episodes,
            "test_episodes": test.selected_episodes,
            "train_positive_fraction": float(train.present.mean()),
            "test_positive_fraction": float(test.present.mean()),
        }), flush=True)
        for epoch in range(args.epochs):
            model.train()
            sums, batches = {}, 0
            for events, labels in loader:
                events, labels = events.to(device), labels.to(device)
                prediction = model(events)
                loss, parts = localization_loss(
                    prediction, labels, events.shape[-2], events.shape[-1]
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                for key, value in {"loss": loss, **parts}.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach())
                batches += 1
            model.eval()
            row = {
                "epoch": epoch,
                **{key: value / batches for key, value in sums.items()},
                **validation_metrics(
                    model, test, device, args.batch_size, args.num_workers
                ),
            }
            metric = (
                row["presence_f1"]
                + row["centre_within_8px_320eq"]
                - 0.01 * row["centre_median_320eq_px"]
            )
            if metric > best_metric:
                best_metric = metric
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_metric": metric,
                    "model_type": "spatial_event_localizer",
                    "input_shape": list(sample_events.shape),
                    "label_format": [
                        "presence", "u_norm", "v_norm", "radius_norm", "depth_zfar"
                    ],
                }, checkpoint)
            history.append(row)
            print(json.dumps(row), flush=True)
        output.write_text(json.dumps({
            "history": history,
            "best_validation_metric": best_metric,
            "best_checkpoint": str(checkpoint),
        }, indent=2))
    finally:
        train.close()
        test.close()


if __name__ == "__main__":
    main()
