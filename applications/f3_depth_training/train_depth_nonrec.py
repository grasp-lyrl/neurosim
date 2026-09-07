"""
Monocular Depth Training with F3 + the online_data pipeline.

Trains a monocular depth model (F3 / EventPatchFF backbone + DepthAnythingV2
decoder) on time-aligned events+depth streamed from the neurosim
``OnlineDataLoader``: ``num_producers`` SynchronousSimulators (events + depth)
run in separate processes on the producer GPUs, push time-aligned samples to a
bounded bus, and this (trainer) process builds batches and trains on the
``trainer_gpu``.

Reference: https://github.com/grasp-lyrl/fast-feature-fields/tree/main/src/f3/tasks/depth
"""

import os
import cv2
import yaml
import torch
import wandb
import logging
import argparse
import datetime
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from matplotlib import colormaps

from neurosim.online_data import OnlineDataLoader, TimeAlignedSample

from .nets import (
    EventFFDepthAnythingV2,
    batch_cropper,
    get_resize_shapes,
    load_depth_weights,
)
from .utils import (
    ScaleAndShiftInvariantLoss,
    eval_disparity,
    get_disparity_image,
    get_random_crop_params,
    set_best_results,
)

# Disparity metrics from eval_disparity; the loss name is appended per run.
METRICS = ("1pe", "2pe", "3pe", "rmse", "rmse_log", "log10", "silog")


def setup_torch() -> None:
    """f3's training defaults: fixed seed, tf32 matmuls, dynamo tracing dynamic shapes."""
    torch.manual_seed(403)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.compiled_autograd = True


def log_dict(logger, values: dict) -> None:
    """One log line of ``k: v``, tensors unwrapped."""
    logger.info(
        ", ".join(
            f"{k}: {v.item():.4f}" if isinstance(v, torch.Tensor) else f"{k}: {v:.4f}"
            for k, v in values.items()
        )
    )


def log_wandb(args, metrics: dict):
    """Log under the ``train/``, ``val/`` and ``opt/`` namespaces."""
    if args.wandb:
        wandb.log(metrics)


def uncompiled_state_dict(model) -> dict:
    """Weights without ``torch.compile``'s ``_orig_mod.`` prefix."""
    return {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}


def save_checkpoint(path, epoch, results, model, optimizer, scheduler):
    """Write the resumable checkpoint."""
    state = uncompiled_state_dict(model)
    torch.save(
        {
            "epoch": epoch,
            "results": results,
            "model": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )
    path = Path(path)
    torch.save(state, path.with_stem(f"{path.stem}_weights"))


def ev_to_frames_with_polarity(events, counts, w, h):
    """Convert events to RGB frames with polarity coloring (pos=red, neg=blue)."""
    scale = torch.tensor([w, h, 1, 1], device=events.device)
    events = (events * scale).round().to(torch.int32)

    B = counts.shape[0]
    event_frames = torch.zeros(B, h, w, 3, dtype=torch.uint8, device=events.device)
    c = torch.cumsum(torch.cat((torch.zeros(1).to(counts.device), counts)), 0).to(
        torch.int32
    )

    for i in range(B):
        x_coords = events[c[i] : c[i + 1], 0]
        y_coords = events[c[i] : c[i + 1], 1]
        polarities = events[c[i] : c[i + 1], 3]
        pos_mask = polarities == 1
        neg_mask = polarities == 0
        event_frames[i, y_coords[pos_mask], x_coords[pos_mask], 0] = 255  # Red
        event_frames[i, y_coords[neg_mask], x_coords[neg_mask], 2] = 255  # Blue

    return event_frames


def setup_experiment(args, base_path: str, models_path: str):
    """Setup experiment directories and check for resume."""
    for path in (
        models_path,
        f"{base_path}/predictions",
        f"{base_path}/training_events",
    ):
        os.makedirs(path, exist_ok=True)

    resume = os.path.exists(f"{models_path}/last.pth")

    # Carry the wandb run id across resumes, otherwise each restart forks a new run.
    config_path = f"{base_path}/config.yaml"
    args.wandb_run_id = None
    if resume and os.path.exists(config_path):
        with open(config_path, "r") as f:
            args.wandb_run_id = yaml.safe_load(f).get("wandb_run_id")
    if args.wandb_run_id is None:
        args.wandb_run_id = wandb.util.generate_id()

    logging.basicConfig(
        filename=f"{base_path}/training.log",
        filemode="a" if resume else "w",
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    with open(config_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    return logger, resume


def usable_sample_filter(
    depth_uuid: str,
    event_sensor: str,
    max_disparity: float,
    min_events: int,
    min_valid_frac: float = 0.5,
) -> Callable[[TimeAlignedSample], bool]:
    """Predicate dropping samples the model cannot learn from.

    Rejects a mostly-invalid depth anchor (0 m reads, e.g. a camera facing open
    sky) and packets too sparse for the backbone. Applied before batching, so a
    rejected sample costs a row rather than shrinking the batch.
    """
    min_depth = 1.0 / max_disparity

    def keep(sample: TimeAlignedSample) -> bool:
        depth = sample.sensors[depth_uuid]
        if np.count_nonzero(depth > min_depth) < min_valid_frac * depth.size:
            return False
        return len(sample.sensors[event_sensor].get("x", ())) >= min_events

    return keep


def build_loader(
    data_cfg: dict,
    *,
    batch_size: int,
    log_dir: str | None = None,
    sample_filter: Callable[[TimeAlignedSample], bool] | None = None,
) -> OnlineDataLoader:
    """Build an OnlineDataLoader from the ``data.online_data`` config block.

    Delegates to :meth:`OnlineDataLoader.from_config` (the same YAML schema used
    everywhere — roles, scenes, DR, and loader knobs live in ``data.online_data``).
    ``batch_size`` comes from the training config (``train.mini_batch``), and an
    optional ``data.sim_time`` overrides the base settings' episode length.
    """
    od = dict(data_cfg["online_data"])

    sim_time = data_cfg.get("sim_time")
    if sim_time is not None:
        base = od.get("base_settings")
        if isinstance(base, str):
            with open(base, "r") as f:
                base = yaml.safe_load(f)
        base.setdefault("simulator", {})["sim_time"] = sim_time
        od["base_settings"] = base

    return OnlineDataLoader.from_config(
        {"online_data": od},
        batch_size=batch_size,
        log_dir=log_dir,
        sample_filter=sample_filter,
    )


def process_batch(batch, args, device):
    """Process a batch from the OnlineDataLoader into model-ready tensors.

    ``batch[event_sensor]`` is ``(counts, events)`` where events are *raw*
    ``[x, y, t_anchor - t, p]`` (pixel coords, anchor-relative µs); the loader no
    longer normalizes. We normalize here to the model's frame sizes:
    ``[x/W, y/H, t_rel/window_us, p]`` (on-GPU, cheap). ``args.event_norm`` is
    ``(W, H, window_us)``. ``batch[depth_sensor]`` is ``(B, H, W)``.
    Returns ``(ff_events, event_counts, disparity, color_images)``.
    """
    event_sensor = args.event_sensor
    depth_sensor = args.depth_sensor
    color_sensor = args.color_sensor

    if event_sensor not in batch:
        raise ValueError(
            f"Event sensor '{event_sensor}' not in batch. Available: {list(batch.keys())}"
        )
    counts, events = batch[event_sensor]
    ff_events = torch.from_numpy(events).float().to(device)  # (N, 4) raw [x,y,t_rel,p]
    event_counts = torch.from_numpy(counts).to(device)  # (B,)
    # Normalize to the model's frame sizes (loader ships raw events).
    norm_w, norm_h, norm_window = args.event_norm
    ff_events[:, 0] /= norm_w
    ff_events[:, 1] /= norm_h
    ff_events[:, 2] /= norm_window

    if depth_sensor not in batch:
        raise ValueError(
            f"Depth sensor '{depth_sensor}' not in batch. Available: {list(batch.keys())}"
        )
    # Invalid depths are 0.0; clip then convert to disparity (inverse depth).
    depth = torch.from_numpy(batch[depth_sensor]).to(device, torch.float32)
    disparity = 1.0 / depth.clamp(0.5 / args.max_disparity, 1 / args.min_disparity)

    color_images = (
        batch[color_sensor].astype(np.uint8)
        if color_sensor and color_sensor in batch
        else None
    )
    return ff_events, event_counts, disparity, color_images


def usable_samples(valid_mask):
    """Rows with at least half their depth pixels valid."""
    return valid_mask.flatten(1).sum(1) * 2 >= valid_mask[0].numel()


def predict_full_frame(model, ff_events, event_counts, height, width):
    """Whole-frame disparity, aspect preserved — a batched ``model.infer_image``.

    Training crops a random square, but every f3 eval path (``evaluate``,
    ``dsec_benchmark``, its own validator) runs ``infer_image``, which resizes the
    short edge to the DAv2 size and keeps aspect. ``infer_image`` handles one sample
    at a time, so its steps are inlined here to keep validation batched.
    """
    field = model.field(ff_events, event_counts)  # (B, C, H, W)
    fh, fw = get_resize_shapes(height, width, model.size, 14)
    field = F.interpolate(field, (fh, fw), mode="bilinear", align_corners=False)
    pred = model.dav2(field).unsqueeze(1)  # (B, 1, fh, fw)
    return F.interpolate(
        pred, (height, width), mode="bilinear", align_corners=True
    ).squeeze(1)


def train_epoch(
    args,
    logger,
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    epoch,
    max_batches,
    iters_to_accumulate=1,
):
    """Train for one epoch (consumes ``max_batches`` batches from the loader)."""
    model.train()
    train_loss = 0.0
    iter_loss = 0.0
    idx = 0

    pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=max_batches)
    for idx, batch in pbar:
        if idx >= max_batches:
            break

        ff_events, event_counts, disparity, _ = process_batch(batch, args, args.device)

        B, H, W = disparity.shape
        cparams = get_random_crop_params((H, W), (H, H), batch_size=B).to(
            ff_events.device
        )
        disparity = batch_cropper(disparity.unsqueeze(1), cparams).squeeze(1)

        with torch.autocast(device_type="cuda", enabled=args.amp, dtype=torch.bfloat16):
            disparity_pred = model(ff_events, event_counts, cparams)[0]  # (B, H, W)

        # The SSI loss normalizes by a per-sample MAD, so it runs in fp32 even
        # under autocast; bf16 there would divide by an 8-bit-mantissa scale.
        disparity_pred = disparity_pred.float()
        disparity_valid_mask = disparity < args.max_disparity
        keep = usable_samples(disparity_valid_mask)
        if not keep.any():
            continue
        loss = loss_fn(
            disparity_pred[keep], disparity[keep], disparity_valid_mask[keep]
        )
        loss = loss / iters_to_accumulate

        loss.backward()
        train_loss += loss.item()
        iter_loss += loss.item()

        if (idx + 1) % iters_to_accumulate == 0:
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(
                {"loss": f"{iter_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
            )
            log_wandb(
                args,
                {
                    "train/iter_loss": iter_loss,
                    "opt/lr": scheduler.get_last_lr()[0],
                    "opt/epoch": epoch,
                    "opt/iteration": idx,
                },
            )
            iter_loss = 0.0

    num_iters = max(1, (idx + 1) // iters_to_accumulate)
    train_loss /= num_iters

    logger.info("#" * 50)
    logger.info(f"Training: Epoch: {epoch}, Loss: {train_loss:.4f}")
    logger.info("#" * 50)
    return train_loss


@torch.no_grad()
def validate(
    args, logger, model, dataloader, loss_fn, epoch, max_batches=50, save_preds=False
):
    """Validate the model on ``max_batches`` batches, at inference framing."""
    model.eval()
    cmap = colormaps["magma"]

    results = {k: torch.tensor([0.0]).cuda() for k in (*METRICS, loss_fn.name)}
    nsamples = torch.tensor([0.0]).cuda()

    for idx, batch in tqdm(enumerate(dataloader), total=max_batches, desc="Validation"):
        if idx >= max_batches:
            break

        ff_events, event_counts, disparity, color_images = process_batch(
            batch, args, args.device
        )
        B, H, W = disparity.shape
        disparity_pred = predict_full_frame(
            model, ff_events, event_counts, H, W
        ).float()
        valid_mask = disparity < args.max_disparity
        keep = usable_samples(valid_mask)
        if not keep.any():
            continue
        kept_mask = valid_mask[keep]

        cur_results = eval_disparity(
            disparity_pred[keep][kept_mask], disparity[keep][kept_mask]
        )
        for k in cur_results:
            results[k] += cur_results[k]
        results[loss_fn.name] += loss_fn(
            disparity_pred[keep], disparity[keep], kept_mask
        ).item()
        nsamples += 1

        if idx % 10 == 0 and save_preds:
            base_path = f"outputs/monoculardepth/{args.name}"
            event_frames = (
                ev_to_frames_with_polarity(ff_events, event_counts, W, H).cpu().numpy()
            )
            for i in range(disparity_pred.shape[0]):
                if not keep[i]:
                    continue
                images = {
                    "training_events/disparity": get_disparity_image(
                        disparity[i], valid_mask[i], cmap
                    ),
                    "predictions/disparity_pred": get_disparity_image(
                        disparity_pred[i],
                        torch.ones_like(disparity_pred[i], dtype=torch.bool),
                        cmap,
                    ),
                    "training_events/events": event_frames[i],
                }
                if color_images is not None:
                    images["training_events/color"] = color_images[i]
                # Every panel above is RGB (matplotlib cmap, polarity channels); cv2 wants BGR.
                for stem, image in images.items():
                    cv2.imwrite(
                        f"{base_path}/{stem}_{epoch}_{idx}_{i}.png",
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    )

    for k in results:
        results[k] /= nsamples

    logger.info("#" * 50)
    logger.info(f"Validation: Epoch: {epoch}")
    log_dict(logger, results)
    logger.info("#" * 50)
    return results


def get_args():
    parser = argparse.ArgumentParser(
        description="Train F3 monocular depth using the online_data pipeline"
    )
    parser.add_argument("--conf", type=str, required=True, help="Training config YAML")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the decoder"
    )
    parser.add_argument(
        "--retrain-f3",
        action="store_true",
        help="Fine-tune F3 backbone (default frozen)",
    )
    parser.add_argument(
        "--init", type=str, default=None, help="Initial weights to finetune"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--amp", action="store_true", help="Mixed precision (bf16)")
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.conf, "r") as f:
        conf = yaml.safe_load(f)
    for key, value in conf.items():
        if not hasattr(args, key):
            setattr(args, key, value)
        else:
            raise ValueError(f"Config key '{key}' overrides a command-line arg")

    data_cfg = conf["data"]
    # Sensor UUIDs come from the loader roles (anchor=depth, stream=events) so they
    # are defined in exactly one place (the `online_data` block).
    roles = data_cfg["online_data"]["roles"]
    args.depth_sensor = roles["anchor"][0]
    args.event_sensor = roles["stream"][0]
    args.color_sensor = data_cfg.get("color_sensor")

    setup_torch()

    # Trainer GPU (producers run on their own GPUs in separate processes).
    trainer_gpu = int(data_cfg.get("trainer_gpu", 0))
    torch.cuda.set_device(trainer_gpu)
    args.device = torch.device(f"cuda:{trainer_gpu}")

    if args.name is None:
        args.name = (
            f"f3depth_neurosim_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    base_path = f"outputs/monoculardepth/{args.name}"
    models_path = f"{base_path}/models"
    logger, resume = setup_experiment(args, base_path, models_path)

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Initializing EventFFDepthAnythingV2 model...")
    model = EventFFDepthAnythingV2(
        args.eventff["config"], args.dav2_config, args.retrain_f3
    )
    model.load_eventff_weights(args.eventff["ckpt"])
    model.save_configs(models_path)
    model = model.to(args.device)
    if args.init is not None:
        logger.info(f"Loading initial weights from {args.init}")
        load_depth_weights(model, torch.load(args.init)["model"])
        torch.cuda.empty_cache()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {trainable}")

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
    # DAv2 encoder at lr, F3 backbone at half, DAv2 decoder head at 10x. Grouped
    # rather than one group per parameter, so AdamW's foreach path can batch them.
    def lr_scale(name):
        if "pretrained" in name:
            return 1.0
        return 0.5 if "eventff" in name else 10.0

    def decays(name, param):
        # Norms and biases are 1-D; the re-initialised patch embed is left free too.
        return param.ndim > 1 and "patch_embed.proj" not in name

    grouped: dict[tuple[float, bool], list] = {}
    for name, param in model.named_parameters():
        grouped.setdefault((lr_scale(name), decays(name, param)), []).append(param)
    param_groups = [
        {"params": params, "lr": scale * args.lr, "weight_decay": 0.01 if wd else 0.0}
        for (scale, wd), params in grouped.items()
    ]
    logger.info(
        "Optimizer: %d param groups over %d tensors",
        len(param_groups),
        sum(len(g["params"]) for g in param_groups),
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    # Warm up, hold, then cosine to zero. Held flat rather than decayed throughout
    # because the data never repeats: there is no overfitting to decay away, and the
    # loss is still falling at block resolution well past the midpoint.
    # Stepped once per epoch, so both phases are measured in epochs (~512 steps each).
    phases = [
        torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=args.epochs
        )
    ]
    milestones = []
    if args.warmup_epochs:
        phases.insert(
            0,
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1 / (args.warmup_epochs + 1),
                end_factor=1.0,
                total_iters=args.warmup_epochs,
            ),
        )
        milestones.append(args.warmup_epochs)
    if args.cooldown_epochs:
        phases.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.cooldown_epochs
            )
        )
        milestones.append(args.epochs - args.cooldown_epochs)
    scheduler = (
        torch.optim.lr_scheduler.SequentialLR(optimizer, phases, milestones=milestones)
        if milestones
        else phases[0]
    )
    logger.info(
        "LR schedule: warmup %d, hold %d, cosine cooldown %d (of %d epochs)",
        args.warmup_epochs,
        args.epochs - args.warmup_epochs - args.cooldown_epochs,
        args.cooldown_epochs,
        args.epochs,
    )

    assert args.loss == "ssimae", (
        "ScaleAndShiftInvariantLoss for monocular relative depth"
    )
    loss_fn = ScaleAndShiftInvariantLoss(alpha=args.alpha, scales=args.scales)
    best_results = {k: 100.0 for k in (*METRICS, loss_fn.name)}
    start = 0

    if resume:
        logger.info(f"Resuming from {models_path}/last.pth")
        last_dict = torch.load(f"{models_path}/last.pth")
        load_depth_weights(model, last_dict["model"])
        optimizer.load_state_dict(last_dict["optimizer"])
        scheduler.load_state_dict(last_dict["scheduler"])
        start = last_dict["epoch"] + 1
        del last_dict
        try:
            best_results = torch.load(f"{models_path}/best.pth").get(
                "results", best_results
            )
        except FileNotFoundError:
            logger.info("No best model found; using default best results")
        torch.cuda.empty_cache()

    # Compiled last: OptimizedModule prefixes its state_dict keys, so every load above
    # runs on the plain modules and only the saves carry the prefix.
    model.eventff = torch.compile(model.eventff, fullgraph=False, dynamic=True)
    if args.compile:
        model.dav2 = torch.compile(model.dav2)

    # ── Data loader (built after the model so the event window matches frame T) ─
    # Events ship raw from the loader; process_batch normalizes to the model's
    # frame sizes using args.event_norm = (W, H, window_us).
    event_W, event_H, event_T = model.eventff.frame_sizes
    window_us = data_cfg.get("event_time_window_us", event_T * 1000)
    args.event_norm = (event_W, event_H, window_us)
    logger.info("Initializing OnlineDataLoader (event norm window=%s us)...", window_us)
    dataloader = build_loader(
        data_cfg,
        batch_size=int(args.train["mini_batch"]),
        log_dir=f"{base_path}/logs",
        sample_filter=usable_sample_filter(
            args.depth_sensor,
            args.event_sensor,
            args.max_disparity,
            int(data_cfg.get("min_events_per_sample", 10000)),
        ),
    )

    if args.wandb:
        wandb.init(
            project="f3-depth-neurosim",
            name=args.name,
            id=args.wandb_run_id,
            resume="allow",
            config=vars(args),
        )

    val_results = dict(best_results)
    iters_to_accumulate = args.train["batch"] // args.train["mini_batch"]

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    try:
        for epoch in range(start, args.epochs):
            train_loss = train_epoch(
                args,
                logger,
                model,
                dataloader,
                optimizer,
                scheduler,
                loss_fn,
                epoch,
                args.batches_per_epoch,
                iters_to_accumulate,
            )
            log_wandb(args, {"train/loss": train_loss, "opt/epoch": epoch})

            if (epoch + 1) % args.val_interval == 0:
                save_preds = (epoch + 1) % args.log_interval == 0
                with torch.autocast(
                    device_type="cuda", enabled=args.amp, dtype=torch.bfloat16
                ):
                    val_results = validate(
                        args,
                        logger,
                        model,
                        dataloader,
                        loss_fn,
                        epoch,
                        max_batches=args.batches_per_epoch // 2,
                        save_preds=save_preds,
                    )
                better_ssimae = val_results[loss_fn.name] < best_results[loss_fn.name]
                set_best_results(best_results, val_results)
                if better_ssimae:
                    save_checkpoint(
                        f"{models_path}/best.pth",
                        epoch,
                        best_results,
                        model,
                        optimizer,
                        scheduler,
                    )
                    logger.info(f"Saved best model at epoch {epoch}")
                log_wandb(
                    args,
                    {"opt/epoch": epoch}
                    | {
                        f"val/{k}": (v.item() if isinstance(v, torch.Tensor) else v)
                        for k, v in val_results.items()
                    },
                )

            scheduler.step()
            save_checkpoint(
                f"{models_path}/last.pth",
                epoch,
                val_results,
                model,
                optimizer,
                scheduler,
            )
            if (epoch + 1) % args.log_interval == 0:
                torch.save(
                    uncompiled_state_dict(model),
                    f"{models_path}/checkpoint_{epoch}.pth",
                )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        dataloader.close()
        if args.wandb:
            wandb.finish()
        logger.info("Training completed!")
        log_dict(logger, best_results)


if __name__ == "__main__":
    main()
