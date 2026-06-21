"""
Monocular Depth Training with F3 + the online_data pipeline.

Trains a monocular depth model (F3 / EventPatchFF backbone + DepthAnythingV2
decoder) on time-aligned events+depth streamed from the neurosim
``OnlineDataLoader``: ``num_producers`` SynchronousSimulators (events + depth)
run in separate processes on the producer GPUs, push time-aligned samples to a
bounded bus, and this (trainer) process builds batches and trains on the
``trainer_gpu``.

Use ``--smoke-data`` to validate the data path (build the loader, pull a few
batches, print shapes) without the model — handy on a fresh setup.

Reference: https://github.com/grasp-lyrl/fast-feature-fields/tree/main/src/f3/tasks/depth
"""

import os
import cv2
import copy
import yaml
import torch
import logging
import argparse
import datetime
import itertools
import numpy as np
from tqdm import tqdm
from matplotlib import colormaps

# NOTE: f3 (fast-feature-fields) is imported lazily inside the functions that use
# it, so this module imports — and ``--smoke-data`` runs — without f3 installed.
# wandb is optional (only when --wandb is passed).
try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from neurosim.online_data import OnlineDataLoader, SampleSchema


def ev_to_frames_with_polarity(events, counts, w, h):
    """Convert events to RGB frames with polarity coloring (pos=red, neg=blue)."""
    from f3.utils import unnormalize_events

    if events.max() <= 1.0:
        events = unnormalize_events(events, (w, h, 1, 1))

    B = counts.shape[0]

    if isinstance(events, torch.Tensor):
        event_frames = torch.zeros(B, h, w, 3, dtype=torch.uint8).to(events.device)
        c = torch.cumsum(torch.cat((torch.zeros(1).to(counts.device), counts)), 0).to(
            torch.int32
        )
    elif isinstance(events, np.ndarray):
        event_frames = np.zeros((B, h, w, 3), dtype=np.uint8)
        c = np.cumsum(np.concatenate((np.zeros(1), counts))).astype(np.int32)

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
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(f"{base_path}/predictions", exist_ok=True)
    os.makedirs(f"{base_path}/training_events", exist_ok=True)

    resume = os.path.exists(f"{models_path}/last.pth")

    logging.basicConfig(
        filename=f"{base_path}/training.log",
        filemode="a" if resume else "w",
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    with open(f"{base_path}/config.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    return logger, resume


def build_loader(
    data_cfg: dict,
    *,
    batch_size: int,
    log_dir: str | None = None,
) -> tuple[OnlineDataLoader, SampleSchema]:
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

    loader = OnlineDataLoader.from_config(
        {"online_data": od}, batch_size=batch_size, log_dir=log_dir
    )
    return loader, loader.schema


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
    depth = batch[depth_sensor]
    # Invalid depths are 0.0; clip then convert to disparity (inverse depth).
    depth = np.clip(depth, 0.5 / args.max_disparity, 1 / args.min_disparity)
    disparity = torch.from_numpy(1.0 / depth).float().to(device)  # (B, H, W)

    color_images = (
        batch[color_sensor].astype(np.uint8)
        if color_sensor and color_sensor in batch
        else None
    )
    return ff_events, event_counts, disparity, color_images


def train_epoch(
    args,
    logger,
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    scaler,
    epoch,
    max_batches,
    iters_to_accumulate=1,
):
    """Train for one epoch (consumes ``max_batches`` batches from the loader)."""
    from f3.utils import get_random_crop_params, batch_cropper

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
            disparity_valid_mask = disparity < args.max_disparity
            if disparity_valid_mask.sum() < W * H / 2:
                continue
            loss = loss_fn(disparity_pred, disparity, disparity_valid_mask)
            loss = loss / iters_to_accumulate

        scaler.scale(loss).backward()
        train_loss += loss.item()
        iter_loss += loss.item()

        if (idx + 1) % iters_to_accumulate == 0:
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(
                {"loss": f"{iter_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
            )
            if args.wandb:
                wandb.log(
                    {
                        "train_iter_loss": iter_loss,
                        "train_lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "iteration": idx,
                    }
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
    """Validate the model on ``max_batches`` batches."""
    from f3.utils import log_dict, batch_cropper
    from f3.tasks.depth.utils import eval_disparity, get_disparity_image

    model.eval()
    cmap = colormaps["magma"]

    results = {
        k: torch.tensor([0.0]).cuda()
        for k in (
            "1pe",
            "2pe",
            "3pe",
            "rmse",
            "rmse_log",
            "log10",
            "silog",
            loss_fn.name,
        )
    }
    nsamples = torch.tensor([0.0]).cuda()

    for idx, batch in tqdm(enumerate(dataloader), total=max_batches, desc="Validation"):
        if idx >= max_batches:
            break

        ff_events, event_counts, disparity, color_images = process_batch(
            batch, args, args.device
        )
        B, H, W = disparity.shape
        crop_params = torch.tensor(
            [[0, (W - H) // 2, H, (W + H) // 2]],
            dtype=torch.int32,
            device=ff_events.device,
        ).repeat(B, 1)
        disparity = batch_cropper(disparity.unsqueeze(1), crop_params).squeeze(1)

        disparity_pred = model(ff_events, event_counts, crop_params)[0]  # (B, H, H)
        valid_mask = disparity < args.max_disparity
        if valid_mask.sum() < W * H / 2:
            continue

        cur_results = eval_disparity(disparity_pred[valid_mask], disparity[valid_mask])
        for k in cur_results:
            results[k] += cur_results[k]
        results[loss_fn.name] += loss_fn(disparity_pred, disparity, valid_mask).item()
        nsamples += 1

        if idx % 10 == 0 and save_preds:
            base_path = f"outputs/monoculardepth/{args.name}"
            event_frames = (
                ev_to_frames_with_polarity(ff_events, event_counts, W, H).cpu().numpy()
            )
            for i in range(disparity_pred.shape[0]):
                disp_img = get_disparity_image(disparity[i], valid_mask[i], cmap)
                pred_img = get_disparity_image(
                    disparity_pred[i],
                    torch.ones_like(disparity_pred[i], dtype=torch.bool),
                    cmap,
                )
                cv2.imwrite(
                    f"{base_path}/training_events/disparity_{epoch}_{idx}_{i}.png",
                    disp_img,
                )
                cv2.imwrite(
                    f"{base_path}/predictions/disparity_pred_{epoch}_{idx}_{i}.png",
                    pred_img,
                )
                cv2.imwrite(
                    f"{base_path}/training_events/events_{epoch}_{idx}_{i}.png",
                    event_frames[i],
                )
                if color_images is not None:
                    cv2.imwrite(
                        f"{base_path}/training_events/color_{epoch}_{idx}_{i}.png",
                        cv2.cvtColor(color_images[i], cv2.COLOR_RGB2BGR),
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
    parser.add_argument(
        "--smoke-data",
        action="store_true",
        help="Build the loader, pull a few batches, print shapes, and exit (no model).",
    )
    parser.add_argument("--smoke-batches", type=int, default=3)
    return parser.parse_args()


def _smoke_data(args, data_cfg, logger=None):
    """Validate the data path without the model: pull a few batches, log shapes."""
    log = logger or logging.getLogger("smoke")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    batch_size = int(args.train["mini_batch"]) if hasattr(args, "train") else 4
    loader, schema = build_loader(data_cfg, batch_size=batch_size)
    log.info("smoke: deliver=%s, batch_size=%d", schema.deliver_uuids(), batch_size)
    try:
        for i, batch in enumerate(itertools.islice(loader, args.smoke_batches)):
            depth = batch[args.depth_sensor]
            counts, events = batch[args.event_sensor]
            log.info(
                "batch %d: depth=%s events=%s counts.sum=%d spec_ids=%s",
                i,
                tuple(depth.shape),
                tuple(events.shape),
                int(counts.sum()),
                sorted(set(batch.meta.spec_id.tolist())),
            )
    finally:
        loader.close()
    log.info("smoke: OK")


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

    if args.smoke_data:
        _smoke_data(args, data_cfg)
        return

    from f3.utils import num_params, setup_torch, log_dict
    from f3.tasks.depth.utils import (
        EventFFDepthAnythingV2,
        ScaleAndShiftInvariantLoss,
        set_best_results,
    )

    setup_torch(cudnn_benchmark=True)

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
    model_uncompiled = model
    model.eventff = torch.compile(model.eventff, fullgraph=False)
    if args.compile:
        model.dav2 = torch.compile(model.dav2)
    model.load_weights(args.eventff["ckpt"])
    model.save_configs(models_path)
    if args.init is not None:
        logger.info(f"Loading initial weights from {args.init}")
        model.load_state_dict(torch.load(args.init)["model"])
        torch.cuda.empty_cache()
    logger.info(f"Total trainable parameters: {num_params(model)}")

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
    param_groups = []
    for name, param in model.named_parameters():
        if "pretrained" in name:
            if "patch_embed.proj" in name:
                param_groups.append(
                    {"params": param, "lr": args.lr, "weight_decay": 0.0}
                )
            else:
                param_groups.append({"params": param, "lr": args.lr})
        elif "eventff" in name:
            param_groups.append({"params": param, "lr": 0.5 * args.lr})
        else:
            param_groups.append({"params": param, "lr": 10 * args.lr})
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=args.lr_end_factor,
        total_iters=args.epochs,
    )

    assert args.loss == "ssimae", (
        "ScaleAndShiftInvariantLoss for monocular relative depth"
    )
    loss_fn = ScaleAndShiftInvariantLoss(alpha=args.alpha, scales=args.scales)
    best_results = {
        k: 100.0
        for k in (
            "1pe",
            "2pe",
            "3pe",
            "rmse",
            "rmse_log",
            "log10",
            "silog",
            loss_fn.name,
        )
    }
    start = 0

    if resume:
        logger.info(f"Resuming from {models_path}/last.pth")
        last_dict = torch.load(f"{models_path}/last.pth")
        model.load_state_dict(last_dict["model"])
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

    # ── Data loader (built after the model so the event window matches frame T) ─
    # Events ship raw from the loader; process_batch normalizes to the model's
    # frame sizes using args.event_norm = (W, H, window_us).
    event_W, event_H, event_T = model.eventff.frame_sizes
    window_us = data_cfg.get("event_time_window_us", event_T * 1000)
    args.event_norm = (event_W, event_H, window_us)
    logger.info("Initializing OnlineDataLoader (event norm window=%s us)...", window_us)
    dataloader, _ = build_loader(
        data_cfg,
        batch_size=int(args.train["mini_batch"]),
        log_dir=f"{base_path}/logs",
    )

    if args.wandb:
        wandb.init(project="f3-depth-neurosim", name=args.name, config=vars(args))

    val_results = copy.deepcopy(best_results)
    iters_to_accumulate = args.train["batch"] // args.train["mini_batch"]
    scaler = torch.GradScaler(enabled=args.amp)

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
                scaler,
                epoch,
                args.batches_per_epoch,
                iters_to_accumulate,
            )
            if args.wandb:
                wandb.log({"train_loss": train_loss, "epoch": epoch})

            if (epoch + 1) % args.val_interval == 0:
                save_preds = (epoch + 1) % args.log_interval == 0
                with torch.autocast(
                    device_type="cuda", enabled=args.amp, dtype=torch.float16
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
                    torch.save(
                        {
                            "epoch": epoch,
                            "results": best_results,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        },
                        f"{models_path}/best.pth",
                    )
                    logger.info(f"Saved best model at epoch {epoch}")
                if args.wandb:
                    wandb.log(
                        {
                            f"val_{k}": (v.item() if isinstance(v, torch.Tensor) else v)
                            for k, v in val_results.items()
                        }
                        | {"epoch": epoch}
                    )

            scheduler.step()
            torch.save(
                {
                    "epoch": epoch,
                    "results": val_results,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                f"{models_path}/last.pth",
            )
            if (epoch + 1) % args.log_interval == 0:
                torch.save(
                    model_uncompiled.state_dict(),
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
