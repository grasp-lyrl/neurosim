"""
Monocular Depth Training with F3 + the online_data pipeline.

Trains a monocular depth model (F3 / EventPatchFF backbone + DepthAnythingV2
decoder) on time-aligned events+depth streamed from the neurosim ``OnlineDataLoader``.

Reference: https://github.com/grasp-lyrl/fast-feature-fields/tree/main/src/f3/tasks/depth
"""

import argparse
import datetime

import cv2
import torch
import torch.nn.functional as F
import wandb
import yaml
from matplotlib import colormaps
from tqdm import tqdm

from .data import (
    build_m3ed_loader,
    build_online_loader,
    evaluate_m3ed,
    process_batch,
    usable_sample_filter,
    usable_samples,
)
from .nets import (
    EventFFDepthAnythingV2,
    batch_cropper,
    get_resize_shapes,
    load_depth_weights,
)
from .utils.experiment import (
    log_dict,
    log_wandb,
    save_checkpoint,
    setup_experiment,
    setup_torch,
    uncompiled_state_dict,
)
from .utils import (
    HIGHER_IS_BETTER,
    ScaleAndShiftInvariantLoss,
    build_optimizer,
    build_scheduler,
    ev_to_frames_with_polarity,
    eval_relative_depth,
    get_disparity_image,
    get_random_crop_params,
    improved,
    set_best_results,
)

# Metrics from eval_relative_depth; the loss name is appended per run.
METRICS = ("abs_rel", "sq_rel", "d1", "d2", "d3", "rmse", "rmse_log", "log10", "silog")


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
def make_validator(args, logger, model, dataloader, loss_fn, data_cfg, frame):
    """Resolve `validation.source` once into the callable the training loop calls."""
    source = getattr(args, "validation", {}).get("source", "simulator")

    if source == "m3ed":
        loader = build_m3ed_loader(data_cfg.get("m3ed", {}), *frame)
        assert loader is not None, "validation.source is m3ed but data.m3ed is empty"

        def run(epoch):
            results = evaluate_m3ed(
                model,
                loader,
                predict_full_frame,
                args.device,
                args.min_disparity,
                loss_fn,
            )
            logger.info("Validation (M3ED): Epoch: %d", epoch)
            log_dict(logger, results)
            return results

    elif source == "simulator":

        def run(epoch):
            return validate(
                args,
                logger,
                model,
                dataloader,
                loss_fn,
                epoch,
                max_batches=args.batches_per_epoch // 2,
                save_preds=(epoch + 1) % args.log_interval == 0,
            )

    else:
        raise ValueError(f"validation.source is {source!r}, not 'simulator' or 'm3ed'")

    return run


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

        cur_results = eval_relative_depth(
            disparity_pred[keep], disparity[keep], kept_mask, args.min_disparity
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

    optimizer = build_optimizer(model, args.lr)
    scheduler = build_scheduler(
        optimizer, args.epochs, args.warmup_epochs, args.cooldown_epochs
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
    best_results = {
        k: 0.0 if k in HIGHER_IS_BETTER else 100.0 for k in (*METRICS, loss_fn.name)
    }
    start = 0

    if resume:
        logger.info(f"Resuming from {models_path}/last.pth")
        last_dict = torch.load(f"{models_path}/last.pth")
        load_depth_weights(model, last_dict["model"])
        optimizer.load_state_dict(last_dict["optimizer"])
        scheduler.load_state_dict(last_dict["scheduler"])
        start = last_dict["epoch"] + 1
        saved = last_dict.get("results", {})
        best_results.update({k: v for k, v in saved.items() if k in best_results})
        torch.cuda.empty_cache()

    # Compiled last: OptimizedModule prefixes its state_dict keys, so every load above
    # runs on the plain modules and only the saves carry the prefix.
    model.eventff = torch.compile(model.eventff, fullgraph=False, dynamic=True)
    if args.compile:
        model.dav2 = torch.compile(model.dav2)

    # ── Data loader (built after the model so the event window matches frame T) ─
    event_W, event_H, event_T = model.eventff.frame_sizes
    window_us = data_cfg.get("event_time_window_us", event_T * 1000)
    args.event_norm = (event_W, event_H, window_us)
    logger.info("Initializing OnlineDataLoader (event norm window=%s us)...", window_us)
    dataloader = build_online_loader(
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

    track_best = getattr(args, "validation", {}).get("track_best", [loss_fn.name])
    run_validation = make_validator(
        args, logger, model, dataloader, loss_fn, data_cfg, (event_W, event_H, event_T)
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
                with torch.autocast(
                    device_type="cuda", enabled=args.amp, dtype=torch.bfloat16
                ):
                    val_results = run_validation(epoch)

                beaten = [
                    m
                    for m in track_best
                    if improved(m, val_results[m], best_results[m])
                ]
                set_best_results(best_results, val_results)
                for metric in beaten:
                    save_checkpoint(
                        f"{models_path}/best_{metric}.pth",
                        epoch,
                        best_results,
                        model,
                        optimizer,
                        scheduler,
                    )
                    logger.info("Saved best_%s at epoch %d", metric, epoch)
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
                best_results,
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
