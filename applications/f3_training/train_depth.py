"""
Monocular Depth Training Script with F3 and OnlineDataLoader

This script trains a monocular depth model using F3 (EventPatchFF) as the backbone
and DepthAnythingV2 as the depth decoder. Data is streamed from the neurosim
OnlineDataLoader.

The F3 backbone extracts dense feature representations from events, which are
then processed by DepthAnythingV2 to predict monocular depth.

Reference: https://github.com/grasp-lyrl/fast-feature-fields/tree/main/src/f3/tasks/depth
"""

import os
import cv2
import copy
import yaml
import time
import torch
import wandb
import logging
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from matplotlib import colormaps

from f3.utils import (
    num_params,
    setup_torch,
    log_dict,
    get_random_crop_params,
    batch_cropper,
    unnormalize_events,
)
from f3.tasks.depth.utils import (
    EventFFDepthAnythingV2,
    ScaleAndShiftInvariantLoss,
    eval_disparity,
    get_disparity_image,
    set_best_results,
)

# Import from neurosim
from neurosim.sims.online_dataloader import OnlineDataLoader, DatasetConfig


def ev_to_frames_with_polarity(events, counts, w, h):
    """
    Converts events to RGB frames with polarity-based coloring.
    Positive polarity (1) = Red, Negative polarity (0) = Blue

    events: (N, 4) where events[:, 3] is polarity (0 or 1)
    counts: (B,) event counts per batch

    Returns: (B, H, W, 3) RGB tensor (uint8)
    """
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

        # Positive polarity (1) -> Red (255, 0, 0)
        # Negative polarity (0) -> Blue (0, 0, 255)
        pos_mask = polarities == 1
        neg_mask = polarities == 0
        event_frames[i, y_coords[pos_mask], x_coords[pos_mask], 0] = 255  # Red channel
        event_frames[i, y_coords[neg_mask], x_coords[neg_mask], 2] = 255  # Blue channel

    return event_frames


def setup_experiment(
    args, base_path: str, models_path: str
) -> tuple[logging.Logger, bool]:
    """Setup experiment directories and check for resume."""
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(f"{base_path}/predictions", exist_ok=True)
    os.makedirs(f"{base_path}/training_events", exist_ok=True)

    # Check if we should resume
    resume = os.path.exists(f"{models_path}/last.pth")

    # Setup logging
    logging.basicConfig(
        filename=f"{base_path}/training.log",
        filemode="a" if resume else "w",
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Save config
    with open(f"{base_path}/config.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    return logger, resume


def process_batch(
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray | None]:
    """
    Process a batch from the OnlineDataLoader.

    The OnlineDataLoader provides:
    - Event sensor: (counts, events) tuple where:
        - counts: (B,) array of event counts per sample
        - events: (N_total, 4) array of events [x, y, t, p]
    - Depth sensor: (B, H, W) depth images
    - Color camera: (B, H, W, 3) RGB images

    Events are assumed to already be normalized by the simulator.

    Args:
        batch: Dictionary from OnlineDataLoader
        args: Training arguments
        device: Target device

    Returns:
        (ff_events, event_counts, disparity, color_images)
    """
    event_sensor = args.event_sensor
    depth_sensor = args.depth_sensor
    color_sensor = args.color_sensor

    # Get event data
    if event_sensor not in batch:
        raise ValueError(
            f"Event sensor '{event_sensor}' not found in batch. Available: {list(batch.keys())}"
        )

    counts, events = batch[event_sensor]

    # Convert to torch tensors
    ff_events = torch.from_numpy(events).float().to(device)  # (N, 4)
    event_counts = torch.from_numpy(counts).to(device)  # (B,)

    # Get depth data
    if depth_sensor not in batch:
        raise ValueError(
            f"Depth sensor '{depth_sensor}' not found in batch. Available: {list(batch.keys())}"
        )

    depth = batch[depth_sensor]
    # Invalid depths are marked as 0.0

    # Convert depth to disparity (inverse depth)
    # Avoid division by zero
    depth = np.clip(
        depth, 0.5 / args.max_disparity, 1 / args.min_disparity
    )  # (B, H, W)
    disparity = (
        1.0 / depth
    )  # Convert to disparity 0.05 -> 2 * max_disparity, just for safety
    disparity = torch.from_numpy(disparity).float().to(device)  # (B, H, W)

    # Get color camera data
    color_images = (
        batch[color_sensor].astype(np.uint8)  # (B, H, W, 3)
        if color_sensor
        else None
    )

    return ff_events, event_counts, disparity, color_images


def train_epoch(
    args: argparse.Namespace,
    logger: logging.Logger,
    model: torch.nn.Module,
    dataloader: OnlineDataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: torch.nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    max_batches: int,
    iters_to_accumulate: int = 1,
) -> float:
    """
    Train for one epoch.

    Args:
        args: Training arguments
        model: EventFFDepthAnythingV2 model
        dataloader: OnlineDataLoader instance
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function (ScaleAndShiftInvariantLoss)
        epoch: Current epoch
        scaler: Gradient scaler for AMP
        max_batches: Maximum batches per epoch
        iters_to_accumulate: Gradient accumulation steps

    Returns:
        Average training loss
    """
    model.train()
    train_loss = 0.0
    iter_loss = 0.0

    # Timer-based profiling accumulators
    epoch_start_time = time.perf_counter()
    prev_iter_end_time = None
    total_events = 0
    total_event_bytes = 0
    total_data_wait_time = 0.0
    total_batch_prep_time = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_optimizer_time = 0.0
    profiled_steps = 0

    pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=max_batches)

    for idx, batch in pbar:
        if idx >= max_batches:
            break

        iter_start_time = time.perf_counter()
        if prev_iter_end_time is not None:
            total_data_wait_time += max(0.0, iter_start_time - prev_iter_end_time)

        # Process batch
        prep_t0 = time.perf_counter()
        ff_events, event_counts, disparity, _ = process_batch(
            batch, args, torch.device("cuda")
        )
        total_batch_prep_time += time.perf_counter() - prep_t0

        num_events = int(ff_events.shape[0])
        total_events += num_events
        total_event_bytes += num_events * ff_events.shape[1] * ff_events.element_size()

        B, H, W = disparity.shape
        cparams = get_random_crop_params((H, W), (H, H), batch_size=B).to(
            ff_events.device
        )
        disparity = batch_cropper(disparity.unsqueeze(1), cparams).squeeze(1)

        # Forward pass with AMP
        if args.profile_timers:
            torch.cuda.synchronize()
            fw_t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", enabled=args.amp, dtype=torch.bfloat16):
            disparity_pred = model(ff_events, event_counts, cparams)[0]  # (B, H, W)

            # Mask invalid disparities
            disparity_valid_mask = disparity < args.max_disparity

            if (
                disparity_valid_mask.sum() < W * H / 2
            ):  # at least half of the disparities need to be valid
                continue

            loss = loss_fn(disparity_pred, disparity, disparity_valid_mask)
            loss = loss / iters_to_accumulate
        if args.profile_timers:
            torch.cuda.synchronize()
            total_forward_time += time.perf_counter() - fw_t0

        # Backward pass with GradScaler
        if args.profile_timers:
            torch.cuda.synchronize()
            bw_t0 = time.perf_counter()
        scaler.scale(loss).backward()
        if args.profile_timers:
            torch.cuda.synchronize()
            total_backward_time += time.perf_counter() - bw_t0
            profiled_steps += 1

        train_loss += loss.item()
        iter_loss += loss.item()

        # Gradient accumulation step
        if (idx + 1) % iters_to_accumulate == 0:
            # Unscale before clipping when using AMP
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            # Optimizer step via scaler
            if args.profile_timers:
                torch.cuda.synchronize()
                opt_t0 = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if args.profile_timers:
                torch.cuda.synchronize()
                total_optimizer_time += time.perf_counter() - opt_t0

            pbar.set_postfix(
                {
                    "loss": f"{iter_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
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

        # Periodic profiling log
        if args.profile_timers and (idx + 1) % args.profile_log_interval == 0:
            elapsed = max(1e-6, time.perf_counter() - epoch_start_time)
            events_per_sec = total_events / elapsed
            bytes_per_event = (
                ff_events.shape[1] * ff_events.element_size()
            )  # usually 16B
            mb_per_sec = events_per_sec * bytes_per_event / (1024**2)
            gb_per_hour = mb_per_sec * 3600 / 1024
            wait_frac = total_data_wait_time / elapsed
            gpu_busy_frac = (
                total_forward_time + total_backward_time + total_optimizer_time
            ) / elapsed
            logger.info(
                "[PROFILE][Epoch %d][Batch %d] events/s=%.2fM, data_rate=%.2f MB/s, "
                "disk_est=%.2f GB/hr, wait=%.1f%%, gpu_busy~=%.1f%%",
                epoch,
                idx + 1,
                events_per_sec / 1e6,
                mb_per_sec,
                gb_per_hour,
                100.0 * wait_frac,
                100.0 * gpu_busy_frac,
            )

        prev_iter_end_time = time.perf_counter()

    # Normalize loss
    num_iters = (idx + 1) // iters_to_accumulate
    train_loss /= num_iters

    # End-of-epoch profiling summary
    if args.profile_timers:
        elapsed = max(1e-6, time.perf_counter() - epoch_start_time)
        events_per_sec = total_events / elapsed
        bytes_per_event = (
            (total_event_bytes / total_events) if total_events > 0 else 16.0
        )
        mb_per_sec = events_per_sec * bytes_per_event / (1024**2)
        gb_per_hour = mb_per_sec * 3600 / 1024
        wait_frac = total_data_wait_time / elapsed
        gpu_busy_frac = (
            total_forward_time + total_backward_time + total_optimizer_time
        ) / elapsed

        logger.info("[PROFILE][Epoch %d Summary]", epoch)
        logger.info(
            "[PROFILE] events=%d, elapsed=%.2fs, events/s=%.2fM",
            total_events,
            elapsed,
            events_per_sec / 1e6,
        )
        logger.info(
            "[PROFILE] bytes/event=%.1f, stream_rate=%.2f MB/s, disk_est=%.2f GB/hr",
            bytes_per_event,
            mb_per_sec,
            gb_per_hour,
        )
        logger.info(
            "[PROFILE] time_breakdown: dataloader_wait=%.1f%%, batch_prep=%.1f%%, "
            "forward=%.1f%%, backward=%.1f%%, optimizer=%.1f%%, gpu_busy~=%.1f%%",
            100.0 * wait_frac,
            100.0 * (total_batch_prep_time / elapsed),
            100.0 * (total_forward_time / elapsed),
            100.0 * (total_backward_time / elapsed),
            100.0 * (total_optimizer_time / elapsed),
            100.0 * gpu_busy_frac,
        )

    logger.info("#" * 50)
    logger.info(f"Training: Epoch: {epoch}, Loss: {train_loss:.4f}")
    logger.info("#" * 50)

    return train_loss


@torch.no_grad()
def validate(
    args: argparse.Namespace,
    logger: logging.Logger,
    model: torch.nn.Module,
    dataloader: OnlineDataLoader,
    loss_fn: torch.nn.Module,
    epoch: int,
    max_batches: int = 50,
    save_preds: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Validate the model.

    Args:
        args: Training arguments
        model: EventFFDepthAnythingV2 model
        dataloader: OnlineDataLoader instance
        loss_fn: Loss function
        epoch: Current epoch
        max_batches: Maximum batches for validation
        save_preds: Whether to save prediction visualizations

    Returns:
        Dictionary of validation results
    """
    model.eval()

    cmap = colormaps["magma"]

    results = {
        "1pe": torch.tensor([0.0]).cuda(),
        "2pe": torch.tensor([0.0]).cuda(),
        "3pe": torch.tensor([0.0]).cuda(),
        "rmse": torch.tensor([0.0]).cuda(),
        "rmse_log": torch.tensor([0.0]).cuda(),
        "log10": torch.tensor([0.0]).cuda(),
        "silog": torch.tensor([0.0]).cuda(),
        loss_fn.name: torch.tensor([0.0]).cuda(),
    }
    nsamples = torch.tensor([0.0]).cuda()

    for idx, batch in tqdm(enumerate(dataloader), total=max_batches, desc="Validation"):
        if idx >= max_batches:
            break

        ff_events, event_counts, disparity, color_images = process_batch(
            batch, args, torch.device("cuda")
        )

        B, H, W = disparity.shape

        # Crop parameters. Center crop to square.
        # TODO: Handle non-square images better. This is an issue in F3 codebase
        crop_params = torch.tensor(
            [[0, (W - H) // 2, H, (W + H) // 2]],
            dtype=torch.int32,
            device=ff_events.device,
        ).repeat(B, 1)

        # Apply cropping
        disparity = batch_cropper(disparity.unsqueeze(1), crop_params).squeeze(1)

        # Inference
        disparity_pred = model(ff_events, event_counts, crop_params)[0]  # (B, H, H)

        valid_mask = disparity < args.max_disparity
        if (
            valid_mask.sum() < W * H / 2
        ):  # at least half of the disparities need to be valid
            continue

        # Evaluate
        cur_results = eval_disparity(disparity_pred[valid_mask], disparity[valid_mask])

        for k in cur_results:
            results[k] += cur_results[k]

        results[loss_fn.name] += loss_fn(disparity_pred, disparity, valid_mask).item()
        nsamples += 1

        # Save visualizations
        if idx % 10 == 0 and save_preds:
            base_path = f"outputs/monoculardepth/{args.name}"

            # Create event visualizations with polarity coloring
            event_frames_polarity = (
                ev_to_frames_with_polarity(ff_events, event_counts, W, H).cpu().numpy()
            )  # (B, H, W, 3)

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
                    event_frames_polarity[i],
                )

                if color_images is not None:
                    cv2.imwrite(
                        f"{base_path}/training_events/color_{epoch}_{idx}_{i}.png",
                        cv2.cvtColor(color_images[i], cv2.COLOR_RGB2BGR),
                    )

    # Average results
    for k in results:
        results[k] /= nsamples

    logger.info("#" * 50)
    logger.info(f"Validation: Epoch: {epoch}")
    log_dict(logger, results)
    logger.info("#" * 50)

    return results


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Monocular Depth model with F3 backbone using OnlineDataLoader"
    )

    # Config files
    parser.add_argument(
        "--conf", type=str, required=True, help="Path to training configuration YAML"
    )

    # Training options
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile for faster training"
    )
    parser.add_argument(
        "--retrain-f3",
        action="store_true",
        help="Allow F3 backbone to be fine-tuned (default: frozen)",
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        help="Path to initial weights for the model (to finetune)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--batches-per-epoch", type=int, default=100, help="Number of batches per epoch"
    )

    # Mixed precision (AMP)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision training (fp16) with GradScaler",
    )
    parser.add_argument(
        "--profile-timers",
        action="store_true",
        help="Enable timer-based profiling logs (events/s, disk estimate, GPU/data wait)",
    )
    parser.add_argument(
        "--profile-log-interval",
        type=int,
        default=20,
        help="How often (in batches) to log profiling stats",
    )

    args = parser.parse_args()
    return args


def main():
    """Main training function."""
    # ═══════════════════════════════════════════════════════════════════════════════
    # CONFIGURATION & SETUP
    # ═══════════════════════════════════════════════════════════════════════════════

    args = get_args()

    # Load and merge training configuration
    with open(args.conf, "r") as f:
        conf = yaml.safe_load(f)

    for key, value in conf.items():
        if not hasattr(args, key):
            setattr(args, key, value)
        else:
            raise ValueError(f"Argument '{key}' in config overrides command-line arg")

    # Extract data loader settings
    dl_conf = conf["dataloader"]
    args.event_sensor = dl_conf["event_sensor"]
    args.depth_sensor = dl_conf["depth_sensor"]
    args.color_sensor = dl_conf.get("color_sensor", None)

    # Setup PyTorch and create experiment directories
    setup_torch(cudnn_benchmark=True)

    if args.name is None:
        args.name = (
            f"f3depth_neurosim_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

    base_path = f"outputs/monoculardepth/{args.name}"
    models_path = f"outputs/monoculardepth/{args.name}/models"

    logger, resume = setup_experiment(args, base_path, models_path)

    # ═══════════════════════════════════════════════════════════════════════════════
    # MODEL INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════════

    logger.info("#" * 50)
    logger.info("Initializing EventFFDepthAnythingV2 model...")

    model = EventFFDepthAnythingV2(
        args.eventff["config"], args.dav2_config, args.retrain_f3
    )
    model_uncompiled = model  # Keep reference for saving uncompiled weights

    # Optional: Compile model for faster training
    model.eventff = torch.compile(model.eventff, fullgraph=False)
    if args.compile:
        model.dav2 = torch.compile(model.dav2)

    # Load pretrained F3 backbone weights
    model.load_weights(args.eventff["ckpt"])
    model.save_configs(models_path)

    # Optional: Load additional initialization weights for fine-tuning
    if args.init is not None:
        logger.info(f"Loading initial weights from {args.init}")
        state_dict = torch.load(args.init)
        model.load_state_dict(state_dict["model"])
        torch.cuda.empty_cache()
        logger.info("Initial weights loaded successfully.")
    else:
        logger.info("No initial weights provided, starting from scratch.")

    logger.info(f"Feature Field + Monocular Depth: {model}")
    logger.info(f"Trainable parameters in Depth Anything V2: {num_params(model.dav2)}")
    logger.info(f"Total Trainable parameters: {num_params(model)}")
    logger.info("#" * 50)

    # ═══════════════════════════════════════════════════════════════════════════════
    # OPTIMIZER & SCHEDULER SETUP
    # ═══════════════════════════════════════════════════════════════════════════════

    # Create parameter groups with different learning rates:
    # - Pretrained layers: standard LR
    # - Patch embedding (first layer): standard LR, no weight decay (adapts RGB→events)
    # - F3 backbone: 0.5x LR (fine-tuning or frozen)
    # - Decoder head: 10x LR (trained from scratch)
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

    # ═══════════════════════════════════════════════════════════════════════════════
    # LOSS FUNCTION & METRICS TRACKING
    # ═══════════════════════════════════════════════════════════════════════════════

    assert args.loss == "ssimae", (
        "ScaleAndShiftInvariantLoss for monocular relative depth"
    )
    loss_fn = ScaleAndShiftInvariantLoss(alpha=args.alpha, scales=args.scales)

    # Initialize best results tracker (lower is better for all metrics)
    best_results = {
        "1pe": 100.0,
        "2pe": 100.0,
        "3pe": 100.0,
        "rmse": 100.0,
        "rmse_log": 100.0,
        "log10": 100.0,
        "silog": 100.0,
        loss_fn.name: 100.0,
    }

    start = 0

    # ═══════════════════════════════════════════════════════════════════════════════
    # RESUME FROM CHECKPOINT
    # ═══════════════════════════════════════════════════════════════════════════════

    if resume:
        logger.info(f"Resuming from {models_path}/last.pth")
        last_dict = torch.load(f"{models_path}/last.pth")
        model.load_state_dict(last_dict["model"])
        optimizer.load_state_dict(last_dict["optimizer"])
        scheduler.load_state_dict(last_dict["scheduler"])
        start = last_dict["epoch"] + 1
        del last_dict

        try:
            best_dict = torch.load(f"{models_path}/best.pth")
            best_results = best_dict.get("results", best_results)
            del best_dict
        except FileNotFoundError:
            logger.info("No best model found, so falling back on default best results")

        torch.cuda.empty_cache()
        logger.info(f"Resuming from epoch: {start}")
        log_dict(logger, best_results)

    # ═══════════════════════════════════════════════════════════════════════════════
    # DATALOADER INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════════

    logger.info("Initializing OnlineDataLoader...")

    # Load dataset configuration from YAML
    if (dataset_config_path := dl_conf.get("dataset_config")) is None:
        raise ValueError(
            "dataloader.dataset_config must be specified in training config"
        )

    logger.info(f"Loading dataset config from: {dataset_config_path}")
    dataset_config = DatasetConfig.from_yaml(dataset_config_path)

    # Configure sensors and batch sizes
    sensor_uuids = [args.event_sensor, args.depth_sensor, args.color_sensor]
    sensor_batch_sizes = {
        sensor_uuids[0]: args.train["mini_batch"],
        sensor_uuids[1]: args.train["mini_batch"],
        sensor_uuids[2]: args.train["mini_batch"],
    }

    # Create dataloader (starts subscriber process automatically)
    event_W, event_H, event_T = model.eventff.frame_sizes
    dataloader = OnlineDataLoader(
        config=dataset_config,
        sensor_uuids=sensor_uuids,
        sensor_batch_sizes=sensor_batch_sizes,
        ipc_sub_addr=dataset_config.ipc_pub_addr,
        queue_maxsize=dl_conf.get("queue_maxsize", 1000),
        max_events=dl_conf.get("max_events", 8_000_000),
        prefetch_factor=dl_conf.get("prefetch_factor", 8),
        verbose=True,
        batch_building_config={
            "normalize_events": True,
            "event_width": event_W,
            "event_height": event_H,
            "event_time_window": event_T * 1000,  # in us
        },
    )

    logger.info(f"  Event sensor: {dl_conf['event_sensor']}")
    logger.info(f"  Depth sensor: {dl_conf['depth_sensor']}")
    logger.info(f"  Color sensor: {args.color_sensor}")
    logger.info(f"  Mini batch size: {args.train['mini_batch']}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # EXPERIMENT TRACKING
    # ═══════════════════════════════════════════════════════════════════════════════

    if args.wandb:
        wandb.init(project="f3-depth-neurosim", name=args.name, config=vars(args))

    # ═══════════════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════════════

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    val_results = copy.deepcopy(best_results)
    iters_to_accumulate = args.train["batch"] // args.train["mini_batch"]
    # Initialize GradScaler for AMP
    scaler = torch.GradScaler(enabled=args.amp)

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
                iters_to_accumulate=iters_to_accumulate,
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

                # Check for improvements
                better_ssimae = val_results[loss_fn.name] < best_results[loss_fn.name]
                better_2pe = val_results["2pe"] < best_results["2pe"]

                set_best_results(best_results, val_results)

                # Save best model based on loss
                if better_ssimae:
                    best_dict = {
                        "epoch": epoch,
                        "results": best_results,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    torch.save(best_dict, f"{models_path}/best.pth")
                    if args.compile:
                        torch.save(
                            model_uncompiled.state_dict(),
                            f"{models_path}/best_uncompiled.pth",
                        )
                    logger.info(f"Saving best model at epoch: {epoch}")
                    log_dict(logger, best_results)

                # Save best model based on 2-pixel error
                if better_2pe:
                    best_2pe_dict = {
                        "epoch": epoch,
                        "results": best_results,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    torch.save(best_2pe_dict, f"{models_path}/best_2pe.pth")
                    if args.compile:
                        torch.save(
                            model_uncompiled.state_dict(),
                            f"{models_path}/best_2pe_uncompiled.pth",
                        )
                    logger.info(f"Saving best 2pe model at epoch: {epoch}")

                if args.wandb:
                    wandb_dict = {
                        f"val_{k}": v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in val_results.items()
                    }
                    wandb_dict["epoch"] = epoch
                    wandb.log(wandb_dict)

            # ─────────────────────────────────────────────────────────────────────
            # Checkpoint Saving
            # ─────────────────────────────────────────────────────────────────────
            scheduler.step()

            # Save latest checkpoint (every epoch)
            last_dict = {
                "epoch": epoch,
                "results": val_results,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(last_dict, f"{models_path}/last.pth")
            if args.compile:
                torch.save(
                    model_uncompiled.state_dict(), f"{models_path}/last_uncompiled.pth"
                )

            # Save periodic checkpoint (for recovery/analysis)
            if (epoch + 1) % args.log_interval == 0:
                torch.save(last_dict, f"{models_path}/checkpoint_{epoch}.pth")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    finally:
        dataloader.close()
        if args.wandb:
            wandb.finish()

        logger.info("Training completed!")
        logger.info("Best results:")
        log_dict(logger, best_results)


if __name__ == "__main__":
    main()
