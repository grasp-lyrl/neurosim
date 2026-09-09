"""Run scaffolding shared by the training entry points: dirs, logging, checkpoints."""

import logging
import os
from pathlib import Path

import torch
import wandb
import yaml


def setup_torch() -> None:
    """f3's training defaults: fixed seed, tf32 matmuls, dynamo tracing dynamic shapes."""
    torch.manual_seed(403)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.compiled_autograd = True


def setup_experiment(args, base_path: str, models_path: str):
    """Create the run's directories and logger, and say whether it is resuming."""
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

    with open(config_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    return logging.getLogger("f3_depth_training"), resume


def log_dict(logger, values: dict) -> None:
    """One log line of ``k: v``, tensors unwrapped."""
    logger.info(
        ", ".join(
            f"{k}: {v.item():.4f}" if isinstance(v, torch.Tensor) else f"{k}: {v:.4f}"
            for k, v in values.items()
        )
    )


def log_wandb(args, metrics: dict) -> None:
    """Log under the ``train/``, ``val/``, ``opt/`` namespaces."""
    if args.wandb:
        wandb.log(metrics)


def uncompiled_state_dict(model) -> dict:
    """Weights without ``torch.compile``'s ``_orig_mod.`` prefix."""
    return {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}


def save_checkpoint(path, epoch, results, model, optimizer, scheduler) -> None:
    """Write the resumable checkpoint, and the bare weights beside it."""
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
