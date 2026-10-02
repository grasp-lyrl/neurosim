"""The optimizer groups and LR schedule both depth trainers use."""

import torch


def build_optimizer(model, lr: float) -> torch.optim.Optimizer:
    """AdamW at `lr`, 10x for DAv2's head and the age table."""

    def lr_scale(name: str) -> float:
        retargets = "depth_head" in name or "multi_hash_encoder.table" in name
        return 10.0 if retargets else 1.0

    def decays(name: str, param) -> bool:
        return param.ndim > 1 and "patch_embed.proj" not in name

    grouped: dict[tuple[float, bool], list] = {}
    for name, param in model.named_parameters():
        grouped.setdefault((lr_scale(name), decays(name, param)), []).append(param)

    groups = [
        {"params": params, "lr": scale * lr, "weight_decay": 0.01 if wd else 0.0}
        for (scale, wd), params in grouped.items()
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))


def build_scheduler(optimizer, epochs: int, warmup_epochs: int, cooldown_epochs: int):
    """Warm up, hold, then cosine to `lr`/1000, stepped once per epoch."""
    phases = [
        torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs)
    ]
    milestones = []
    if warmup_epochs:
        phases.insert(
            0,
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1 / (warmup_epochs + 1),
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
        )
        milestones.append(warmup_epochs)
    if cooldown_epochs:
        phases.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cooldown_epochs,
                eta_min=optimizer.defaults["lr"] / 1000,
            )
        )
        milestones.append(epochs - cooldown_epochs)

    if not milestones:
        return phases[0]
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, phases, milestones=milestones
    )


def build_finetune_optimizer(model, lr: float) -> torch.optim.Optimizer:
    """AdamW at one `lr` for every parameter: a trained model has nothing to retarget."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        keeps_decay = param.ndim > 1 and "patch_embed.proj" not in name
        (decay if keeps_decay else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))
