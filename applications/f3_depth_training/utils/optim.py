"""The optimizer groups and LR schedule both depth trainers use."""

import torch


def build_optimizer(model, lr: float) -> torch.optim.Optimizer:
    """AdamW with DAv2's encoder at `lr`, the F3 backbone at half, the head at 10x.

    Grouped by (scale, decay) rather than one group per parameter, so AdamW's foreach
    path can batch them. Norms, biases and the re-initialised patch embed skip decay.
    """

    def lr_scale(name: str) -> float:
        if "pretrained" in name:
            return 1.0
        return 0.5 if "eventff" in name else 10.0

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
    """Warm up, hold, then cosine to zero, stepped once per epoch.

    Held flat rather than decayed throughout because the data never repeats: there is no
    overfitting to decay away, and the loss is still falling well past the midpoint.
    """
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
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cooldown_epochs)
        )
        milestones.append(epochs - cooldown_epochs)

    if not milestones:
        return phases[0]
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, phases, milestones=milestones
    )
