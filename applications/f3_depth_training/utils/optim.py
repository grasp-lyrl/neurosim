"""The optimizer groups and LR schedule both depth trainers use."""

import torch


def build_optimizer(model, lr: float) -> torch.optim.Optimizer:
    """AdamW at `lr`, 10x for the two parts that retarget: DAv2's head and the age table.

    Everything else is a loaded backbone being fine-tuned. Grouped by (scale, decay)
    rather than one group per parameter, so AdamW's foreach path can batch them. Norms,
    biases and the re-initialised patch embed skip decay.
    """

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
    """Warm up, hold, then cosine to `lr`/1000, stepped once per epoch.

    Held flat rather than decayed throughout because the data never repeats: there is no
    overfitting to decay away, and the loss is still falling well past the midpoint. The
    floor is one absolute LR, so the faster groups land proportionally further down.
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
