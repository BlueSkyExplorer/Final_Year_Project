from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def build_lr_scheduler(optimizer, training_cfg, num_epochs, last_epoch: int = -1):
    scheduler_cfg = training_cfg.get("lr_scheduler", {"type": "cosine"})
    scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()

    if last_epoch >= 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

    if scheduler_type in {"none", "off", "disabled"}:
        return None, False

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("t_max", num_epochs),
            eta_min=scheduler_cfg.get("eta_min", 0.0),
            last_epoch=last_epoch,
        )
        return scheduler, False

    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 3),
            min_lr=scheduler_cfg.get("min_lr", 0.0),
        )
        if last_epoch >= 0:
            scheduler.last_epoch = last_epoch
        return scheduler, True

    if scheduler_type == "step":
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 10),
            gamma=scheduler_cfg.get("gamma", 0.1),
            last_epoch=last_epoch,
        )
        return scheduler, False

    raise ValueError(f"Unknown lr_scheduler type: {scheduler_type}")
