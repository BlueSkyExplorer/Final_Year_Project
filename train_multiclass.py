import argparse
import json
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
from src.utils.config import load_and_merge, validate_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging, get_logger
from src.utils.paths import resolve_output_dir
from src.utils.data_checks import ensure_dataset_not_empty
from src.utils.lr_scheduler import build_lr_scheduler
from src.data.limuc_dataset import LIMUCDataset
from src.models.backbones import build_backbone, get_backbone_output_dim
from src.models.heads import MultiClassHead
from src.losses import multiclass as mc_losses
from src.metrics.classification_metrics import evaluate_all


def _build_dataset_class_weights(train_dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = torch.tensor(train_dataset.samples["label"].tolist(), dtype=torch.long)
    class_weights = mc_losses.compute_class_weights(labels, num_classes=num_classes)
    return class_weights.to(device)


def _parse_focal_alpha(alpha_cfg, num_classes: int, device: torch.device) -> Optional[torch.Tensor | float]:
    if alpha_cfg is None:
        return None
    if isinstance(alpha_cfg, (float, int)):
        return float(alpha_cfg)
    if isinstance(alpha_cfg, list):
        if len(alpha_cfg) != num_classes:
            raise ValueError(
                f"model.alpha list length must equal num_classes ({num_classes}), got {len(alpha_cfg)}"
            )
        return torch.tensor(alpha_cfg, dtype=torch.float32, device=device)
    raise TypeError("model.alpha must be a float/int, list, or null")

def train_one_epoch(model, loader, criterion, optimizer, device):
    ensure_dataset_not_empty(loader, "Training")
    model.train()
    total_loss = 0.0
    for images, labels, _, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def set_requires_grad(module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad


def get_param_group_lrs(optimizer, freeze_epochs: int, epoch: int):
    if len(optimizer.param_groups) == 1:
        name = "head" if freeze_epochs > 0 and epoch < freeze_epochs else "all"
        return {name: optimizer.param_groups[0]["lr"]}
    return {
        "head": optimizer.param_groups[0]["lr"],
        "backbone": optimizer.param_groups[1]["lr"],
    }


def validate(model, loader, criterion, device):
    ensure_dataset_not_empty(loader, "Validation")
    model.eval()
    total_loss = 0.0
    preds, targets, probas = [], [], []
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
            probas.extend(probs.cpu().tolist())
    metrics = evaluate_all(targets, preds, y_proba=probas)
    return total_loss / len(loader.dataset), metrics, preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    cfg = load_and_merge("configs/dataset/limuc.yaml", args.config)
    if args.fold is not None:
        cfg.setdefault("cv", {})["current_fold"] = args.fold
    validate_config(cfg)
    set_seed(cfg.get("seed", 42))

    output_dir = resolve_output_dir(cfg)
    setup_logging(output_dir)
    logger = get_logger(__name__)
    logger.info(f"Using config {cfg}")

    train_ds = LIMUCDataset(cfg, split="train")
    val_ds = LIMUCDataset(cfg, split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg["training"].get("batch_size", 16), shuffle=True,
                              num_workers=cfg["images"].get("num_workers", 4), pin_memory=cfg["images"].get("pin_memory", True))
    val_loader = DataLoader(val_ds, batch_size=cfg["training"].get("batch_size", 16), shuffle=False,
                            num_workers=cfg["images"].get("num_workers", 4), pin_memory=cfg["images"].get("pin_memory", True))

    backbone = build_backbone(cfg["model"].get("backbone", "resnet18"), pretrained=True)
    feature_dim = get_backbone_output_dim(cfg["model"].get("backbone", "resnet18"))
    head = MultiClassHead(feature_dim, num_classes=4)
    model = torch.nn.Sequential(backbone, head)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_name = cfg["model"].get("loss", "ce")
    if loss_name == "ce":
        criterion = mc_losses.cross_entropy_loss
    elif loss_name == "cbce":
        class_weights = _build_dataset_class_weights(train_ds, num_classes=4, device=device)
        logger.info(f"Using dataset-level CBCE class weights: {class_weights.tolist()}")
        criterion = lambda logits, targets: mc_losses.class_balanced_ce(logits, targets, class_weights=class_weights)
    elif loss_name == "focal":
        focal_alpha = _parse_focal_alpha(cfg["model"].get("alpha", 0.25), num_classes=4, device=device)
        logger.info(f"Using focal loss alpha: {focal_alpha if isinstance(focal_alpha, float) or focal_alpha is None else focal_alpha.tolist()}")
        criterion = lambda logits, targets: mc_losses.focal_loss(
            logits,
            targets,
            gamma=cfg["model"].get("gamma", 2.0),
            alpha=focal_alpha,
        )
    else:
        raise ValueError(f"Unknown loss {loss_name}")

    training_cfg = cfg["training"]
    num_epochs = training_cfg.get("num_epochs", 40)
    base_lr = training_cfg.get("learning_rate", 1e-4)
    weight_decay = training_cfg.get("weight_decay", 1e-4)
    freeze_epochs = training_cfg.get("freeze_epochs", 0)
    backbone_lr = training_cfg.get("backbone_learning_rate", base_lr)
    early_stopping_patience = training_cfg.get("early_stopping_patience", 0)
    early_stopping_min_delta = training_cfg.get("early_stopping_min_delta", 0.0)
    logger.info(
        "Early stopping config: patience=%s, min_delta=%s",
        early_stopping_patience,
        early_stopping_min_delta,
    )

    if freeze_epochs > 0:
        set_requires_grad(backbone, False)
        optimizer = torch.optim.Adam(head.parameters(), lr=base_lr, weight_decay=weight_decay)
        logger.info(f"Freezing backbone for first {freeze_epochs} epochs (head lr={base_lr})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    scheduler, scheduler_on_val = build_lr_scheduler(optimizer, training_cfg, num_epochs)

    best_qwk = -1
    epochs_without_improvement = 0
    history = []
    for epoch in range(num_epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            logger.info("Unfreezing backbone for fine-tuning")
            set_requires_grad(backbone, True)
            head_lr_before_unfreeze = optimizer.param_groups[0]["lr"]
            optimizer.add_param_group({"params": backbone.parameters(), "lr": backbone_lr})
            optimizer.param_groups[0]["lr"] = head_lr_before_unfreeze
            logger.info(
                "Unfreeze transition lr continuity: head(before)=%.8f, head(after)=%.8f, backbone(start)=%.8f",
                head_lr_before_unfreeze,
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
            )
            if scheduler is not None:
                scheduler, scheduler_on_val = build_lr_scheduler(
                    optimizer,
                    training_cfg,
                    num_epochs,
                    last_epoch=epoch - 1,
                )

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics, preds, targets = validate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "auroc_source": "probability",
                **metrics,
            }
        )
        if scheduler is not None:
            if scheduler_on_val:
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lrs = get_param_group_lrs(optimizer, freeze_epochs, epoch)
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"qwk={metrics['qwk']:.4f} param_group_lrs={current_lrs}"
        )
        if metrics["qwk"] > best_qwk + early_stopping_min_delta:
            best_qwk = metrics["qwk"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), Path(output_dir) / "best_model.pt")
            logger.info("Saved new best model")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} after "
                    f"{epochs_without_improvement} epoch(s) without QWK improvement "
                    f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})."
                )
                break

    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
