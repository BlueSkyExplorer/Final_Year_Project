import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.utils.config import load_and_merge, validate_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging, get_logger
from src.utils.paths import resolve_output_dir
from src.utils.data_checks import ensure_dataset_not_empty
from src.utils.lr_scheduler import build_lr_scheduler
from src.utils.evaluation_io import build_split_manifest, persist_json_atomic, write_predictions_csv
from src.data.limuc_dataset import LIMUCDataset
from src.models.backbones import build_backbone, get_backbone_output_dim
from src.models.heads import OrdinalHead, MultiClassHead
from src.models import ordinal as ordinal_utils
from src.losses import ordinal as ordinal_losses
from src.losses import multiclass as mc_losses
from src.metrics.classification_metrics import evaluate_all


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    ensure_dataset_not_empty(loader, "Training")
    model.train()
    total_loss = 0.0
    for images, labels, _, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
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


def _resolve_unfreeze_backbone_lr_mode(training_cfg) -> str:
    mode = str(training_cfg.get("unfreeze_backbone_lr_mode", "global_schedule")).lower()
    allowed_modes = {"global_schedule", "full_value_start"}
    if mode not in allowed_modes:
        raise ValueError(
            "training.unfreeze_backbone_lr_mode must be one of "
            f"{sorted(allowed_modes)}, got {mode!r}"
        )
    return mode


def _register_new_group_with_scheduler(scheduler, optimizer, backbone_lr: float, training_cfg):
    if scheduler is None:
        return
    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs.append(backbone_lr)
    scheduler_last_lr = getattr(scheduler, "_last_lr", None)
    if isinstance(scheduler_last_lr, list):
        scheduler_last_lr.append(backbone_lr)
    if hasattr(scheduler, "min_lrs"):
        min_lr = training_cfg.get("lr_scheduler", {}).get("min_lr", 0.0)
        scheduler.min_lrs.append(min_lr)
    if hasattr(scheduler, "optimizer"):
        scheduler.optimizer = optimizer


def evaluate_split(model, loader, loss_fn, device, decode_fn, proba_fn, auroc_source, split_name: str):
    ensure_dataset_not_empty(loader, split_name.capitalize())
    model.eval()
    total_loss = 0.0
    preds, targets, probas = [], [], []
    prediction_rows = []
    with torch.no_grad():
        for images, labels, patient_ids, image_paths in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.size(0)
            pred_labels = decode_fn(logits)
            class_probs = proba_fn(logits, num_classes=4)
            batch_preds = pred_labels.cpu().tolist()
            batch_targets = labels.cpu().tolist()
            batch_probs = class_probs.cpu().tolist()
            preds.extend(batch_preds)
            targets.extend(batch_targets)
            probas.append(class_probs.cpu())
            for patient_id, image_path, true_label, pred_label, class_scores in zip(
                patient_ids,
                image_paths,
                batch_targets,
                batch_preds,
                batch_probs,
            ):
                prediction_rows.append(
                    {
                        "image_path": str(image_path),
                        "patient_id": str(patient_id),
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "split": split_name,
                        "score_0": float(class_scores[0]),
                        "score_1": float(class_scores[1]),
                        "score_2": float(class_scores[2]),
                        "score_3": float(class_scores[3]),
                    }
                )
    y_proba = torch.cat(probas, dim=0).numpy()
    metrics = evaluate_all(targets, preds, y_proba=y_proba)
    metrics["auroc_source"] = auroc_source
    return total_loss / len(loader.dataset), metrics, prediction_rows


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

    train_ds = LIMUCDataset(cfg, split="train")
    val_ds = LIMUCDataset(cfg, split="val")
    test_ds = LIMUCDataset(cfg, split="test")
    train_loader = DataLoader(train_ds, batch_size=cfg["training"].get("batch_size", 16), shuffle=True,
                              num_workers=cfg["images"].get("num_workers", 4), pin_memory=cfg["images"].get("pin_memory", True))
    val_loader = DataLoader(val_ds, batch_size=cfg["training"].get("batch_size", 16), shuffle=False,
                            num_workers=cfg["images"].get("num_workers", 4), pin_memory=cfg["images"].get("pin_memory", True))
    test_loader = DataLoader(test_ds, batch_size=cfg["training"].get("batch_size", 16), shuffle=False,
                             num_workers=cfg["images"].get("num_workers", 4), pin_memory=cfg["images"].get("pin_memory", True))

    backbone = build_backbone(cfg["model"].get("backbone", "resnet18"), pretrained=True)
    feature_dim = get_backbone_output_dim(cfg["model"].get("backbone", "resnet18"))
    head_dropout = cfg["model"].get("head_dropout", 0.0)

    # Determine loss first — CDW-CE (distance) uses MultiClassHead (softmax),
    # while CORAL/CORN use OrdinalHead (K-1 binary thresholds).
    loss_name = cfg["model"].get("loss", "coral")
    if loss_name == "distance":
        head = MultiClassHead(feature_dim, num_classes=4, dropout=head_dropout)
        cdw_alpha = cfg["model"].get("alpha", 1.0)
        logger.info(f"Using CDW-CE loss (Polat et al. 2022) with alpha={cdw_alpha}")
        loss_fn = lambda logits, targets: mc_losses.cdw_ce_loss(
            logits, targets, num_classes=4, alpha=cdw_alpha,
        )
        decode_fn = lambda logits: logits.argmax(dim=1)
        proba_fn = lambda logits, num_classes: torch.softmax(logits, dim=1)
        auroc_source = "class_probs"
    else:
        head = OrdinalHead(feature_dim, num_classes=4, dropout=head_dropout)
        if loss_name == "coral":
            loss_fn = lambda logits, targets: ordinal_losses.coral_loss(logits, targets, num_classes=4)
            decode_fn = ordinal_utils.coral_logits_to_label
            proba_fn = ordinal_utils._ordinal_logits_to_class_probs
            auroc_source = "ordinal_class_probs_coral"
        elif loss_name == "corn":
            loss_fn = lambda logits, targets: ordinal_losses.corn_loss(logits, targets, num_classes=4)
            decode_fn = ordinal_utils.corn_logits_to_label
            proba_fn = ordinal_utils.corn_logits_to_class_probs
            auroc_source = "ordinal_class_probs_corn"
        else:
            raise ValueError(f"Unknown ordinal loss {loss_name}")
    logger.info("Using loss '%s' with head '%s'", loss_name, head.__class__.__name__)

    model = torch.nn.Sequential(backbone, head)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_cfg = cfg["training"]
    num_epochs = training_cfg.get("num_epochs", 40)
    base_lr = training_cfg.get("learning_rate", 1e-4)
    original_head_base_lr = base_lr
    weight_decay = training_cfg.get("weight_decay", 1e-4)
    freeze_epochs = training_cfg.get("freeze_epochs", 0)
    backbone_lr = training_cfg.get("backbone_learning_rate", base_lr)
    unfreeze_backbone_lr_mode = _resolve_unfreeze_backbone_lr_mode(training_cfg)
    early_stopping_patience = training_cfg.get("early_stopping_patience", 0)
    early_stopping_min_delta = training_cfg.get("early_stopping_min_delta", 0.0)
    logger.info(
        "Early stopping config: patience=%s, min_delta=%s",
        early_stopping_patience,
        early_stopping_min_delta,
    )
    logger.info(
        "Freeze strategy: freeze_epochs=%s, backbone_learning_rate=%.8f, unfreeze_backbone_lr_mode=%s",
        freeze_epochs,
        backbone_lr,
        unfreeze_backbone_lr_mode,
    )

    if freeze_epochs > 0:
        set_requires_grad(backbone, False)
        optimizer = torch.optim.Adam(head.parameters(), lr=base_lr, weight_decay=weight_decay)
        logger.info(f"Freezing backbone for first {freeze_epochs} epochs (head lr={base_lr})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    scheduler, scheduler_on_val = build_lr_scheduler(optimizer, training_cfg, num_epochs)

    best_qwk = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(num_epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            logger.info("Unfreezing backbone for fine-tuning")
            set_requires_grad(backbone, True)
            head_lr_before_unfreeze = optimizer.param_groups[0]["lr"]
            optimizer.param_groups[0]["lr"] = original_head_base_lr
            optimizer.add_param_group({"params": backbone.parameters(), "lr": backbone_lr})
            logger.info(
                "Unfreeze transition lr reset: head(before)=%.8f, head(reset_to_base)=%.8f, backbone(start)=%.8f",
                head_lr_before_unfreeze,
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
            )
            if scheduler is not None:
                if unfreeze_backbone_lr_mode == "global_schedule":
                    scheduler, scheduler_on_val = build_lr_scheduler(
                        optimizer,
                        training_cfg,
                        num_epochs,
                        last_epoch=epoch - 1,
                    )
                else:
                    _register_new_group_with_scheduler(scheduler, optimizer, backbone_lr, training_cfg)
            scheduler_last_epoch = getattr(scheduler, "last_epoch", None) if scheduler is not None else None
            scheduler_base_lrs = getattr(scheduler, "base_lrs", None) if scheduler is not None else None
            current_lrs = [group["lr"] for group in optimizer.param_groups]
            logger.info(
                "Unfreeze event detail: epoch=%d base_lrs=%s current_lrs=%s last_epoch=%s",
                epoch,
                scheduler_base_lrs,
                current_lrs,
                scheduler_last_epoch,
            )

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, metrics, _ = evaluate_split(
            model,
            val_loader,
            loss_fn,
            device,
            decode_fn,
            proba_fn,
            auroc_source,
            "val",
        )
        assert "auroc_source" in metrics, "evaluate_split() must provide auroc_source in metrics"
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        if scheduler is not None:
            if scheduler_on_val:
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lrs = get_param_group_lrs(optimizer, freeze_epochs, epoch)
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"qwk={metrics['qwk']:.4f} auroc_source={metrics['auroc_source']} "
            f"param_group_lrs={current_lrs}"
        )
        if metrics["qwk"] > best_qwk + early_stopping_min_delta:
            best_qwk = metrics["qwk"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), Path(output_dir) / "best_model.pt")
            persist_json_atomic(history, Path(output_dir) / "metrics.json")
            logger.info("Saved new best model")
        else:
            persist_json_atomic(history, Path(output_dir) / "metrics.json")
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} after "
                    f"{epochs_without_improvement} epoch(s) without QWK improvement "
                    f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})."
                )
                break

    if best_epoch is None:
        raise RuntimeError("Training completed without saving a best_model.pt checkpoint.")

    best_model_path = Path(output_dir) / "best_model.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    best_val_loss, best_val_metrics, val_predictions = evaluate_split(
        model, val_loader, loss_fn, device, decode_fn, proba_fn, auroc_source, "val"
    )
    test_loss, test_metrics, test_predictions = evaluate_split(
        model, test_loader, loss_fn, device, decode_fn, proba_fn, auroc_source, "test"
    )
    persist_json_atomic(
        {
            "epoch": best_epoch,
            "loss": best_val_loss,
            **best_val_metrics,
        },
        Path(output_dir) / "best_val_metrics.json",
    )
    persist_json_atomic(
        {
            "epoch": best_epoch,
            "loss": test_loss,
            **test_metrics,
        },
        Path(output_dir) / "test_metrics.json",
    )
    write_predictions_csv(val_predictions, Path(output_dir) / "val_predictions.csv")
    write_predictions_csv(test_predictions, Path(output_dir) / "test_predictions.csv")
    persist_json_atomic(
        build_split_manifest(cfg=cfg, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds),
        Path(output_dir) / "split_manifest.json",
    )
    logger.info(
        "Best checkpoint summary: best_epoch=%s best_val_qwk=%.4f test_qwk=%.4f test_auroc_source=%s",
        best_epoch,
        best_val_metrics["qwk"],
        test_metrics["qwk"],
        test_metrics["auroc_source"],
    )


if __name__ == "__main__":
    main()
