import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.utils.config import load_and_merge, validate_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging, get_logger
from src.utils.paths import resolve_output_dir
from src.data.limuc_dataset import LIMUCDataset
from src.models.backbones import build_backbone, get_backbone_output_dim
from src.models.heads import MultiClassHead
from src.losses import multiclass as mc_losses
from src.metrics.classification_metrics import evaluate_all


def train_one_epoch(model, loader, criterion, optimizer, device):
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


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    metrics = evaluate_all(targets, preds)
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
        criterion = mc_losses.class_balanced_ce
    elif loss_name == "focal":
        criterion = lambda logits, targets: mc_losses.focal_loss(logits, targets, gamma=cfg["model"].get("gamma", 2.0), alpha=cfg["model"].get("alpha", 0.25))
    else:
        raise ValueError(f"Unknown loss {loss_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"].get("learning_rate", 1e-4), weight_decay=cfg["training"].get("weight_decay", 1e-4))

    best_qwk = -1
    history = []
    for epoch in range(cfg["training"].get("num_epochs", 10)):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics, preds, targets = validate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} qwk={metrics['qwk']:.4f}")
        if metrics["qwk"] > best_qwk:
            best_qwk = metrics["qwk"]
            torch.save(model.state_dict(), Path(output_dir) / "best_model.pt")
            logger.info("Saved new best model")

    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
