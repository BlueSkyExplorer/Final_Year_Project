import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.limuc_dataset import LIMUCDataset
from src.losses import multiclass as mc_losses
from src.losses import ordinal as ordinal_losses
from src.losses import regression as reg_losses
from src.metrics.classification_metrics import evaluate_all
from src.models import ordinal as ordinal_utils
from src.models.backbones import build_backbone, get_backbone_output_dim
from src.models.heads import MultiClassHead, OrdinalHead, RegressionHead
from src.utils.config import load_and_merge, validate_config
from src.utils.data_checks import ensure_dataset_not_empty
from src.utils.evaluation_io import persist_json_atomic, write_predictions_csv
from src.utils.paths import resolve_output_dir
from src.utils.seed import set_seed


def build_model_and_eval_fn(cfg):
    backbone_name = cfg["model"].get("backbone", "resnet18")
    backbone = build_backbone(backbone_name, pretrained=False)
    feature_dim = get_backbone_output_dim(backbone_name)
    head_dropout = cfg["model"].get("head_dropout", 0.0)
    paradigm = cfg["model"].get("paradigm", "multiclass")

    if paradigm == "multiclass":
        head = MultiClassHead(feature_dim, num_classes=4, dropout=head_dropout)
        model = torch.nn.Sequential(backbone, head)
        loss_name = cfg["model"].get("loss", "ce")
        if loss_name == "ce":
            criterion = mc_losses.cross_entropy_loss
        elif loss_name == "cbce":
            criterion = mc_losses.cross_entropy_loss
        elif loss_name == "focal":
            criterion = lambda logits, targets: mc_losses.focal_loss(
                logits,
                targets,
                gamma=cfg["model"].get("gamma", 2.0),
                alpha=None,
            )
        else:
            raise ValueError(f"Unknown multiclass loss {loss_name}")

        def evaluate_split_fn(model, loader, device, split_name):
            ensure_dataset_not_empty(loader, split_name.capitalize())
            model.eval()
            total_loss = 0.0
            preds, targets, probas = [], [], []
            rows = []
            with torch.no_grad():
                for images, labels, patient_ids, image_paths in loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    total_loss += criterion(logits, labels).item() * images.size(0)
                    batch_preds = logits.argmax(dim=1).cpu().tolist()
                    batch_targets = labels.cpu().tolist()
                    batch_probs = probs.cpu().tolist()
                    preds.extend(batch_preds)
                    targets.extend(batch_targets)
                    probas.extend(batch_probs)
                    for patient_id, image_path, true_label, pred_label, class_scores in zip(
                        patient_ids, image_paths, batch_targets, batch_preds, batch_probs
                    ):
                        rows.append(
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
            metrics = evaluate_all(targets, preds, y_proba=probas)
            return total_loss / len(loader.dataset), metrics, rows

        return model, evaluate_split_fn

    if paradigm == "ordinal":
        loss_name = cfg["model"].get("loss", "coral")
        if loss_name == "distance":
            head = MultiClassHead(feature_dim, num_classes=4, dropout=head_dropout)
            model = torch.nn.Sequential(backbone, head)
            loss_fn = lambda logits, targets: mc_losses.cdw_ce_loss(
                logits, targets, num_classes=4, alpha=cfg["model"].get("alpha", 1.0),
            )
            decode_fn = lambda logits: logits.argmax(dim=1)
            proba_fn = lambda logits, num_classes: torch.softmax(logits, dim=1)
            auroc_source = "class_probs"
        else:
            head = OrdinalHead(feature_dim, num_classes=4, dropout=head_dropout)
            model = torch.nn.Sequential(backbone, head)
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

        def evaluate_split_fn(model, loader, device, split_name):
            ensure_dataset_not_empty(loader, split_name.capitalize())
            model.eval()
            total_loss = 0.0
            preds, targets, probas = [], [], []
            rows = []
            with torch.no_grad():
                for images, labels, patient_ids, image_paths in loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    total_loss += loss_fn(logits, labels).item() * images.size(0)
                    class_probs = proba_fn(logits, num_classes=4)
                    batch_preds = decode_fn(logits).cpu().tolist()
                    batch_targets = labels.cpu().tolist()
                    batch_probs = class_probs.cpu().tolist()
                    preds.extend(batch_preds)
                    targets.extend(batch_targets)
                    probas.append(class_probs.cpu())
                    for patient_id, image_path, true_label, pred_label, class_scores in zip(
                        patient_ids, image_paths, batch_targets, batch_preds, batch_probs
                    ):
                        rows.append(
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
            metrics = evaluate_all(targets, preds, y_proba=torch.cat(probas, dim=0).numpy())
            metrics["auroc_source"] = auroc_source
            return total_loss / len(loader.dataset), metrics, rows

        return model, evaluate_split_fn

    if paradigm == "regression":
        head = RegressionHead(feature_dim, dropout=head_dropout)
        model = torch.nn.Sequential(backbone, head)
        loss_name = cfg["model"].get("loss", "mse")
        if loss_name == "mse":
            loss_fn = reg_losses.mse_loss
        elif loss_name == "huber":
            loss_fn = lambda preds, targets: reg_losses.huber_loss(preds, targets, delta=cfg["model"].get("delta", 1.0))
        else:
            raise ValueError(f"Unknown regression loss {loss_name}")

        def evaluate_split_fn(model, loader, device, split_name):
            ensure_dataset_not_empty(loader, split_name.capitalize())
            model.eval()
            total_loss = 0.0
            preds, targets, raw_scores = [], [], []
            rows = []
            with torch.no_grad():
                for images, labels, patient_ids, image_paths in loader:
                    images, labels = images.to(device), labels.to(device).float()
                    outputs = model(images)
                    total_loss += loss_fn(outputs, labels).item() * images.size(0)
                    batch_raw_scores = outputs.squeeze(-1).cpu().tolist()
                    batch_preds = reg_losses.regression_to_class(outputs).cpu().tolist()
                    batch_targets = labels.cpu().long().tolist()
                    preds.extend(batch_preds)
                    targets.extend(batch_targets)
                    raw_scores.extend(batch_raw_scores)
                    for patient_id, image_path, true_label, pred_label, raw_score in zip(
                        patient_ids, image_paths, batch_targets, batch_preds, batch_raw_scores
                    ):
                        rows.append(
                            {
                                "image_path": str(image_path),
                                "patient_id": str(patient_id),
                                "true_label": int(true_label),
                                "pred_label": int(pred_label),
                                "split": split_name,
                                "raw_score": float(raw_score),
                            }
                        )
            metrics = evaluate_all(targets, preds, y_proba=raw_scores)
            return total_loss / len(loader.dataset), metrics, rows

        return model, evaluate_split_fn

    raise ValueError(f"Unsupported paradigm {paradigm}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_and_merge("configs/dataset/limuc.yaml", args.config)
    cfg.setdefault("cv", {})["current_fold"] = args.fold
    validate_config(cfg)
    set_seed(cfg.get("seed", 42))

    dataset = LIMUCDataset(cfg, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"].get("batch_size", 16),
        shuffle=False,
        num_workers=cfg["images"].get("num_workers", 4),
        pin_memory=cfg["images"].get("pin_memory", True),
    )
    model, evaluate_split_fn = build_model_and_eval_fn(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dir = resolve_output_dir(cfg)
    ckpt = Path(output_dir) / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    loss, metrics, rows = evaluate_split_fn(model, loader, device, args.split)
    artifact_prefix = "best_val" if args.split == "val" else args.split
    persist_json_atomic(
        {
            "epoch": None,
            "loss": loss,
            **metrics,
        },
        Path(output_dir) / f"{artifact_prefix}_metrics.json",
    )
    write_predictions_csv(rows, Path(output_dir) / f"{args.split}_predictions.csv")


if __name__ == "__main__":
    main()
