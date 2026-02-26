import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.limuc_dataset import LIMUCDataset
from src.explain.grad_cam import GradCAM, overlay_heatmap
from src.losses.regression import regression_to_class
from src.models import ordinal as ordinal_utils
from src.models.backbones import build_backbone, get_backbone_output_dim
from src.models.heads import MultiClassHead, OrdinalHead, RegressionHead
from src.utils.config import load_and_merge, validate_config
from src.utils.seed import set_seed


def build_model(cfg):
    paradigm = cfg["model"].get("paradigm", "multiclass")
    backbone_name = cfg["model"].get("backbone", "resnet18")
    backbone = build_backbone(backbone_name, pretrained=False)
    feature_dim = get_backbone_output_dim(backbone_name)

    if paradigm == "multiclass":
        head = MultiClassHead(feature_dim, num_classes=4)
    elif paradigm == "ordinal":
        head = OrdinalHead(feature_dim, num_classes=4)
    elif paradigm == "regression":
        head = RegressionHead(feature_dim)
    else:
        raise ValueError(f"Unsupported paradigm for CAM generation: {paradigm}")

    return torch.nn.Sequential(backbone, head), paradigm


def get_label_and_target_scores(outputs: torch.Tensor, paradigm: str):
    if paradigm == "multiclass":
        labels = outputs.argmax(dim=1)
        return labels, None

    if paradigm == "ordinal":
        labels = ordinal_utils.coral_logits_to_label(outputs)
        per_sample_scores = []
        for sample_logits, sample_label in zip(outputs, labels):
            if sample_label.item() == 0:
                score = -sample_logits[0]
            else:
                positive_idx = sample_label.item() - 1
                score = sample_logits[positive_idx]
            per_sample_scores.append(score)
        return labels, torch.stack(per_sample_scores)

    if paradigm == "regression":
        labels = regression_to_class(outputs)
        return labels, outputs

    raise ValueError(f"Unsupported paradigm for CAM generation: {paradigm}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_per_class", type=int, default=2)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_and_merge("configs/dataset/limuc.yaml", args.config)
    cfg.setdefault("cv", {})["current_fold"] = args.fold
    validate_config(cfg)
    set_seed(cfg.get("seed", 42))

    ds = LIMUCDataset(cfg, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    model, paradigm = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    exp_name = cfg.get("experiment_name", Path(cfg["paths"].get("experiment_config", "exp")).stem)
    ckpt = Path(cfg["paths"].get("output_root", "results")) / exp_name / f"fold_{args.fold}" / "best_model.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    cam_extractor = GradCAM(model)
    saved = {i: 0 for i in range(4)}
    out_dir = Path(cfg["paths"].get("output_root", "results")) / exp_name / f"fold_{args.fold}" / "cam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for images, _, _, paths in loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        pred_labels, target_scores = get_label_and_target_scores(outputs, paradigm)

        label = int(pred_labels.item())
        if saved[label] >= args.num_per_class:
            continue

        cam = cam_extractor(images, target_scores=target_scores)[0].detach().cpu().numpy()
        img_np = cv2.cvtColor(cv2.imread(str(paths[0])), cv2.COLOR_BGR2RGB) / 255.0
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        overlay = overlay_heatmap(img_np, cam_resized)

        out_path = out_dir / f"MES{label}_{saved[label]}_{paradigm}_{args.split}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        saved[label] += 1

        if all(v >= args.num_per_class for v in saved.values()):
            break


if __name__ == "__main__":
    main()
