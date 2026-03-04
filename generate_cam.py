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

    loss_name = cfg["model"].get("loss", "")

    if paradigm == "multiclass":
        head = MultiClassHead(feature_dim, num_classes=4)
    elif paradigm == "ordinal":
        # CDW-CE (distance) uses MultiClassHead; CORAL/CORN use OrdinalHead
        if loss_name == "distance":
            head = MultiClassHead(feature_dim, num_classes=4)
        else:
            head = OrdinalHead(feature_dim, num_classes=4)
    elif paradigm == "regression":
        head = RegressionHead(feature_dim)
    else:
        raise ValueError(f"Unsupported paradigm for CAM generation: {paradigm}")

    return torch.nn.Sequential(backbone, head), paradigm, loss_name


def get_predicted_labels(outputs: torch.Tensor, paradigm: str, loss_name: str = ""):
    if paradigm == "multiclass":
        return outputs.argmax(dim=1)

    if paradigm == "ordinal":
        if loss_name == "distance":
            return outputs.argmax(dim=1)
        return ordinal_utils.coral_logits_to_label(outputs)

    if paradigm == "regression":
        return regression_to_class(outputs)

    raise ValueError(f"Unsupported paradigm for CAM generation: {paradigm}")


def parse_target_classes(raw_target_classes: str, num_classes: int = 4):
    if raw_target_classes.strip().lower() == "all":
        return list(range(num_classes))

    target_classes = []
    for token in raw_target_classes.split(","):
        token = token.strip()
        if not token:
            continue
        target_class = int(token)
        if target_class < 0 or target_class >= num_classes:
            raise ValueError(f"target class {target_class} is out of range [0, {num_classes - 1}]")
        target_classes.append(target_class)

    if not target_classes:
        raise ValueError("No valid target class is provided")

    return sorted(set(target_classes))


def build_target_for_class(outputs: torch.Tensor, paradigm: str, target_class: int, loss_name: str = ""):
    batch_size = outputs.size(0)
    device = outputs.device
    target_category = torch.full((batch_size,), target_class, dtype=torch.long, device=device)

    if paradigm == "multiclass":
        return target_category, None

    if paradigm == "ordinal":
        if loss_name == "distance":
            # CDW-CE uses softmax outputs, same as multiclass
            return target_category, None
        class_probs = ordinal_utils._ordinal_logits_to_class_probs(outputs, num_classes=4)
        target_scores = class_probs[:, target_class]
        return None, target_scores

    if paradigm == "regression":
        target_value = torch.full_like(outputs, float(target_class))
        target_scores = -((outputs - target_value) ** 2).view(batch_size)
        return None, target_scores

    raise ValueError(f"Unsupported paradigm for CAM generation: {paradigm}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_per_class", type=int, default=2)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--target_classes",
        default="all",
        help="CAM target classes. Use 'all' or comma-separated class ids (e.g., 0,1,2,3)",
    )
    args = parser.parse_args()

    cfg = load_and_merge("configs/dataset/limuc.yaml", args.config)
    cfg.setdefault("cv", {})["current_fold"] = args.fold
    validate_config(cfg)
    set_seed(cfg.get("seed", 42))

    ds = LIMUCDataset(cfg, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    model, paradigm, loss_name = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    exp_name = cfg.get("experiment_name", Path(cfg["paths"].get("experiment_config", "exp")).stem)
    ckpt = Path(cfg["paths"].get("output_root", "results")) / exp_name / f"fold_{args.fold}" / "best_model.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    cam_extractor = GradCAM(model)
    target_classes = parse_target_classes(args.target_classes, num_classes=4)
    saved = {target_class: 0 for target_class in target_classes}
    out_dir = Path(cfg["paths"].get("output_root", "results")) / exp_name / f"fold_{args.fold}" / "cam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for images, _, _, paths in loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            pred_labels = get_predicted_labels(outputs, paradigm, loss_name)

        pending_classes = [target_class for target_class in target_classes if saved[target_class] < args.num_per_class]
        if not pending_classes:
            break

        img_np = cv2.cvtColor(cv2.imread(str(paths[0])), cv2.COLOR_BGR2RGB) / 255.0
        pred_label = int(pred_labels.item())

        for target_class in pending_classes:
            target_category, target_scores = build_target_for_class(outputs, paradigm, target_class, loss_name)
            cam = cam_extractor(images, target_category=target_category, target_scores=target_scores)[0].detach().cpu().numpy()
            cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
            overlay = overlay_heatmap(img_np, cam_resized)

            out_path = out_dir / (
                f"target_MES{target_class}_pred_MES{pred_label}_{saved[target_class]}_{paradigm}_{args.split}.png"
            )
            cv2.imwrite(str(out_path), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            saved[target_class] += 1

        if all(v >= args.num_per_class for v in saved.values()):
            break


if __name__ == "__main__":
    main()
