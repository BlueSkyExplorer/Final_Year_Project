import argparse
from pathlib import Path
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from src.utils.config import load_and_merge
from src.utils.seed import set_seed
from src.data.limuc_dataset import LIMUCDataset
from src.models.backbones import build_backbone, get_backbone_output_dim
from src.models.heads import MultiClassHead
from src.explain.grad_cam import GradCAM, overlay_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_per_class", type=int, default=2)
    args = parser.parse_args()

    cfg = load_and_merge("configs/dataset/limuc.yaml", args.config)
    cfg.setdefault("cv", {})["current_fold"] = args.fold
    set_seed(cfg.get("seed", 42))

    val_ds = LIMUCDataset(cfg, split="val")
    loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    backbone = build_backbone(cfg["model"].get("backbone", "resnet18"), pretrained=False)
    feature_dim = get_backbone_output_dim(cfg["model"].get("backbone", "resnet18"))
    head = MultiClassHead(feature_dim, num_classes=4)
    model = torch.nn.Sequential(backbone, head)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt = Path(cfg["paths"].get("output_root", "results")) / cfg.get("experiment_name", Path(cfg["paths"].get("experiment_config", "exp")).stem) / f"fold_{args.fold}" / "best_model.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    cam_extractor = GradCAM(model)
    saved = {i: 0 for i in range(4)}
    out_dir = Path(cfg["paths"].get("output_root", "results")) / cfg.get("experiment_name", Path(cfg["paths"].get("experiment_config", "exp")).stem) / f"fold_{args.fold}" / "cam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for images, labels, _, paths in loader:
        label = int(labels.item())
        if saved[label] >= args.num_per_class:
            continue
        images = images.to(device)
        cam = cam_extractor(images)[0].cpu().numpy()
        img_np = cv2.cvtColor(cv2.imread(str(paths[0])), cv2.COLOR_BGR2RGB) / 255.0
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        overlay = overlay_heatmap(img_np, cam_resized)
        out_path = out_dir / f"MES{label}_{saved[label]}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        saved[label] += 1
        if all(v >= args.num_per_class for v in saved.values()):
            break


if __name__ == "__main__":
    main()
