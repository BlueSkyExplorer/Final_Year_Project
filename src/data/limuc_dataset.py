from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from .folds import load_fold_mapping
from .transforms import build_transforms


class LIMUCDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], split: str = "train", fold_mapping: Dict[str, int] = None):
        self.cfg = cfg
        self.split = split
        self.data_root = Path(cfg["paths"]["data_root"])
        self.fold_mapping = fold_mapping or load_fold_mapping(
            cfg["paths"]["data_root"],
            cfg["paths"]["folds_file"],
            num_folds=cfg.get("cv", {}).get("num_folds", 5),
            seed=cfg.get("seed", 42),
        )
        self.current_fold = cfg.get("cv", {}).get("current_fold", 0)
        self.samples = self._build_table()
        self.transform = build_transforms(cfg, train=split == "train")

    def _build_table(self) -> pd.DataFrame:
        rows = []
        for patient_dir in sorted(self.data_root.glob("*")):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name
            fold_idx = self.fold_mapping.get(patient_id, 0)
            for label in range(4):
                class_dir = patient_dir / f"Mayo {label}"
                for img_path in class_dir.glob("*"):
                    rows.append({
                        "image_path": img_path,
                        "label": label,
                        "patient_id": patient_id,
                        "fold": fold_idx,
                    })
        df = pd.DataFrame(rows)
        if self.split == "train":
            return df[df["fold"] != self.current_fold].reset_index(drop=True)
        elif self.split == "val":
            return df[df["fold"] == self.current_fold].reset_index(drop=True)
        else:
            return df.reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"]), row["patient_id"], str(row["image_path"])
