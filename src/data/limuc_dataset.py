import logging
from pathlib import Path
from typing import List, Dict, Any, Set
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from .folds import load_fold_mapping
from .transforms import build_transforms


class LIMUCDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], split: str = "train", fold_mapping=None, test_patients: Set[str] = None):
        self.cfg = cfg
        self.split = split
        self.data_root = Path(cfg["paths"]["data_root"])

        if fold_mapping is not None and test_patients is not None:
            self.fold_mapping = fold_mapping
            self.test_patients = test_patients
            self.fold_metadata = {}
        else:
            self.fold_mapping, self.test_patients, self.fold_metadata = load_fold_mapping(
                cfg["paths"]["data_root"],
                cfg["paths"]["folds_file"],
                num_folds=cfg.get("cv", {}).get("num_folds", 5),
                seed=cfg.get("seed", 42),
                test_ratio=cfg.get("cv", {}).get("test_ratio", 0.2),
            )

        self.current_fold = cfg.get("cv", {}).get("current_fold", 0)
        self.samples = self._build_table()
        self.transform = build_transforms(cfg, train=split == "train")

    def _build_table(self) -> pd.DataFrame:
        logger = logging.getLogger(__name__)
        rows = []
        for patient_dir in sorted(self.data_root.glob("*")):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name

            # Route patient to correct split
            if patient_id in self.test_patients:
                if self.split != "test":
                    continue
            else:
                fold_idx = self.fold_mapping.get(patient_id, 0)
                if self.split == "train" and fold_idx == self.current_fold:
                    continue
                if self.split == "val" and fold_idx != self.current_fold:
                    continue
                if self.split == "test":
                    continue

            for label in range(4):
                class_dir = patient_dir / f"Mayo {label}"
                if not class_dir.exists() or not class_dir.is_dir():
                    logger.warning("Missing class directory %s for patient %s", class_dir, patient_id)
                    continue
                img_paths = list(class_dir.glob("*"))
                if not img_paths:
                    logger.warning("Empty class directory %s for patient %s", class_dir, patient_id)
                    continue
                for img_path in img_paths:
                    if not img_path.is_file():
                        continue
                    rows.append({
                        "image_path": img_path,
                        "label": label,
                        "patient_id": patient_id,
                        "fold": self.fold_mapping.get(patient_id, -1),
                    })

        if not rows:
            logger.error("No image samples found in data_root '%s' for split '%s'", self.data_root, self.split)
            raise ValueError(f"No image samples found in data_root '{self.data_root}' for split '{self.split}'")

        return pd.DataFrame(rows)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"]), row["patient_id"], str(row["image_path"])
