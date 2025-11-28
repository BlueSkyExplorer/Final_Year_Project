import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold


def _extract_patient_id(path: Path) -> str:
    return path.name


def _load_patient_labels(data_root: Path) -> Tuple[List[str], List[int]]:
    patients = []
    labels = []
    for patient_dir in sorted(data_root.glob("*")):
        if not patient_dir.is_dir():
            continue
        mes_counts = []
        for i in range(4):
            mes_dir = patient_dir / f"Mayo {i}"
            mes_counts.append(len(list(mes_dir.glob("*"))))
        dominant = int(np.argmax(mes_counts)) if any(mes_counts) else 0
        patients.append(_extract_patient_id(patient_dir))
        labels.append(dominant)
    return patients, labels


def generate_folds(data_root: str, folds_file: str, num_folds: int = 5, seed: int = 42) -> Dict[str, int]:
    data_root_path = Path(data_root)
    folds_path = Path(folds_file)
    if folds_path.exists():
        try:
            with open(folds_path, "r") as f:
                stored = json.load(f)
            if isinstance(stored, dict) and stored.get("folds"):
                mapping = {pid: idx for idx, fold in enumerate(stored["folds"]) for pid in fold.get("patients", [])}
                if mapping:
                    return mapping
        except json.JSONDecodeError:
            pass

    patients, labels = _load_patient_labels(data_root_path)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_assignments = {}
    folds = []
    for fold_idx, (_, val_idx) in enumerate(skf.split(patients, labels)):
        fold_patients = [patients[i] for i in val_idx]
        folds.append({"patients": fold_patients})
        for pid in fold_patients:
            fold_assignments[pid] = fold_idx

    folds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(folds_path, "w") as f:
        json.dump({"folds": folds}, f, indent=2)
    return fold_assignments


def load_fold_mapping(data_root: str, folds_file: str, num_folds: int = 5, seed: int = 42) -> Dict[str, int]:
    folds_path = Path(folds_file)
    if folds_path.exists():
        with open(folds_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("folds"):
            return {pid: idx for idx, fold in enumerate(data["folds"]) for pid in fold.get("patients", [])}
    return generate_folds(data_root, folds_file, num_folds=num_folds, seed=seed)
