import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


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


def generate_folds(
    data_root: str,
    folds_file: str,
    num_folds: int = 5,
    seed: int = 42,
    test_ratio: float = 0.2,
) -> Tuple[Dict[str, int], Set[str]]:
    """
    First holds out a stratified test set (test_ratio of all patients),
    then applies StratifiedKFold on the remaining patients.

    Returns:
        fold_assignments: dict mapping patient_id -> fold index (train/val patients only)
        test_patients:    set of patient_ids reserved for the test set
    """
    data_root_path = Path(data_root)
    folds_path = Path(folds_file)

    # Load from cache if valid
    if folds_path.exists():
        try:
            with open(folds_path, "r") as f:
                stored = json.load(f)
            if (
                isinstance(stored, dict)
                and stored.get("folds")
                and stored.get("test_patients") is not None
            ):
                fold_assignments = {
                    pid: idx
                    for idx, fold in enumerate(stored["folds"])
                    for pid in fold.get("patients", [])
                }
                test_patients = set(stored["test_patients"])
                if fold_assignments and test_patients:
                    return fold_assignments, test_patients
        except json.JSONDecodeError:
            pass

    patients, labels = _load_patient_labels(data_root_path)

    # Step 1: stratified test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss.split(patients, labels))

    test_patients = {patients[i] for i in test_idx}
    trainval_patients = [patients[i] for i in trainval_idx]
    trainval_labels = [labels[i] for i in trainval_idx]

    # Step 2: 5-fold CV on remaining patients
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    fold_assignments = {}
    folds = []
    for fold_idx, (_, val_idx) in enumerate(skf.split(trainval_patients, trainval_labels)):
        fold_patients = [trainval_patients[i] for i in val_idx]
        folds.append({"patients": fold_patients})
        for pid in fold_patients:
            fold_assignments[pid] = fold_idx

    # Cache to disk
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(folds_path, "w") as f:
        json.dump({"folds": folds, "test_patients": sorted(test_patients)}, f, indent=2)

    return fold_assignments, test_patients


def load_fold_mapping(
    data_root: str,
    folds_file: str,
    num_folds: int = 5,
    seed: int = 42,
    test_ratio: float = 0.2,
) -> Tuple[Dict[str, int], Set[str]]:
    folds_path = Path(folds_file)
    if folds_path.exists():
        try:
            with open(folds_path, "r") as f:
                data = json.load(f)
            if (
                isinstance(data, dict)
                and data.get("folds")
                and data.get("test_patients") is not None
            ):
                fold_assignments = {
                    pid: idx
                    for idx, fold in enumerate(data["folds"])
                    for pid in fold.get("patients", [])
                }
                test_patients = set(data["test_patients"])
                if fold_assignments and test_patients:
                    return fold_assignments, test_patients
        except json.JSONDecodeError:
            pass
    return generate_folds(data_root, folds_file, num_folds=num_folds, seed=seed, test_ratio=test_ratio)
