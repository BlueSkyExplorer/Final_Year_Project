import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

FOLD_SCHEMA_VERSION = 2


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


def _build_patient_label_signature(patients: List[str], labels: List[int]) -> str:
    payload = [
        {"patient_id": patient_id, "dominant_label": int(label)}
        for patient_id, label in sorted(zip(patients, labels), key=lambda item: item[0])
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_cache_metadata(
    *,
    data_root: Path,
    seed: int,
    num_folds: int,
    test_ratio: float,
    patient_label_signature: str,
) -> Dict[str, Any]:
    return {
        "schema_version": FOLD_SCHEMA_VERSION,
        "data_root": str(data_root.resolve()),
        "seed": int(seed),
        "num_folds": int(num_folds),
        "test_ratio": float(test_ratio),
        "patient_label_signature": patient_label_signature,
    }


def _build_cache_payload(
    *,
    folds: List[Dict[str, List[str]]],
    test_patients: Set[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "metadata": metadata,
        "folds": folds,
        "test_patients": sorted(test_patients),
    }


def _deserialize_cache_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, int], Set[str], Dict[str, Any]] | None:
    metadata = payload.get("metadata")
    folds = payload.get("folds")
    test_patients_raw = payload.get("test_patients")
    if not isinstance(metadata, dict) or not isinstance(folds, list) or test_patients_raw is None:
        return None
    fold_assignments = {
        pid: idx
        for idx, fold in enumerate(folds)
        for pid in fold.get("patients", [])
    }
    test_patients = set(test_patients_raw)
    return fold_assignments, test_patients, metadata


def _load_cached_payload(folds_path: Path) -> Tuple[Dict[str, int], Set[str], Dict[str, Any]] | None:
    if not folds_path.exists():
        return None
    try:
        stored = json.loads(folds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(stored, dict):
        return None
    return _deserialize_cache_payload(stored)


def _metadata_matches(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    comparable_keys = {
        "schema_version",
        "data_root",
        "seed",
        "num_folds",
        "test_ratio",
        "patient_label_signature",
    }
    return all(expected.get(key) == actual.get(key) for key in comparable_keys)


def generate_folds(
    data_root: str,
    folds_file: str,
    num_folds: int = 5,
    seed: int = 42,
    test_ratio: float = 0.2,
) -> Tuple[Dict[str, int], Set[str], Dict[str, Any]]:
    """
    First holds out a stratified test set (test_ratio of all patients),
    then applies StratifiedKFold on the remaining patients.

    Returns:
        fold_assignments: dict mapping patient_id -> fold index (train/val patients only)
        test_patients:    set of patient_ids reserved for the test set
        metadata:         cache metadata used to validate reuse
    """
    data_root_path = Path(data_root)
    folds_path = Path(folds_file)

    patients, labels = _load_patient_labels(data_root_path)
    if not patients:
        raise ValueError(f"No patient directories found under data_root: {data_root_path}")
    patient_label_signature = _build_patient_label_signature(patients, labels)
    expected_metadata = _build_cache_metadata(
        data_root=data_root_path,
        seed=seed,
        num_folds=num_folds,
        test_ratio=test_ratio,
        patient_label_signature=patient_label_signature,
    )

    cached_payload = _load_cached_payload(folds_path)
    if cached_payload is not None:
        fold_assignments, test_patients, metadata = cached_payload
        if _metadata_matches(expected_metadata, metadata):
            return fold_assignments, test_patients, metadata

    # Step 1: stratified test split
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        trainval_idx, test_idx = next(sss.split(patients, labels))
    except ValueError as exc:
        raise ValueError(
            "Failed to generate stratified test split. "
            f"Check patient-level class counts or adjust cv.test_ratio. Details: {exc}"
        ) from exc

    test_patients = {patients[i] for i in test_idx}
    trainval_patients = [patients[i] for i in trainval_idx]
    trainval_labels = [labels[i] for i in trainval_idx]

    # Step 2: 5-fold CV on remaining patients
    try:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_iter = skf.split(trainval_patients, trainval_labels)
    except ValueError as exc:
        raise ValueError(
            "Failed to configure stratified K-fold split. "
            f"Check patient-level class counts or adjust cv.num_folds. Details: {exc}"
        ) from exc
    fold_assignments = {}
    folds = []
    try:
        for fold_idx, (_, val_idx) in enumerate(split_iter):
            fold_patients = [trainval_patients[i] for i in val_idx]
            folds.append({"patients": fold_patients})
            for pid in fold_patients:
                fold_assignments[pid] = fold_idx
    except ValueError as exc:
        raise ValueError(
            "Failed while generating stratified validation folds. "
            f"Check patient-level class counts or adjust cv.num_folds. Details: {exc}"
        ) from exc

    # Cache to disk
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_cache_payload(
        folds=folds,
        test_patients=test_patients,
        metadata=expected_metadata,
    )
    with open(folds_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return fold_assignments, test_patients, expected_metadata


def load_fold_mapping(
    data_root: str,
    folds_file: str,
    num_folds: int = 5,
    seed: int = 42,
    test_ratio: float = 0.2,
) -> Tuple[Dict[str, int], Set[str], Dict[str, Any]]:
    return generate_folds(data_root, folds_file, num_folds=num_folds, seed=seed, test_ratio=test_ratio)
