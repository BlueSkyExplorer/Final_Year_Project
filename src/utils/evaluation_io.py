import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def persist_json_atomic(payload: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def write_predictions_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No prediction rows available for {path}")
    fieldnames = list(rows[0].keys())
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def dataset_patient_ids(dataset) -> list[str]:
    return sorted({str(patient_id) for patient_id in dataset.samples["patient_id"].tolist()})


def build_split_manifest(
    *,
    cfg: dict[str, Any],
    train_ds,
    val_ds,
    test_ds,
) -> dict[str, Any]:
    fold_metadata = getattr(train_ds, "fold_metadata", {}) or getattr(val_ds, "fold_metadata", {}) or getattr(test_ds, "fold_metadata", {})
    serialized_metadata = json.dumps(fold_metadata, sort_keys=True, separators=(",", ":"))
    return {
        "current_fold": int(cfg.get("cv", {}).get("current_fold", 0)),
        "paths": {
            "data_root": str(cfg["paths"]["data_root"]),
            "folds_file": str(cfg["paths"]["folds_file"]),
        },
        "fold_metadata": fold_metadata,
        "fold_metadata_hash": hashlib.sha256(serialized_metadata.encode("utf-8")).hexdigest(),
        "train_patients": dataset_patient_ids(train_ds),
        "val_patients": dataset_patient_ids(val_ds),
        "test_patients": dataset_patient_ids(test_ds),
    }
