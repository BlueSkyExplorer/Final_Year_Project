import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without modifying inputs."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def load_and_merge(dataset_config: str, experiment_config: str) -> Dict[str, Any]:
    dataset_cfg = load_config(dataset_config)
    experiment_cfg = load_config(experiment_config)
    merged = merge_dicts(dataset_cfg, experiment_cfg)
    merged.setdefault("seed", dataset_cfg.get("seed", 42))
    merged.setdefault("paths", {})
    merged["paths"].setdefault("dataset_config", dataset_config)
    merged["paths"].setdefault("experiment_config", experiment_config)
    validate_config(merged)
    return merged


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _coerce_float(value: Any, field_name: str, allow_zero: bool = True) -> float:
    """Convert value to float with helpful error messages."""

    try:
        coerced = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field_name}' must be a float, but got {value!r}") from None

    if allow_zero:
        if coerced < 0:
            raise ValueError(f"'{field_name}' must be >= 0, but got {coerced}")
    else:
        if coerced <= 0:
            raise ValueError(f"'{field_name}' must be > 0, but got {coerced}")
    return coerced


def _coerce_int(value: Any, field_name: str) -> int:
    """Convert value to int with helpful error messages."""

    try:
        coerced = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field_name}' must be an integer, but got {value!r}") from None

    if coerced <= 0:
        raise ValueError(f"'{field_name}' must be > 0, but got {coerced}")
    return coerced


def validate_config(cfg: Dict[str, Any]) -> None:
    required_sections = ["paths", "images", "training", "model"]
    missing_sections = [key for key in required_sections if key not in cfg]
    if missing_sections:
        raise ValueError(
            f"Missing required configuration sections: {', '.join(missing_sections)}"
        )

    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("'paths' section must be a dictionary")

    required_paths = ["data_root", "folds_file"]
    missing_paths = [key for key in required_paths if key not in paths]
    if missing_paths:
        raise ValueError(
            f"Missing required path settings: {', '.join(missing_paths)}"
        )

    data_root = Path(paths["data_root"])
    if not data_root.exists():
        raise ValueError(f"Configured data_root does not exist: {data_root}")
    if not data_root.is_dir():
        raise ValueError(f"Configured data_root is not a directory: {data_root}")
    if not os.access(data_root, os.R_OK | os.X_OK):
        raise ValueError(f"Configured data_root is not accessible: {data_root}")

    folds_file = Path(paths["folds_file"])
    if not folds_file.exists():
        raise ValueError(f"Configured folds_file does not exist: {folds_file}")
    if not folds_file.is_file():
        raise ValueError(f"Configured folds_file is not a file: {folds_file}")
    if not os.access(folds_file, os.R_OK):
        raise ValueError(f"Configured folds_file is not readable: {folds_file}")

    training = cfg.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("'training' section must be a dictionary")

    if "learning_rate" in training:
        training["learning_rate"] = _coerce_float(
            training["learning_rate"], "training.learning_rate", allow_zero=False
        )

    if "weight_decay" in training:
        training["weight_decay"] = _coerce_float(
            training["weight_decay"], "training.weight_decay", allow_zero=True
        )

    if "num_epochs" in training:
        training["num_epochs"] = _coerce_int(
            training["num_epochs"], "training.num_epochs"
        )

    if "batch_size" in training:
        training["batch_size"] = _coerce_int(
            training["batch_size"], "training.batch_size"
        )
