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
    return merged


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
