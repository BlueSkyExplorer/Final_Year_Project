from pathlib import Path


def resolve_output_dir(cfg) -> Path:
    output_root = Path(cfg["paths"].get("output_root", "results"))
    exp_name = cfg.get("experiment_name", Path(cfg["paths"].get("experiment_config", "exp")).stem)
    fold = cfg.get("cv", {}).get("current_fold", 0)
    out_dir = output_root / exp_name / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
