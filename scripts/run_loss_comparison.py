#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

LOSS_PARAM_DEFAULTS = {
    "focal": {
        "gamma": [1.5, 2.0, 2.5],
        "alpha": [0.25, 0.5],
    },
    "huber": {
        "delta": [0.5, 1.0, 1.5],
    },
    "distance": {
        "alpha": [3, 5, 7],
    },
}

TRAIN_SCRIPT_BY_PARADIGM = {
    "multiclass": "train_multiclass.py",
    "ordinal": "train_ordinal.py",
    "regression": "train_regression.py",
}


@dataclass
class Candidate:
    candidate_id: str
    config_path: Path
    hp_source: str
    lr: float
    wd: float
    freeze_epochs: int
    batch_size: int
    loss_name: str
    loss_params: Dict[str, float]



def parse_float_list(raw: str) -> List[float]:
    return [float(v.strip()) for v in raw.split(",") if v.strip()]



def parse_int_list(raw: str) -> List[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]



def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def dump_yaml(path: Path, content: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)



def get_best_qwk(result_root: Path, experiment_name: str, fold: int) -> float | None:
    metrics_path = result_root / experiment_name / f"fold_{fold}" / "metrics.json"
    if not metrics_path.exists():
        return None

    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_qwk = max((float(ep.get("qwk", float("-inf"))) for ep in history), default=float("-inf"))
    if best_qwk == float("-inf"):
        return None
    return best_qwk



def build_candidates(
    *,
    base_cfg: dict,
    config_stem: str,
    baseline_lr: float,
    baseline_wd: float,
    baseline_freeze_epochs: int,
    baseline_batch_size: int,
    lr_grid: Iterable[float],
    wd_grid: Iterable[float],
    temp_dir: Path,
) -> List[Candidate]:
    model_cfg = base_cfg.get("model", {})
    loss_name = str(model_cfg.get("loss", "")).strip().lower()

    candidates: List[Candidate] = []

    # Baseline shared HP candidate.
    shared_cfg = json.loads(json.dumps(base_cfg))
    shared_cfg["experiment_name"] = f"{config_stem}_shared_hp_preview"
    shared_cfg.setdefault("training", {})
    shared_cfg["training"]["learning_rate"] = baseline_lr
    shared_cfg["training"]["lr"] = baseline_lr
    shared_cfg["training"]["weight_decay"] = baseline_wd
    shared_cfg["training"]["freeze_epochs"] = baseline_freeze_epochs
    shared_cfg["training"]["batch_size"] = baseline_batch_size

    shared_path = temp_dir / f"{config_stem}_shared_hp.yaml"
    dump_yaml(shared_path, shared_cfg)
    candidates.append(
        Candidate(
            candidate_id="shared_hp",
            config_path=shared_path,
            hp_source="shared_hp",
            lr=baseline_lr,
            wd=baseline_wd,
            freeze_epochs=baseline_freeze_epochs,
            batch_size=baseline_batch_size,
            loss_name=loss_name,
            loss_params={
                k: float(v) for k, v in model_cfg.items() if k in LOSS_PARAM_DEFAULTS.get(loss_name, {})
            },
        )
    )

    # per-loss tuned grid.
    param_grid = LOSS_PARAM_DEFAULTS.get(loss_name, {})
    param_keys = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_keys]
    param_product = list(itertools.product(*param_values)) if param_keys else [tuple()]

    seen = {
        (
            baseline_lr,
            baseline_wd,
            tuple(sorted(candidates[0].loss_params.items())),
        )
    }
    idx = 0
    for lr, wd, pvals in itertools.product(lr_grid, wd_grid, param_product):
        loss_params = dict(zip(param_keys, pvals))
        key = (float(lr), float(wd), tuple(sorted((k, float(v)) for k, v in loss_params.items())))
        if key in seen:
            continue
        seen.add(key)

        idx += 1
        tuned_cfg = json.loads(json.dumps(base_cfg))
        tuned_cfg["experiment_name"] = f"{config_stem}_per_loss_tuned_{idx}"
        tuned_cfg.setdefault("training", {})
        tuned_cfg["training"]["learning_rate"] = float(lr)
        tuned_cfg["training"]["lr"] = float(lr)
        tuned_cfg["training"]["weight_decay"] = float(wd)
        tuned_cfg["training"]["freeze_epochs"] = baseline_freeze_epochs
        tuned_cfg["training"]["batch_size"] = baseline_batch_size

        tuned_cfg.setdefault("model", {})
        for pk, pv in loss_params.items():
            tuned_cfg["model"][pk] = float(pv)

        tuned_path = temp_dir / f"{config_stem}_per_loss_tuned_{idx}.yaml"
        dump_yaml(tuned_path, tuned_cfg)

        candidates.append(
            Candidate(
                candidate_id=f"per_loss_tuned_{idx}",
                config_path=tuned_path,
                hp_source="per_loss_tuned",
                lr=float(lr),
                wd=float(wd),
                freeze_epochs=baseline_freeze_epochs,
                batch_size=baseline_batch_size,
                loss_name=loss_name,
                loss_params={k: float(v) for k, v in loss_params.items()},
            )
        )

    return candidates



def run_train(script: str, config_path: Path, fold: int) -> None:
    cmd = ["python", script, "--config", str(config_path), "--fold", str(fold)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)



def summarize_preview(
    *,
    result_root: Path,
    preview_folds: List[int],
    loss_config: Path,
    candidates: List[Candidate],
) -> List[dict]:
    rows = []
    for c in candidates:
        exp_name = load_yaml(c.config_path).get("experiment_name")
        fold_qwk = {fold: get_best_qwk(result_root, exp_name, fold) for fold in preview_folds}
        available = [q for q in fold_qwk.values() if q is not None]
        mean_qwk = statistics.fmean(available) if available else float("-inf")
        std_qwk = statistics.pstdev(available) if len(available) > 1 else 0.0
        rows.append(
            {
                "loss_config": str(loss_config),
                "loss_name": c.loss_name,
                "experiment_name": exp_name,
                "candidate_id": c.candidate_id,
                "hp_source": c.hp_source,
                "learning_rate": c.lr,
                "weight_decay": c.wd,
                "freeze_epochs": c.freeze_epochs,
                "batch_size": c.batch_size,
                "loss_params": json.dumps(c.loss_params, ensure_ascii=False),
                "preview_fold_scores": json.dumps(fold_qwk, ensure_ascii=False),
                "mean_qwk": mean_qwk,
                "std_qwk": std_qwk,
            }
        )

    rows.sort(key=lambda r: (-r["mean_qwk"], r["std_qwk"], r["candidate_id"]))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows



def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def main() -> None:
    parser = argparse.ArgumentParser(description="Per-loss tuning before formal 5-fold loss comparison.")
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment config yaml paths for each loss.")
    parser.add_argument("--baseline-lr", type=float, required=True)
    parser.add_argument("--baseline-wd", type=float, required=True)
    parser.add_argument("--baseline-freeze-epochs", type=int, default=5)
    parser.add_argument("--baseline-batch-size", type=int, default=16)
    parser.add_argument("--lr-grid", type=str, default="1e-4,3e-4,1e-3")
    parser.add_argument("--wd-grid", type=str, default="1e-5,1e-4,3e-4")
    parser.add_argument("--preview-folds", type=str, default="0,1")
    parser.add_argument("--full-folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output-root", type=str, default="results/loss_comparison")
    args = parser.parse_args()

    lr_grid = parse_float_list(args.lr_grid)
    wd_grid = parse_float_list(args.wd_grid)
    preview_folds = parse_int_list(args.preview_folds)
    full_folds = parse_int_list(args.full_folds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"per_loss_tuning_{timestamp}"
    temp_dir = run_dir / "generated_configs"
    run_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    all_preview_rows: List[dict] = []
    selected_rows: List[dict] = []
    final_rows: List[dict] = []

    for cfg_path_str in args.configs:
        cfg_path = Path(cfg_path_str)
        cfg = load_yaml(cfg_path)
        paradigm = cfg.get("model", {}).get("paradigm", "")
        train_script = TRAIN_SCRIPT_BY_PARADIGM.get(paradigm)
        if not train_script:
            raise ValueError(f"Unsupported paradigm in {cfg_path}: {paradigm}")

        config_stem = cfg_path.stem
        candidates = build_candidates(
            base_cfg=cfg,
            config_stem=config_stem,
            baseline_lr=args.baseline_lr,
            baseline_wd=args.baseline_wd,
            baseline_freeze_epochs=args.baseline_freeze_epochs,
            baseline_batch_size=args.baseline_batch_size,
            lr_grid=lr_grid,
            wd_grid=wd_grid,
            temp_dir=temp_dir,
        )

        for cand in candidates:
            for fold in preview_folds:
                run_train(train_script, cand.config_path, fold)

        preview_rows = summarize_preview(
            result_root=Path("results"),
            preview_folds=preview_folds,
            loss_config=cfg_path,
            candidates=candidates,
        )
        all_preview_rows.extend(preview_rows)

        tuned_rows = [r for r in preview_rows if r["hp_source"] == "per_loss_tuned"]
        if not tuned_rows:
            tuned_rows = [r for r in preview_rows if r["hp_source"] == "shared_hp"]

        picked = tuned_rows[: max(1, args.top_k)]
        selected_rows.extend(picked)

        exp_to_candidate = {load_yaml(c.config_path)["experiment_name"]: c for c in candidates}
        for chosen in picked:
            chosen_exp = chosen["experiment_name"]
            cand = exp_to_candidate[chosen_exp]
            for fold in full_folds:
                run_train(train_script, cand.config_path, fold)

            fold_scores = {fold: get_best_qwk(Path("results"), chosen_exp, fold) for fold in full_folds}
            vals = [v for v in fold_scores.values() if v is not None]
            final_rows.append(
                {
                    "loss_config": str(cfg_path),
                    "loss_name": cand.loss_name,
                    "experiment_name": chosen_exp,
                    "hp_source": cand.hp_source,
                    "learning_rate": cand.lr,
                    "weight_decay": cand.wd,
                    "freeze_epochs": cand.freeze_epochs,
                    "batch_size": cand.batch_size,
                    "loss_params": json.dumps(cand.loss_params, ensure_ascii=False),
                    "full_fold_scores": json.dumps(fold_scores, ensure_ascii=False),
                    "mean_qwk": statistics.fmean(vals) if vals else float("-inf"),
                    "std_qwk": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                }
            )

    write_csv(run_dir / "preview_ranking.csv", all_preview_rows)
    write_csv(run_dir / "selected_candidates.csv", selected_rows)
    write_csv(run_dir / "final_5fold_summary.csv", final_rows)

    print(f"Done. Artifacts under: {run_dir}")


if __name__ == "__main__":
    main()
