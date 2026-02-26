#!/usr/bin/env python3
"""Two-layer sweep orchestrator (Python version)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

EVAL_FOLDS = [0, 1]
BATCH_SIZE = 16
PROMOTION_TOP_K = 4
PROMOTION_RATIO: float | None = None
LAYER2_LRS = [1e-3, 7e-4, 5e-4, 3e-4, 2e-4, 1e-4]
LAYER2_WDS = [1e-4, 7e-5, 5e-5, 3e-5, 1e-5]


@dataclass(frozen=True)
class MemberGrid:
    lrs: list[float]
    wds: list[float]
    freeze_epochs: list[int]


MEMBER_GRIDS: dict[str, MemberGrid] = {
    "A": MemberGrid(lrs=[1e-3, 3e-4], wds=[1e-4, 1e-5], freeze_epochs=[0]),
    "B": MemberGrid(lrs=[1e-3, 3e-4], wds=[1e-4, 1e-5], freeze_epochs=[5]),
    "C": MemberGrid(lrs=[1e-4, 5e-5], wds=[1e-4, 1e-5], freeze_epochs=[0, 5]),
    "ALL": MemberGrid(lrs=[1e-3, 3e-4], wds=[1e-4, 1e-5], freeze_epochs=[0, 5]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer1/Layer2 sweep orchestrator")
    parser.add_argument("--member-id", default="A", choices=sorted(MEMBER_GRIDS.keys()))
    parser.add_argument(
        "--base-config",
        default="configs/experiments/multiclass_resnet18_ce.yaml",
    )
    parser.add_argument("--resume-sweep-dir", default="")
    return parser.parse_args()


def get_train_script(base_config: Path) -> tuple[str, str]:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    paradigm = cfg.get("model", {}).get("paradigm", "")
    mapping = {
        "multiclass": "train_multiclass.py",
        "ordinal": "train_ordinal.py",
        "regression": "train_regression.py",
    }
    if paradigm not in mapping:
        raise ValueError(f"Unsupported paradigm '{paradigm}' in {base_config}.")
    return paradigm, mapping[paradigm]


def build_config(
    *,
    base_config: Path,
    exp_name: str,
    lr: float,
    wd: float,
    freeze_epochs: int,
    out_cfg: Path,
    batch_size: int,
) -> None:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["experiment_name"] = exp_name
    cfg.setdefault("training", {})
    cfg["training"]["learning_rate"] = float(lr)
    cfg["training"]["lr"] = float(lr)
    cfg["training"]["weight_decay"] = float(wd)
    cfg["training"]["batch_size"] = int(batch_size)
    cfg["training"]["freeze_epochs"] = int(freeze_epochs)
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def append_registry_row(registry_path: Path, row: dict[str, Any]) -> None:
    row = {
        "ts": datetime.utcnow().isoformat() + "Z",
        **row,
    }
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_registry_latest(registry_path: Path) -> dict[tuple[str, float, float, int, int], dict[str, Any]]:
    latest: dict[tuple[str, float, float, int, int], dict[str, Any]] = {}
    if not registry_path.exists():
        return latest
    ignored_rows = 0
    for raw in registry_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            ignored_rows += 1
            continue

        try:
            key = (
                row.get("phase"),
                float(row.get("lr")),
                float(row.get("weight_decay")),
                int(row.get("freeze_epochs")),
                int(row.get("fold")),
            )
        except (TypeError, ValueError):
            ignored_rows += 1
            continue
        latest[key] = row

    if ignored_rows > 0:
        print(f"[WARN] read_registry_latest ignored {ignored_rows} malformed row(s): {registry_path}")
    return latest


def should_skip_completed_fold(
    registry_path: Path,
    *,
    phase: str,
    lr: float,
    wd: float,
    freeze_epochs: int,
    fold: int,
) -> bool:
    latest = read_registry_latest(registry_path)
    row = latest.get((phase, float(lr), float(wd), int(freeze_epochs), int(fold)))
    return bool(row and row.get("status") == "completed")


def read_best_metric(result_root: Path, exp_name: str, fold: int) -> float | None:
    metrics_path = result_root / exp_name / f"fold_{fold}" / "metrics.json"
    if not metrics_path.exists():
        return None
    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    vals = [float(ep.get("qwk")) for ep in history if ep.get("qwk") is not None]
    return max(vals) if vals else None


def tail_error(log_path: Path) -> str:
    if not log_path.exists():
        return "training exited without logs"
    lines = [ln.strip() for ln in log_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    tail = lines[-8:] if lines else ["training exited without logs"]
    return " | ".join(tail).replace('"', "'")


def is_oom(log_path: Path) -> bool:
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower() if log_path.exists() else ""
    return (
        "out of memory" in text
        or "cuda out of memory" in text
        or "cudnn_status_alloc_failed" in text
    )


def run_single_fold_with_guard(
    *,
    phase: str,
    exp_name: str,
    cfg_path: Path,
    lr: float,
    wd: float,
    freeze_epochs: int,
    fold: int,
    base_config: Path,
    train_script: str,
    result_root: Path,
    sweep_dir: Path,
    registry_path: Path,
) -> None:
    current_bs = BATCH_SIZE
    run_status = "failed"
    run_error = ""
    best_metric: float | None = None

    while True:
        build_config(
            base_config=base_config,
            exp_name=exp_name,
            lr=lr,
            wd=wd,
            freeze_epochs=freeze_epochs,
            out_cfg=cfg_path,
            batch_size=current_bs,
        )
        run_log = sweep_dir / f"{exp_name}_fold{fold}_bs{current_bs}.log"
        print(f"  -> fold {fold} (batch_size={current_bs})")
        with run_log.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                [sys.executable, train_script, "--config", str(cfg_path), "--fold", str(fold)],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if proc.returncode == 0:
            run_status = "completed"
            best_metric = read_best_metric(result_root, exp_name, fold)
            run_error = ""
            break

        run_error = tail_error(run_log)
        if is_oom(run_log):
            if current_bs > 1:
                current_bs = (current_bs + 1) // 2
                print(f"    OOM detected, fallback batch_size -> {current_bs} and retry")
                continue
            run_error = f"OOM persists at batch_size=1. {run_error}"

        run_status = "failed"
        break

    append_registry_row(
        registry_path,
        {
            "phase": phase,
            "experiment_name": exp_name,
            "lr": float(lr),
            "weight_decay": float(wd),
            "freeze_epochs": int(freeze_epochs),
            "fold": int(fold),
            "metric": best_metric,
            "status": run_status,
            "error": run_error if run_error else None,
            "batch_size": int(current_bs),
        },
    )


def run_combo_all_folds(**kwargs: Any) -> None:
    print(f"[Run {kwargs['phase']}:{kwargs['exp_name']}] eval folds: {EVAL_FOLDS}")
    for fold in EVAL_FOLDS:
        if should_skip_completed_fold(
            kwargs["registry_path"],
            phase=kwargs["phase"],
            lr=kwargs["lr"],
            wd=kwargs["wd"],
            freeze_epochs=kwargs["freeze_epochs"],
            fold=fold,
        ):
            print(f"  -> fold {fold} already completed in registry, skip (resume).")
            continue
        run_single_fold_with_guard(fold=fold, **kwargs)


def load_tsv_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row:
                rows.append(row)
    return rows


def write_tsv_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def summarize_layer1(sweep_dir: Path, registry_path: Path) -> None:
    if PROMOTION_RATIO is not None and not (0 < PROMOTION_RATIO <= 1):
        raise ValueError("PROMOTION_RATIO must be in (0,1].")

    latest = read_registry_latest(registry_path)
    rows: list[dict[str, Any]] = []
    for combo in load_tsv_rows(sweep_dir / "layer1_combos.tsv"):
        exp_name, lr, wd, frz = combo
        lr_f = float(lr)
        wd_f = float(wd)
        frz_i = int(frz)

        fold_qwks: dict[int, float] = {}
        failed_reasons: list[str] = []
        fold_states: list[str] = []
        for fold in EVAL_FOLDS:
            rec = latest.get(("layer1", lr_f, wd_f, frz_i, fold))
            if not rec:
                failed_reasons.append(f"fold_{fold}:not_run")
                fold_states.append("not_run")
                continue
            status = rec.get("status")
            fold_states.append(status)
            if status == "completed" and rec.get("metric") is not None:
                fold_qwks[fold] = float(rec["metric"])
            elif status == "failed":
                failed_reasons.append(f"fold_{fold}:{rec.get('error') or 'failed_without_message'}")
            else:
                failed_reasons.append(f"fold_{fold}:{status}")

        rank_vals = [fold_qwks[f] for f in EVAL_FOLDS if f in fold_qwks]
        mean_qwk = statistics.fmean(rank_vals) if rank_vals else float("-inf")
        std_qwk = statistics.pstdev(rank_vals) if len(rank_vals) > 1 else 0.0

        if all(s == "completed" for s in fold_states):
            combo_status = "completed"
        elif any(s == "failed" for s in fold_states):
            combo_status = "failed"
        else:
            combo_status = "partial"

        rows.append(
            {
                "experiment_name": exp_name,
                "lr": lr_f,
                "weight_decay": wd_f,
                "freeze_epochs": frz_i,
                "fold0_qwk": fold_qwks.get(0),
                "fold1_qwk": fold_qwks.get(1),
                "mean_qwk": mean_qwk,
                "std_qwk": std_qwk,
                "combo_status": combo_status,
                "failed_reason": " | ".join(failed_reasons) if failed_reasons else "",
            }
        )

    rows.sort(key=lambda r: (-r["mean_qwk"], r["std_qwk"], r["experiment_name"]))
    total = len(rows)

    if PROMOTION_RATIO is not None:
        promoted_n = max(1, math.ceil(total * PROMOTION_RATIO))
        policy_text = f"ratio={PROMOTION_RATIO:.2f}"
    else:
        promoted_n = min(PROMOTION_TOP_K, total)
        policy_text = f"top_k={PROMOTION_TOP_K}"

    for i, row in enumerate(rows, start=1):
        promoted = i <= promoted_n and row["combo_status"] == "completed"
        row["rank"] = i
        row["promoted"] = promoted
        row["promotion_reason"] = (
            f"promoted_by_{policy_text}; rank={i}/{total}; sort=(-mean_qwk,+std_qwk)"
            if promoted
            else f"not_promoted_by_{policy_text}; rank={i}/{total}; sort=(-mean_qwk,+std_qwk); status={row['combo_status']}"
        )

    csv_path = sweep_dir / "layer1_ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank",
            "experiment_name",
            "lr",
            "weight_decay",
            "freeze_epochs",
            "fold0_qwk",
            "fold1_qwk",
            "mean_qwk",
            "std_qwk",
            "combo_status",
            "promoted",
            "promotion_reason",
            "failed_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    jsonl_path = sweep_dir / "layer1_ranking.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    promoted_rows = [r for r in rows if r["promoted"]]
    (sweep_dir / "promoted_layer1.json").write_text(
        json.dumps(promoted_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved ranking CSV: {csv_path}")
    print(f"Saved ranking JSONL: {jsonl_path}")
    print(f"Promoted: {len(promoted_rows)}/{total}")


def nearest_idx(values: list[float], target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def build_layer2_expansions(sweep_dir: Path) -> list[dict[str, Any]]:
    promoted_path = sweep_dir / "promoted_layer1.json"
    promoted = json.loads(promoted_path.read_text(encoding="utf-8")) if promoted_path.exists() else []

    expansions = []
    for p in promoted:
        base_lr = float(p["lr"])
        base_wd = float(p["weight_decay"])
        frz = int(p["freeze_epochs"])

        i = nearest_idx(LAYER2_LRS, base_lr)
        j = nearest_idx(LAYER2_WDS, base_wd)

        lr_neighbors = sorted({LAYER2_LRS[k] for k in [i - 1, i, i + 1] if 0 <= k < len(LAYER2_LRS)})
        wd_neighbors = sorted({LAYER2_WDS[k] for k in [j - 1, j, j + 1] if 0 <= k < len(LAYER2_WDS)})

        for lr2 in lr_neighbors:
            for wd2 in wd_neighbors:
                expansions.append(
                    {
                        "from_experiment": p["experiment_name"],
                        "from_rank": p["rank"],
                        "from_reason": p["promotion_reason"],
                        "lr": lr2,
                        "weight_decay": wd2,
                        "freeze_epochs": frz,
                    }
                )

    uniq: dict[tuple[float, float, int], dict[str, Any]] = {}
    for entry in expansions:
        key = (float(entry["lr"]), float(entry["weight_decay"]), int(entry["freeze_epochs"]))
        if key not in uniq:
            uniq[key] = entry

    layer2 = list(uniq.values())
    (sweep_dir / "layer2_expansions.json").write_text(
        json.dumps(layer2, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Layer 2 expansions (deduped): {len(layer2)}")
    return layer2


def main() -> None:
    args = parse_args()
    member_grid = MEMBER_GRIDS[args.member_id]
    base_config = Path(args.base_config)
    paradigm, train_script = get_train_script(base_config)

    result_root = Path("results")
    if args.resume_sweep_dir:
        sweep_dir = Path(args.resume_sweep_dir)
        print(f"[Resume] using existing sweep dir: {sweep_dir}")
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = result_root / "sweeps" / f"layer1_layer2_{run_tag}"

    sweep_dir.mkdir(parents=True, exist_ok=True)
    run_tag_file = sweep_dir / "run_tag.txt"
    if run_tag_file.exists():
        run_tag = run_tag_file.read_text(encoding="utf-8").strip()
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag_file.write_text(f"{run_tag}\n", encoding="utf-8")

    registry_path = sweep_dir / "sweep_registry.jsonl"
    registry_path.touch(exist_ok=True)

    print(
        f"[Config] MEMBER_ID={args.member_id}, BASE_CONFIG={base_config}, "
        f"PARADIGM={paradigm}, TRAIN_SCRIPT={train_script}"
    )
    print(
        f"[Config] Layer1 LRs={member_grid.lrs} WDs={member_grid.wds} "
        f"FREEZE_EPOCHS={member_grid.freeze_epochs}"
    )

    with tempfile.TemporaryDirectory() as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)

        combos_path = sweep_dir / "layer1_combos.tsv"
        combo_rows = load_tsv_rows(combos_path)
        existing_combo_names = {row[0] for row in combo_rows if len(row) >= 1}

        combo_idx = 0
        for lr in member_grid.lrs:
            for wd in member_grid.wds:
                for frz in member_grid.freeze_epochs:
                    combo_idx += 1
                    exp_name = f"layer1_{run_tag}_combo_{combo_idx}"
                    combo_cfg = tmp_dir / f"{exp_name}.yaml"

                    print(f"[Layer1 {combo_idx}] lr={lr}, wd={wd}, batch_size={BATCH_SIZE}, freeze_epochs={frz}")
                    run_combo_all_folds(
                        phase="layer1",
                        exp_name=exp_name,
                        cfg_path=combo_cfg,
                        lr=lr,
                        wd=wd,
                        freeze_epochs=frz,
                        base_config=base_config,
                        train_script=train_script,
                        result_root=result_root,
                        sweep_dir=sweep_dir,
                        registry_path=registry_path,
                    )

                    if exp_name not in existing_combo_names:
                        combo_rows.append([exp_name, str(lr), str(wd), str(frz)])
                        existing_combo_names.add(exp_name)

        write_tsv_rows(combos_path, combo_rows)

        summarize_layer1(sweep_dir, registry_path)
        layer2 = build_layer2_expansions(sweep_dir)

        if not layer2:
            print("No Layer 2 expansions generated. Done.")
            return

        for idx, cfg in enumerate(layer2, start=1):
            exp_name = f"layer2_{run_tag}_combo_{idx}"
            combo_cfg = tmp_dir / f"{exp_name}.yaml"
            lr = float(cfg["lr"])
            wd = float(cfg["weight_decay"])
            frz = int(cfg["freeze_epochs"])

            print(f"[Layer2 {idx}/{len(layer2)}] lr={lr}, wd={wd}, freeze_epochs={frz}")
            run_combo_all_folds(
                phase="layer2",
                exp_name=exp_name,
                cfg_path=combo_cfg,
                lr=lr,
                wd=wd,
                freeze_epochs=frz,
                base_config=base_config,
                train_script=train_script,
                result_root=result_root,
                sweep_dir=sweep_dir,
                registry_path=registry_path,
            )

    print(f"Sweep complete. Artifacts under: {sweep_dir}")


if __name__ == "__main__":
    main()
