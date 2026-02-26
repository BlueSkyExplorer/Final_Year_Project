#!/usr/bin/env bash
set -euo pipefail

# Two-layer sweep orchestrator
# - Layer 1: coarse sweep and ranking by dual-fold mean_qwk, std_qwk tie-breaker.
# - Promotion: fixed Top-K or ratio.
# - Layer 2: only promoted combos perform ±1 LR/WD neighborhood expansion.
#
# Resume usage:
#   RESUME_SWEEP_DIR=results/sweeps/layer1_layer2_YYYYmmdd_HHMMSS bash scripts/run_layer1_sweep.sh

BASE_CONFIG="${BASE_CONFIG:-configs/experiments/multiclass_resnet18_ce.yaml}"
EVAL_FOLDS=(0 1)
BATCH_SIZE=16

# Static workload split for collaboration.
# Example:
#   MEMBER_ID=A bash scripts/run_layer1_sweep.sh
#   MEMBER_ID=B bash scripts/run_layer1_sweep.sh
MEMBER_ID="${MEMBER_ID:-A}"
case "${MEMBER_ID}" in
  A)
    LAYER1_LRS=(1e-3 3e-4)
    LAYER1_WDS=(1e-4 1e-5)
    FREEZE_EPOCHS=(0)
    ;;
  B)
    LAYER1_LRS=(1e-3 3e-4)
    LAYER1_WDS=(1e-4 1e-5)
    FREEZE_EPOCHS=(5)
    ;;
  C)
    LAYER1_LRS=(1e-4 5e-5)
    LAYER1_WDS=(1e-4 1e-5)
    FREEZE_EPOCHS=(0 5)
    ;;
  ALL)
    LAYER1_LRS=(1e-3 3e-4)
    LAYER1_WDS=(1e-4 1e-5)
    FREEZE_EPOCHS=(0 5)
    ;;
  *)
    echo "[ERROR] Unsupported MEMBER_ID=${MEMBER_ID}. Use A/B/C/ALL."
    exit 1
    ;;
esac

PARADIGM=$(python - <<PY
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path("${BASE_CONFIG}").read_text(encoding="utf-8"))
print(cfg.get("model", {}).get("paradigm", ""))
PY
)
case "${PARADIGM}" in
  multiclass)
    TRAIN_SCRIPT="train_multiclass.py"
    ;;
  ordinal)
    TRAIN_SCRIPT="train_ordinal.py"
    ;;
  regression)
    TRAIN_SCRIPT="train_regression.py"
    ;;
  *)
    echo "[ERROR] Unsupported paradigm '${PARADIGM}' in ${BASE_CONFIG}."
    exit 1
    ;;
esac

# Promotion policy (choose one)
PROMOTION_TOP_K=4              # fixed small top-k
PROMOTION_RATIO=""             # e.g. 0.4 ; takes precedence when set

# Layer 2 neighborhood grids (used for ±1 expansion around promoted combo's LR/WD)
LAYER2_LRS=(1e-3 7e-4 5e-4 3e-4 2e-4 1e-4)
LAYER2_WDS=(1e-4 7e-5 5e-5 3e-5 1e-5)

RESULT_ROOT="results"
if [[ -n "${RESUME_SWEEP_DIR:-}" ]]; then
  SWEEP_DIR="${RESUME_SWEEP_DIR}"
  echo "[Resume] using existing sweep dir: ${SWEEP_DIR}"
else
  RUN_TAG="$(date +%Y%m%d_%H%M%S)"
  SWEEP_DIR="${RESULT_ROOT}/sweeps/layer1_layer2_${RUN_TAG}"
fi
mkdir -p "${SWEEP_DIR}"
RUN_TAG_FILE="${SWEEP_DIR}/run_tag.txt"
if [[ -f "${RUN_TAG_FILE}" ]]; then
  RUN_TAG="$(<"${RUN_TAG_FILE}")"
else
  if [[ -z "${RUN_TAG:-}" ]]; then
    RUN_TAG="$(date +%Y%m%d_%H%M%S)"
  fi
  printf '%s\n' "${RUN_TAG}" > "${RUN_TAG_FILE}"
fi
REGISTRY_PATH="${SWEEP_DIR}/sweep_registry.jsonl"
touch "${REGISTRY_PATH}"

echo "[Config] MEMBER_ID=${MEMBER_ID}, BASE_CONFIG=${BASE_CONFIG}, PARADIGM=${PARADIGM}, TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[Config] Layer1 LRs=${LAYER1_LRS[*]} WDs=${LAYER1_WDS[*]} FREEZE_EPOCHS=${FREEZE_EPOCHS[*]}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

build_config() {
  local exp_name="$1"
  local lr="$2"
  local wd="$3"
  local frz="$4"
  local out_cfg="$5"
  local batch_size="$6"

  python - <<PY
import yaml
from pathlib import Path

base_path = Path("${BASE_CONFIG}")
out_path = Path("${out_cfg}")

cfg = yaml.safe_load(base_path.read_text())
cfg["experiment_name"] = "${exp_name}"
cfg.setdefault("training", {})
cfg["training"]["learning_rate"] = float("${lr}")
# Backward compatibility if old code still reads `lr`
cfg["training"]["lr"] = float("${lr}")
cfg["training"]["weight_decay"] = float("${wd}")
cfg["training"]["batch_size"] = int("${batch_size}")
cfg["training"]["freeze_epochs"] = int("${frz}")

out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY
}

append_registry_row() {
  local phase="$1"
  local exp_name="$2"
  local lr="$3"
  local wd="$4"
  local frz="$5"
  local fold="$6"
  local metric="$7"
  local status="$8"
  local error_msg="$9"
  local used_batch_size="${10}"

  python - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

row = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "phase": "${phase}",
    "experiment_name": "${exp_name}",
    "lr": float("${lr}"),
    "weight_decay": float("${wd}"),
    "freeze_epochs": int("${frz}"),
    "fold": int("${fold}"),
    "metric": None if "${metric}" == "" else float("${metric}"),
    "status": "${status}",
    "error": "${error_msg}" if "${error_msg}" else None,
    "batch_size": int("${used_batch_size}"),
}

path = Path("${REGISTRY_PATH}")
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\\n")
PY
}

should_skip_completed_fold() {
  local phase="$1"
  local lr="$2"
  local wd="$3"
  local frz="$4"
  local fold="$5"
  python - <<PY
import json
from pathlib import Path

registry = Path("${REGISTRY_PATH}")
phase = "${phase}"
lr = float("${lr}")
wd = float("${wd}")
frz = int("${frz}")
fold = int("${fold}")

if not registry.exists():
    print("0")
    raise SystemExit

for raw in registry.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    row = json.loads(raw)
    try:
        row_lr = float(row.get("lr", -1))
        row_wd = float(row.get("weight_decay", -1))
    except (TypeError, ValueError):
        continue
    if (
        row.get("phase") == phase
        and abs(row_lr - lr) < 1e-12
        and abs(row_wd - wd) < 1e-12
        and int(row.get("freeze_epochs", -1)) == frz
        and int(row.get("fold", -1)) == fold
        and row.get("status") == "completed"
    ):
        print("1")
        break
else:
    print("0")
PY
}

run_single_fold_with_guard() {
  local phase="$1"
  local exp_name="$2"
  local cfg_path="$3"
  local lr="$4"
  local wd="$5"
  local frz="$6"
  local fold="$7"

  local current_bs="${BATCH_SIZE}"
  local run_status="failed"
  local run_error=""
  local best_metric=""

  while true; do
    build_config "${exp_name}" "${lr}" "${wd}" "${frz}" "${cfg_path}" "${current_bs}"
    local run_log="${SWEEP_DIR}/${exp_name}_fold${fold}_bs${current_bs}.log"

    echo "  -> fold ${fold} (batch_size=${current_bs})"
    set +e
    python "${TRAIN_SCRIPT}" --config "${cfg_path}" --fold "${fold}" > "${run_log}" 2>&1
    local rc=$?
    set -e

    if [[ "${rc}" -eq 0 ]]; then
      run_status="completed"
      best_metric=$(python - <<PY
import json
from pathlib import Path
metrics_path = Path("${RESULT_ROOT}") / "${exp_name}" / "fold_${fold}" / "metrics.json"
if not metrics_path.exists():
    print("")
    raise SystemExit
history = json.loads(metrics_path.read_text(encoding="utf-8"))
vals = [float(ep.get("qwk")) for ep in history if ep.get("qwk") is not None]
print(max(vals) if vals else "")
PY
)
      run_error=""
      break
    fi

    run_error=$(python - <<PY
from pathlib import Path
log_path = Path("${run_log}")
text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
tail = lines[-8:] if lines else ["training exited without logs"]
print(" | ".join(tail).replace('"', "'"))
PY
)

    local oom_flag
    oom_flag=$(python - <<PY
from pathlib import Path
text = Path("${run_log}").read_text(encoding="utf-8", errors="ignore").lower()
oom = ("out of memory" in text) or ("cuda out of memory" in text) or ("cudnn_status_alloc_failed" in text)
print("1" if oom else "0")
PY
)

    if [[ "${oom_flag}" == "1" ]]; then
      if [[ "${current_bs}" -gt 1 ]]; then
        current_bs=$(((current_bs + 1) / 2))
        echo "    OOM detected, fallback batch_size -> ${current_bs} and retry"
        continue
      fi
      run_error="OOM persists at batch_size=1. ${run_error}"
    fi

    run_status="failed"
    break
  done

  append_registry_row "${phase}" "${exp_name}" "${lr}" "${wd}" "${frz}" "${fold}" "${best_metric}" "${run_status}" "${run_error}" "${current_bs}"
}

run_combo_all_folds() {
  local phase="$1"
  local exp_name="$2"
  local cfg_path="$3"
  local lr="$4"
  local wd="$5"
  local frz="$6"

  echo "[Run ${phase}:${exp_name}] eval folds: ${EVAL_FOLDS[*]}"
  for fold in "${EVAL_FOLDS[@]}"; do
    if should_skip_completed_fold "${phase}" "${lr}" "${wd}" "${frz}" "${fold}" | grep -q '^1$'; then
      echo "  -> fold ${fold} already completed in registry, skip (resume)."
      continue
    fi
    run_single_fold_with_guard "${phase}" "${exp_name}" "${cfg_path}" "${lr}" "${wd}" "${frz}" "${fold}"
  done
}

# -----------------------------
# Layer 1: coarse sweep training
# -----------------------------
: > "${SWEEP_DIR}/layer1_combos.tsv"
combo_idx=0
for lr in "${LAYER1_LRS[@]}"; do
  for wd in "${LAYER1_WDS[@]}"; do
    for frz in "${FREEZE_EPOCHS[@]}"; do
      combo_idx=$((combo_idx + 1))
      exp_name="layer1_${RUN_TAG}_combo_${combo_idx}"
      combo_cfg="${TMP_DIR}/${exp_name}.yaml"

      echo "[Layer1 ${combo_idx}] lr=${lr}, wd=${wd}, batch_size=${BATCH_SIZE}, freeze_epochs=${frz}"
      run_combo_all_folds "layer1" "${exp_name}" "${combo_cfg}" "${lr}" "${wd}" "${frz}"

      printf '%s\t%s\t%s\t%s\n' "${exp_name}" "${lr}" "${wd}" "${frz}" >> "${SWEEP_DIR}/layer1_combos.tsv"
    done
  done
done

# --------------------------------------
# Summarize Layer 1 and decide promotion
# --------------------------------------
python - <<PY
import csv
import json
import math
import statistics
from pathlib import Path

sweep_dir = Path("${SWEEP_DIR}")
rank_folds = [0, 1]
promotion_top_k = int("${PROMOTION_TOP_K}")
promotion_ratio_raw = "${PROMOTION_RATIO}".strip()
promotion_ratio = float(promotion_ratio_raw) if promotion_ratio_raw else None
registry_path = Path("${REGISTRY_PATH}")

if promotion_ratio is not None and not (0 < promotion_ratio <= 1):
    raise ValueError("PROMOTION_RATIO must be in (0,1].")

latest = {}
if registry_path.exists():
    for raw in registry_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        key = (
            row.get("phase"),
            float(row.get("lr")),
            float(row.get("weight_decay")),
            int(row.get("freeze_epochs")),
            int(row.get("fold")),
        )
        latest[key] = row

rows = []
combos_tsv = sweep_dir / "layer1_combos.tsv"
for line in combos_tsv.read_text(encoding="utf-8").strip().splitlines():
    exp_name, lr, wd, frz = line.split("\t")
    lr_f = float(lr)
    wd_f = float(wd)
    frz_i = int(frz)

    fold_qwks = {}
    failed_reasons = []
    fold_states = []
    for fold in rank_folds:
        key = ("layer1", lr_f, wd_f, frz_i, fold)
        rec = latest.get(key)
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

    rank_vals = [fold_qwks[f] for f in rank_folds if f in fold_qwks]
    mean_qwk = statistics.fmean(rank_vals) if rank_vals else float("-inf")
    std_qwk = statistics.pstdev(rank_vals) if len(rank_vals) > 1 else 0.0

    if all(s == "completed" for s in fold_states):
        combo_status = "completed"
    elif any(s == "failed" for s in fold_states):
        combo_status = "failed"
    else:
        combo_status = "partial"

    rows.append({
        "experiment_name": exp_name,
        "lr": lr_f,
        "weight_decay": wd_f,
        "freeze_epochs": frz_i,
        "fold0_qwk": fold_qwks.get(0),
        "fold1_qwk": fold_qwks.get(1),
        "mean_qwk": mean_qwk,
        "std_qwk": std_qwk,
        "rank_folds": rank_folds,
        "fold_qwks": fold_qwks,
        "combo_status": combo_status,
        "failed_reason": " | ".join(failed_reasons) if failed_reasons else "",
    })

rows.sort(key=lambda r: (-r["mean_qwk"], r["std_qwk"], r["experiment_name"]))

total = len(rows)
if promotion_ratio is not None:
    promoted_n = max(1, math.ceil(total * promotion_ratio))
    policy_text = f"ratio={promotion_ratio:.2f}"
else:
    promoted_n = min(promotion_top_k, total)
    policy_text = f"top_k={promotion_top_k}"

for i, row in enumerate(rows, start=1):
    promoted = i <= promoted_n and row["combo_status"] == "completed"
    row["rank"] = i
    row["promoted"] = promoted
    row["promotion_reason"] = (
        f"promoted_by_{policy_text}; rank={i}/{total}; sort=(-mean_qwk,+std_qwk)"
        if promoted else
        f"not_promoted_by_{policy_text}; rank={i}/{total}; sort=(-mean_qwk,+std_qwk); status={row['combo_status']}"
    )

csv_path = sweep_dir / "layer1_ranking.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "rank", "experiment_name", "lr", "weight_decay", "freeze_epochs",
            "fold0_qwk", "fold1_qwk", "mean_qwk", "std_qwk",
            "combo_status", "promoted", "promotion_reason", "failed_reason",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in writer.fieldnames})

jsonl_path = sweep_dir / "layer1_ranking.jsonl"
with jsonl_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

promoted_path = sweep_dir / "promoted_layer1.json"
promoted_rows = [r for r in rows if r["promoted"]]
promoted_path.write_text(json.dumps(promoted_rows, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Saved ranking CSV: {csv_path}")
print(f"Saved ranking JSONL: {jsonl_path}")
print(f"Promoted: {len(promoted_rows)}/{total}")
PY

# ------------------------------------------------------------
# Layer 2: neighborhood expansion ONLY for promoted candidates
# ------------------------------------------------------------
python - <<PY
import itertools
import json
from pathlib import Path

sweep_dir = Path("${SWEEP_DIR}")
layer2_lrs = [float(x) for x in "${LAYER2_LRS[*]}".split()]
layer2_wds = [float(x) for x in "${LAYER2_WDS[*]}".split()]

promoted = json.loads((sweep_dir / "promoted_layer1.json").read_text(encoding="utf-8"))


def nearest_idx(values, target):
    return min(range(len(values)), key=lambda i: abs(values[i] - target))

expansions = []
for p in promoted:
    base_lr = float(p["lr"])
    base_wd = float(p["weight_decay"])
    frz = int(p["freeze_epochs"])

    i = nearest_idx(layer2_lrs, base_lr)
    j = nearest_idx(layer2_wds, base_wd)

    lr_neighbors = sorted(set(layer2_lrs[k] for k in [i - 1, i, i + 1] if 0 <= k < len(layer2_lrs)))
    wd_neighbors = sorted(set(layer2_wds[k] for k in [j - 1, j, j + 1] if 0 <= k < len(layer2_wds)))

    for lr2, wd2 in itertools.product(lr_neighbors, wd_neighbors):
        expansions.append({
            "from_experiment": p["experiment_name"],
            "from_rank": p["rank"],
            "from_reason": p["promotion_reason"],
            "lr": lr2,
            "weight_decay": wd2,
            "freeze_epochs": frz,
        })

# Deduplicate by params + freeze
uniq = {}
for e in expansions:
    key = (e["lr"], e["weight_decay"], e["freeze_epochs"])
    if key not in uniq:
        uniq[key] = e

layer2 = list(uniq.values())
(sweep_dir / "layer2_expansions.json").write_text(json.dumps(layer2, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Layer 2 expansions (deduped): {len(layer2)}")
PY

layer2_count=$(python - <<PY
import json
from pathlib import Path
path = Path("${SWEEP_DIR}/layer2_expansions.json")
print(len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else 0)
PY
)

if [[ "${layer2_count}" -eq 0 ]]; then
  echo "No Layer 2 expansions generated. Done."
  exit 0
fi

idx=0
while IFS=$'\t' read -r _ lr wd frz; do
  idx=$((idx + 1))
  exp_name="layer2_${RUN_TAG}_combo_${idx}"
  combo_cfg="${TMP_DIR}/${exp_name}.yaml"

  echo "[Layer2 ${idx}/${layer2_count}] lr=${lr}, wd=${wd}, freeze_epochs=${frz}"
  run_combo_all_folds "layer2" "${exp_name}" "${combo_cfg}" "${lr}" "${wd}" "${frz}"
done < <(python - <<PY
import json
from pathlib import Path
sweep_dir = Path("${SWEEP_DIR}")
layer2 = json.loads((sweep_dir / "layer2_expansions.json").read_text(encoding="utf-8"))
for idx, cfg in enumerate(layer2, start=1):
    print(f"{idx}\t{cfg['lr']}\t{cfg['weight_decay']}\t{cfg['freeze_epochs']}")
PY
)

echo "Sweep complete. Artifacts under: ${SWEEP_DIR}"
