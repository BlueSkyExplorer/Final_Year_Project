#!/usr/bin/env bash
set -euo pipefail

# Two-layer sweep orchestrator
# - Layer 1: coarse sweep and ranking by dual-fold mean_qwk, std_qwk tie-breaker.
# - Promotion: fixed Top-K or ratio.
# - Layer 2: only promoted combos perform ±1 LR/WD neighborhood expansion.

BASE_CONFIG="configs/experiments/multiclass_resnet18_ce.yaml"
FOLDS=(0 1 2 3 4)
LAYER1_LRS=(1e-3 3e-4)
LAYER1_WDS=(1e-4 1e-5)
FREEZE_EPOCHS=(0 5)
BATCH_SIZE=16

# Promotion policy (choose one)
PROMOTION_TOP_K=4              # fixed small top-k
PROMOTION_RATIO=""             # e.g. 0.4 ; takes precedence when set
RANK_BY_FOLDS=(0 1)            # dual-fold average for ranking

# Layer 2 neighborhood grids (used for ±1 expansion around promoted combo's LR/WD)
LAYER2_LRS=(1e-3 7e-4 5e-4 3e-4 2e-4 1e-4)
LAYER2_WDS=(1e-4 7e-5 5e-5 3e-5 1e-5)

RESULT_ROOT="results"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="${RESULT_ROOT}/sweeps/layer1_layer2_${RUN_TAG}"
mkdir -p "${SWEEP_DIR}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

build_config() {
  local exp_name="$1"
  local lr="$2"
  local wd="$3"
  local frz="$4"
  local out_cfg="$5"

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
cfg["training"]["batch_size"] = int("${BATCH_SIZE}")
cfg["training"]["freeze_epochs"] = int("${frz}")

out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY
}

run_combo_all_folds() {
  local exp_name="$1"
  local cfg_path="$2"
  echo "[Run ${exp_name}] all folds: ${FOLDS[*]}"
  for fold in "${FOLDS[@]}"; do
    echo "  -> fold ${fold}"
    python train_multiclass.py --config "${cfg_path}" --fold "${fold}"
  done
}

# -----------------------------
# Layer 1: coarse sweep training
# -----------------------------
combo_idx=0
for lr in "${LAYER1_LRS[@]}"; do
  for wd in "${LAYER1_WDS[@]}"; do
    for frz in "${FREEZE_EPOCHS[@]}"; do
      combo_idx=$((combo_idx + 1))
      exp_name="layer1_${RUN_TAG}_combo_${combo_idx}"
      combo_cfg="${TMP_DIR}/${exp_name}.yaml"
      build_config "${exp_name}" "${lr}" "${wd}" "${frz}" "${combo_cfg}"

      echo "[Layer1 ${combo_idx}] lr=${lr}, wd=${wd}, batch_size=${BATCH_SIZE}, freeze_epochs=${frz}"
      run_combo_all_folds "${exp_name}" "${combo_cfg}"

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
result_root = Path("${RESULT_ROOT}")
rank_folds = [int(x) for x in "${RANK_BY_FOLDS[*]}".split()]
promotion_top_k = int("${PROMOTION_TOP_K}")
promotion_ratio_raw = "${PROMOTION_RATIO}".strip()
promotion_ratio = float(promotion_ratio_raw) if promotion_ratio_raw else None

if promotion_ratio is not None and not (0 < promotion_ratio <= 1):
    raise ValueError("PROMOTION_RATIO must be in (0,1].")

rows = []
combos_tsv = sweep_dir / "layer1_combos.tsv"
for line in combos_tsv.read_text().strip().splitlines():
    exp_name, lr, wd, frz = line.split("\t")
    fold_qwks = {}
    for fold in [0,1,2,3,4]:
        metrics_path = result_root / exp_name / f"fold_{fold}" / "metrics.json"
        if not metrics_path.exists():
            continue
        history = json.loads(metrics_path.read_text())
        best_qwk = max((float(ep.get("qwk", float("-inf"))) for ep in history), default=float("-inf"))
        if best_qwk != float("-inf"):
            fold_qwks[fold] = best_qwk

    rank_vals = [fold_qwks[f] for f in rank_folds if f in fold_qwks]
    mean_qwk = statistics.fmean(rank_vals) if rank_vals else float("-inf")
    std_qwk = statistics.pstdev(rank_vals) if len(rank_vals) > 1 else 0.0
    rows.append({
        "experiment_name": exp_name,
        "lr": float(lr),
        "weight_decay": float(wd),
        "freeze_epochs": int(frz),
        "mean_qwk": mean_qwk,
        "std_qwk": std_qwk,
        "rank_folds": rank_folds,
        "fold_qwks": fold_qwks,
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
    promoted = i <= promoted_n
    row["rank"] = i
    row["promoted"] = promoted
    row["promotion_reason"] = (
        f"promoted_by_{policy_text}; rank={i}/{total}; sort=(-mean_qwk,+std_qwk)"
        if promoted else
        f"not_promoted_by_{policy_text}; rank={i}/{total}; sort=(-mean_qwk,+std_qwk)"
    )

csv_path = sweep_dir / "layer1_ranking.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "rank", "experiment_name", "lr", "weight_decay", "freeze_epochs",
            "mean_qwk", "std_qwk", "promoted", "promotion_reason",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row[k] for k in writer.fieldnames})

jsonl_path = sweep_dir / "layer1_ranking.jsonl"
with jsonl_path.open("w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

promoted_path = sweep_dir / "promoted_layer1.json"
promoted_rows = [r for r in rows if r["promoted"]]
promoted_path.write_text(json.dumps(promoted_rows, indent=2, ensure_ascii=False))

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
result_root = Path("${RESULT_ROOT}")
run_tag = "${RUN_TAG}"
layer2_lrs = [float(x) for x in "${LAYER2_LRS[*]}".split()]
layer2_wds = [float(x) for x in "${LAYER2_WDS[*]}".split()]

promoted = json.loads((sweep_dir / "promoted_layer1.json").read_text())


def nearest_idx(values, target):
    return min(range(len(values)), key=lambda i: abs(values[i] - target))

expansions = []
for p in promoted:
    base_lr = float(p["lr"])
    base_wd = float(p["weight_decay"])
    frz = int(p["freeze_epochs"])

    i = nearest_idx(layer2_lrs, base_lr)
    j = nearest_idx(layer2_wds, base_wd)

    lr_neighbors = sorted(set(layer2_lrs[k] for k in [i-1, i, i+1] if 0 <= k < len(layer2_lrs)))
    wd_neighbors = sorted(set(layer2_wds[k] for k in [j-1, j, j+1] if 0 <= k < len(layer2_wds)))

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
(sweep_dir / "layer2_expansions.json").write_text(json.dumps(layer2, indent=2, ensure_ascii=False))
print(f"Layer 2 expansions (deduped): {len(layer2)}")
PY

layer2_count=$(python - <<PY
import json
from pathlib import Path
path = Path("${SWEEP_DIR}/layer2_expansions.json")
print(len(json.loads(path.read_text())) if path.exists() else 0)
PY
)

if [[ "${layer2_count}" -eq 0 ]]; then
  echo "No Layer 2 expansions generated. Done."
  exit 0
fi

# Train Layer 2 expansions
python - <<PY
import json
from pathlib import Path

sweep_dir = Path("${SWEEP_DIR}")
layer2 = json.loads((sweep_dir / "layer2_expansions.json").read_text())
for idx, cfg in enumerate(layer2, start=1):
    print(f"{idx}\t{cfg['lr']}\t{cfg['weight_decay']}\t{cfg['freeze_epochs']}")
PY

idx=0
while IFS=$'\t' read -r _ lr wd frz; do
  idx=$((idx + 1))
  exp_name="layer2_${RUN_TAG}_combo_${idx}"
  combo_cfg="${TMP_DIR}/${exp_name}.yaml"
  build_config "${exp_name}" "${lr}" "${wd}" "${frz}" "${combo_cfg}"

  echo "[Layer2 ${idx}/${layer2_count}] lr=${lr}, wd=${wd}, freeze_epochs=${frz}"
  run_combo_all_folds "${exp_name}" "${combo_cfg}"

done < <(python - <<PY
import json
from pathlib import Path
sweep_dir = Path("${SWEEP_DIR}")
layer2 = json.loads((sweep_dir / "layer2_expansions.json").read_text())
for idx, cfg in enumerate(layer2, start=1):
    print(f"{idx}\t{cfg['lr']}\t{cfg['weight_decay']}\t{cfg['freeze_epochs']}")
PY
)

echo "Sweep complete. Artifacts under: ${SWEEP_DIR}"
