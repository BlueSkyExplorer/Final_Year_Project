#!/usr/bin/env bash
set -euo pipefail

# Layer 1（粗搜）
# LR ∈ {1e-3, 3e-4}、WD ∈ {1e-4, 1e-5}、BS 固定 16、freeze_epochs ∈ {0, 5}
# 組合公式：|LR| × |WD| × |BS| × |freeze| = 2 × 2 × 1 × 2 = 8

BASE_CONFIG="configs/experiments/multiclass_resnet18_ce.yaml"
FOLDS=(0 1 2 3 4)
LRS=(1e-3 3e-4)
WDS=(1e-4 1e-5)
FREEZE_EPOCHS=(0 5)
BATCH_SIZE=16

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

build_config() {
  local lr="$1"
  local wd="$2"
  local frz="$3"
  local out_cfg="$4"

  python - <<PY
import yaml
from pathlib import Path

base_path = Path("${BASE_CONFIG}")
out_path = Path("${out_cfg}")

cfg = yaml.safe_load(base_path.read_text())
cfg.setdefault("training", {})
cfg["training"]["lr"] = float("${lr}")
cfg["training"]["weight_decay"] = float("${wd}")
cfg["training"]["batch_size"] = int("${BATCH_SIZE}")
cfg["training"]["freeze_epochs"] = int("${frz}")

out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY
}

combo_idx=0
for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for frz in "${FREEZE_EPOCHS[@]}"; do
      combo_idx=$((combo_idx + 1))
      combo_cfg="${TMP_DIR}/layer1_combo_${combo_idx}.yaml"
      build_config "${lr}" "${wd}" "${frz}" "${combo_cfg}"

      echo "[Layer1 ${combo_idx}/8] lr=${lr}, wd=${wd}, batch_size=${BATCH_SIZE}, freeze_epochs=${frz}"
      for fold in "${FOLDS[@]}"; do
        echo "  -> fold ${fold}"
        python train_multiclass.py --config "${combo_cfg}" --fold "${fold}"
      done
    done
  done
done

# 訓練成本估算：
# 總訓練次數 N = 8（組合） × 5（fold）= 40
# 總時數 H = 40 × T（T 為單次訓練小時數）
