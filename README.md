# LIMUC UC Severity Baseline (PyTorch)

此專案提供以 **LIMUC** 內視鏡影像資料集進行潰瘍性結腸炎（UC）嚴重度建模的 PyTorch baseline，支援三種任務範式：

- **Multi-class classification**（MES 0/1/2/3）
- **Ordinal prediction**（CORAL / CORN / distance-aware）
- **Regression-to-class**（連續分數回歸再映射至 MES）

另外內建：
- patient-level fold 切分
- held-out test patient split
- 統一 config 驅動訓練
- 結果彙整與 Grad-CAM 可解釋化

---

## 1. Repository 結構

```text
.
├── configs/
│   ├── dataset/
│   │   └── limuc.yaml
│   ├── experiments/
│   │   ├── multiclass_*.yaml
│   │   ├── ordinal_*.yaml
│   │   └── regression_*.yaml
│   └── folds/
│       └── limuc_5fold_patient.json   # fold cache（需存在，內容可先為 {}）
├── src/
│   ├── data/
│   │   ├── limuc_dataset.py
│   │   ├── folds.py
│   │   └── transforms.py
│   ├── models/
│   ├── losses/
│   ├── metrics/
│   ├── explain/
│   └── utils/
├── scripts/
│   └── run_layer1_sweep.sh
├── train_multiclass.py
├── train_ordinal.py
├── train_regression.py
├── analyze_results.py
└── generate_cam.py
```

---

## 2. 環境安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. 資料準備

### 3.1 LIMUC 目錄格式

請將資料放在（預設）`data/LIMUC/patient_based_classified_images`，每位病人底下有 `Mayo 0` ~ `Mayo 3`：

```text
data/LIMUC/patient_based_classified_images/
├── patient_001/
│   ├── Mayo 0/
│   ├── Mayo 1/
│   ├── Mayo 2/
│   └── Mayo 3/
├── patient_002/
│   └── ...
└── ...
```

### 3.2 fold cache 注意事項（重要）

目前 `validate_config()` 會要求 `paths.folds_file` **必須存在**。因此第一次使用請先建立檔案：

```bash
mkdir -p configs/folds
echo "{}" > configs/folds/limuc_5fold_patient.json
```

之後 `src/data/folds.py` 會在需要時重新寫入正確格式（包含 `folds` 與 `test_patients`）。

---

## 4. 設定檔說明

### 4.1 Dataset config：`configs/dataset/limuc.yaml`

重點欄位：

- `paths.data_root`：資料根目錄
- `paths.folds_file`：patient split cache 路徑
- `images.image_size`：輸入尺寸
- `cv.num_folds`：K-fold 的 K
- `cv.test_ratio`：先從所有病人抽出 test patient 的比例
- `seed`：隨機種子

### 4.2 Experiment config：`configs/experiments/*.yaml`

重點欄位：

- `experiment_name`
- `model.paradigm`：`multiclass` / `ordinal` / `regression`
- `model.backbone`：例如 `resnet18`、`efficientnet_b0`
- `model.loss`：依任務不同（`ce`、`focal`、`cbce`、`coral`、`corn`、`mse`、`huber`...）
- `training.*`：batch size、epoch、lr、weight decay、freeze epochs 等
- `cv.current_fold`：目前訓練 fold（也可由 CLI `--fold` 覆蓋）

---

## 5. Sweep orchestrator（Layer 1 / Layer 2）

### 5.1 Layer 1 搜尋空間（粗搜）

- `LR ∈ {1e-3, 3e-4}`（2 組）
- `WD ∈ {1e-4, 1e-5}`（2 組）
- `BS = 16`（固定 1 組）
- `freeze_epochs ∈ {0, 5}`（2 組）

### 5.2 組合數公式

Layer 1（粗搜）組合數：

\[
N_{L1} = |LR| \times |WD| \times |BS| \times |freeze| = 2 \times 2 \times 1 \times 2 = 8
\]

> Layer 1 為 **8 組**（固定空間）。

### 5.3 晉級策略（Promotion policy）

`scripts/run_layer1_sweep.sh` 目前採用兩階段流程：

1. 先跑完 Layer 1 的所有組合。
2. 用 **雙 fold 平均**（預設 fold 0 + fold 1）計算 `mean_qwk` 排名。
3. 以 `std_qwk` 做 tie-breaker（**std 較小優先**）。
4. 晉級數可設定為：
   - 固定 Top-K（預設 `PROMOTION_TOP_K=4`）
   - 或比例（設定 `PROMOTION_RATIO`，例如 0.3~0.5）
5. 只對晉級組合做 Layer 2 的 **±1 LR/WD 鄰域展開**。

> 這可避免對所有 Layer 1 組合做全量 Layer 2 展開，降低訓練成本。

### 5.4 晉級可追蹤輸出

orchestrator 會在 `results/sweeps/layer1_layer2_<timestamp>/` 產生：

- `layer1_ranking.csv`
- `layer1_ranking.jsonl`

每筆均含以下欄位，便於追蹤決策：

- `rank`
- `promoted`
- `promotion_reason`
- `fold0_qwk`
- `fold1_qwk`
- `mean_qwk`
- `std_qwk`

### 5.5 訓練成本估算（避免排程錯誤）

若每個組合在 Phase 1 固定跑 2 folds（fold 0 + fold 1），且每次訓練平均耗時 `T` 小時：

- 訓練次數：

\[
N_{train} = 8 \times 2 = 16
\]

- 總訓練時數：

\[
H_{total} = 16 \times T
\]

範例（若 `T = 1.5` 小時）：

- 總訓練時數 `= 16 × 1.5 = 24` 小時。


### 5.6 執行 sweep

```bash
bash scripts/run_layer1_sweep.sh
```

### 5.7 Loss comparison：加入 per-loss tuning（避免單一 shared HP）

為避免「所有 loss 直接套同一組 HP」造成比較偏差，建議改為以下流程：

1. 先用既有通用流程（例如 Layer 1/Layer 2）取得一組 baseline HP（shared HP）。
2. 每個 loss 在正式 5-fold 前，先做小網格（LR + WD + loss 專屬參數）：
   - LR：2~3 組
   - WD：2~3 組
   - loss 專屬參數：
     - focal：`gamma` / `alpha`
     - huber：`delta`
     - distance-aware：`alpha`
3. 小網格先只跑 fold 0 + fold 1，依 `mean_qwk` 排名挑 top 1~2。
4. 只有 top 組進入該 loss 的完整 5-fold。
5. 結果表明確標註 `hp_source`：`shared_hp` 或 `per_loss_tuned`，避免報告解讀混淆。

可使用下列 orchestrator 執行：

```bash
python scripts/run_loss_comparison.py \
  --configs \
    configs/experiments/multiclass_resnet18_ce.yaml \
    configs/experiments/multiclass_resnet18_focal.yaml \
    configs/experiments/multiclass_resnet18_cbce.yaml \
  --baseline-lr 3e-4 \
  --baseline-wd 1e-4 \
  --baseline-freeze-epochs 5 \
  --baseline-batch-size 16 \
  --lr-grid 1e-4,3e-4,1e-3 \
  --wd-grid 1e-5,1e-4,3e-4 \
  --top-k 2
```

輸出目錄（`results/loss_comparison/per_loss_tuning_<timestamp>/`）包含：

- `preview_ranking.csv`：fold 0/1 小網格排名（含 `hp_source`）
- `selected_candidates.csv`：每個 loss 晉級到 full 5-fold 的 top 候選
- `final_5fold_summary.csv`：最終 5-fold 結果表（含 `hp_source`）

---

## 6. 訓練

### 6.1 Multi-class

```bash
python train_multiclass.py \
  --config configs/experiments/multiclass_resnet18_ce.yaml \
  --fold 0
```

### 6.2 Ordinal

```bash
python train_ordinal.py \
  --config configs/experiments/ordinal_resnet18_coral.yaml \
  --fold 0
```

### 6.3 Regression

```bash
python train_regression.py \
  --config configs/experiments/regression_resnet18_mse.yaml \
  --fold 0
```

> 建議依序跑完 `--fold 0..4` 完成 5-fold CV。

---

## 7. split 邏輯（目前實作）

在 `src/data/folds.py`：

1. 先依病人主導類別做 **stratified hold-out test split**（比例 `cv.test_ratio`）
2. 再對剩餘 train/val 病人做 `StratifiedKFold`
3. cache 至 `folds_file`：
   - `folds`: 每 fold 的病人清單
   - `test_patients`: 保留給 test split 的病人

在 `src/data/limuc_dataset.py`：

- `split="train"`：排除 test patients，且排除 `current_fold`（當作 validation fold）
- `split="val"`：排除 test patients，只保留 `current_fold`
- `split="test"`：只保留 `test_patients`

使用方式：

```python
from src.data.limuc_dataset import LIMUCDataset

train_ds = LIMUCDataset(cfg, split="train")
val_ds = LIMUCDataset(cfg, split="val")
test_ds = LIMUCDataset(cfg, split="test")
```

---

## 8. 輸出結果

每次訓練輸出到：

```text
results/<experiment_name>/fold_<k>/
├── best_model.pt
├── metrics.json
└── train.log (若 logger 有寫檔)
```

---

## 9. 結果彙整

```bash
python analyze_results.py --experiment multiclass_resnet18_ce
```

會讀取 `results/<experiment>/fold_*/metrics.json` 並輸出整體 summary（mean/std/cv），同時預設寫入：
- `results/<experiment>/cross_fold_summary.json`
- `results/<experiment>/cross_fold_summary.csv`

也可透過 `--output-json` 與 `--output-csv` 指定輸出路徑。

---

## 10. Grad-CAM

```bash
python generate_cam.py \
  --config configs/experiments/multiclass_resnet18_ce.yaml \
  --fold 0 \
  --num_per_class 3
```

輸出路徑：

```text
results/<experiment_name>/fold_<k>/cam/
```

---

## 11. 常見問題（FAQ）

### Q1. 為什麼一開始就報 `Configured folds_file does not exist`？
因為目前 config 驗證要求 fold cache 檔案存在。先建立空檔（內容 `{}`）即可，後續會自動覆寫成完整 split。

### Q2. `No image samples found ... split='val'/'test'`？
通常是：
- `data_root` 路徑不對
- 資料夾結構非 `patient/Mayo i/*`
- 資料量太小或分布不均，導致某 split 沒資料

### Q3. 要不要手動刪除舊 fold cache？
如果你改了 `test_ratio`、`num_folds` 或資料集內容，建議刪除 `folds_file` 後重建（或改檔名）以避免沿用舊切分。

---

## 12. 快速開始（最短流程）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p configs/folds
echo "{}" > configs/folds/limuc_5fold_patient.json

python train_multiclass.py --config configs/experiments/multiclass_resnet18_ce.yaml --fold 0
python analyze_results.py --experiment multiclass_resnet18_ce
```
