# LIMUC UC Severity Baseline（Windows 版使用說明）

本專案提供以 **PyTorch** 建立的 LIMUC 內視鏡影像 baseline，用於潰瘍性結腸炎（UC）嚴重度建模，支援：

- **Multi-class classification**（MES 0/1/2/3）
- **Ordinal prediction**（CORAL / CORN / distance-aware）
- **Regression-to-class**（先回歸連續分數，再映射回 MES）

本 README 以 **Windows（PowerShell）** 為主要操作環境撰寫。

---

## 1. 專案結構

```text
.
├── configs/
│   ├── dataset/limuc.yaml
│   ├── experiments/*.yaml
│   └── folds/limuc_5fold_patient.json
├── src/
├── scripts/
├── train_multiclass.py
├── train_ordinal.py
├── train_regression.py
├── analyze_results.py
└── generate_cam.py
```

---

## 2. Windows 環境安裝（PowerShell）

> 建議使用 Python 3.10+。

```powershell
# 1) 建立虛擬環境
python -m venv .venv

# 2) 啟用虛擬環境
.\.venv\Scripts\Activate.ps1

# 3) 安裝套件
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若遇到 PowerShell 執行政策問題，可先在「目前視窗」暫時允許：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 3. 資料準備（Windows 路徑）

預設資料夾：

```text
data\LIMUC\patient_based_classified_images\
```

目錄結構需為：

```text
data\LIMUC\patient_based_classified_images\
├── patient_001\
│   ├── Mayo 0\
│   ├── Mayo 1\
│   ├── Mayo 2\
│   └── Mayo 3\
├── patient_002\
└── ...
```

### 建立 fold cache（第一次必做）

`paths.folds_file` 必須先存在，請執行：

```powershell
New-Item -ItemType Directory -Force -Path configs\folds | Out-Null
"{}" | Set-Content configs\folds\limuc_5fold_patient.json
```

---

## 4. 設定檔重點

### `configs/dataset/limuc.yaml`

- `paths.data_root`：資料根目錄
- `paths.folds_file`：fold 快取檔案路徑
- `images.image_size`：輸入影像尺寸
- `cv.num_folds`：K-fold 的 K 值
- `cv.test_ratio`：先切出 test patients 的比例
- `seed`：隨機種子

### `configs/experiments/*.yaml`

- `experiment_name`
- `model.paradigm`：`multiclass` / `ordinal` / `regression`
- `model.backbone`：如 `resnet18`、`efficientnet_b0`
- `model.loss`：如 `ce`、`focal`、`coral`、`corn`、`mse`、`huber`
- `training.*`：batch size、epoch、lr、weight decay、freeze epochs
- `cv.current_fold`：目前 fold（可被 CLI `--fold` 覆蓋）

---

## 5. Windows 上的訓練指令

請先確定已啟用 `.venv`。

### 5.1 Multi-class

```powershell
python train_multiclass.py --config configs\experiments\multiclass_resnet18_ce.yaml --fold 0
```

### 5.2 Ordinal

```powershell
python train_ordinal.py --config configs\experiments\ordinal_resnet18_coral.yaml --fold 0
```

### 5.3 Regression

```powershell
python train_regression.py --config configs\experiments\regression_resnet18_mse.yaml --fold 0
```

建議依序執行 `--fold 0` 到 `--fold 4` 完成 5-fold。

---

## 6. Sweep / Loss Comparison（Windows）

### Layer1/Layer2 Sweep

```powershell
python scripts\run_layer1_sweep.py --member-id A
python scripts\run_layer1_sweep.py --member-id B
```

### 指定 base config

```powershell
python scripts\run_layer1_sweep.py --member-id A --base-config configs\experiments\ordinal_resnet18_coral.yaml
```

### Resume 既有 sweep

```powershell
python scripts\run_layer1_sweep.py --member-id A --resume-sweep-dir results\sweeps\layer1_layer2_YYYYmmdd_HHMMSS
```

### Loss comparison（含 per-loss tuning）

```powershell
python scripts\run_loss_comparison.py `
  --configs `
    configs\experiments\multiclass_resnet18_ce.yaml `
    configs\experiments\multiclass_resnet18_focal.yaml `
    configs\experiments\multiclass_resnet18_cbce.yaml `
  --baseline-lr 3e-4 `
  --baseline-wd 1e-4 `
  --baseline-freeze-epochs 5 `
  --baseline-batch-size 16 `
  --lr-grid 1e-4,3e-4,1e-3 `
  --wd-grid 1e-5,1e-4,3e-4 `
  --top-k 2
```

### 合併多人結果

```powershell
python scripts\merge_results.py `
  memberA\results\sweeps\layer1_layer2_xxx `
  memberB\results\sweeps\layer1_layer2_yyy `
  memberC\results\loss_comparison\per_loss_tuning_zzz
```

---

## 7. 結果彙整與 Grad-CAM

### Cross-fold 統計

```powershell
python analyze_results.py --experiment multiclass_resnet18_ce
```

預設輸出：

- `results/<experiment>/cross_fold_summary.json`
- `results/<experiment>/cross_fold_summary.csv`

### 產生 Grad-CAM

```powershell
python generate_cam.py --config configs\experiments\multiclass_resnet18_ce.yaml --fold 0 --num_per_class 3
```

輸出目錄：

```text
results\<experiment_name>\fold_<k>\cam\
```

---

## 8. 常見問題（Windows）

### Q1. 出現 `Configured folds_file does not exist`
請先建立：

```powershell
New-Item -ItemType Directory -Force -Path configs\folds | Out-Null
"{}" | Set-Content configs\folds\limuc_5fold_patient.json
```

### Q2. 出現 `No image samples found ... split='val'/'test'`
常見原因：

- `data_root` 設錯
- 資料夾不是 `patient\Mayo i\*`
- 資料量太少或分布不均

### Q3. 變更 `test_ratio` / `num_folds` 後結果怪怪的
請刪除舊的 `folds_file` 後重建，避免沿用舊切分。

---

## 9. Windows 快速開始（最短流程）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

New-Item -ItemType Directory -Force -Path configs\folds | Out-Null
"{}" | Set-Content configs\folds\limuc_5fold_patient.json

python train_multiclass.py --config configs\experiments\multiclass_resnet18_ce.yaml --fold 0
python analyze_results.py --experiment multiclass_resnet18_ce
```
