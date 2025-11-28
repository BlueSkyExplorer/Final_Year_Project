# UC Severity Prediction on LIMUC

This repository provides reproducible PyTorch code for assessing ulcerative colitis severity from endoscopic images using the public LIMUC dataset. It supports multi-class classification of Mayo Endoscopic Score (MES 0–3), ordinal regression (CORAL/CORN style), and regression-to-class pipelines, with 5-fold patient-level cross-validation.

## Key Features
- Config-driven experiments (YAML) controlling data paths, models, optimizers, and losses.
- Patient-level 5-fold cross-validation automatically generated and cached.
- Multiple learning paradigms: multi-class, ordinal, and regression.
- ResNet-18 (main) and EfficientNet-B0 (control) backbones.
- Rich metrics: Quadratic Weighted Kappa (primary), macro/per-class F1, AUROC for derived binaries, MAE, etc.
- Grad-CAM/Grad-CAM++ visualizations for interpretability.

## Repository Structure
- `configs/`
  - `dataset/limuc.yaml`: dataset paths and augmentation settings.
  - `folds/limuc_5fold_patient.json`: cached patient-level folds (auto-generated if empty/missing).
  - `experiments/*.yaml`: experiment-specific hyperparameters per paradigm/backbone/loss.
- `src/`
  - `data/`: dataset loader, transforms, and fold utilities.
  - `models/`: backbone builders, heads, and ordinal utilities.
  - `losses/`: multi-class, ordinal, and regression losses.
  - `metrics/`: evaluation metrics for multi-class and derived binary tasks.
  - `explain/`: Grad-CAM implementations.
  - `utils/`: config handling, seeding, logging, and path helpers.
- `train_multiclass.py`, `train_ordinal.py`, `train_regression.py`: training entry points.
- `analyze_results.py`: aggregate cross-validation results and run statistical tests.
- `generate_cam.py`: produce Grad-CAM heatmaps for trained models.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation
1. Download the LIMUC dataset and place it under `data/LIMUC/patient_based_classified_images/`.
2. The structure should be `patient_based_classified_images/<patient_id>/Mayo 0/*.bmp` (and Mayo 1/2/3).
3. On the first run, folds are created automatically and stored at `configs/folds/limuc_5fold_patient.json` using a stratified patient-level split (seeded).

## Running Experiments
Example: multi-class cross-entropy with ResNet-18, fold 0:
```bash
python train_multiclass.py --config configs/experiments/multiclass_resnet18_ce.yaml --fold 0
```
Run folds 0–4 to complete cross-validation. Replace config for different paradigms or backbones.

## Aggregating Cross-Validation
After completing 5 folds for an experiment, summarize metrics:
```bash
python analyze_results.py --experiment multiclass_resnet18_ce
```
This computes mean ± std, coefficient of variation, and bootstrap confidence intervals; pairwise tests are supported when comparing experiments.

## Grad-CAM Visualization
Generate Grad-CAM overlays for a trained checkpoint:
```bash
python generate_cam.py --config configs/experiments/multiclass_resnet18_ce.yaml --fold 0 --num_per_class 3
```
Outputs are stored under `results/<experiment>/fold_<k>/cam/`.

## Citation
If you use this codebase, please cite the LIMUC dataset and acknowledge this repository.
