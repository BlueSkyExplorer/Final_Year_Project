import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats


def load_fold_metrics(exp_dir: Path):
    metrics = []
    for fold_dir in sorted(exp_dir.glob("fold_*/metrics.json")):
        with open(fold_dir, "r") as f:
            hist = json.load(f)
            metrics.append(hist[-1])
    return metrics


def summarize(metrics):
    summary = {}
    keys = [k for k in metrics[0] if isinstance(metrics[0][k], (int, float))]
    for key in keys:
        values = np.array([m[key] for m in metrics])
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "cv": float(values.std() / (values.mean() + 1e-8)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="experiment name (matches config stem)")
    args = parser.parse_args()

    exp_dir = Path("results") / args.experiment
    metrics = load_fold_metrics(exp_dir)
    summary = summarize(metrics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
