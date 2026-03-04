import argparse
import csv
import json
from pathlib import Path
import numpy as np


def flatten_metrics(payload: dict, prefix: str = "") -> dict[str, float]:
    flattened = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float)):
            flattened[name] = float(value)
    return flattened


def load_fold_metrics(exp_dir: Path, source: str):
    metrics = []
    metric_filename = "test_metrics.json" if source == "test" else "best_val_metrics.json"
    for fold_dir in sorted(exp_dir.glob(f"fold_*/{metric_filename}")):
        with open(fold_dir, "r") as f:
            payload = json.load(f)
            if not payload:
                continue
            metrics.append(flatten_metrics(payload))
    return metrics


def summarize(metrics):
    summary = {}
    keys = sorted({key for metric in metrics for key in metric.keys()})
    for key in keys:
        values = np.array([m[key] for m in metrics if key in m])
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "cv": float(values.std() / (values.mean() + 1e-8)),
        }
    return summary


def write_summary_json(summary: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def write_summary_csv(summary: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "cv"])
        for metric_name, stats_dict in summary.items():
            writer.writerow(
                [
                    metric_name,
                    stats_dict["mean"],
                    stats_dict["std"],
                    stats_dict["cv"],
                ]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="experiment name (matches config stem)")
    parser.add_argument("--source", choices=["test", "val"], default="test")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="path to write cross-fold summary JSON (default: results/<experiment>/cross_fold_summary.json)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="path to write cross-fold summary CSV (default: results/<experiment>/cross_fold_summary.csv)",
    )
    args = parser.parse_args()

    exp_dir = Path("results") / args.experiment
    output_json = args.output_json or (exp_dir / "cross_fold_summary.json")
    output_csv = args.output_csv or (exp_dir / "cross_fold_summary.csv")

    metrics = load_fold_metrics(exp_dir, args.source)
    if not metrics:
        raise ValueError(f"No fold metrics found under: {exp_dir} for source={args.source}")

    summary = summarize(metrics)
    print(json.dumps(summary, indent=2))
    write_summary_json(summary, output_json)
    write_summary_csv(summary, output_csv)
    print(f"Saved JSON summary to: {output_json}")
    print(f"Saved CSV summary to: {output_csv}")


if __name__ == "__main__":
    main()
