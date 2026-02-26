#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List


def parse_mean_qwk(row: Dict[str, str]) -> float:
    raw = row.get("mean_qwk", "")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return -math.inf
    if math.isnan(val):
        return -math.inf
    return val


def read_csv_rows(path: Path, kind: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["source_dir"] = str(path.parent)
            row["source_file"] = path.name
            row["result_kind"] = kind
            rows.append(row)
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/merge_results.py <result_dir_or_csv> [more_paths...]")
        raise SystemExit(1)

    inputs = [Path(p) for p in sys.argv[1:]]
    all_rows: List[Dict[str, str]] = []

    for entry in inputs:
        if entry.is_file() and entry.suffix.lower() == ".csv":
            kind = "layer1_ranking" if entry.name == "layer1_ranking.csv" else "final_5fold_summary"
            all_rows.extend(read_csv_rows(entry, kind))
            continue

        if not entry.exists():
            print(f"[WARN] Skip missing path: {entry}")
            continue

        layer1_csv = entry / "layer1_ranking.csv"
        if layer1_csv.exists():
            all_rows.extend(read_csv_rows(layer1_csv, "layer1_ranking"))

        final_csv = entry / "final_5fold_summary.csv"
        if final_csv.exists():
            all_rows.extend(read_csv_rows(final_csv, "final_5fold_summary"))

    all_rows.sort(key=parse_mean_qwk, reverse=True)

    print(f"Total results merged: {len(all_rows)}")
    for row in all_rows[:10]:
        exp_name = row.get("experiment_name", "")
        mean_qwk = row.get("mean_qwk", "")
        source = row.get("source_dir", "")
        print(f"- {exp_name} | mean_qwk={mean_qwk} | source={source}")


if __name__ == "__main__":
    main()
