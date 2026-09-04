#!/usr/bin/env python3
"""Aggregate proxy loss CSVs across seeds."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["selector"], float(row["budget"]))].append(row)

    out = []
    for (selector, budget), items in sorted(grouped.items()):
        losses = np.array([float(row["best_val_loss"]) for row in items], dtype=np.float64)
        ratios = [float(row["unique_action_ratio"]) for row in items if row["unique_action_ratio"]]
        out.append({
            "selector": selector,
            "budget": budget,
            "seeds": len(items),
            "unique_action_ratio_mean": float(np.mean(ratios)) if ratios else "",
            "best_val_loss_mean": float(np.mean(losses)),
            "best_val_loss_std": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            "best_val_loss_min": float(np.min(losses)),
            "best_val_loss_max": float(np.max(losses)),
        })
    return out


def plot(rows: list[dict], output_path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["selector"]].append(row)
    plt.figure(figsize=(7, 4.5))
    for selector, items in sorted(grouped.items()):
        items = sorted(items, key=lambda row: float(row["unique_action_ratio_mean"]))
        xs = [float(row["unique_action_ratio_mean"]) for row in items]
        ys = [float(row["best_val_loss_mean"]) for row in items]
        yerr = [float(row["best_val_loss_std"]) for row in items]
        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=selector)
    plt.xlabel("Unique action-label budget ratio")
    plt.ylabel("Best validation MSE, mean +/- std")
    plt.title("Stage2 proxy loss across seeds")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate(load_rows(args.csv))
    csv_path = args.output_dir / "proxy_loss_aggregate.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output_dir / "proxy_loss_aggregate_curve.png")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
