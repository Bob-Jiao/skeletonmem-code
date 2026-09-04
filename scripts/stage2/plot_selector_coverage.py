#!/usr/bin/env python3
"""Plot timestep coverage for Stage2 selector manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_selector_budget(path: Path) -> tuple[str, float]:
    selector, budget_token = path.stem.rsplit("_b", 1)
    return selector, int(budget_token) / 100.0


def coverage_for_manifest(path: Path, episode_lengths: dict[int, int], bins: int) -> list[dict]:
    selector, budget = parse_selector_budget(path)
    covered: dict[int, set[int]] = {episode: set() for episode in episode_lengths}
    for segment in read_jsonl(path):
        episode = int(segment["episode_index"])
        first = int(segment["action_loss_first"])
        last = int(segment["action_loss_last"])
        covered.setdefault(episode, set()).update(range(first, last + 1))

    rows = []
    for bin_index in range(bins):
        covered_count = 0
        total_count = 0
        for episode, length in episode_lengths.items():
            start = round(bin_index * length / bins)
            end = round((bin_index + 1) * length / bins)
            total_count += max(0, end - start)
            covered_count += sum(1 for timestep in range(start, end) if timestep in covered.get(episode, set()))
        rows.append({
            "selector": selector,
            "budget": budget,
            "bin_index": bin_index,
            "bin_start_ratio": bin_index / bins,
            "bin_end_ratio": (bin_index + 1) / bins,
            "covered_unique_actions": covered_count,
            "total_unique_actions": total_count,
            "coverage_ratio": covered_count / max(1, total_count),
        })
    return rows


def plot(rows: list[dict], output_path: Path) -> None:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["selector"], row["budget"]), []).append(row)

    plt.figure(figsize=(9, 4.8))
    for (selector, budget), items in sorted(grouped.items()):
        if budget >= 0.999:
            continue
        items = sorted(items, key=lambda row: row["bin_index"])
        xs = [(row["bin_start_ratio"] + row["bin_end_ratio"]) / 2 for row in items]
        ys = [row["coverage_ratio"] for row in items]
        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3, label=f"{selector}-{int(budget * 100)}")
    plt.xlabel("Normalized episode time")
    plt.ylabel("Unique action coverage")
    plt.title("Selector temporal coverage")
    plt.ylim(-0.03, 1.03)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = read_jsonl(args.dataset_root / "meta" / "episodes.jsonl")
    episode_lengths = {int(item["episode_index"]): int(item["length"]) for item in episodes}

    rows = []
    for manifest in sorted(args.manifest_dir.glob("*_b*.jsonl")):
        rows.extend(coverage_for_manifest(manifest, episode_lengths, args.bins))

    csv_path = args.output_dir / "selector_coverage_by_time.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output_dir / "selector_coverage_by_time.png")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
