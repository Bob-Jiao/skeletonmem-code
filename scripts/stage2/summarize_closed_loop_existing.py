#!/usr/bin/env python3
"""Summarize existing pick_objects_in_order closed-loop results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_result(path: Path) -> dict:
    row = {"path": str(path), "success_rate": "", "reward": ""}
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("Success Rate:"):
            row["success_rate"] = line.split(":", 1)[1].strip()
        if line.startswith("Reward:"):
            row["reward"] = line.split(":", 1)[1].strip()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = [parse_result(path) for path in sorted(args.result_root.glob("**/_result.txt"))]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "success_rate", "reward"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_csv}")
    print(f"rows {len(rows)}")
    if rows:
        success = [float(row["success_rate"]) for row in rows if row["success_rate"]]
        reward = [float(row["reward"]) for row in rows if row["reward"]]
        print(f"success_min_max {min(success):.4f} {max(success):.4f}")
        print(f"reward_min_max {min(reward):.4f} {max(reward):.4f}")


if __name__ == "__main__":
    main()
