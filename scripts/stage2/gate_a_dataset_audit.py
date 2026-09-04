#!/usr/bin/env python3
"""Audit candidate full-data benchmark datasets for Stage2 Gate A."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_repo(repo: Path) -> dict:
    meta = repo / "meta" / "episodes.jsonl"
    info = repo / "meta" / "info.json"
    tasks = repo / "meta" / "tasks.jsonl"
    empty_parent = repo.parent / "empty_emb.pt"
    empty_local = repo / "empty_emb.pt"
    row = {
        "repo": str(repo),
        "benchmark_guess": "",
        "task_guess": repo.name,
        "has_info": info.exists(),
        "has_tasks": tasks.exists(),
        "has_empty_emb_parent": empty_parent.exists(),
        "has_empty_emb_local": empty_local.exists(),
        "episodes": 0,
        "frames": 0,
        "segments": 0,
        "latent_files": 0,
        "camera_dirs": "",
        "action_dim": "",
        "status": "missing_meta",
    }
    if "libero" in str(repo).lower():
        row["benchmark_guess"] = "libero"
    elif "rmbench" in str(repo).lower():
        row["benchmark_guess"] = "rmbench"
    elif "robotwin" in str(repo).lower():
        row["benchmark_guess"] = "robotwin"
    if not meta.exists():
        return row
    episodes = read_jsonl(meta)
    row["episodes"] = len(episodes)
    row["frames"] = sum(int(item.get("length", 0)) for item in episodes)
    row["segments"] = sum(len(item.get("action_config", [])) for item in episodes)
    latent_dirs = sorted((repo / "latents" / "chunk-000").glob("*")) if (repo / "latents" / "chunk-000").exists() else []
    row["camera_dirs"] = ";".join(path.name for path in latent_dirs if path.is_dir())
    row["latent_files"] = sum(1 for _ in (repo / "latents").glob("**/*.pth")) if (repo / "latents").exists() else 0
    parquet = next((repo / "data").glob("chunk-*/*.parquet"), None) if (repo / "data").exists() else None
    if parquet is not None:
        try:
            import pandas as pd

            frame = pd.read_parquet(parquet, columns=["action"])
            if len(frame):
                row["action_dim"] = len(frame["action"].iloc[0])
        except Exception as exc:  # noqa: BLE001
            row["action_dim"] = f"read_error:{type(exc).__name__}"
    if row["episodes"] and row["segments"] and row["latent_files"]:
        row["status"] = "train_data_present"
    elif row["episodes"] and row["segments"]:
        row["status"] = "metadata_only"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    repos: list[Path] = []
    for root in args.root:
        if not root.exists():
            continue
        if (root / "meta" / "episodes.jsonl").exists():
            repos.append(root)
        repos.extend(sorted(path.parent.parent for path in root.glob("**/meta/episodes.jsonl")))
    unique = []
    seen = set()
    for repo in repos:
        resolved = str(repo.resolve())
        if resolved not in seen:
            unique.append(repo)
            seen.add(resolved)

    rows = [audit_repo(repo) for repo in unique]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["repo"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_csv}")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
