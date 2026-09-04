#!/usr/bin/env python3
"""CPU/I/O-only downloader for standard LIBERO LeRobot image datasets.

This script intentionally does not prepare Wan latents or start training. It only
downloads candidate LeRobot-format image datasets for Gate B0 audit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import time

from huggingface_hub import snapshot_download


SUITES = {
    "object": {
        "repo_id": "lerobot/libero_object_image",
        "local_name": "libero-object-image-lerobot",
    },
    "goal": {
        "repo_id": "lerobot/libero_goal_image",
        "local_name": "libero-goal-image-lerobot",
    },
    "spatial": {
        "repo_id": "lerobot/libero_spatial_image",
        "local_name": "libero-spatial-image-lerobot",
    },
}


def summarize_path(path: Path) -> dict:
    files = 0
    bytes_total = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            try:
                bytes_total += item.stat().st_size
            except OSError:
                pass
    return {"files": files, "bytes": bytes_total}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/jiaoguanbo/LIBERO"),
        help="Benchmark data root. Must remain under /data/jiaoguanbo/LIBERO.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/jiaoguanbo/skeletonmem/runtime/cache/huggingface"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "/data/jiaoguanbo/skeletonmem/results/stage2/gate_b0/"
            "libero_multisuite_download_summary.json"
        ),
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=sorted(SUITES),
        default=sorted(SUITES),
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HOME"] = str(args.cache_root)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    args.root.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for suite in args.suites:
        spec = SUITES[suite]
        local_dir = args.root / spec["local_name"]
        record = {
            "suite": suite,
            "repo_id": spec["repo_id"],
            "local_dir": str(local_dir),
            "started_at_unix": time(),
            "status": "DRY_RUN" if args.dry_run else "STARTED",
            "note": (
                "LeRobot image/parquet candidate only; Wan latents and LingBot "
                "training contract must be audited separately."
            ),
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.dry_run:
            records.append(record)
            continue
        try:
            snapshot_download(
                repo_id=spec["repo_id"],
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                max_workers=args.max_workers,
            )
            record["status"] = "DOWNLOADED"
            record["summary"] = summarize_path(local_dir)
        except Exception as exc:  # noqa: BLE001
            record["status"] = "FAILED"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
        record["finished_at_unix"] = time()
        print(json.dumps(record, ensure_ascii=False), flush=True)
        records.append(record)

    args.summary.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    return 0 if all(r["status"] in {"DOWNLOADED", "DRY_RUN"} for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
