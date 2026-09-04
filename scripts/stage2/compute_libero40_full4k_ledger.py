#!/usr/bin/env python3
"""Compute the LIBERO40 full-data training ledger from the active dataloader.

This script is intentionally read-only with respect to datasets and checkpoints.
It writes only the requested audit artifacts under skeletonmem/results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path


def parse_train_log(path: Path) -> dict:
    by_step: dict[int, int] = {}
    mem_gb = []
    first_ts = None
    last_ts = None
    ts_re = re.compile(r"(2026-\d\d-\d\d \d\d:\d\d:\d\d)")
    step_re = re.compile(r"step=(\d+)")
    eff_re = re.compile(r"eff_labels=(\d+)")
    mem_re = re.compile(r"mem_gb=([0-9.]+)")

    text = path.read_text(errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        ts_match = ts_re.search(line)
        if ts_match:
            cur = dt.datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
            first_ts = first_ts or cur
            last_ts = cur
        if "Training:" not in line or "eff_labels=" not in line:
            continue
        step_match = step_re.search(line)
        eff_match = eff_re.search(line)
        mem_match = mem_re.search(line)
        if step_match and eff_match:
            by_step[int(step_match.group(1))] = int(eff_match.group(1))
        if mem_match:
            mem_gb.append(float(mem_match.group(1)))

    wall_hours = None
    if first_ts and last_ts:
        wall_hours = (last_ts - first_ts).total_seconds() / 3600.0

    return {
        "unique_optimizer_steps_logged": len(by_step),
        "step_min": min(by_step) if by_step else None,
        "step_max": max(by_step) if by_step else None,
        "c_seen_label": int(sum(by_step.values())),
        "eff_labels_min": min(by_step.values()) if by_step else None,
        "eff_labels_max": max(by_step.values()) if by_step else None,
        "eff_labels_mean": (sum(by_step.values()) / len(by_step)) if by_step else None,
        "max_mem_gb": max(mem_gb) if mem_gb else None,
        "first_timestamp": first_ts.isoformat(sep=" ") if first_ts else None,
        "last_timestamp": last_ts.isoformat(sep=" ") if last_ts else None,
        "wall_clock_hours_timestamp_span": wall_hours,
        "gpu_hours_2gpu_timestamp_span": wall_hours * 2.0 if wall_hours is not None else None,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_ledger(root: Path, source_root: Path) -> dict:
    os.environ["LINGBOT_VA_DATASET_PATH"] = str(root)
    os.environ["LINGBOT_VA_LOAD_WORKERS"] = "0"
    os.environ["LINGBOT_VA_INIT_WORKERS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(source_root))

    from wan_va.configs import VA_CONFIGS  # noqa: WPS433
    from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset  # noqa: WPS433

    config = VA_CONFIGS["libero_train"]
    dataset = MultiLatentLeRobotDataset(config, num_init_worker=1)

    suite_rows = []
    total_segments = 0
    total_unique_label = 0
    total_unique_raw_frames_from_meta = 0
    per_segment_labels: list[int] = []

    for sub in dataset._datasets:
        suite_root = Path(sub.repo_id)
        suite_label = 0
        suite_raw_frames = 0
        suite_segments = len(sub)
        for idx in range(len(sub)):
            meta = sub.new_metas[idx]
            suite_raw_frames += int(meta["end_frame"]) - int(meta["start_frame"])
            item = sub[idx]
            labels = int(item["actions_mask"].sum().item())
            suite_label += labels
            per_segment_labels.append(labels)
        suite_rows.append(
            {
                "suite": suite_root.name,
                "segments": suite_segments,
                "unique_raw_frame_span_sum": suite_raw_frames,
                "unique_label_actions_mask_sum": suite_label,
            }
        )
        total_segments += suite_segments
        total_unique_label += suite_label
        total_unique_raw_frames_from_meta += suite_raw_frames

    return {
        "dataset_root": str(root),
        "segments": total_segments,
        "c_unique_label": total_unique_label,
        "c_unique_raw_frame_span_sum": total_unique_raw_frames_from_meta,
        "per_segment_label_min": min(per_segment_labels) if per_segment_labels else None,
        "per_segment_label_max": max(per_segment_labels) if per_segment_labels else None,
        "per_segment_label_mean": (sum(per_segment_labels) / len(per_segment_labels))
        if per_segment_labels
        else None,
        "suites": suite_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    train = parse_train_log(args.train_log)
    data = compute_dataset_ledger(args.dataset_root, args.source_root)
    model_file = args.checkpoint / "transformer" / "diffusion_pytorch_model.safetensors"
    config_file = args.checkpoint / "transformer" / "config.json"
    checkpoint = {
        "checkpoint": str(args.checkpoint),
        "model_file": str(model_file),
        "model_file_exists": model_file.is_file(),
        "model_file_size_bytes": model_file.stat().st_size if model_file.is_file() else None,
        "model_file_sha256": sha256_file(model_file) if model_file.is_file() else None,
        "config_file_exists": config_file.is_file(),
    }
    c_seen = train["c_seen_label"]
    c_unique = data["c_unique_label"]
    replay_multiplier = c_seen / c_unique if c_seen and c_unique else None
    # Only average replay is exactly observable from the current train log. A
    # per-label replay histogram requires sampler index instrumentation.
    replay_entropy_note = (
        "not available from existing train log; requires per-sample/per-label sampler exposure logging"
    )
    doc = {
        "created_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "run_name": "libero40_full4k_gacc10_after_full4k_eval_then_libero40_20260819T122727Z",
        "dataset": data,
        "training_log": train,
        "checkpoint_step_4000": checkpoint,
        "ledger": {
            "c_unique_raw": data["c_unique_raw_frame_span_sum"],
            "c_unique_label": c_unique,
            "c_seen_label": c_seen,
            "c_compute": {
                "optimizer_steps": train["unique_optimizer_steps_logged"],
                "gradient_accumulation_steps": 10,
                "world_size": 2,
                "wall_clock_hours_timestamp_span": train["wall_clock_hours_timestamp_span"],
                "gpu_hours_2gpu_timestamp_span": train["gpu_hours_2gpu_timestamp_span"],
                "peak_mem_gb_per_rank_logged": train["max_mem_gb"],
            },
            "replay_multiplier": replay_multiplier,
            "replay_entropy": None,
            "replay_entropy_note": replay_entropy_note,
        },
        "script_runtime_seconds": time.time() - started,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(doc, indent=2) + "\n")
    print(args.output_json)


if __name__ == "__main__":
    main()
