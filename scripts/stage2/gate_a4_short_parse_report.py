#!/usr/bin/env python3
"""Parse A4-short train/reload logs and write metrics/report."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"latent_loss=(?P<latent>[0-9.]+), action_loss=(?P<action>[0-9.]+), "
    r"total_loss=(?P<total>[0-9.]+), step=(?P<step>[0-9]+), "
    r"grad_norm=(?P<grad>[0-9.]+), lr=(?P<lr>[0-9.eE+-]+), "
    r"eff_labels=(?P<labels>[0-9]+), mem_gb=(?P<mem>[0-9.]+), reserved_gb=(?P<reserved>[0-9.]+)"
)


def parse_metrics(path: Path) -> list[dict]:
    text = path.read_text(errors="replace")
    rows = []
    seen = set()
    for match in METRIC_RE.finditer(text):
        row = {
            "step": int(match.group("step")),
            "latent_loss": float(match.group("latent")),
            "action_loss": float(match.group("action")),
            "total_loss": float(match.group("total")),
            "grad_norm": float(match.group("grad")),
            "learning_rate": float(match.group("lr")),
            "effective_action_labels": int(match.group("labels")),
            "max_memory_allocated_gb": float(match.group("mem")),
            "max_memory_reserved_gb": float(match.group("reserved")),
        }
        key = (row["step"], row["latent_loss"], row["action_loss"], row["total_loss"])
        if key not in seen:
            rows.append(row)
            seen.add(key)
    return rows


def parse_gpu_csv(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    max_used = {}
    samples = 0
    for i, line in enumerate(path.read_text(errors="replace").splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        samples += 1
        gpu = parts[1]
        used = float(parts[2])
        max_used[gpu] = max(max_used.get(gpu, 0.0), used)
    return {"exists": True, "samples": samples, "max_used_mib_by_gpu": max_used}


def finite_rows(rows: list[dict]) -> bool:
    for row in rows:
        for key in ["latent_loss", "action_loss", "total_loss", "grad_norm"]:
            if not math.isfinite(row[key]):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--reload-log", required=True)
    parser.add_argument("--gpu-log", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--title", default="Gate A4-short Full-data Training Smoke")
    parser.add_argument("--expected-train-steps", type=int, default=10)
    parser.add_argument("--checkpoint-step", type=int, default=10)
    parser.add_argument("--phase-name", default="training smoke")
    parser.add_argument("--reload-required", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    run_id = args.run_id
    train_log = Path(args.train_log)
    reload_log = Path(args.reload_log)
    gpu_log = Path(args.gpu_log)
    save_root = Path(args.save_root)
    metrics_jsonl = Path(args.metrics_jsonl)
    report_md = Path(args.report_md)
    checkpoint_dir = save_root / "checkpoints" / f"checkpoint_step_{args.checkpoint_step}" / "transformer"
    checkpoint_file = checkpoint_dir / "diffusion_pytorch_model.safetensors"
    checkpoint_config = checkpoint_dir / "config.json"

    train_rows = parse_metrics(train_log)
    reload_rows = parse_metrics(reload_log) if args.reload_required else []
    gpu_summary = parse_gpu_csv(gpu_log)

    metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with metrics_jsonl.open("w", encoding="utf-8") as f:
        for row in train_rows:
            out = {"run_id": run_id, "phase": "train", **row}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        for row in reload_rows:
            out = {"run_id": run_id, "phase": "reload", **row}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    train_pass = len(train_rows) == args.expected_train_steps and finite_rows(train_rows)
    ckpt_pass = checkpoint_file.exists() and checkpoint_config.exists()
    reload_pass = (len(reload_rows) >= 1 and finite_rows(reload_rows)) if args.reload_required else True
    status = "pass" if train_pass and ckpt_pass and reload_pass else "partial_or_failed"

    warnings = []
    warning_targets = [("train", train_log)]
    if args.reload_required:
        warning_targets.append(("reload", reload_log))
    for label, path in warning_targets:
        text = path.read_text(errors="replace")
        for pattern in ["unused", "Warning", "WARNING", "deprecated", "destroy_process_group"]:
            if pattern in text:
                warnings.append(f"{label}: contains `{pattern}`")

    lines = [
        f"# {args.title}",
        "",
        f"Status: {status}",
        "",
        f"This is a {args.phase_name}, not a stable baseline and not closed-loop evidence.",
        "",
        "## Run",
        "",
        f"- run_id: `{run_id}`",
        f"- train_log: `{train_log}`",
            f"- reload_log: `{reload_log if args.reload_required else 'not_run'}`",
        f"- gpu_log: `{gpu_log}`",
        f"- save_root: `{save_root}`",
        "",
        "## Train metrics",
        "",
        f"- optimizer_steps_observed: `{len(train_rows)}`",
        f"- losses_finite: `{finite_rows(train_rows)}`",
    ]
    if train_rows:
        lines.extend(
            [
                f"- first_total_loss: `{train_rows[0]['total_loss']}`",
                f"- last_total_loss: `{train_rows[-1]['total_loss']}`",
                f"- max_total_loss: `{max(r['total_loss'] for r in train_rows)}`",
                f"- min_total_loss: `{min(r['total_loss'] for r in train_rows)}`",
                f"- max_memory_allocated_gb: `{max(r['max_memory_allocated_gb'] for r in train_rows)}`",
                f"- max_memory_reserved_gb: `{max(r['max_memory_reserved_gb'] for r in train_rows)}`",
                f"- effective_action_labels_range: `[{min(r['effective_action_labels'] for r in train_rows)}, {max(r['effective_action_labels'] for r in train_rows)}]`",
            ]
        )
    lines.extend(
        [
            "",
            "## Checkpoint smoke",
            "",
            f"- checkpoint_dir: `{checkpoint_dir}`",
            f"- checkpoint_exists: `{checkpoint_file.exists()}`",
            f"- checkpoint_bytes: `{checkpoint_file.stat().st_size if checkpoint_file.exists() else 0}`",
            f"- reload_required: `{args.reload_required}`",
            f"- reload_steps_observed: `{len(reload_rows)}`",
            f"- reload_losses_finite: `{finite_rows(reload_rows) if reload_rows else 'not_applicable'}`",
        ]
    )
    if reload_rows:
        lines.append(f"- reload_total_loss: `{reload_rows[-1]['total_loss']}`")
    lines.extend(
        [
            "",
            "## GPU monitor",
            "",
            f"- gpu_summary: `{gpu_summary}`",
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        for warning in sorted(set(warnings)):
            lines.append(f"- {warning}")
    else:
        lines.append("- none detected")
    lines.extend(
        [
            "",
            "## Gate implication",
            "",
            f"- This {args.phase_name} passes if status is `pass`.",
            f"- Passing this {args.phase_name} only means the requested full-data training substage and checkpoint save/reload requirements are runnable.",
            "- It does not prove baseline stability or task learning.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "train_steps": len(train_rows), "reload_steps": len(reload_rows)}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
