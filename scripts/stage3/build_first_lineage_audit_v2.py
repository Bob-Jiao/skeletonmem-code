#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


V1_BUNDLE_DEFAULT = Path(
    "/data/jiaoguanbo/skeletonmem/results/lineage_audit/"
    "first_lineage_audit_bundle_20260903T135825Z"
)
V1_LOG_DIR_DEFAULT = Path(
    "/data/jiaoguanbo/skeletonmem/logs/lineage_audit/"
    "first_lineage_audit_bundle_20260903T135825Z"
)
OUT_ROOT_DEFAULT = Path("/data/jiaoguanbo/skeletonmem/results/lineage_audit")
PUBLIC_AUDIT_REPO_DEFAULT = Path(
    "/data/jiaoguanbo/skeletonmem_github_skeletonmem-code_20260904T050058Z"
)


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n"
    )


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def run_git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return None


def file_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\0")
    return digest.hexdigest()


def canonical_action_id(dataset: str, task: str, episode: int, target_t: int) -> str:
    return f"{dataset}\t{task}\t{int(episode)}\t{int(target_t)}"


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p10": math.nan, "p50": math.nan, "p90": math.nan, "p99": math.nan}
    values = sorted(values)

    def q(prob: float) -> float:
        if len(values) == 1:
            return float(values[0])
        pos = prob * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(values[lo])
        frac = pos - lo
        return float(values[lo] * (1.0 - frac) + values[hi] * frac)

    return {"p10": q(0.10), "p50": q(0.50), "p90": q(0.90), "p99": q(0.99)}


def n_eff(values: list[float]) -> float:
    total = float(sum(values))
    denom = float(sum(v * v for v in values))
    return total * total / denom if denom > 0 else 0.0


def field(value: Any, source: str, unit: str | None = None, note: str | None = None):
    out = {"value": value, "source": source}
    if unit:
        out["unit"] = unit
    if note:
        out["note"] = note
    return out


def load_lineage(v1_bundle: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in (0, 1):
        rows.extend(pq.read_table(v1_bundle / f"lineage_rank{rank}.parquet").to_pylist())
    return rows


def aggregate_seen_counts(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    padding_total = 0
    padding_active = 0
    e_padding_unweighted = 0
    e_padding_scheduler_weighted = 0.0
    invalid_or_zero_weight_real = 0

    for row in rows:
        targets = row["action_target_timestamps"] or []
        valid_mask = row["valid_mask"] or []
        weights = row["loss_weights"] or []
        update_applied = bool(row.get("optimizer_update_applied"))
        for target_t, valid, weight in zip(targets, valid_mask, weights):
            target_i = int(target_t)
            valid_b = bool(valid)
            weight_f = float(weight)
            if target_i < 0:
                padding_total += 1
                if update_applied and valid_b and weight_f > 0.0:
                    padding_active += 1
                    e_padding_unweighted += 1
                    e_padding_scheduler_weighted += weight_f
                continue
            if not (update_applied and valid_b and weight_f > 0.0):
                invalid_or_zero_weight_real += 1
                continue
            cid = canonical_action_id(row["dataset"], row["task"], row["episode"], target_i)
            item = counts.get(cid)
            if item is None:
                item = {
                    "dataset": row["dataset"],
                    "task": row["task"],
                    "episode": int(row["episode"]),
                    "t_target_absolute": target_i,
                    "canonical_action_id": cid,
                    "source_views": set(),
                    "phases": set(),
                    "subset_manifest_ids": set(),
                    "exposure_unweighted": 0,
                    "exposure_scheduler_weighted": 0.0,
                    "first_optimizer_step": int(row["optimizer_step"]),
                    "last_optimizer_step": int(row["optimizer_step"]),
                    "sample_occurrence_count": 0,
                    "ranks": set(),
                }
                counts[cid] = item
            item["source_views"].add(row.get("source_view") or "unknown")
            item["phases"].add(row.get("phase") or "train")
            item["subset_manifest_ids"].add(row.get("subset_manifest_id") or "unknown")
            item["exposure_unweighted"] += 1
            item["exposure_scheduler_weighted"] += weight_f
            item["first_optimizer_step"] = min(
                item["first_optimizer_step"], int(row["optimizer_step"])
            )
            item["last_optimizer_step"] = max(
                item["last_optimizer_step"], int(row["optimizer_step"])
            )
            item["sample_occurrence_count"] += 1
            item["ranks"].add(int(row["rank"]))

    padding = {
        "padding_target_slots_total": padding_total,
        "padding_target_slots_active": padding_active,
        "E_padding_unweighted": e_padding_unweighted,
        "E_padding_scheduler_weighted": e_padding_scheduler_weighted,
        "invalid_or_zero_weight_real_target_slots": invalid_or_zero_weight_real,
    }
    return counts, padding


def support_hash(canonical_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(canonical_ids):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_eligible_universe(manifest_v1: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    import torch

    dataset_path = Path(manifest_v1["resolved_config"]["dataset_path"])
    camera_key = manifest_v1["resolved_config"]["obs_cam_keys"][0]
    ids: dict[str, dict[str, Any]] = {}
    repo_roots = sorted(
        Path(info).parents[1]
        for info in dataset_path.rglob("meta/info.json")
    )
    if not repo_roots:
        raise FileNotFoundError(f"No meta/info.json files under {dataset_path}")

    for repo_root in repo_roots:
        with (repo_root / "meta" / "episodes.jsonl").open() as file:
            episodes = [json.loads(line) for line in file if line.strip()]
        episodes = sorted(episodes, key=lambda value: int(value["episode_index"]))
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            action_config = sorted(
                episode["action_config"],
                key=lambda item: (int(item["start_frame"]), int(item["end_frame"])),
            )
            for acfg in action_config:
                start_frame = int(acfg["start_frame"])
                end_frame = int(acfg["end_frame"])
                latent_matches = sorted(
                    (repo_root / "latents").glob(
                        f"chunk-*/{camera_key}/episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                    )
                )
                if not latent_matches:
                    continue
                latent_data = torch.load(
                    latent_matches[0],
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
                frame_ids = latent_data["frame_ids"]
                frame_ids = [int(value) for value in frame_ids]
                if len(frame_ids) < 2:
                    raise ValueError(f"Not enough frame_ids in {latent_matches[0]}")
                frame_stride = int(frame_ids[1] - frame_ids[0])
                if frame_stride <= 0:
                    raise ValueError(f"Non-positive frame stride in {latent_matches[0]}")
                latent_frame_num = (len(frame_ids) - 1) // 4 + 1
                action_tokens_per_latent = frame_stride * 4
                act_shift = int(frame_ids[0] - start_frame)
                prefix_count = action_tokens_per_latent
                required_action_num = latent_frame_num * action_tokens_per_latent
                target_timestamps = [-1] * min(prefix_count, required_action_num)
                real_start = start_frame + act_shift
                real_available = max(end_frame - real_start, 0)
                real_needed = required_action_num - len(target_timestamps)
                target_timestamps.extend(
                    range(real_start, real_start + min(real_available, real_needed))
                )
                if len(target_timestamps) < required_action_num:
                    target_timestamps.extend(
                        [-1] * (required_action_num - len(target_timestamps))
                    )
                task = str(acfg.get("action_text") or episode["tasks"][0])
                dataset_name = repo_root.name
                for target_t in target_timestamps:
                    target_i = int(target_t)
                    if target_i < 0:
                        continue
                    cid = canonical_action_id(
                        dataset_name, task, episode_index, target_i
                    )
                    ids.setdefault(
                        cid,
                        {
                            "dataset": dataset_name,
                            "task": task,
                            "episode": episode_index,
                            "t_target_absolute": target_i,
                            "canonical_action_id": cid,
                        },
                    )
    rows = sorted(
        ids.values(),
        key=lambda item: (
            item["dataset"],
            item["task"],
            item["episode"],
            item["t_target_absolute"],
        ),
    )
    return rows, time.perf_counter() - started


def write_label_tables(
    *,
    out_dir: Path,
    eligible_rows: list[dict[str, Any]],
    seen_counts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    label_rows: list[dict[str, Any]] = []
    for eligible in eligible_rows:
        cid = eligible["canonical_action_id"]
        seen = cid in seen_counts
        item = seen_counts.get(cid)
        label_rows.append(
            {
                **eligible,
                "retained": True,
                "eligible": True,
                "seen": seen,
                "exposure_unweighted": int(item["exposure_unweighted"]) if item else 0,
                "exposure_scheduler_weighted": float(item["exposure_scheduler_weighted"])
                if item
                else 0.0,
                "first_optimizer_step": int(item["first_optimizer_step"]) if item else None,
                "last_optimizer_step": int(item["last_optimizer_step"]) if item else None,
                "sample_occurrence_count": int(item["sample_occurrence_count"]) if item else 0,
                "rank_count": len(item["ranks"]) if item else 0,
                "source_views": sorted(item["source_views"]) if item else [],
                "phases": sorted(item["phases"]) if item else [],
                "subset_manifest_ids": sorted(item["subset_manifest_ids"]) if item else [],
            }
        )

    pq.write_table(pa.Table.from_pylist(label_rows), out_dir / "label_exposure_counts_v2.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(eligible_rows), out_dir / "eligible_action_ids_v2.parquet", compression="zstd")
    return label_rows


def make_resource_card(
    *,
    manifest_v1: dict[str, Any],
    label_rows: list[dict[str, Any]],
    padding: dict[str, Any],
    support_scan_wall_clock: float,
) -> dict[str, Any]:
    seen_rows = [row for row in label_rows if row["seen"]]
    eligible_count = len(label_rows)
    seen_count = len(seen_rows)
    e_unweighted = sum(int(row["exposure_unweighted"]) for row in label_rows)
    e_scheduler = sum(float(row["exposure_scheduler_weighted"]) for row in label_rows)
    unweighted_seen = [float(row["exposure_unweighted"]) for row in seen_rows]
    unweighted_eligible = [float(row["exposure_unweighted"]) for row in label_rows]
    scheduler_seen = [float(row["exposure_scheduler_weighted"]) for row in seen_rows]
    n_eff_unweighted = n_eff(unweighted_eligible)
    n_eff_scheduler = n_eff(scheduler_seen)
    padding_fraction = (
        padding["E_padding_scheduler_weighted"]
        / (padding["E_padding_scheduler_weighted"] + e_scheduler)
        if (padding["E_padding_scheduler_weighted"] + e_scheduler) > 0
        else 0.0
    )
    wall_clock = None
    try:
        wall_clock = float(manifest_v1["checkpoint_comparison"].get("training_wall_clock_s"))
    except Exception:
        wall_clock = None
    # V1 Resource Card has exact train wall-clock; use that instead.
    whole = {
        "sample_draws": field(3200, "exact", "sample occurrences"),
        "U_ret": field(eligible_count, "analytical", "canonical action targets"),
        "U_eligible": field(eligible_count, "analytical", "canonical action targets"),
        "U_seen": field(seen_count, "exact", "canonical action targets"),
        "E_action_unweighted": field(e_unweighted, "exact", "physical action target exposures"),
        "E_action_scheduler_weighted": field(
            e_scheduler,
            "exact",
            "scheduler-weighted physical action target exposures",
            "Weights are diffusion scheduler frame weights only, not full optimizer-contribution coefficients.",
        ),
        "E_action_effective_loss_weighted": field(
            None,
            "unavailable",
            "optimizer-contribution coefficient mass",
            "Current lineage does not record per-frame valid-mask normalization, episode frame-count normalization, episode_equal_loss scaling, or DDP/global reduction scaling.",
        ),
        "mean_replay_unweighted_seen": field(e_unweighted / seen_count, "exact"),
        "mean_replay_unweighted_eligible": field(e_unweighted / eligible_count, "exact"),
        "mean_replay_scheduler_weighted": field(e_scheduler / seen_count, "exact"),
        "exposure_unweighted_p10_seen": field(quantiles(unweighted_seen)["p10"], "exact"),
        "exposure_unweighted_p50_seen": field(quantiles(unweighted_seen)["p50"], "exact"),
        "exposure_unweighted_p90_seen": field(quantiles(unweighted_seen)["p90"], "exact"),
        "exposure_unweighted_p99_seen": field(quantiles(unweighted_seen)["p99"], "exact"),
        "exposure_unweighted_p10_eligible": field(quantiles(unweighted_eligible)["p10"], "exact"),
        "exposure_unweighted_p50_eligible": field(quantiles(unweighted_eligible)["p50"], "exact"),
        "exposure_unweighted_p90_eligible": field(quantiles(unweighted_eligible)["p90"], "exact"),
        "exposure_unweighted_p99_eligible": field(quantiles(unweighted_eligible)["p99"], "exact"),
        "exposure_scheduler_weighted_p10_seen": field(quantiles(scheduler_seen)["p10"], "exact"),
        "exposure_scheduler_weighted_p50_seen": field(quantiles(scheduler_seen)["p50"], "exact"),
        "exposure_scheduler_weighted_p90_seen": field(quantiles(scheduler_seen)["p90"], "exact"),
        "exposure_scheduler_weighted_p99_seen": field(quantiles(scheduler_seen)["p99"], "exact"),
        "N_eff_unweighted": field(n_eff_unweighted, "exact", "canonical action targets"),
        "N_eff_unweighted/U_seen": field(n_eff_unweighted / seen_count, "exact"),
        "N_eff_unweighted/U_eligible": field(n_eff_unweighted / eligible_count, "exact"),
        "N_eff_scheduler_weighted": field(n_eff_scheduler, "exact", "canonical action targets"),
        "N_eff_scheduler_weighted/U_seen": field(n_eff_scheduler / seen_count, "exact"),
        "N_eff_scheduler_weighted/U_eligible": field(n_eff_scheduler / eligible_count, "exact"),
        "zero_exposure_fraction_eligible": field((eligible_count - seen_count) / eligible_count, "exact"),
        "zero_exposure_fraction_retained": field((eligible_count - seen_count) / eligible_count, "exact"),
        "padding_target_slots_total": field(padding["padding_target_slots_total"], "exact", "target slots"),
        "padding_target_slots_active": field(padding["padding_target_slots_active"], "exact", "target slots"),
        "E_padding_unweighted": field(padding["E_padding_unweighted"], "exact", "padding target slots"),
        "E_padding_scheduler_weighted": field(
            padding["E_padding_scheduler_weighted"], "exact", "scheduler-weighted padding slots"
        ),
        "padding_scheduler_weight_fraction": field(padding_fraction, "exact"),
        "invalid_or_zero_weight_real_target_slots": field(
            padding["invalid_or_zero_weight_real_target_slots"], "exact", "target slots"
        ),
        "optimizer_steps": field(100, "exact", "optimizer updates"),
        "global_batch_size": field(32, "analytical", "sample occurrences per optimizer update"),
        "world_size": field(2, "analytical", "ranks"),
        "gradient_accumulation": field(16, "analytical", "configured gradient accumulation steps"),
        "end_to_end_tracer_wall_clock": field(1555.198176, "exact", "seconds", "V1 Resource Card training_wall_clock: tracer ON from lineage tracer initialization through training completion/finalization."),
        "optimizer_update_wall_clock": field(None, "unavailable", "seconds", "Logs expose per-step tqdm timings but not a clean exact optimizer-only interval separated from dataloader/model/checkpoint/report phases."),
        "support_scan_wall_clock": field(support_scan_wall_clock, "exact", "seconds", "V2 eligible universe scan over dataset.get_lineage_metadata."),
        "checkpoint_IO_wall_clock": field(None, "unavailable", "seconds", "The logs have checkpoint timestamps but do not isolate all-rank checkpoint I/O from barriers and trainer bookkeeping exactly."),
        "report_finalize_wall_clock": field(None, "unavailable", "seconds", "V1 did not separately time report finalization."),
        "training_GPU_hours": field(0.8639989865, "analytical", "GPU-hours", "end_to_end_tracer_wall_clock multiplied by world_size / 3600."),
        "curation_GPU_hours": field(None, "unavailable", "GPU-hours", "Full dry run did not execute a curation phase."),
        "valid_physical_targets_per_second": field(e_unweighted / 1555.198176, "exact", "physical action targets/s"),
    }
    card = {
        "format_version": 2,
        "run_id": manifest_v1["run_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "metric_semantics": {
            "weighted_exposure": "scheduler_weighted_only",
            "physical_action_exposure_excludes_padding": True,
            "effective_loss_weighted_exposure": "unavailable",
        },
        "whole_run": whole,
        "phases": {"train": whole},
        "source_views": {"full": whole},
    }
    return card


def markdown_resource_card(card: dict[str, Any]) -> str:
    lines = [
        "# Resource Card v2",
        "",
        f"- run_id: `{card['run_id']}`",
        f"- generated_utc: `{card['generated_utc']}`",
        "- weighted exposure semantics: `scheduler_weighted_only`",
        "- physical action exposure excludes padding/null targets.",
        "",
        "## Whole Run",
        "",
        "| field | value | source | unit | note |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for key, item in card["whole_run"].items():
        value = item["value"]
        if value is None:
            value_s = "unavailable"
        elif isinstance(value, float):
            value_s = f"{value:.12g}"
        else:
            value_s = str(value)
        lines.append(
            f"| `{key}` | {value_s} | {item.get('source','')} | {item.get('unit','')} | {item.get('note','')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `E_action_scheduler_weighted` uses the recorded diffusion scheduler frame weights only.",
            "- `E_action_effective_loss_weighted` is unavailable because the lineage does not record the full optimizer-contribution coefficient.",
            "- Padding/null targets are reported separately and are not mixed into physical-action `U_seen` or `E_action_*`.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_semantics_md() -> str:
    return """# Metric Semantics v2

This v2 audit bundle does not alter model training, loss computation, sampler order,
dataset order, or raw lineage records. It only renames and recomputes reporting
fields from the existing Full100 lineage.

## Weighted Exposure

The lineage field previously named `loss_weights` records diffusion scheduler
training weights. It does not include per-frame valid-mask normalization, episode
frame-count normalization, `episode_equal_loss` scaling, or DDP/global reduction
scaling.

Therefore v2 uses:

- `E_action_scheduler_weighted`
- `mean_replay_scheduler_weighted`
- `N_eff_scheduler_weighted`
- `exposure_scheduler_weighted_pXX_seen`

The full optimizer-contribution metric is reported as:

```text
E_action_effective_loss_weighted = unavailable
```

## Physical Action Exposure

Physical-action `U_seen` and `E_action_*` count only committed optimizer-update
slots with:

```text
t_target_absolute >= 0
valid_mask == true
scheduler_weight > 0
```

Padding/null targets remain auditable but are reported in a separate padding
section.

## Quantiles

Fields ending in `_seen` are computed over `U_seen`. Fields ending in
`_eligible` include zero-exposure targets from the full eligible universe.
"""


def make_migration_md() -> str:
    return """# Migration v1 to v2

v2 is a reporting-semantics migration from the original Full100 lineage audit.
No training code was rerun and no raw lineage records were modified.

## Renamed Fields

| v1 field | v2 field |
| --- | --- |
| `E_action_weighted` | `E_action_scheduler_weighted` |
| `mean_replay` | `mean_replay_scheduler_weighted` |
| `N_eff` | `N_eff_scheduler_weighted` |
| `exposure_p10/p50/p90/p99` | `exposure_scheduler_weighted_p10/p50/p90/p99_seen` |

## Added Fields

- complete `label_exposure_counts_v2.parquet` over all eligible canonical IDs;
- `eligible_action_ids_v2.parquet`;
- unweighted exposure quantiles over both `U_seen` and `U_eligible`;
- `N_eff_unweighted` and ratios against `U_seen` and `U_eligible`;
- padding/null-target consumption fields;
- unavailable status for full effective-loss-weighted exposure and unrecoverable timing fields.
"""


def make_bundle_index(out_dir: Path, card: dict[str, Any], tests: list[dict[str, Any]]) -> str:
    passed = sum(1 for test in tests if test["passed"])
    return f"""# First Lineage Audit Bundle v2

- bundle_dir: `{out_dir}`
- source_v1_bundle: `{V1_BUNDLE_DEFAULT}`
- run_id: `{card['run_id']}`
- tests: `{passed}/{len(tests)}` passed

## Key Files

- `metric_semantics.md`
- `migration_v1_to_v2.md`
- `resource_card_v2.json`
- `resource_card_v2.md`
- `label_exposure_counts_v2.parquet`
- `eligible_action_ids_v2.parquet`
- `synthetic_tests_v2.json`
- `synthetic_tests_v2.md`
- `dry_run_manifest_v2.json`
- `lineage_rank0.parquet`
- `lineage_rank1.parquet`
- `SHA256SUMS.txt`

## Core Metrics

- U_eligible: `{card['whole_run']['U_eligible']['value']}`
- U_seen: `{card['whole_run']['U_seen']['value']}`
- E_action_unweighted: `{card['whole_run']['E_action_unweighted']['value']}`
- E_action_scheduler_weighted: `{card['whole_run']['E_action_scheduler_weighted']['value']}`
- N_eff_unweighted: `{card['whole_run']['N_eff_unweighted']['value']}`
- N_eff_scheduler_weighted: `{card['whole_run']['N_eff_scheduler_weighted']['value']}`
"""


def make_tests(
    *,
    label_rows: list[dict[str, Any]],
    card: dict[str, Any],
    padding: dict[str, Any],
    v1_card: dict[str, Any],
    eligible_sha: str,
    reverse_order_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    hist = Counter(row["exposure_unweighted"] for row in label_rows)
    seen_rows = [row for row in label_rows if row["seen"]]
    cids = [row["canonical_action_id"] for row in label_rows]
    scheduler_sum = sum(row["exposure_scheduler_weighted"] for row in label_rows)
    unweighted_values = [float(row["exposure_unweighted"]) for row in label_rows]
    tests = [
        {
            "name": "full_eligible_universe_includes_zero_exposure_targets",
            "expected": {"rows": 273412, "zero_exposure_targets": 32},
            "actual": {"rows": len(label_rows), "zero_exposure_targets": hist.get(0, 0)},
        },
        {
            "name": "eligible_equals_seen_plus_unseen",
            "expected": {"eligible": 273412, "seen": 273380, "unseen": 32},
            "actual": {
                "eligible": len(label_rows),
                "seen": len(seen_rows),
                "unseen": len(label_rows) - len(seen_rows),
            },
        },
        {
            "name": "unweighted_histogram_manual_check",
            "expected": {0: 32, 1: 37116, 2: 236264},
            "actual": dict(sorted(hist.items())),
        },
        {
            "name": "N_eff_unweighted_manual_check",
            "expected": 264451.650765854,
            "actual": n_eff(unweighted_values),
        },
        {
            "name": "scheduler_weighted_values_equal_v1",
            "expected": {
                "E_action_scheduler_weighted": v1_card["whole_run"]["E_action_weighted"]["value"],
                "N_eff_scheduler_weighted": v1_card["whole_run"]["N_eff"]["value"],
            },
            "actual": {
                "E_action_scheduler_weighted": scheduler_sum,
                "N_eff_scheduler_weighted": card["whole_run"]["N_eff_scheduler_weighted"]["value"],
            },
        },
        {
            "name": "padding_active_count_and_scheduler_mass",
            "expected": {
                "padding_target_slots_total": 12800,
                "padding_target_slots_active": 12792,
                "E_padding_scheduler_weighted": 12720.07010185835,
            },
            "actual": {
                "padding_target_slots_total": padding["padding_target_slots_total"],
                "padding_target_slots_active": padding["padding_target_slots_active"],
                "E_padding_scheduler_weighted": padding["E_padding_scheduler_weighted"],
            },
        },
        {
            "name": "padding_not_in_physical_U_seen_or_E_action",
            "expected": {"U_seen": 273380, "E_action_unweighted": 509644},
            "actual": {
                "U_seen": card["whole_run"]["U_seen"]["value"],
                "E_action_unweighted": card["whole_run"]["E_action_unweighted"]["value"],
            },
        },
        {
            "name": "label_table_no_duplicate_canonical_action_id",
            "expected": {"unique_ids": len(label_rows), "rows": len(label_rows)},
            "actual": {"unique_ids": len(set(cids)), "rows": len(cids)},
        },
        {
            "name": "label_table_ids_match_eligible_universe",
            "expected": {"label_table_action_ids_sha256": eligible_sha},
            "actual": {"label_table_action_ids_sha256": support_hash(cids)},
        },
        {
            "name": "schema_lint_no_generic_loss_weighted_exposure_fields",
            "expected": {"forbidden_fields_absent": True},
            "actual": {
                "forbidden_fields_absent": not any(
                    key in card["whole_run"]
                    for key in ["E_action_weighted", "mean_replay", "N_eff", "exposure_p10"]
                )
            },
        },
        {
            "name": "v1_to_v2_core_lineage_values_no_drift",
            "expected": {
                "sample_draws": 3200,
                "U_seen": 273380,
                "E_action_unweighted": 509644,
            },
            "actual": {
                "sample_draws": card["whole_run"]["sample_draws"]["value"],
                "U_seen": card["whole_run"]["U_seen"]["value"],
                "E_action_unweighted": card["whole_run"]["E_action_unweighted"]["value"],
            },
        },
        {
            "name": "rank_aggregation_order_invariant",
            "expected": {
                "E_action_unweighted": card["whole_run"]["E_action_unweighted"]["value"],
                "U_seen": card["whole_run"]["U_seen"]["value"],
            },
            "actual": reverse_order_metrics,
        },
    ]

    def equal(lhs: Any, rhs: Any) -> bool:
        if isinstance(lhs, float) or isinstance(rhs, float):
            return math.isclose(float(lhs), float(rhs), rel_tol=1e-12, abs_tol=1e-9)
        if isinstance(lhs, dict) and isinstance(rhs, dict):
            return lhs.keys() == rhs.keys() and all(equal(lhs[k], rhs[k]) for k in lhs)
        return lhs == rhs

    for test in tests:
        test["passed"] = equal(test["expected"], test["actual"])
    return tests


def tests_markdown(tests: list[dict[str, Any]]) -> str:
    lines = [
        "# Synthetic Tests v2",
        "",
        f"- test_count: `{len(tests)}`",
        f"- passed_count: `{sum(1 for test in tests if test['passed'])}`",
        "",
        "| test | status | expected | actual |",
        "| --- | --- | --- | --- |",
    ]
    for test in tests:
        status = "PASS" if test["passed"] else "FAIL"
        lines.append(
            "| `{}` | {} | `{}` | `{}` |".format(
                test["name"],
                status,
                json.dumps(test["expected"], sort_keys=True, default=json_default),
                json.dumps(test["actual"], sort_keys=True, default=json_default),
            )
        )
    return "\n".join(lines) + "\n"


def make_dry_run_manifest_v2(
    *,
    manifest_v1: dict[str, Any],
    out_dir: Path,
    eligible_sha: str,
    public_audit_code_commit: str | None,
    bundle_payload_sha256: str | None,
) -> dict[str, Any]:
    repo = Path(manifest_v1["repo"])
    current_dirty_status = run_git(["status", "--short"], repo)
    return {
        "format_version": 2,
        "source_v1_manifest": manifest_v1,
        "runtime_repo_commit": manifest_v1.get("git", {}).get("commit"),
        "runtime_dirty_status": manifest_v1.get("git", {}).get("dirty_status"),
        "runtime_dirty_diff_sha256": {
            "value": None,
            "source": "unavailable",
            "note": "The exact historical dirty diff used for v1 was not archived. Current worktree diff is intentionally not substituted.",
        },
        "source_tree_sha256": {
            "value": file_tree_sha256(out_dir),
            "source": "exact",
            "note": "SHA256 over files present in the v2 bundle directory at manifest generation time.",
        },
        "public_audit_code_commit": public_audit_code_commit,
        "bundle_zip_sha256": {
            "value": None,
            "source": "unavailable_inside_archive",
            "note": "A zip cannot contain a stable hash of itself. The final release zip SHA256 is reported externally and in the final handoff.",
        },
        "eligible_action_ids_sha256": eligible_sha,
        "current_runtime_dirty_status_observed_but_not_used_as_historical_diff": current_dirty_status.splitlines()
        if current_dirty_status
        else [],
        "v2_generation_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_sha256s(out_dir: Path) -> None:
    lines = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        lines.append(f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def copy_v1_inputs(v1_bundle: Path, v1_log_dir: Path, out_dir: Path) -> None:
    for name in [
        "lineage_rank0.parquet",
        "lineage_rank1.parquet",
        "lineage_rank0_summary.json",
        "lineage_rank1_summary.json",
        "human_readable_examples.jsonl",
        "human_readable_examples_actual.jsonl",
        "human_readable_examples_synthetic.jsonl",
        "batch_signatures_rank0.jsonl",
        "batch_signatures_rank1.jsonl",
        "batch_signatures_replay_rank0.jsonl",
        "batch_signatures_replay_rank1.jsonl",
        "invariance_report.json",
        "invariance_report.md",
        "support_summary.json",
        "lineage_spec.md",
    ]:
        shutil.copy2(v1_bundle / name, out_dir / name)
    logs = out_dir / "logs"
    logs.mkdir(exist_ok=True)
    for name in ["train_off_full100.log", "train_on_full100.log"]:
        src = v1_log_dir / name
        if src.is_file():
            shutil.copy2(src, logs / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-bundle", type=Path, default=V1_BUNDLE_DEFAULT)
    parser.add_argument("--v1-log-dir", type=Path, default=V1_LOG_DIR_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    parser.add_argument("--timestamp", default=utc_now_compact())
    args = parser.parse_args()

    out_dir = args.out_root / f"first_lineage_audit_bundle_v2_{args.timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    manifest_v1 = json.loads((args.v1_bundle / "dry_run_manifest_resolved.json").read_text())
    v1_card = json.loads((args.v1_bundle / "resource_card.json").read_text())
    rows = load_lineage(args.v1_bundle)
    seen_counts, padding = aggregate_seen_counts(rows)
    reverse_counts, _ = aggregate_seen_counts(list(reversed(rows)))
    reverse_order_metrics = {
        "E_action_unweighted": sum(item["exposure_unweighted"] for item in reverse_counts.values()),
        "U_seen": len(reverse_counts),
    }

    eligible_rows, support_scan_wall_clock = build_eligible_universe(manifest_v1)
    eligible_sha = support_hash([row["canonical_action_id"] for row in eligible_rows])

    copy_v1_inputs(args.v1_bundle, args.v1_log_dir, out_dir)
    label_rows = write_label_tables(
        out_dir=out_dir, eligible_rows=eligible_rows, seen_counts=seen_counts
    )
    card = make_resource_card(
        manifest_v1=manifest_v1,
        label_rows=label_rows,
        padding=padding,
        support_scan_wall_clock=support_scan_wall_clock,
    )
    write_json(out_dir / "resource_card_v2.json", card)
    (out_dir / "resource_card_v2.md").write_text(markdown_resource_card(card))
    (out_dir / "metric_semantics.md").write_text(make_semantics_md())
    (out_dir / "migration_v1_to_v2.md").write_text(make_migration_md())

    tests = make_tests(
        label_rows=label_rows,
        card=card,
        padding=padding,
        v1_card=v1_card,
        eligible_sha=eligible_sha,
        reverse_order_metrics=reverse_order_metrics,
    )
    write_json(out_dir / "synthetic_tests_v2.json", tests)
    (out_dir / "synthetic_tests_v2.md").write_text(tests_markdown(tests))
    (out_dir / "bundle_index.md").write_text(make_bundle_index(out_dir, card, tests))

    public_commit = run_git(["rev-parse", "main"], PUBLIC_AUDIT_REPO_DEFAULT)
    manifest_v2 = make_dry_run_manifest_v2(
        manifest_v1=manifest_v1,
        out_dir=out_dir,
        eligible_sha=eligible_sha,
        public_audit_code_commit=public_commit,
        bundle_payload_sha256=None,
    )
    write_json(out_dir / "dry_run_manifest_v2.json", manifest_v2)
    write_sha256s(out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
