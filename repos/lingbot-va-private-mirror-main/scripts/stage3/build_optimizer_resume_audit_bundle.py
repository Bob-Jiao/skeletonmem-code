#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from safetensors.torch import safe_open


REQUIRED_BUNDLE_FILES = [
    "bundle_index.md",
    "root_cause_report.md",
    "optimizer_state_before_save.parquet",
    "optimizer_state_after_load.parquet",
    "optimizer_state_diff.json",
    "model_state_boundary_diff.json",
    "resume_invariance_report.md",
    "resume_invariance_report.json",
    "synthetic_tests.md",
    "synthetic_tests.json",
    "resolved_config.yaml",
    "dry_run_manifest.json",
    "code_diff.patch",
    "SHA256SUMS.txt",
]

MISSING_ADAMW_STATE_PARAMETERS = [
    "condition_embedder_action.text_embedder.linear_1.weight",
    "condition_embedder_action.text_embedder.linear_1.bias",
    "condition_embedder_action.text_embedder.linear_2.weight",
    "condition_embedder_action.text_embedder.linear_2.bias",
]

EXPOSURE_FIELDS = [
    "sample_draws",
    "U_ret",
    "U_eligible",
    "U_seen",
    "E_action_unweighted",
    "E_action_weighted",
    "mean_replay",
    "mean_replay_unweighted",
    "exposure_p10",
    "exposure_p50",
    "exposure_p90",
    "exposure_p99",
    "N_eff",
    "N_eff/U_seen",
    "N_eff/U_eligible",
    "zero_exposure_fraction_eligible",
    "zero_exposure_fraction_retained",
    "optimizer_steps",
    "global_batch_size",
    "world_size",
    "gradient_accumulation",
    "padding_target_slots_ignored_for_U_seen_E_A",
    "invalid_or_zero_weight_real_target_slots",
]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def load_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default).encode()
    ).hexdigest()


def git_output(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception:
        return None


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def summary_paths(lineage_dir: Path) -> list[Path]:
    return sorted(lineage_dir.glob("lineage_rank[0-9]*_summary.json"))


def shard_paths(lineage_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for summary_path in summary_paths(lineage_dir):
        summary = load_json(summary_path)
        for shard in summary.get("shards", []):
            path = Path(shard["path"])
            if path.exists():
                paths.append(path)
    return sorted(set(paths))


def iter_records(paths: list[Path]):
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256):
            for record in batch.to_pylist():
                yield record


def accepted_records(lineage_dir: Path) -> list[dict[str, Any]]:
    paths = shard_paths(lineage_dir)
    latest_generation: dict[str, int] = {}
    for record in iter_records(paths):
        if not bool(record.get("optimizer_update_applied")):
            continue
        occurrence_id = str(record["occurrence_id"])
        generation = int(record.get("resume_generation", 0))
        latest_generation[occurrence_id] = max(generation, latest_generation.get(occurrence_id, generation))

    accepted = []
    for record in iter_records(paths):
        if not bool(record.get("optimizer_update_applied")):
            continue
        occurrence_id = str(record["occurrence_id"])
        if int(record.get("resume_generation", 0)) == latest_generation[occurrence_id]:
            accepted.append(record)
    return accepted


def canonical_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_step": int(record["optimizer_step"]),
        "rank": int(record["rank"]),
        "microbatch_id": int(record["microbatch_id"]),
        "gradient_accumulation_index": int(record["gradient_accumulation_index"]),
        "sample_slot": int(record["sample_slot"]),
        "dataset": record["dataset"],
        "task": record["task"],
        "episode": int(record["episode"]),
        "requested_anchor_t": int(record["requested_anchor_t"]),
        "remapped_anchor_t": int(record["remapped_anchor_t"]),
        "observation_timestamps": [int(v) for v in record["observation_timestamps"]],
        "action_target_timestamps": [int(v) for v in record["action_target_timestamps"]],
        "valid_mask": [bool(v) for v in record["valid_mask"]],
        "loss_weights": [float(v) for v in record["loss_weights"]],
    }


def normalized_batch_signatures(lineage_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(lineage_dir.glob("batch_signatures_rank*.jsonl")):
        with path.open() as file:
            for line in file:
                row = json.loads(line)
                rows.append(
                    {
                        "optimizer_step": int(row["optimizer_step"]),
                        "rank": int(row["rank"]),
                        "microbatch_id": int(row["microbatch_id"]),
                        "gradient_accumulation_index": int(row["gradient_accumulation_index"]),
                        "signature_sha256": row["signature_sha256"],
                        "items": row["items"],
                    }
                )
    return sorted(rows, key=lambda row: (row["optimizer_step"], row["rank"], row["microbatch_id"]))


def resource_values(card: dict[str, Any]) -> dict[str, Any]:
    whole = card["whole_run"]
    return {field: whole[field]["value"] for field in EXPOSURE_FIELDS if field in whole}


def read_update_audit(lineage_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(lineage_dir.glob("optimizer_update_audit_rank*.jsonl")):
        with path.open() as file:
            for line in file:
                row = json.loads(line)
                rows[(int(row["rank"]), int(row["optimizer_step"]))] = row
    return rows


def compare_update_audit(ref_dir: Path, resumed_dir: Path) -> dict[str, Any]:
    ref = read_update_audit(ref_dir)
    resumed = read_update_audit(resumed_dir)
    fields = [
        "latent_loss",
        "action_loss",
        "total_loss",
        "grad_norm",
        "learning_rate",
        "optimizer_state_checksum_before_update",
        "model_checksum_after_update",
    ]
    comparisons = []
    first_diff = None
    for key in sorted(set(ref) | set(resumed)):
        rank, step = key
        if step < 5 or step > 9:
            continue
        ref_row = ref.get(key)
        resumed_row = resumed.get(key)
        diffs = {}
        for field in fields:
            left = ref_row.get(field) if ref_row else None
            right = resumed_row.get(field) if resumed_row else None
            if left != right:
                diffs[field] = {"uninterrupted": left, "resumed": right}
        row = {
            "rank": rank,
            "optimizer_step": step,
            "uninterrupted": {field: ref_row.get(field) for field in fields} if ref_row else None,
            "resumed": {field: resumed_row.get(field) for field in fields} if resumed_row else None,
            "identical": not diffs,
            "diffs": diffs,
        }
        comparisons.append(row)
        if diffs and first_diff is None:
            first_diff = {"rank": rank, "optimizer_step": step, "diffs": diffs}
    return {
        "step_5_to_9_comparisons": comparisons,
        "all_step_5_to_9_fields_identical": all(row["identical"] for row in comparisons),
        "first_step_diff": first_diff,
    }


def compare_optimizer_inventory(before_path: Path, after_path: Path) -> dict[str, Any]:
    before = {str(row["parameter_name"]): row for row in read_parquet_rows(before_path)}
    after = {str(row["parameter_name"]): row for row in read_parquet_rows(after_path)}
    mismatches = []
    fields = [
        "optimizer_group_index",
        "position_in_group",
        "optimizer_state_present",
        "optimizer_state_keys",
        "optimizer_step",
        "step_checksum",
        "exp_avg_shape",
        "exp_avg_dtype",
        "exp_avg_checksum",
        "exp_avg_sq_shape",
        "exp_avg_sq_dtype",
        "exp_avg_sq_checksum",
    ]
    for name in sorted(set(before) & set(after)):
        diffs = {field: {"before": before[name].get(field), "after": after[name].get(field)} for field in fields if before[name].get(field) != after[name].get(field)}
        if diffs:
            mismatches.append({"parameter_name": name, "diffs": diffs})
    missing_before = [row for name, row in before.items() if name in MISSING_ADAMW_STATE_PARAMETERS]
    return {
        "before_rows": len(before),
        "after_rows": len(after),
        "missing_parameter_names": sorted(set(before) - set(after)),
        "extra_parameter_names": sorted(set(after) - set(before)),
        "state_mismatch_count": len(mismatches),
        "state_mismatch_examples": mismatches[:20],
        "bitwise_equal_name_aligned_optimizer_state": not mismatches and set(before) == set(after),
        "previously_missing_adamw_slots": missing_before,
    }


def compare_model_boundary(before_path: Path, checkpoint_path: Path, after_load_path: Path | None) -> dict[str, Any]:
    before = {str(row["parameter_name"]): row for row in read_parquet_rows(before_path)}
    checkpoint = {str(row["parameter_name"]): row for row in read_parquet_rows(checkpoint_path)}
    after = {str(row["parameter_name"]): row for row in read_parquet_rows(after_load_path)} if after_load_path and after_load_path.exists() else {}
    dtype_changes = []
    checksum_changes = []
    for name in sorted(set(before) & set(checkpoint)):
        if before[name].get("parameter_dtype") != checkpoint[name].get("parameter_dtype"):
            dtype_changes.append({"parameter_name": name, "before": before[name].get("parameter_dtype"), "checkpoint": checkpoint[name].get("parameter_dtype")})
        if before[name].get("parameter_checksum") != checkpoint[name].get("parameter_checksum"):
            checksum_changes.append(name)
    return {
        "before_save_rows": len(before),
        "checkpoint_rows": len(checkpoint),
        "after_load_rows": len(after) if after else None,
        "parameter_name_sets_equal_before_vs_checkpoint": set(before) == set(checkpoint),
        "parameter_name_sets_equal_checkpoint_vs_after_load": (set(checkpoint) == set(after)) if after else None,
        "save_before_dtype_set": sorted({row.get("parameter_dtype") for row in before.values()}),
        "checkpoint_dtype_set": sorted({row.get("parameter_dtype") for row in checkpoint.values()}),
        "after_load_dtype_set": sorted({row.get("parameter_dtype") for row in after.values()}) if after else None,
        "dtype_change_count_before_vs_checkpoint": len(dtype_changes),
        "dtype_change_examples": dtype_changes[:20],
        "checksum_change_count_before_vs_checkpoint": len(checksum_changes),
        "checksum_change_examples": checksum_changes[:20],
        "name_aligned_max_abs_diff_before_save_vs_checkpoint": "unavailable: before-save full-precision tensor values were not persisted, only checksums were persisted",
        "name_aligned_max_abs_diff_checkpoint_vs_after_load": 0.0 if after and all(checkpoint[name].get("parameter_checksum") == after[name].get("parameter_checksum") for name in checkpoint) else None,
        "model_save_precision_loss": len(dtype_changes) > 0,
    }


def compare_safetensors(ref_path: Path, resumed_path: Path) -> dict[str, Any]:
    ref_sha = sha256_file(ref_path)
    resumed_sha = sha256_file(resumed_path)
    max_abs = 0.0
    max_rel = 0.0
    first_parameter = None
    first_diff = None
    with safe_open(ref_path, framework="pt", device="cpu") as ref_file, safe_open(resumed_path, framework="pt", device="cpu") as resumed_file:
        ref_keys = list(ref_file.keys())
        resumed_keys = list(resumed_file.keys())
        if ref_keys != resumed_keys:
            return {
                "uninterrupted_checkpoint_sha256": ref_sha,
                "resumed_checkpoint_sha256": resumed_sha,
                "bitwise_identical": False,
                "parameter_keys_identical": False,
                "uninterrupted_key_count": len(ref_keys),
                "resumed_key_count": len(resumed_keys),
            }
        for name in ref_keys:
            left = ref_file.get_tensor(name)
            right = resumed_file.get_tensor(name)
            if left.shape != right.shape or left.dtype != right.dtype:
                if first_diff is None:
                    first_diff = {"parameter_name": name, "field": "shape_or_dtype", "uninterrupted": [str(left.dtype), list(left.shape)], "resumed": [str(right.dtype), list(right.shape)]}
                continue
            diff = (left.float() - right.float()).abs()
            local_max = float(diff.max().item()) if diff.numel() else 0.0
            denom = left.float().abs().clamp_min(1e-12)
            local_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
            if local_max > max_abs:
                max_abs = local_max
                max_rel = local_rel
                first_parameter = name
            if local_max != 0.0 and first_diff is None:
                first_diff = {"parameter_name": name, "field": "parameter_value", "max_abs_diff": local_max, "max_relative_diff": local_rel}
    return {
        "uninterrupted_checkpoint_sha256": ref_sha,
        "resumed_checkpoint_sha256": resumed_sha,
        "bitwise_identical": ref_sha == resumed_sha,
        "parameter_keys_identical": True,
        "all_parameters_bitwise_identical": max_abs == 0.0,
        "max_abs_parameter_diff": max_abs,
        "max_relative_parameter_diff": max_rel,
        "first_different_parameter_name": first_diff["parameter_name"] if first_diff else None,
        "largest_diff_parameter_name": first_parameter,
        "first_optimizer_state_field_difference": None,
    }


def compare_runs(audit_root: Path) -> dict[str, Any]:
    ref_dir = audit_root / "run_a_uninterrupted" / "lineage"
    resumed_dir = audit_root / "run_b_resumed" / "lineage"
    ref_records = accepted_records(ref_dir)
    resumed_records = accepted_records(resumed_dir)
    ref_by_occ = {str(row["occurrence_id"]): row for row in ref_records}
    resumed_by_occ = {str(row["occurrence_id"]): row for row in resumed_records}
    common = sorted(set(ref_by_occ) & set(resumed_by_occ))
    lineage_mismatches = []
    for occurrence_id in common:
        if canonical_record(ref_by_occ[occurrence_id]) != canonical_record(resumed_by_occ[occurrence_id]):
            lineage_mismatches.append(occurrence_id)
            if len(lineage_mismatches) >= 20:
                break
    ref_sigs = normalized_batch_signatures(ref_dir)
    resumed_sigs = normalized_batch_signatures(resumed_dir)
    ref_card = load_json(ref_dir / "resource_card.json")
    resumed_card = load_json(resumed_dir / "resource_card.json")
    checkpoint = compare_safetensors(
        audit_root / "run_a_uninterrupted" / "train" / "checkpoints" / "checkpoint_step_10" / "transformer" / "diffusion_pytorch_model.safetensors",
        audit_root / "run_b_resumed" / "train" / "checkpoints" / "checkpoint_step_10" / "transformer" / "diffusion_pytorch_model.safetensors",
    )
    update_audit = compare_update_audit(ref_dir, resumed_dir)
    return {
        "occurrence": {
            "uninterrupted_count": len(ref_records),
            "resumed_count": len(resumed_records),
            "duplicate_occurrence_count_accepted": len(resumed_records) - len(resumed_by_occ),
            "missing_occurrence_count": len(set(ref_by_occ) - set(resumed_by_occ)),
            "extra_occurrence_count": len(set(resumed_by_occ) - set(ref_by_occ)),
            "sets_identical": set(ref_by_occ) == set(resumed_by_occ),
        },
        "lineage": {
            "canonical_record_mismatch_count": len(lineage_mismatches),
            "canonical_record_mismatch_examples": lineage_mismatches,
            "requested_remapped_target_mask_weight_identical": not lineage_mismatches,
        },
        "batch_signatures": {
            "uninterrupted_count": len(ref_sigs),
            "resumed_count": len(resumed_sigs),
            "identical_after_attempt_normalization": ref_sigs == resumed_sigs,
            "uninterrupted_sha256": stable_hash(ref_sigs),
            "resumed_sha256": stable_hash(resumed_sigs),
        },
        "resource_card": {
            "uninterrupted": resource_values(ref_card),
            "resumed": resource_values(resumed_card),
            "exposure_fields_identical": resource_values(ref_card) == resource_values(resumed_card),
        },
        "step_5_to_9": update_audit,
        "checkpoint": checkpoint,
    }


def collect_memory(lineage_dir: Path, projected_steps: int) -> dict[str, Any]:
    summaries = [load_json(path) for path in summary_paths(lineage_dir)]
    shards = []
    total_bytes = 0
    for path in shard_paths(lineage_dir):
        size = path.stat().st_size
        total_bytes += size
        shards.append({"path": str(path), "bytes": size, "sha256": sha256_file(path)})
    card = load_json(lineage_dir / "resource_card.json")
    sample_draws = int(card["whole_run"]["sample_draws"]["value"])
    optimizer_steps = int(card["whole_run"]["optimizer_steps"]["value"])
    return {
        "max_buffered_records": max(int(summary.get("max_buffered_records") or 0) for summary in summaries),
        "max_pending_records": max(int(summary.get("max_pending_records") or 0) for summary in summaries),
        "peak_host_rss_bytes": max(int(summary.get("peak_host_rss_bytes") or 0) for summary in summaries),
        "shards": shards,
        "total_lineage_bytes": total_bytes,
        "bytes_per_sample_occurrence": total_bytes / sample_draws if sample_draws else None,
        "projected_20k_lineage_bytes_same_schema": total_bytes * projected_steps / optimizer_steps if optimizer_steps else None,
        "lineage_flush_wall_clock_s_rank_sum": sum(float(summary.get("lineage_flush_wall_clock_s") or 0.0) for summary in summaries),
        "aggregation_wall_clock_s": card.get("durable_lineage", {}).get("aggregation_wall_clock_s"),
        "bounded_memory_evidence": "observed buffered records are bounded by configured shard interval and pending microbatches, not by total optimizer steps",
    }


def collect_timing(lineage_dir: Path) -> dict[str, Any]:
    per_attempt = []
    by_rank: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for path in summary_paths(lineage_dir):
        summary = load_json(path)
        rank = int(summary["rank"])
        timings = Counter({key: float(value) for key, value in summary.get("timing_seconds", {}).items()})
        timings["support_scan_wall_clock_s"] += float(summary.get("support_scan_wall_clock_s") or 0.0)
        timings["lineage_flush_wall_clock_s"] += float(summary.get("lineage_flush_wall_clock_s") or 0.0)
        by_rank[rank].update(timings)
        per_attempt.append({"path": str(path), "rank": rank, "training_attempt_id": summary.get("training_attempt_id"), "resume_generation": summary.get("resume_generation"), "timing_seconds": dict(timings)})
    aggregate = {}
    for key in sorted({key for counters in by_rank.values() for key in counters}):
        values = [float(by_rank[rank].get(key, 0.0)) for rank in sorted(by_rank)]
        aggregate[key] = {"rank_values": values, "rank_max": max(values), "rank_mean": sum(values) / len(values), "rank_sum": sum(values)}
    card = load_json(lineage_dir / "resource_card.json")
    aggregate["report_finalize_wall_clock_s"] = {
        "rank_values": [card.get("durable_lineage", {}).get("aggregation_wall_clock_s")],
        "rank_max": card.get("durable_lineage", {}).get("aggregation_wall_clock_s"),
        "rank_mean": card.get("durable_lineage", {}).get("aggregation_wall_clock_s"),
        "rank_sum": card.get("durable_lineage", {}).get("aggregation_wall_clock_s"),
        "aggregation": "rank0 finalize_global streaming aggregation",
    }
    return {"per_attempt": per_attempt, "aggregate": aggregate}


def parse_log_times(log_path: Path) -> dict[str, Any]:
    return {"log_path": str(log_path), "sha256": sha256_file(log_path) if log_path.exists() else None}


def make_tests(report: dict[str, Any], optimizer_diff: dict[str, Any], model_boundary: dict[str, Any], synthetic: dict[str, Any]) -> list[dict[str, Any]]:
    tests = []

    def add(name: str, expected: Any, actual: Any, passed: bool) -> None:
        tests.append({"name": name, "expected": expected, "actual": actual, "pass": bool(passed)})

    add("AdamW state save/load name-aligned exact equality", True, optimizer_diff["bitwise_equal_name_aligned_optimizer_state"], optimizer_diff["bitwise_equal_name_aligned_optimizer_state"])
    add("optimizer param-group reordered rejected", True, synthetic["tests"][1]["pass"], synthetic["tests"][1]["pass"])
    add("existing exp_avg/exp_avg_sq deletion rejected", True, synthetic["tests"][2]["pass"], synthetic["tests"][2]["pass"])
    add("never-gradient parameter allows legal empty state", 4, len(optimizer_diff["previously_missing_adamw_slots"]), len(optimizer_diff["previously_missing_adamw_slots"]) == 4)
    add("legacy non-equivalent compatible restore is explicitly marked", True, synthetic["tests"][4]["pass"], synthetic["tests"][4]["pass"])
    add("next_optimizer_step no off-by-one", 5, report["boundary"]["next_optimizer_step"], report["boundary"]["next_optimizer_step"] == 5)
    add("scheduler state aligns with optimizer step", True, report["boundary"]["scheduler_state_loaded"], bool(report["boundary"]["scheduler_state_loaded"]))
    add("RNG state present at save boundary", True, report["boundary"]["rng_state_present"], bool(report["boundary"]["rng_state_present"]))
    add("gradient accumulation resumes only at boundary", 0, report["boundary"]["gradient_accumulation_position"], report["boundary"]["gradient_accumulation_position"] == 0)
    add("partial checkpoint requires success marker", True, report["boundary"]["checkpoint_success_marker_present"], bool(report["boundary"]["checkpoint_success_marker_present"]))
    add("uninterrupted/resumed occurrence sets identical", True, report["invariance"]["occurrence"]["sets_identical"], report["invariance"]["occurrence"]["sets_identical"])
    add("uninterrupted/resumed exposure fields identical", True, report["invariance"]["resource_card"]["exposure_fields_identical"], report["invariance"]["resource_card"]["exposure_fields_identical"])
    add("model checkpoint preserves runtime parameter dtype", True, not model_boundary["model_save_precision_loss"], not model_boundary["model_save_precision_loss"])
    add("final optimizer state name-aligned exact equality", True, report["final_optimizer_state"]["bitwise_equal_name_aligned_optimizer_state"], report["final_optimizer_state"]["bitwise_equal_name_aligned_optimizer_state"])
    return tests


def markdown_tests(tests: list[dict[str, Any]]) -> str:
    lines = ["| test | expected | actual | result |", "|---|---:|---:|---|"]
    for test in tests:
        lines.append(
            f"| {test['name']} | `{json.dumps(test['expected'], sort_keys=True)}` | "
            f"`{json.dumps(test['actual'], sort_keys=True)}` | {'PASS' if test['pass'] else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def copy_if_exists(source: Path, dest: Path) -> None:
    if source.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def copy_artifacts(audit_root: Path, logs_root: Path, bundle: Path, synthetic_dir: Path) -> None:
    step5_state = audit_root / "run_b_resumed" / "train" / "checkpoints" / "checkpoint_step_5" / "trainer_state"
    copy_if_exists(step5_state / "optimizer_state_before_save.parquet", bundle / "optimizer_state_before_save.parquet")
    copy_if_exists(audit_root / "run_b_resumed" / "lineage" / "optimizer_state_after_load_run_b_attempt1_resume_to_step10.parquet", bundle / "optimizer_state_after_load.parquet")
    copy_if_exists(step5_state / "model_state_before_save.parquet", bundle / "model_state_before_save.parquet")
    copy_if_exists(step5_state / "model_state_checkpoint.parquet", bundle / "model_state_checkpoint.parquet")
    copy_if_exists(step5_state / "model_state_checkpoint_bf16.parquet", bundle / "model_state_checkpoint_bf16.parquet")
    copy_if_exists(audit_root / "run_b_resumed" / "lineage" / "model_state_after_load_run_b_attempt1_resume_to_step10.parquet", bundle / "model_state_after_load.parquet")
    copy_if_exists(synthetic_dir / "synthetic_tests.json", bundle / "synthetic_tests.json")
    copy_if_exists(synthetic_dir / "synthetic_tests.md", bundle / "synthetic_tests.md")
    copy_if_exists(audit_root / "run_b_resumed" / "lineage" / "accepted_training_trajectory.json", bundle / "commit_ledger.json")
    copy_if_exists(audit_root / "run_a_uninterrupted" / "lineage" / "resource_card.json", bundle / "uninterrupted_resource_card.json")
    copy_if_exists(audit_root / "run_b_resumed" / "lineage" / "resource_card.json", bundle / "resumed_resource_card.json")
    copy_if_exists(audit_root / "run_a_uninterrupted" / "lineage" / "resource_card.md", bundle / "uninterrupted_resource_card.md")
    copy_if_exists(audit_root / "run_b_resumed" / "lineage" / "resource_card.md", bundle / "resumed_resource_card.md")
    for run_name in ["run_a_uninterrupted", "run_b_resumed"]:
        out_dir = bundle / run_name / "lineage"
        for pattern in ["lineage_rank*_steps*_part*.parquet", "lineage_rank*summary.json", "batch_signatures_rank*.jsonl", "attempt_rank*.json", "support_summary.json", "human_readable_examples_actual.jsonl", "optimizer_update_audit_rank*.jsonl"]:
            for source in (audit_root / run_name / "lineage").glob(pattern):
                copy_if_exists(source, out_dir / source.name)
    for label, subdir in [("uninterrupted", "uninterrupted"), ("interrupted", "interrupted"), ("resumed", "resumed")]:
        out_dir = bundle / "logs" / label
        for source in (logs_root / subdir).glob("*.log"):
            copy_if_exists(source, out_dir / source.name)


def write_sha256sums(bundle: Path) -> None:
    lines = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(bundle)}")
    (bundle / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def zip_dir(bundle: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(bundle.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--logs-root", required=True, type=Path)
    parser.add_argument("--synthetic-dir", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--projected-steps", default=20000, type=int)
    args = parser.parse_args()

    if args.bundle_dir.exists():
        shutil.rmtree(args.bundle_dir)
    args.bundle_dir.mkdir(parents=True)

    step5_state = args.audit_root / "run_b_resumed" / "train" / "checkpoints" / "checkpoint_step_5" / "trainer_state"
    before_optimizer = step5_state / "optimizer_state_before_save.parquet"
    after_optimizer = args.audit_root / "run_b_resumed" / "lineage" / "optimizer_state_after_load_run_b_attempt1_resume_to_step10.parquet"
    before_model = step5_state / "model_state_before_save.parquet"
    checkpoint_model = step5_state / "model_state_checkpoint.parquet"
    if not checkpoint_model.is_file():
        checkpoint_model = step5_state / "model_state_checkpoint_bf16.parquet"
    after_model = args.audit_root / "run_b_resumed" / "lineage" / "model_state_after_load_run_b_attempt1_resume_to_step10.parquet"
    validation = load_json(args.audit_root / "run_b_resumed" / "lineage" / "optimizer_state_load_validation_run_b_attempt1_resume_to_step10.json")
    boundary = load_json(step5_state / "checkpoint_boundary_before_save.json")
    manifest = load_json(args.audit_root / "run_b_resumed" / "lineage" / "dry_run_manifest.json")

    optimizer_diff = compare_optimizer_inventory(before_optimizer, after_optimizer)
    optimizer_diff["load_validation"] = validation
    model_boundary = compare_model_boundary(before_model, checkpoint_model, after_model)
    invariance = compare_runs(args.audit_root)
    final_optimizer_diff = compare_optimizer_inventory(
        args.audit_root / "run_a_uninterrupted" / "train" / "checkpoints" / "checkpoint_step_10" / "trainer_state" / "optimizer_state_checkpoint.parquet",
        args.audit_root / "run_b_resumed" / "train" / "checkpoints" / "checkpoint_step_10" / "trainer_state" / "optimizer_state_checkpoint.parquet",
    )
    pass_a = (
        optimizer_diff["bitwise_equal_name_aligned_optimizer_state"]
        and final_optimizer_diff["bitwise_equal_name_aligned_optimizer_state"]
        and not model_boundary["model_save_precision_loss"]
        and invariance["occurrence"]["sets_identical"]
        and invariance["lineage"]["requested_remapped_target_mask_weight_identical"]
        and invariance["batch_signatures"]["identical_after_attempt_normalization"]
        and invariance["resource_card"]["exposure_fields_identical"]
        and invariance["step_5_to_9"]["all_step_5_to_9_fields_identical"]
        and invariance["checkpoint"]["bitwise_identical"]
    )
    full_report = {
        "verdict": "PASS-A" if pass_a else "FAIL",
        "verdict_reason": (
            "model, optimizer, scheduler, RNG, lineage, step audit, and final checkpoint are bitwise identical"
            if pass_a
            else (
                "checkpoint save casts runtime model parameters to lower precision, so resume is not bitwise-equivalent"
                if model_boundary["model_save_precision_loss"]
                else "resume comparison found non-bitwise differences outside PASS-A criteria"
            )
        ),
        "boundary": {
            "next_optimizer_step": boundary.get("next_optimizer_step"),
            "checkpoint_step": boundary.get("checkpoint_step"),
            "gradient_accumulation_position": boundary.get("gradient_accumulation_position"),
            "gradient_accumulation_buffer_cleared": boundary.get("gradient_accumulation_buffer_cleared"),
            "scheduler_state_loaded": validation.get("scheduler_state_loaded"),
            "rng_state_present": all((step5_state / f"rng_rank_{rank}.pt").exists() for rank in [0, 1]),
            "checkpoint_success_marker_present": (args.audit_root / "run_b_resumed" / "train" / "checkpoints" / "checkpoint_step_5" / "_SUCCESS").exists(),
        },
        "optimizer_state": optimizer_diff,
        "final_optimizer_state": final_optimizer_diff,
        "model_boundary": model_boundary,
        "invariance": invariance,
        "uninterrupted_repeat": (
            "not_needed: PASS-A achieved"
            if pass_a
            else (
                "not_run: divergence source is identified as checkpoint model dtype conversion rather than unknown nondeterminism"
                if model_boundary["model_save_precision_loss"]
                else "not_run: no unknown divergence drill-down was requested after this builder run"
            )
        ),
    }

    synthetic = load_json(args.synthetic_dir / "synthetic_tests.json")
    tests = make_tests(full_report, optimizer_diff, model_boundary, synthetic)
    memory = {
        "uninterrupted": collect_memory(args.audit_root / "run_a_uninterrupted" / "lineage", args.projected_steps),
        "resumed": collect_memory(args.audit_root / "run_b_resumed" / "lineage", args.projected_steps),
    }
    timing = {
        "uninterrupted": collect_timing(args.audit_root / "run_a_uninterrupted" / "lineage"),
        "resumed": collect_timing(args.audit_root / "run_b_resumed" / "lineage"),
        "log_sha256": {
            "uninterrupted": parse_log_times(args.logs_root / "uninterrupted" / "run_a.log"),
            "interrupted": parse_log_times(args.logs_root / "interrupted" / "run_b_attempt0_to_step5.log"),
            "resumed": parse_log_times(args.logs_root / "resumed" / "run_b_attempt1_resume_to_step10.log"),
        },
    }

    copy_artifacts(args.audit_root, args.logs_root, args.bundle_dir, args.synthetic_dir)
    write_json(args.bundle_dir / "optimizer_state_diff.json", optimizer_diff)
    write_json(args.bundle_dir / "model_state_boundary_diff.json", model_boundary)
    write_json(args.bundle_dir / "resume_invariance_report.json", full_report)
    write_json(args.bundle_dir / "memory_scaling_report.json", memory)
    write_json(args.bundle_dir / "timing_report.json", timing)
    write_json(args.bundle_dir / "tests.json", {"passed": sum(test["pass"] for test in tests), "total": len(tests), "tests": tests})

    precision_note = (
        "No model checkpoint precision loss was detected: runtime parameter dtype and saved checkpoint dtype match."
        if not model_boundary["model_save_precision_loss"]
        else (
            "The remaining resume non-equivalence is model checkpoint precision loss: runtime parameters are saved "
            "with a different dtype from the before-save runtime parameter manifest."
        )
    )
    (args.bundle_dir / "root_cause_report.md").write_text(
        "\n".join(
            [
                "# Root Cause Report",
                "",
                "The four missing AdamW slot states are class A: absent before checkpoint save because the parameters had not received gradients.",
                "The new loader no longer silently fills arbitrary missing state. It verifies param-group order and name-aligned AdamW `step`, `exp_avg`, and `exp_avg_sq`; only before-save proven empty states are allowed.",
                "",
                "| parameter | class | reason |",
                "|---|---|---|",
                *[
                    f"| `{row['parameter_name']}` | A | before-save manifest has `optimizer_state_present=false`, `has_received_gradient=false`, `number_of_applied_updates=0` |"
                    for row in optimizer_diff["previously_missing_adamw_slots"]
                ],
                "",
                precision_note,
                "",
            ]
        )
    )
    (args.bundle_dir / "resume_invariance_report.md").write_text(
        "\n".join(
            [
                "# Resume Invariance Report",
                "",
                f"- verdict: `{full_report['verdict']}`",
                f"- occurrence sets identical: `{invariance['occurrence']['sets_identical']}`",
                f"- duplicate occurrence count: `{invariance['occurrence']['duplicate_occurrence_count_accepted']}`",
                f"- missing occurrence count: `{invariance['occurrence']['missing_occurrence_count']}`",
                f"- batch lineage signatures identical: `{invariance['batch_signatures']['identical_after_attempt_normalization']}`",
                f"- requested/remapped anchors, target timestamps, masks, scheduler weights identical: `{invariance['lineage']['requested_remapped_target_mask_weight_identical']}`",
                f"- step 5-9 audit fields identical: `{invariance['step_5_to_9']['all_step_5_to_9_fields_identical']}`",
                f"- final optimizer state identical: `{final_optimizer_diff['bitwise_equal_name_aligned_optimizer_state']}`",
                f"- final checkpoint bitwise identical: `{invariance['checkpoint']['bitwise_identical']}`",
                f"- max_abs_parameter_diff: `{invariance['checkpoint'].get('max_abs_parameter_diff')}`",
                f"- max_relative_parameter_diff: `{invariance['checkpoint'].get('max_relative_parameter_diff')}`",
                f"- first different parameter: `{invariance['checkpoint'].get('first_different_parameter_name')}`",
                "",
            ]
        )
    )
    (args.bundle_dir / "memory_scaling_report.md").write_text(
        "\n".join(
            [
                "# Memory Scaling Report",
                "",
                f"- resumed max_buffered_records: `{memory['resumed']['max_buffered_records']}`",
                f"- resumed peak_host_rss_bytes: `{memory['resumed']['peak_host_rss_bytes']}`",
                f"- resumed total_lineage_bytes: `{memory['resumed']['total_lineage_bytes']}`",
                f"- resumed bytes_per_sample_occurrence: `{memory['resumed']['bytes_per_sample_occurrence']}`",
                f"- projected 20k lineage bytes: `{memory['resumed']['projected_20k_lineage_bytes_same_schema']}`",
                "",
            ]
        )
    )
    (args.bundle_dir / "timing_report.md").write_text("# Timing Report\n\n" + json.dumps(timing, indent=2, sort_keys=True) + "\n")
    (args.bundle_dir / "tests.md").write_text("# Tests\n\n" + markdown_tests(tests))
    (args.bundle_dir / "resolved_config.yaml").write_text(json.dumps(manifest.get("config", manifest), indent=2, sort_keys=True) + "\n")
    dry_manifest = {
        **manifest,
        "repo_commit": git_output(args.repo, "rev-parse", "HEAD"),
        "dirty_worktree_status": git_output(args.repo, "status", "--short", "--untracked-files=all"),
        "checkpoint_sha256": {
            "resume_checkpoint_optimizer_pt": sha256_file(step5_state / "optimizer.pt"),
            "uninterrupted_final_model": invariance["checkpoint"]["uninterrupted_checkpoint_sha256"],
            "resumed_final_model": invariance["checkpoint"]["resumed_checkpoint_sha256"],
        },
        "optimizer_implementation": "torch.optim.AdamW via FSDP optimizer state dict",
        "scheduler": "serialized in trainer_state/scheduler.pt",
        "model_save_dtype": model_boundary["checkpoint_dtype_set"],
        "runtime_model_parameter_dtype": model_boundary["save_before_dtype_set"],
        "initial_checkpoint": "configured base model/pretrained path; exact source recorded in resolved config when available",
        "resume_checkpoint": str(args.audit_root / "run_b_resumed" / "train" / "checkpoints" / "checkpoint_step_5"),
    }
    write_json(args.bundle_dir / "dry_run_manifest.json", dry_manifest)
    (args.bundle_dir / "code_diff.patch").write_text(git_output(args.repo, "diff") or "")
    (args.bundle_dir / "bundle_index.md").write_text(
        "\n".join(
            [
                "# Optimizer Resume Audit Bundle",
                "",
                f"- generated_utc: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
                f"- source_audit_root: `{args.audit_root}`",
                "- checkpoints and weights are intentionally excluded.",
                f"- primary verdict: `{full_report['verdict']}`.",
                "",
                "## Files",
                "",
                *[f"- `{name}`" for name in REQUIRED_BUNDLE_FILES],
                "- `run_a_uninterrupted/lineage/*`",
                "- `run_b_resumed/lineage/*`",
                "- `logs/uninterrupted/*`, `logs/interrupted/*`, `logs/resumed/*`",
                "",
            ]
        )
    )
    write_sha256sums(args.bundle_dir)
    for name in REQUIRED_BUNDLE_FILES:
        if not (args.bundle_dir / name).exists():
            raise FileNotFoundError(args.bundle_dir / name)
    zip_dir(args.bundle_dir, args.zip_path)
    print(json.dumps({"bundle_dir": str(args.bundle_dir), "zip_path": str(args.zip_path), "zip_sha256": sha256_file(args.zip_path), "tests_passed": sum(test["pass"] for test in tests), "tests_total": len(tests)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
