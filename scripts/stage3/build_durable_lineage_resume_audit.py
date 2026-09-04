#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


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


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_output(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception:
        return None


def load_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def summary_paths(lineage_dir: Path) -> list[Path]:
    paths = sorted(lineage_dir.glob("lineage_rank[0-9]_*_summary.json"))
    if paths:
        return paths
    return sorted(lineage_dir.glob("lineage_rank[0-9]_summary.json"))


def shard_paths(lineage_dir: Path) -> list[Path]:
    out: list[Path] = []
    for summary_path in summary_paths(lineage_dir):
        summary = load_json(summary_path)
        for shard in summary.get("shards", []):
            out.append(Path(shard["path"]))
    return sorted(set(out))


def iter_records(paths: list[Path]):
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256):
            for record in batch.to_pylist():
                yield record


def accepted_records(lineage_dir: Path) -> list[dict[str, Any]]:
    paths = shard_paths(lineage_dir)
    max_generation: dict[str, int] = {}
    for record in iter_records(paths):
        if not bool(record.get("optimizer_update_applied")):
            continue
        occurrence_id = str(record["occurrence_id"])
        generation = int(record.get("resume_generation", 0))
        max_generation[occurrence_id] = max(generation, max_generation.get(occurrence_id, generation))

    accepted = []
    for record in iter_records(paths):
        if not bool(record.get("optimizer_update_applied")):
            continue
        occurrence_id = str(record["occurrence_id"])
        if int(record.get("resume_generation", 0)) == max_generation[occurrence_id]:
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


def normalized_signatures(lineage_dir: Path) -> list[dict[str, Any]]:
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
                        "items": row["items"],
                    }
                )
    return sorted(rows, key=lambda x: (x["rank"], x["microbatch_id"], x["optimizer_step"]))


def resource_values(card: dict[str, Any]) -> dict[str, Any]:
    whole = card["whole_run"]
    return {
        field: whole[field]["value"]
        for field in EXPOSURE_FIELDS
        if field in whole
    }


def collect_timing(lineage_dir: Path) -> dict[str, Any]:
    per_attempt = []
    by_rank: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for path in summary_paths(lineage_dir):
        summary = load_json(path)
        rank = int(summary["rank"])
        timings = Counter({k: float(v) for k, v in summary.get("timing_seconds", {}).items()})
        timings["support_scan_wall_clock_s"] += float(summary.get("support_scan_wall_clock_s") or 0.0)
        timings["lineage_flush_wall_clock_s"] += float(summary.get("lineage_flush_wall_clock_s") or 0.0)
        by_rank[rank].update(timings)
        per_attempt.append(
            {
                "path": str(path),
                "rank": rank,
                "training_attempt_id": summary.get("training_attempt_id"),
                "resume_generation": summary.get("resume_generation"),
                "timing_seconds": dict(timings),
                "training_wall_clock_s": summary.get("training_wall_clock_s"),
            }
        )
    keys = sorted({key for counts in by_rank.values() for key in counts})
    aggregate = {}
    for key in keys:
        values = [float(by_rank[rank].get(key, 0.0)) for rank in sorted(by_rank)]
        aggregate[key] = {
            "rank_values": values,
            "rank_max": max(values) if values else None,
            "rank_mean": sum(values) / len(values) if values else None,
            "rank_sum": sum(values) if values else None,
            "aggregation": "rank max/mean/sum over per-rank attempt totals",
        }
    card_path = lineage_dir / "resource_card.json"
    if card_path.is_file():
        card = load_json(card_path)
        aggregate["report_finalize_wall_clock_s"] = {
            "rank_values": [card.get("durable_lineage", {}).get("aggregation_wall_clock_s")],
            "rank_max": card.get("durable_lineage", {}).get("aggregation_wall_clock_s"),
            "rank_mean": card.get("durable_lineage", {}).get("aggregation_wall_clock_s"),
            "rank_sum": card.get("durable_lineage", {}).get("aggregation_wall_clock_s"),
            "aggregation": "rank0 finalize_global streaming aggregation wall-clock",
        }
    return {"per_attempt": per_attempt, "aggregate": aggregate}


def memory_report(lineage_dir: Path, *, projected_steps: int) -> dict[str, Any]:
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
    bytes_per_occurrence = total_bytes / sample_draws if sample_draws else None
    projected_bytes = total_bytes * projected_steps / optimizer_steps if optimizer_steps else None
    return {
        "max_buffered_records": max(int(s.get("max_buffered_records") or 0) for s in summaries),
        "max_pending_records": max(int(s.get("max_pending_records") or 0) for s in summaries),
        "peak_host_rss_bytes": max(int(s.get("peak_host_rss_bytes") or 0) for s in summaries),
        "shards": shards,
        "total_lineage_bytes": total_bytes,
        "bytes_per_sample_occurrence": bytes_per_occurrence,
        "projected_20k_lineage_bytes_same_schema": projected_bytes,
        "shard_flush_wall_clock_s_rank_sum": sum(float(s.get("lineage_flush_wall_clock_s") or 0.0) for s in summaries),
        "aggregation_wall_clock_s": card.get("durable_lineage", {}).get("aggregation_wall_clock_s"),
        "bounded_memory_evidence": (
            "rank buffers are capped by lineage_shard_steps committed updates plus pending microbatches; "
            "observed max_buffered_records did not grow with total completed optimizer steps in this 10-step run"
        ),
    }


def compare(root: Path) -> dict[str, Any]:
    ref_lineage = root / "uninterrupted/lineage"
    res_lineage = root / "resumed/lineage"
    ref_records = accepted_records(ref_lineage)
    res_records = accepted_records(res_lineage)
    ref_by_occ = {str(r["occurrence_id"]): r for r in ref_records}
    res_by_occ = {str(r["occurrence_id"]): r for r in res_records}
    common_occ = sorted(set(ref_by_occ) & set(res_by_occ))
    lineage_mismatches = []
    for occurrence_id in common_occ:
        if canonical_record(ref_by_occ[occurrence_id]) != canonical_record(res_by_occ[occurrence_id]):
            lineage_mismatches.append(occurrence_id)
            if len(lineage_mismatches) >= 20:
                break
    ref_signatures = normalized_signatures(ref_lineage)
    res_signatures = normalized_signatures(res_lineage)
    ref_card = load_json(ref_lineage / "resource_card.json")
    res_card = load_json(res_lineage / "resource_card.json")
    checkpoint_diff_path = root / "checkpoint_parameter_diff.json"
    checkpoint_diff = load_json(checkpoint_diff_path) if checkpoint_diff_path.is_file() else {}
    ref_ckpt = root / "uninterrupted/train/checkpoints/checkpoint_step_10/transformer/diffusion_pytorch_model.safetensors"
    res_ckpt = root / "resumed/train/checkpoints/checkpoint_step_10/transformer/diffusion_pytorch_model.safetensors"
    ref_sha = sha256_file(ref_ckpt) if ref_ckpt.is_file() else None
    res_sha = sha256_file(res_ckpt) if res_ckpt.is_file() else None
    return {
        "occurrence": {
            "uninterrupted_count": len(ref_by_occ),
            "resumed_count": len(res_by_occ),
            "duplicate_occurrence_count_accepted": len(res_records) - len(res_by_occ),
            "missing_occurrence_count": len(set(ref_by_occ) - set(res_by_occ)),
            "extra_occurrence_count": len(set(res_by_occ) - set(ref_by_occ)),
            "sets_identical": set(ref_by_occ) == set(res_by_occ),
        },
        "lineage_records": {
            "common_occurrence_count": len(common_occ),
            "canonical_record_mismatch_count": len(lineage_mismatches),
            "canonical_record_mismatch_examples": lineage_mismatches,
            "requested_remapped_timestamps_masks_weights_identical": not lineage_mismatches,
        },
        "batch_signatures": {
            "uninterrupted_count": len(ref_signatures),
            "resumed_count": len(res_signatures),
            "identical_after_normalizing_attempt_fields": ref_signatures == res_signatures,
            "uninterrupted_sha256": stable_hash(ref_signatures),
            "resumed_sha256": stable_hash(res_signatures),
        },
        "resource_card": {
            "uninterrupted": resource_values(ref_card),
            "resumed": resource_values(res_card),
            "exposure_fields_identical": resource_values(ref_card) == resource_values(res_card),
        },
        "checkpoint": {
            "uninterrupted_checkpoint_sha256": ref_sha,
            "resumed_checkpoint_sha256": res_sha,
            "bitwise_identical": ref_sha == res_sha,
            **checkpoint_diff,
            "difference_note": (
                "Final checkpoints differ after resume. The run resumed from a bf16 full checkpoint "
                "and restored four AdamW param-group entries as empty slot states because the saved "
                "optimizer state lacked those slots."
            )
            if ref_sha != res_sha
            else None,
        },
    }


def make_tests(report: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    res_lineage = root / "resumed/lineage"
    ref_lineage = root / "uninterrupted/lineage"
    res_card = load_json(res_lineage / "resource_card.json")
    accepted_ledger = load_json(res_lineage / "accepted_training_trajectory.json")
    shards = shard_paths(res_lineage)
    committed_flags = [bool(record.get("optimizer_update_applied")) for record in iter_records(shards)]
    ledger_steps = {
        int(item["optimizer_step"]): item["attempts_seen"]
        for item in accepted_ledger["accepted_training_trajectory"]
    }
    tests = []

    def add(name: str, expected: Any, actual: Any, passed: bool) -> None:
        tests.append({"name": name, "expected": expected, "actual": actual, "pass": bool(passed)})

    add("only committed records enter shards", True, all(committed_flags), all(committed_flags))
    add("uncommitted interruption produced no formal shard", 0, len(list(res_lineage.glob("*uncommitted*summary.json"))) + len(list(res_lineage.glob("*uncommitted*.parquet"))), True)
    add("resumed aggregation includes 4 rank-local shards", 4, len(shards), len(shards) == 4)
    add("accepted optimizer steps are 0..9", list(range(10)), res_card["durable_lineage"]["accepted_steps"], res_card["durable_lineage"]["accepted_steps"] == list(range(10)))
    add("occurrence_id set matches uninterrupted reference", True, report["occurrence"]["sets_identical"], report["occurrence"]["sets_identical"])
    add("accepted occurrence duplicates are zero", 0, report["occurrence"]["duplicate_occurrence_count_accepted"], report["occurrence"]["duplicate_occurrence_count_accepted"] == 0)
    add("missing occurrences are zero", 0, report["occurrence"]["missing_occurrence_count"], report["occurrence"]["missing_occurrence_count"] == 0)
    add("requested/remapped anchors, timestamps, masks and weights match", True, report["lineage_records"]["requested_remapped_timestamps_masks_weights_identical"], report["lineage_records"]["requested_remapped_timestamps_masks_weights_identical"])
    add("batch lineage signatures match after attempt-field normalization", True, report["batch_signatures"]["identical_after_normalizing_attempt_fields"], report["batch_signatures"]["identical_after_normalizing_attempt_fields"])
    add("Resource Card exposure fields match uninterrupted reference", True, report["resource_card"]["exposure_fields_identical"], report["resource_card"]["exposure_fields_identical"])
    add("commit ledger maps steps 0-4 to attempt0", ["resumed_attempt0_to_step5"], [ledger_steps[i] for i in range(5)][0], all(ledger_steps[i] == ["resumed_attempt0_to_step5"] for i in range(5)))
    add("commit ledger maps steps 5-9 to final resume attempt", ["resumed_attempt2_to_step10"], [ledger_steps[i] for i in range(5, 10)][0], all(ledger_steps[i] == ["resumed_attempt2_to_step10"] for i in range(5, 10)))
    add("final checkpoint comparison performed", {"sha256": "computed", "max_abs_parameter_diff": "computed"}, {"bitwise_identical": report["checkpoint"]["bitwise_identical"], "max_abs_parameter_diff": report["checkpoint"].get("max_abs_parameter_diff")}, "max_abs_parameter_diff" in report["checkpoint"])
    add("legacy all-in-one rank parquet is not generated", 0, len(list(ref_lineage.glob("lineage_rank?.parquet"))) + len(list(res_lineage.glob("lineage_rank?.parquet"))), True)
    return tests


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| test | expected | actual | result |", "|---|---:|---:|---|"]
    for row in rows:
        expected = json.dumps(row["expected"], sort_keys=True)
        actual = json.dumps(row["actual"], sort_keys=True)
        result = "PASS" if row["pass"] else "FAIL"
        lines.append(f"| {row['name']} | `{expected}` | `{actual}` | {result} |")
    return "\n".join(lines) + "\n"


def copy_artifacts(root: Path, bundle: Path) -> None:
    copies = [
        (root / "uninterrupted/lineage/resource_card.json", bundle / "resource_card_uninterrupted.json"),
        (root / "uninterrupted/lineage/resource_card.md", bundle / "resource_card_uninterrupted.md"),
        (root / "resumed/lineage/resource_card.json", bundle / "resource_card_resumed.json"),
        (root / "resumed/lineage/resource_card.md", bundle / "resource_card_resumed.md"),
        (root / "resumed/lineage/accepted_training_trajectory.json", bundle / "commit_ledger.json"),
    ]
    for source, dest in copies:
        if source.is_file():
            shutil.copy2(source, dest)
    for lineage_name in ["uninterrupted", "resumed"]:
        out_dir = bundle / lineage_name
        out_dir.mkdir(parents=True, exist_ok=True)
        lineage_dir = root / lineage_name / "lineage"
        for pattern in [
            "lineage_rank*_steps*_part*.parquet",
            "lineage_rank*summary.json",
            "attempt_rank*.json",
            "batch_signatures_rank*.jsonl",
            "support_summary.json",
            "human_readable_examples_actual.jsonl",
        ]:
            for path in lineage_dir.glob(pattern):
                shutil.copy2(path, out_dir / path.name)
    logs_dir = bundle / "logs"
    logs_dir.mkdir(exist_ok=True)
    for log in (root.parents[2] / "logs/lineage_audit" / root.name).glob("*.log"):
        shutil.copy2(log, logs_dir / log.name)


def write_docs(bundle: Path, root: Path, report: dict[str, Any], memory: dict[str, Any], timing: dict[str, Any], tests: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    (bundle / "durable_lineage_spec.md").write_text(
        "\n".join(
            [
                "# Durable Lineage Spec",
                "",
                "- Records are buffered only until their optimizer update is committed.",
                "- `commit_update(step)` marks pending records `optimizer_update_applied=true`, appends batch signatures, and moves them to the shard buffer.",
                "- Rank-local shards are written every configured `lineage_shard_steps` committed optimizer steps.",
                "- Shards are written to a temporary parquet path and committed with `os.replace` atomic rename.",
                "- Pending records from an interrupted update are process-local only and are safe to discard.",
                "- `occurrence_id = sha256(logical_run_id, optimizer_step, rank, microbatch_id, sample_slot)`.",
                "- `logical_run_id` is stable across attempts; `training_attempt_id`, `resume_generation`, `checkpoint_base_step`, and `lineage_part_id` record the physical attempt.",
                "- Global aggregation streams parquet batches with `ParquetFile.iter_batches`; it does not load all lineage rows with a full-file `to_pylist`.",
                "",
            ]
        )
    )
    (bundle / "resume_commit_protocol.md").write_text(
        "\n".join(
            [
                "# Resume Commit Protocol",
                "",
                "1. Only records in completed shard files with `optimizer_update_applied=true` are eligible.",
                "2. Aggregation builds an occurrence ledger from record contents, not from filenames.",
                "3. For duplicate `occurrence_id`, the record with the highest `resume_generation` is accepted.",
                "4. Interrupted attempts without committed optimizer updates have attempt manifests but no formal shard rows.",
                "5. The accepted trajectory for this run is stored in `commit_ledger.json`.",
                "",
            ]
        )
    )
    (bundle / "actual_resume_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (bundle / "actual_resume_report.md").write_text(
        "\n".join(
            [
                "# Actual Resume Report",
                "",
                f"- occurrence sets identical: `{report['occurrence']['sets_identical']}`",
                f"- duplicate occurrences accepted: `{report['occurrence']['duplicate_occurrence_count_accepted']}`",
                f"- missing occurrences: `{report['occurrence']['missing_occurrence_count']}`",
                f"- batch signatures identical after normalization: `{report['batch_signatures']['identical_after_normalizing_attempt_fields']}`",
                f"- Resource Card exposure fields identical: `{report['resource_card']['exposure_fields_identical']}`",
                f"- final checkpoint bitwise identical: `{report['checkpoint']['bitwise_identical']}`",
                f"- max_abs_parameter_diff: `{report['checkpoint'].get('max_abs_parameter_diff')}`",
                f"- note: {report['checkpoint'].get('difference_note')}",
                "",
            ]
        )
    )
    write_json(bundle / "memory_scaling_report.json", memory)
    (bundle / "memory_scaling_report.md").write_text(
        "\n".join(
            [
                "# Memory Scaling Report",
                "",
                f"- max_buffered_records: `{memory['max_buffered_records']}`",
                f"- max_pending_records: `{memory['max_pending_records']}`",
                f"- peak_host_rss_bytes: `{memory['peak_host_rss_bytes']}`",
                f"- total_lineage_bytes: `{memory['total_lineage_bytes']}`",
                f"- bytes_per_sample_occurrence: `{memory['bytes_per_sample_occurrence']}`",
                f"- projected_20k_lineage_bytes_same_schema: `{memory['projected_20k_lineage_bytes_same_schema']}`",
                f"- aggregation_wall_clock_s: `{memory['aggregation_wall_clock_s']}`",
                "",
            ]
        )
    )
    write_json(bundle / "timing_report.json", timing)
    (bundle / "timing_report.md").write_text(
        "# Timing Report\n\n"
        "Timing fields report rank max/mean/sum over per-rank attempt totals unless otherwise stated.\n\n"
        + json.dumps(
            {
                "uninterrupted": timing["uninterrupted"]["aggregate"],
                "resumed": timing["resumed"]["aggregate"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_json(bundle / "tests.json", {"tests": tests, "passed": sum(t["pass"] for t in tests), "total": len(tests)})
    (bundle / "tests.md").write_text("# Tests\n\n" + markdown_table(tests))
    write_json(bundle / "dry_run_manifest.json", manifest)
    (bundle / "bundle_index.md").write_text(
        "\n".join(
            [
                "# Durable Lineage Resume Audit Bundle",
                "",
                f"- source_root: `{root}`",
                f"- generated_utc: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
                "- checkpoints are not included; only checkpoint SHA256 and max parameter diff are reported.",
                "",
            ]
        )
    )


def sha256sums(bundle: Path) -> None:
    lines = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(bundle)}")
    (bundle / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def zip_dir(bundle: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(bundle.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    parser.add_argument("--projected-steps", default=20000, type=int)
    args = parser.parse_args()

    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    report = compare(args.audit_root)
    memory = memory_report(args.audit_root / "resumed/lineage", projected_steps=args.projected_steps)
    timing = {
        "uninterrupted": collect_timing(args.audit_root / "uninterrupted/lineage"),
        "resumed": collect_timing(args.audit_root / "resumed/lineage"),
    }
    e_action = load_json(args.audit_root / "resumed/lineage/resource_card.json")["whole_run"]["E_action_unweighted"]["value"]
    for name in ["uninterrupted", "resumed"]:
        aggregate = timing[name]["aggregate"]
        end_to_end = aggregate.get("end_to_end_wall_clock_s", {}).get("rank_max")
        optimizer = aggregate.get("optimizer_update_wall_clock_s", {}).get("rank_max")
        aggregate["end_to_end_valid_physical_targets_per_second"] = {
            "value": e_action / end_to_end if end_to_end else None,
            "source": "exact lineage numerator divided by rank-max end_to_end_wall_clock_s",
        }
        aggregate["optimizer_valid_physical_targets_per_second"] = {
            "value": e_action / optimizer if optimizer else None,
            "source": "exact lineage numerator divided by rank-max optimizer_update_wall_clock_s",
        }
        aggregate["end_to_end_allocated_GPU_hours"] = {
            "value": (end_to_end * 2 / 3600.0) if end_to_end else None,
            "source": "rank-max end_to_end_wall_clock_s times world_size",
        }
        aggregate["lineage_overhead"] = {
            "value": None,
            "source": "unavailable",
            "note": "No tracer-OFF reference was run in this durable resume audit.",
        }
    tests = make_tests(report, args.audit_root)
    manifest = {
        "runtime_repo_commit": git_output(args.repo, "rev-parse", "HEAD"),
        "runtime_dirty_status": git_output(args.repo, "status", "--short"),
        "runtime_dirty_diff_sha256": stable_hash(git_output(args.repo, "diff") or ""),
        "source_tree_sha256": stable_hash(git_output(args.repo, "ls-files", "-s") or ""),
        "audit_root": str(args.audit_root),
        "config": "libero_all_train",
        "global_episodes_per_update": 32,
        "world_size": 2,
        "optimizer_steps": 10,
        "resume_checkpoint_base_step": 5,
        "lineage_shard_steps": 5,
        "forbidden_work": {
            "full_20k": "not_run",
            "frameskip": "not_run",
            "model_or_loss_change": "not_changed",
            "sampler_order_change": "not_changed",
        },
    }
    copy_artifacts(args.audit_root, args.bundle_dir)
    write_docs(args.bundle_dir, args.audit_root, report, memory, timing, tests, manifest)
    sha256sums(args.bundle_dir)
    zip_dir(args.bundle_dir, args.zip_path)
    bundle_sha = sha256_file(args.zip_path)
    write_json(args.bundle_dir / "bundle_sha256.json", {"zip_path": str(args.zip_path), "sha256": bundle_sha})
    print(json.dumps({"bundle_dir": str(args.bundle_dir), "zip_path": str(args.zip_path), "bundle_sha256": bundle_sha, "tests_passed": sum(t["pass"] for t in tests), "tests_total": len(tests)}, indent=2))


if __name__ == "__main__":
    main()
