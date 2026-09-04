#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2] / "repos" / "lingbot-va-private-mirror-main"
sys.path.insert(0, str(REPO_ROOT / "wan_va"))

from lineage_audit import compute_resource_summary, stable_hash  # noqa: E402


def record(
    *,
    name: str,
    optimizer_step: int = 0,
    rank: int = 0,
    source_view: str = "full",
    dataset: str = "synthetic_dataset",
    task: str = "synthetic_task",
    episode: int = 0,
    requested_anchor_t: int = 0,
    remapped_anchor_t: int = 0,
    targets: list[int],
    valid: list[bool] | None = None,
    weights: list[float] | None = None,
    committed: bool = True,
) -> dict:
    if valid is None:
        valid = [True] * len(targets)
    if weights is None:
        weights = [1.0] * len(targets)
    if not (len(targets) == len(valid) == len(weights)):
        raise ValueError(name)
    return {
        "format_version": 1,
        "run_id": "synthetic_lineage_tests",
        "phase": "train",
        "optimizer_step": optimizer_step,
        "optimizer_update_applied": committed,
        "rank": rank,
        "world_size": 2,
        "microbatch_id": optimizer_step * 10 + rank,
        "gradient_accumulation_index": optimizer_step,
        "sample_slot": 0,
        "source_view": source_view,
        "dataset": dataset,
        "task": task,
        "episode": episode,
        "dataset_index": optimizer_step * 100 + rank,
        "local_dataset_index": optimizer_step * 100 + rank,
        "sample_uid": optimizer_step * 1000 + rank,
        "requested_anchor_t": requested_anchor_t,
        "remapped_anchor_t": remapped_anchor_t,
        "segment_start_t": remapped_anchor_t,
        "segment_end_t": remapped_anchor_t + max([t for t in targets if t >= 0] + [0]) + 1,
        "observation_timestamps": [remapped_anchor_t, remapped_anchor_t + 4],
        "action_target_timestamps": targets,
        "valid_mask": valid,
        "loss_weights": weights,
        "subset_manifest_id": source_view,
        "sampler_probability": 1.0,
        "frame_count": max(1, math.ceil(len(targets) / 4)),
        "action_tokens_per_frame": 4,
        "padding_target_count": sum(1 for target in targets if target < 0),
        "real_target_count": sum(1 for target in targets if target >= 0),
        "micro_count": 1,
        "planned_tokens": 128,
        "synthetic_case": name,
    }


def field_value(summary: dict, field: str):
    return summary[field]["value"]


def run_case(
    name: str,
    records: list[dict],
    *,
    retained: set[str] | None = None,
    eligible: set[str] | None = None,
    expected: dict,
):
    summary = compute_resource_summary(
        records,
        retained_ids=retained,
        eligible_ids=eligible,
        topology={
            "global_batch_size": 2,
            "world_size": 2,
            "gradient_accumulation": 2,
        },
        training_wall_clock_s=10.0,
        peak_vram_bytes=1234,
        group_name=name,
    )
    actual = {key: field_value(summary, key) for key in expected}
    passed = True
    mismatches = {}
    for key, exp in expected.items():
        act = actual[key]
        if isinstance(exp, float):
            ok = abs(float(act) - exp) <= 1e-9
        else:
            ok = act == exp
        if not ok:
            passed = False
            mismatches[key] = {"expected": exp, "actual": act}
    return {
        "name": name,
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tests = []

    tests.append(
        run_case(
            "overlapping_observation_windows",
            [
                record(name="obs_overlap_a", targets=[0, 1, 2, 3]),
                record(name="obs_overlap_b", requested_anchor_t=4, remapped_anchor_t=4, targets=[4, 5, 6, 7]),
            ],
            expected={
                "sample_draws": 2,
                "U_seen": 8,
                "E_action_unweighted": 8,
                "E_action_weighted": 8.0,
                "mean_replay": 1.0,
                "N_eff": 8.0,
            },
        )
    )

    tests.append(
        run_case(
            "overlapping_action_chunks",
            [
                record(name="chunk_a", targets=[0, 1, 2, 3]),
                record(name="chunk_b", requested_anchor_t=2, remapped_anchor_t=2, targets=[2, 3, 4, 5]),
            ],
            expected={
                "sample_draws": 2,
                "U_seen": 6,
                "E_action_unweighted": 8,
                "E_action_weighted": 8.0,
                "mean_replay": 8.0 / 6.0,
                "N_eff": 64.0 / 12.0,
            },
        )
    )

    tests.append(
        run_case(
            "start_end_padding",
            [
                record(name="padding", targets=[-1, -1, 0, 1, 2, -1]),
            ],
            expected={
                "sample_draws": 1,
                "U_seen": 3,
                "E_action_unweighted": 3,
                "E_action_weighted": 3.0,
                "mean_replay": 1.0,
                "padding_target_slots_ignored_for_U_seen_E_A": 3,
            },
        )
    )

    tests.append(
        run_case(
            "invalid_mask_and_nonunit_loss_weight",
            [
                record(
                    name="mask_weight",
                    targets=[0, 1, 2, 3],
                    valid=[True, False, True, True],
                    weights=[1.0, 0.5, 2.0, 0.0],
                ),
            ],
            expected={
                "sample_draws": 1,
                "U_seen": 2,
                "E_action_unweighted": 2,
                "E_action_weighted": 3.0,
                "mean_replay": 1.5,
                "invalid_or_zero_weight_real_target_slots": 2,
            },
        )
    )

    tests.append(
        run_case(
            "same_target_different_chunk_offset",
            [
                record(name="offset_a", targets=[5]),
                record(name="offset_b", requested_anchor_t=3, remapped_anchor_t=3, targets=[5]),
            ],
            expected={
                "sample_draws": 2,
                "U_seen": 1,
                "E_action_unweighted": 2,
                "E_action_weighted": 2.0,
                "mean_replay": 2.0,
                "N_eff": 1.0,
            },
        )
    )

    tests.append(
        run_case(
            "many_to_one_remapping",
            [
                record(name="remap_a", requested_anchor_t=10, remapped_anchor_t=8, targets=[8, 9]),
                record(name="remap_b", requested_anchor_t=11, remapped_anchor_t=8, targets=[8, 9]),
                record(name="remap_c", requested_anchor_t=12, remapped_anchor_t=8, targets=[8, 9]),
            ],
            expected={
                "sample_draws": 3,
                "U_seen": 2,
                "E_action_unweighted": 6,
                "E_action_weighted": 6.0,
                "mean_replay": 3.0,
                "N_eff": 2.0,
            },
        )
    )

    tests.append(
        run_case(
            "replacement_sampling",
            [
                record(name="replacement_a", targets=[0, 1]),
                record(name="replacement_b", targets=[0, 1]),
            ],
            expected={
                "sample_draws": 2,
                "U_seen": 2,
                "E_action_unweighted": 4,
                "E_action_weighted": 4.0,
                "mean_replay": 2.0,
                "N_eff": 2.0,
            },
        )
    )

    tests.append(
        run_case(
            "ddp_padding_duplicate",
            [
                record(name="rank0", rank=0, targets=[0]),
                record(name="rank1_duplicate", rank=1, targets=[0]),
            ],
            expected={
                "sample_draws": 2,
                "U_seen": 1,
                "E_action_unweighted": 2,
                "E_action_weighted": 2.0,
                "mean_replay": 2.0,
                "N_eff": 1.0,
            },
        )
    )

    tests.append(
        run_case(
            "gradient_accumulation",
            [
                record(name="grad_micro0", optimizer_step=0, targets=[0, 1]),
                record(name="grad_micro1", optimizer_step=0, targets=[2, 3]),
            ],
            expected={
                "sample_draws": 2,
                "optimizer_steps": 1,
                "U_seen": 4,
                "E_action_unweighted": 4,
                "E_action_weighted": 4.0,
            },
        )
    )

    tests.append(
        run_case(
            "interrupted_resumed_run",
            [
                record(name="before_interrupt_uncommitted", optimizer_step=0, targets=[0, 1], committed=False),
                record(name="after_resume_committed", optimizer_step=0, targets=[0, 1], committed=True),
            ],
            expected={
                "sample_draws": 1,
                "optimizer_steps": 1,
                "U_seen": 2,
                "E_action_unweighted": 2,
                "E_action_weighted": 2.0,
            },
        )
    )

    tests.append(
        run_case(
            "phase_sampler_switch",
            [
                record(name="warmup_full", source_view="warmup_full", targets=[0, 1]),
                record(name="pruned", source_view="pruned", targets=[1, 2]),
            ],
            expected={
                "sample_draws": 2,
                "U_seen": 3,
                "E_action_unweighted": 4,
                "E_action_weighted": 4.0,
                "mean_replay": 4.0 / 3.0,
            },
        )
    )

    single_records = [
        record(name="single0", rank=0, targets=[0, 1]),
        record(name="single1", rank=0, targets=[2, 3]),
    ]
    multi_records = [
        record(name="multi0", rank=0, targets=[0, 1]),
        record(name="multi1", rank=1, targets=[2, 3]),
    ]
    single_summary = compute_resource_summary(
        single_records,
        retained_ids=None,
        eligible_ids=None,
        topology={"global_batch_size": 2, "world_size": 1, "gradient_accumulation": 1},
        training_wall_clock_s=10.0,
        peak_vram_bytes=1234,
        group_name="single",
    )
    multi_summary = compute_resource_summary(
        multi_records,
        retained_ids=None,
        eligible_ids=None,
        topology={"global_batch_size": 2, "world_size": 2, "gradient_accumulation": 1},
        training_wall_clock_s=10.0,
        peak_vram_bytes=1234,
        group_name="multi",
    )
    consistency_actual = {
        key: field_value(single_summary, key) == field_value(multi_summary, key)
        for key in ("U_seen", "E_action_unweighted", "E_action_weighted", "N_eff")
    }
    tests.append(
        {
            "name": "single_card_multi_rank_aggregation_consistency",
            "passed": all(consistency_actual.values()),
            "expected": {"all_checked_fields_equal": True},
            "actual": consistency_actual,
            "mismatches": {},
            "summary": multi_summary,
        }
    )

    payload = {
        "format_version": 1,
        "test_suite": "first_lineage_synthetic_tests",
        "test_count": len(tests),
        "passed_count": sum(1 for test in tests if test["passed"]),
        "failed": [test["name"] for test in tests if not test["passed"]],
        "tests": tests,
        "signature_sha256": stable_hash(
            [
                {
                    "name": test["name"],
                    "expected": test["expected"],
                    "actual": test["actual"],
                }
                for test in tests
            ]
        ),
    }
    (output_dir / "synthetic_tests.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    md = [
        "# Synthetic Lineage Test Report",
        "",
        f"- test_count: `{payload['test_count']}`",
        f"- passed_count: `{payload['passed_count']}`",
        f"- failed: `{payload['failed']}`",
        f"- signature_sha256: `{payload['signature_sha256']}`",
        "",
        "| test | status | expected | actual |",
        "| --- | --- | --- | --- |",
    ]
    for test in tests:
        status = "PASS" if test["passed"] else "FAIL"
        md.append(
            "| `{}` | {} | `{}` | `{}` |".format(
                test["name"],
                status,
                json.dumps(test["expected"], sort_keys=True),
                json.dumps(test["actual"], sort_keys=True),
            )
        )
    (output_dir / "synthetic_tests.md").write_text("\n".join(md) + "\n")

    examples = []
    for test in tests:
        summary = test.get("summary", {})
        examples.append(
            {
                "example_source": "synthetic_test",
                "example_kind": test["name"],
                "passed": test["passed"],
                "expected": test["expected"],
                "actual": test["actual"],
                "summary_signature": stable_hash(summary),
            }
        )
    with (output_dir / "human_readable_examples_synthetic.jsonl").open("w") as file:
        for example in examples:
            file.write(json.dumps(example, sort_keys=True) + "\n")

    if payload["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
