#!/usr/bin/env python3
"""Validate that every selected RoboTwin task has a complete result file."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _integer_count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{field} is not a non-negative integer")
    return int(number)


def inspect_result(path: Path, expected_total: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "complete": False,
        "path": str(path),
    }
    if not path.is_file():
        result["reason"] = "missing res.json"
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        result["reason"] = f"unreadable res.json: {exc}"
        return result

    if not isinstance(payload, dict):
        result["reason"] = "res.json is not an object"
        return result

    try:
        total = _integer_count(payload.get("total_num"), "total_num")
        success = _integer_count(payload.get("succ_num"), "succ_num")
    except ValueError as exc:
        result["reason"] = str(exc)
        return result

    result["observed_total"] = total
    result["observed_success"] = success
    if total != expected_total:
        result["reason"] = (
            f"total_num={total}, expected exactly {expected_total}"
        )
        return result
    if success > total:
        result["reason"] = f"succ_num={success} exceeds total_num={total}"
        return result

    rate = payload.get("succ_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        result["reason"] = "succ_rate is not numeric"
        return result
    rate = float(rate)
    expected_rate = success / total
    if not math.isfinite(rate) or not math.isclose(
        rate, expected_rate, rel_tol=1e-9, abs_tol=1e-9
    ):
        result["reason"] = (
            f"succ_rate={rate!r} is inconsistent with {success}/{total}"
        )
        return result

    result.update(
        complete=True,
        success_rate=rate,
        reason="complete",
    )
    return result


def build_report(
    save_root: Path,
    test_num: int,
    seed: int,
    tasks: list[str],
) -> dict[str, Any]:
    st_seed = 10000 * (1 + seed)
    metrics_root = save_root / f"stseed-{st_seed}" / "metrics"
    task_results = {
        task: inspect_result(metrics_root / task / "res.json", test_num)
        for task in tasks
    }
    complete_count = sum(
        bool(result["complete"]) for result in task_results.values()
    )
    return {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "save_root": str(save_root),
        "metrics_root": str(metrics_root),
        "seed_index": seed,
        "start_seed": st_seed,
        "expected_test_num": test_num,
        "expected_task_count": len(tasks),
        "complete_task_count": complete_count,
        "incomplete_task_count": len(tasks) - complete_count,
        "complete": complete_count == len(tasks),
        "tasks": task_results,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("save_root", type=Path)
    parser.add_argument("--test-num", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("tasks", nargs="+")
    args = parser.parse_args()

    if args.test_num <= 0:
        parser.error("--test-num must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("tasks must be unique")

    report = build_report(
        args.save_root.expanduser().resolve(),
        args.test_num,
        args.seed,
        args.tasks,
    )
    if args.report is not None:
        write_report(args.report, report)

    incomplete = [
        task for task, result in report["tasks"].items() if not result["complete"]
    ]
    summary = (
        f"RoboTwin completeness: {report['complete_task_count']}/"
        f"{report['expected_task_count']} tasks complete; "
        f"{len(incomplete)} incomplete"
    )
    stream = sys.stderr if args.missing_only else sys.stdout
    print(summary, file=stream)

    if args.verbose:
        for task in incomplete:
            print(
                f"  {task}: {report['tasks'][task]['reason']}",
                file=stream,
            )
    if args.missing_only:
        for task in incomplete:
            print(task)

    return 0 if not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
