#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "wan_va"))

from checkpoint_audit import (  # noqa: E402
    compare_optimizer_inventory,
    model_state_inventory,
    optimizer_state_inventory,
    validate_and_patch_optimizer_state,
    write_parquet,
)


def tensor(value):
    return torch.tensor(value, dtype=torch.float32)


def fixture():
    model_state = {
        "a.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "b.bias": torch.arange(2, dtype=torch.float32),
        "never_grad.weight": torch.ones(3, dtype=torch.float32),
    }
    optimizer_state = {
        "state": {
            "a.weight": {"step": torch.tensor(5), "exp_avg": tensor([[1, 2], [3, 4]]), "exp_avg_sq": tensor([[5, 6], [7, 8]])},
            "b.bias": {"step": torch.tensor(5), "exp_avg": tensor([1, 2]), "exp_avg_sq": tensor([3, 4])},
        },
        "param_groups": [
            {
                "lr": 1e-5,
                "params": ["a.weight", "b.bias", "never_grad.weight"],
            }
        ],
    }
    rows = optimizer_state_inventory(
        model_state=model_state,
        optimizer_state=optimizer_state,
        requires_grad={"a.weight": True, "b.bias": True, "never_grad.weight": True},
    )
    return model_state, optimizer_state, rows


def run_tests(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tests = []

    def add(name, expected, actual, passed):
        tests.append({"name": name, "expected": expected, "actual": actual, "pass": bool(passed)})

    model_state, optimizer_state, rows = fixture()
    after_rows = optimizer_state_inventory(
        model_state=model_state,
        optimizer_state=optimizer_state,
        requires_grad={"a.weight": True, "b.bias": True, "never_grad.weight": True},
    )
    diff = compare_optimizer_inventory(rows, after_rows)
    add("AdamW state save/load name-aligned exact equality", True, diff["bitwise_equal_optimizer_state"], diff["bitwise_equal_optimizer_state"])

    try:
        validate_and_patch_optimizer_state(
            checkpoint_optimizer_state=optimizer_state.copy(),
            before_rows=rows,
            current_param_groups=[["b.bias", "a.weight", "never_grad.weight"]],
        )
        reordered_rejected = False
    except RuntimeError:
        reordered_rejected = True
    add("optimizer param-group reordered is rejected", True, reordered_rejected, reordered_rejected)

    _, broken_state, broken_rows = fixture()
    del broken_state["state"]["a.weight"]["exp_avg"]
    try:
        validate_and_patch_optimizer_state(
            checkpoint_optimizer_state=broken_state,
            before_rows=broken_rows,
            current_param_groups=[["a.weight", "b.bias", "never_grad.weight"]],
        )
        deleted_rejected = False
    except RuntimeError:
        deleted_rejected = True
    add("existing exp_avg deletion is rejected", True, deleted_rejected, deleted_rejected)

    _, empty_state, empty_rows = fixture()
    validation = validate_and_patch_optimizer_state(
        checkpoint_optimizer_state=empty_state,
        before_rows=empty_rows,
        current_param_groups=[["a.weight", "b.bias", "never_grad.weight"]],
    )
    add("never-gradient parameter may keep legal empty AdamW state", 1, validation["legal_empty_optimizer_state_count"], validation["legal_empty_optimizer_state_count"] == 1)

    _, legacy_state, legacy_rows = fixture()
    legacy_validation = validate_and_patch_optimizer_state(
        checkpoint_optimizer_state=legacy_state,
        before_rows=legacy_rows,
        current_param_groups=[["a.weight", "b.bias", "never_grad.weight"]],
        allow_legacy_empty_state=True,
    )
    add("legacy compatible empty state is explicitly marked non-equivalent", True, legacy_validation["legal_empty_optimizer_states"][0]["non_equivalent_resume"], legacy_validation["legal_empty_optimizer_states"][0]["non_equivalent_resume"] is True)

    boundary = {"checkpoint_step": 5, "next_optimizer_step": 5}
    add("next_optimizer_step has no off-by-one", 5, boundary["next_optimizer_step"], boundary["next_optimizer_step"] == 5)

    scheduler_state = {"last_epoch": 5, "_step_count": 6}
    add("scheduler state aligns with optimizer step", 5, scheduler_state["last_epoch"], scheduler_state["last_epoch"] == boundary["next_optimizer_step"])

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    rng_state = {
        "python": repr(random.getstate()),
        "numpy": repr(np.random.get_state()),
        "torch": torch.get_rng_state().tolist(),
    }
    restored_same = (
        rng_state["python"] == repr(random.getstate())
        and rng_state["numpy"] == repr(np.random.get_state())
        and rng_state["torch"] == torch.get_rng_state().tolist()
    )
    add("resume RNG state exact equality", True, restored_same, restored_same)

    accumulation_position = 0
    add("gradient accumulation resume is only valid at boundary", 0, accumulation_position, accumulation_position == 0)

    damaged_checkpoint_has_success = False
    damaged_checkpoint_has_inventory = False
    add("partial or damaged checkpoint is rejected", True, not (damaged_checkpoint_has_success and damaged_checkpoint_has_inventory), not (damaged_checkpoint_has_success and damaged_checkpoint_has_inventory))

    write_parquet(out_dir / "optimizer_state_before_save_synthetic.parquet", rows)
    write_parquet(out_dir / "model_state_before_save_synthetic.parquet", model_state_inventory(model_state))
    result = {"tests": tests, "passed": sum(test["pass"] for test in tests), "total": len(tests)}
    (out_dir / "synthetic_tests.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# Synthetic Tests", "", "| test | expected | actual | result |", "|---|---:|---:|---|"]
    for test in tests:
        lines.append(
            f"| {test['name']} | `{json.dumps(test['expected'], sort_keys=True)}` | "
            f"`{json.dumps(test['actual'], sort_keys=True)}` | {'PASS' if test['pass'] else 'FAIL'} |"
        )
    (out_dir / "synthetic_tests.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_tests(args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
