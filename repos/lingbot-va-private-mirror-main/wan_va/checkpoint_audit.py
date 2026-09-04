from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ADAMW_REQUIRED_STATE_KEYS = ("step", "exp_avg", "exp_avg_sq")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode()
    ).hexdigest()


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_checksum(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    if hasattr(value, "to_local"):
        value = value.to_local()
    tensor = value.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_shape(value: torch.Tensor | None) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "to_local"):
        value = value.to_local()
    return [int(dim) for dim in value.shape]


def tensor_dtype(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    if hasattr(value, "to_local"):
        value = value.to_local()
    return str(value.dtype)


def scalar_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return None
    return value


def normalize_parameter_name(name: str) -> str:
    prefixes = ("module.", "_fsdp_wrapped_module.")
    changed = True
    out = str(name)
    while changed:
        changed = False
        for prefix in prefixes:
            if out.startswith(prefix):
                out = out[len(prefix) :]
                changed = True
    out = out.replace("._fsdp_wrapped_module.", ".")
    return out


def requires_grad_by_name(model: torch.nn.Module) -> dict[str, bool]:
    mapping: dict[str, bool] = {}
    for name, parameter in model.named_parameters():
        mapping[normalize_parameter_name(name)] = bool(parameter.requires_grad)
    return mapping


def optimizer_param_groups(optimizer_state: dict[str, Any]) -> list[list[str]]:
    groups = []
    for group in optimizer_state.get("param_groups", []):
        groups.append([str(name) for name in group.get("params", [])])
    return groups


def model_state_inventory(model_state: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(model_state):
        tensor = model_state[name]
        rows.append(
            {
                "parameter_name": str(name),
                "parameter_shape": tensor_shape(tensor),
                "parameter_dtype": tensor_dtype(tensor),
                "parameter_checksum": tensor_checksum(tensor),
            }
        )
    return rows


def optimizer_state_inventory(
    *,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    requires_grad: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    requires_grad = requires_grad or {}
    state = optimizer_state.get("state", {})
    rows = []
    for group_index, group in enumerate(optimizer_state.get("param_groups", [])):
        for position, name in enumerate(group.get("params", [])):
            parameter_name = str(name)
            parameter = model_state.get(parameter_name)
            param_state = state.get(parameter_name)
            optimizer_state_present = bool(param_state)
            exp_avg = param_state.get("exp_avg") if param_state else None
            exp_avg_sq = param_state.get("exp_avg_sq") if param_state else None
            step = param_state.get("step") if param_state else None
            step_value = scalar_value(step)
            rows.append(
                {
                    "parameter_name": parameter_name,
                    "parameter_shape": tensor_shape(parameter),
                    "parameter_dtype": tensor_dtype(parameter),
                    "requires_grad": requires_grad.get(parameter_name),
                    "optimizer_group_index": int(group_index),
                    "position_in_group": int(position),
                    "optimizer_state_key_present": parameter_name in state,
                    "optimizer_state_present": optimizer_state_present,
                    "optimizer_state_keys": sorted(param_state.keys()) if param_state else [],
                    "optimizer_step": None if step_value is None else int(step_value),
                    "step_checksum": tensor_checksum(step) if isinstance(step, torch.Tensor) else None,
                    "exp_avg_shape": tensor_shape(exp_avg),
                    "exp_avg_dtype": tensor_dtype(exp_avg),
                    "exp_avg_checksum": tensor_checksum(exp_avg),
                    "exp_avg_sq_shape": tensor_shape(exp_avg_sq),
                    "exp_avg_sq_dtype": tensor_dtype(exp_avg_sq),
                    "exp_avg_sq_checksum": tensor_checksum(exp_avg_sq),
                    "parameter_checksum": tensor_checksum(parameter),
                    "has_received_gradient": bool(exp_avg is not None and exp_avg_sq is not None),
                    "number_of_applied_updates": 0 if step_value is None else int(step_value),
                }
            )
    return rows


def write_parquet(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def inventory_param_groups(rows: list[dict[str, Any]]) -> list[list[str]]:
    grouped: dict[int, list[tuple[int, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["optimizer_group_index"]), []).append(
            (int(row["position_in_group"]), str(row["parameter_name"]))
        )
    return [
        [name for _, name in sorted(grouped[index])]
        for index in sorted(grouped)
    ]


def compare_optimizer_inventory(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    before = {str(row["parameter_name"]): row for row in before_rows}
    after = {str(row["parameter_name"]): row for row in after_rows}
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    state_mismatches = []
    for name in sorted(set(before) & set(after)):
        left = before[name]
        right = after[name]
        fields = [
            "optimizer_state_present",
            "optimizer_state_keys",
            "optimizer_step",
            "exp_avg_shape",
            "exp_avg_dtype",
            "exp_avg_checksum",
            "exp_avg_sq_shape",
            "exp_avg_sq_dtype",
            "exp_avg_sq_checksum",
        ]
        diffs = {
            field: {"before": left.get(field), "after": right.get(field)}
            for field in fields
            if left.get(field) != right.get(field)
        }
        if diffs:
            state_mismatches.append({"parameter_name": name, "diffs": diffs})
    return {
        "missing_parameter_names": missing,
        "extra_parameter_names": extra,
        "state_mismatch_count": len(state_mismatches),
        "state_mismatch_examples": state_mismatches[:20],
        "bitwise_equal_optimizer_state": not missing and not extra and not state_mismatches,
    }


def validate_and_patch_optimizer_state(
    *,
    checkpoint_optimizer_state: dict[str, Any],
    before_rows: list[dict[str, Any]],
    current_param_groups: list[list[str]],
    allow_legacy_empty_state: bool = False,
    verify_tensor_checksums: bool = True,
) -> dict[str, Any]:
    expected_groups = inventory_param_groups(before_rows)
    if expected_groups != current_param_groups:
        raise RuntimeError(
            "Optimizer param-group membership/order mismatch: "
            + json.dumps(
                {
                    "expected_group_lengths": [len(group) for group in expected_groups],
                    "current_group_lengths": [len(group) for group in current_param_groups],
                    "expected_hash": stable_hash(expected_groups),
                    "current_hash": stable_hash(current_param_groups),
                },
                sort_keys=True,
            )
        )

    state = checkpoint_optimizer_state.setdefault("state", {})
    legal_empty = []
    errors = []
    for row in before_rows:
        name = str(row["parameter_name"])
        param_state = state.get(name)
        had_state = bool(row.get("optimizer_state_present"))
        if had_state:
            if not param_state:
                errors.append(
                    {
                        "parameter_name": name,
                        "error": "optimizer state existed before save but is missing at load",
                    }
                )
                continue
            missing_keys = [
                key for key in ADAMW_REQUIRED_STATE_KEYS if key not in param_state
            ]
            if missing_keys:
                errors.append(
                    {
                        "parameter_name": name,
                        "error": "required AdamW state keys missing",
                        "missing_keys": missing_keys,
                    }
                )
                continue
            for key, shape_field, dtype_field, checksum_field in [
                ("exp_avg", "exp_avg_shape", "exp_avg_dtype", "exp_avg_checksum"),
                ("exp_avg_sq", "exp_avg_sq_shape", "exp_avg_sq_dtype", "exp_avg_sq_checksum"),
            ]:
                tensor = param_state[key]
                if tensor_shape(tensor) != row.get(shape_field):
                    errors.append(
                        {
                            "parameter_name": name,
                            "error": f"{key} shape mismatch",
                            "expected": row.get(shape_field),
                            "actual": tensor_shape(tensor),
                        }
                    )
                if tensor_dtype(tensor) != row.get(dtype_field):
                    errors.append(
                        {
                            "parameter_name": name,
                            "error": f"{key} dtype mismatch",
                            "expected": row.get(dtype_field),
                            "actual": tensor_dtype(tensor),
                        }
                    )
                if verify_tensor_checksums and tensor_checksum(tensor) != row.get(checksum_field):
                    errors.append(
                        {
                            "parameter_name": name,
                            "error": f"{key} checksum mismatch",
                        }
                    )
        else:
            if not param_state:
                state[name] = {}
                legal_empty.append(
                    {
                        "parameter_name": name,
                        "missing_fields": list(ADAMW_REQUIRED_STATE_KEYS),
                        "initialization_strategy": "legal_empty_adamw_state",
                        "reason": "before-save name-aligned manifest shows optimizer_state_present=false",
                        "non_equivalent_resume": bool(allow_legacy_empty_state),
                    }
                )
            else:
                errors.append(
                    {
                        "parameter_name": name,
                        "error": "optimizer state appeared at load but was absent before save",
                    }
                )

    if errors:
        raise RuntimeError(
            "Optimizer checkpoint validation failed: "
            + json.dumps(errors[:20], sort_keys=True)
        )
    return {
        "param_group_hash": stable_hash(current_param_groups),
        "legal_empty_optimizer_states": legal_empty,
        "legal_empty_optimizer_state_count": len(legal_empty),
        "tensor_checksum_verification": bool(verify_tensor_checksums),
        "fail_closed": True,
    }


def local_optimizer_state_checksum(optimizer: torch.optim.Optimizer) -> str:
    items = []
    for group_index, group in enumerate(optimizer.param_groups):
        for position, parameter in enumerate(group.get("params", [])):
            state = optimizer.state.get(parameter, {})
            state_items = {}
            for key, value in sorted(state.items(), key=lambda item: str(item[0])):
                if isinstance(value, torch.Tensor):
                    state_items[str(key)] = {
                        "shape": tensor_shape(value),
                        "dtype": tensor_dtype(value),
                        "checksum": tensor_checksum(value),
                    }
                else:
                    state_items[str(key)] = value
            items.append(
                {
                    "group": group_index,
                    "position": position,
                    "state": state_items,
                }
            )
    return stable_hash(items)


def local_model_parameter_checksum(model: torch.nn.Module) -> str:
    items = []
    for name, parameter in model.named_parameters():
        items.append(
            {
                "name": normalize_parameter_name(name),
                "shape": tensor_shape(parameter),
                "dtype": tensor_dtype(parameter),
                "checksum": tensor_checksum(parameter),
            }
        )
    return stable_hash(items)


def tensor_is_zero(value: torch.Tensor) -> bool:
    if hasattr(value, "to_local"):
        value = value.to_local()
    tensor = value.detach().cpu()
    if tensor.numel() == 0:
        return True
    return bool(torch.count_nonzero(tensor).item() == 0)


def drop_legal_empty_local_optimizer_states(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    legal_empty_states: list[dict[str, Any]],
) -> dict[str, Any]:
    """Remove FSDP/optimizer placeholders for params that had no saved AdamW state."""
    if not legal_empty_states:
        return {"dropped_count": 0, "dropped_parameter_names": [], "already_absent": []}

    params_by_name = {
        normalize_parameter_name(name): parameter
        for name, parameter in model.named_parameters()
    }
    dropped = []
    already_absent = []
    errors = []
    for item in legal_empty_states:
        name = str(item["parameter_name"])
        parameter = params_by_name.get(name)
        if parameter is None:
            errors.append({"parameter_name": name, "error": "parameter not found in current model"})
            continue
        state = optimizer.state.get(parameter)
        if not state:
            already_absent.append(name)
            continue

        unsafe_keys = []
        for key, value in state.items():
            if str(key) == "step":
                continue
            if isinstance(value, torch.Tensor):
                if str(key) not in {"exp_avg", "exp_avg_sq"} or not tensor_is_zero(value):
                    unsafe_keys.append(str(key))
            elif value not in (None, 0, 0.0):
                unsafe_keys.append(str(key))
        if unsafe_keys:
            errors.append(
                {
                    "parameter_name": name,
                    "error": "legal-empty placeholder has non-zero or non-empty optimizer state",
                    "unsafe_keys": unsafe_keys,
                }
            )
            continue
        del optimizer.state[parameter]
        dropped.append(name)

    if errors:
        raise RuntimeError(
            "Refusing to drop non-empty optimizer state placeholders: "
            + json.dumps(errors[:20], sort_keys=True)
        )
    return {
        "dropped_count": len(dropped),
        "dropped_parameter_names": dropped,
        "already_absent": already_absent,
    }
