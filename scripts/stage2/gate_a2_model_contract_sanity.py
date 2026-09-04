#!/usr/bin/env python3
"""Gate A2 model/config/input-contract sanity for LingBot-VA LIBERO-LONG.

This script intentionally does not load the full transformer weights and does
not run optimization. It validates that:
  1. patched LingBot model modules import in this runtime;
  2. model checkpoint/config files are present and parseable;
  3. the already-audited dataloader tensors match the transformer/config
     contract used by LIBERO training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tensor_shape(report: dict[str, Any], section: str, key: str) -> list[int]:
    return list(report["dataloader"][section][key]["shape"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingbot-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--posttrain-model", required=True)
    parser.add_argument("--dataloader-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    lingbot_root = Path(args.lingbot_root).resolve()
    base_model = Path(args.base_model).resolve()
    posttrain_model = Path(args.posttrain_model).resolve()
    dataloader_json = Path(args.dataloader_json).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()

    result: dict[str, Any] = {
        "status": "unknown",
        "scope": "import_config_input_contract_only_no_weight_load_no_forward_no_training",
        "paths": {
            "lingbot_root": str(lingbot_root),
            "base_model": str(base_model),
            "posttrain_model": str(posttrain_model),
            "dataloader_json": str(dataloader_json),
        },
        "checks": {},
        "errors": [],
    }

    sys.path.insert(0, str(lingbot_root))
    sys.path.insert(0, str(lingbot_root / "wan_va"))

    try:
        from wan_va.modules import model as model_module  # type: ignore

        result["checks"]["model_import"] = {
            "status": "pass",
            "WanTransformer3DModel": hasattr(model_module, "WanTransformer3DModel"),
            "flash_attn_func": getattr(
                getattr(model_module, "flash_attn_func", None), "__name__", None
            ),
        }
    except Exception as exc:  # noqa: BLE001
        result["checks"]["model_import"] = {"status": "failed", "error": repr(exc)}
        result["errors"].append(f"model import failed: {exc!r}")

    dataloader_report = load_json(dataloader_json)
    result["checks"]["dataloader_report_status"] = dataloader_report.get("status")

    model_reports: dict[str, Any] = {}
    for label, model_root in [("base", base_model), ("released_posttrain", posttrain_model)]:
        transformer_dir = model_root / "transformer"
        config_path = transformer_dir / "config.json"
        shards = sorted(transformer_dir.glob("*.safetensors"))
        report: dict[str, Any] = {
            "root": str(model_root),
            "transformer_config": str(config_path),
            "transformer_config_exists": config_path.exists(),
            "safetensor_shards": [str(p) for p in shards],
            "safetensor_shard_count": len(shards),
            "safetensor_total_bytes": sum(p.stat().st_size for p in shards),
        }
        if config_path.exists():
            cfg = load_json(config_path)
            report["config_subset"] = {
                k: cfg.get(k)
                for k in [
                    "_class_name",
                    "in_channels",
                    "out_channels",
                    "patch_size",
                    "text_dim",
                    "attention_head_dim",
                    "num_attention_heads",
                    "num_layers",
                    "action_dim",
                ]
            }
            report["config_sha256"] = sha256_file(config_path)
        model_reports[label] = report
    result["checks"]["models"] = model_reports

    sample_latents = tensor_shape(dataloader_report, "sample", "latents")
    sample_text = tensor_shape(dataloader_report, "sample", "text_emb")
    sample_actions = tensor_shape(dataloader_report, "sample", "actions")
    sample_actions_mask = tensor_shape(dataloader_report, "sample", "actions_mask")

    base_cfg = model_reports["base"].get("config_subset", {})
    post_cfg = model_reports["released_posttrain"].get("config_subset", {})
    input_contract = {
        "latents_shape": sample_latents,
        "text_emb_shape": sample_text,
        "actions_shape": sample_actions,
        "actions_mask_shape": sample_actions_mask,
        "latents_channels_match_base_in_channels": sample_latents[0] == base_cfg.get("in_channels"),
        "text_dim_match_base": sample_text[-1] == base_cfg.get("text_dim"),
        "action_channels_match_libero_action_dim": sample_actions[0] == 30,
        "action_channels_match_base_config_action_dim": sample_actions[0] == base_cfg.get("action_dim"),
        "actions_mask_shape_match_actions": sample_actions_mask == sample_actions,
        "base_posttrain_transformer_config_match": base_cfg == post_cfg,
        "spatial_dims_divisible_by_patch": None,
    }
    patch_size = base_cfg.get("patch_size")
    if isinstance(patch_size, list) and len(patch_size) == 3:
        input_contract["spatial_dims_divisible_by_patch"] = (
            sample_latents[-2] % patch_size[1] == 0
            and sample_latents[-1] % patch_size[2] == 0
        )
    result["checks"]["input_contract"] = input_contract

    hard_checks = [
        result["checks"].get("model_import", {}).get("status") == "pass",
        dataloader_report.get("status") == "pass",
        model_reports["base"]["transformer_config_exists"],
        model_reports["released_posttrain"]["transformer_config_exists"],
        model_reports["base"]["safetensor_shard_count"] > 0,
        model_reports["released_posttrain"]["safetensor_shard_count"] > 0,
        input_contract["latents_channels_match_base_in_channels"],
        input_contract["text_dim_match_base"],
        input_contract["action_channels_match_base_config_action_dim"],
        input_contract["actions_mask_shape_match_actions"],
        input_contract["base_posttrain_transformer_config_match"],
        input_contract["spatial_dims_divisible_by_patch"],
    ]
    result["status"] = "pass" if all(hard_checks) else "failed"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Gate A2 Model Contract Sanity",
        "",
        f"Status: {result['status']}",
        "",
        "Scope: import/config/input-contract only. No full weight load, no forward, no backward, no training.",
        "",
        "## Import",
        "",
        f"- model_import: `{result['checks'].get('model_import', {}).get('status')}`",
        f"- flash_attn_func: `{result['checks'].get('model_import', {}).get('flash_attn_func')}`",
        "",
        "## Checkpoint/config",
        "",
    ]
    for label, report in model_reports.items():
        lines.extend(
            [
                f"### {label}",
                "",
                f"- root: `{report['root']}`",
                f"- transformer_config_exists: `{report['transformer_config_exists']}`",
                f"- safetensor_shard_count: `{report['safetensor_shard_count']}`",
                f"- safetensor_total_bytes: `{report['safetensor_total_bytes']}`",
                f"- config_subset: `{report.get('config_subset')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Input contract",
            "",
            f"- latents_shape: `{sample_latents}`",
            f"- text_emb_shape: `{sample_text}`",
            f"- actions_shape: `{sample_actions}`",
            f"- actions_mask_shape: `{sample_actions_mask}`",
        ]
    )
    for key, value in input_contract.items():
        if key.endswith("_shape"):
            continue
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Pending",
            "",
            "- Full transformer weight load / forward is intentionally not performed in this check because it may require large GPU/CPU memory. It should be run only after this contract sanity is accepted.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "scope": result["scope"]}, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
