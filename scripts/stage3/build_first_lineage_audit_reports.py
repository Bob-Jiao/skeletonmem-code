#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROGRESS_RE = re.compile(
    r"latent_loss=(?P<latent>[0-9.]+), "
    r"action_loss=(?P<action>[0-9.]+), "
    r"step=(?P<step>[0-9]+), "
    r"grad_norm=(?P<grad>[0-9.]+), "
    r"packs=(?P<packs>[0-9]+), "
    r"step_s=(?P<step_s>[0-9.]+), "
    r"lr=(?P<lr>[0-9.e+-]+)"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _read_json(path: Path) -> Any:
    with path.open() as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_progress(path: Path) -> dict[int, dict[str, Any]]:
    text = path.read_text(errors="replace")
    entries: dict[int, dict[str, Any]] = {}
    for match in PROGRESS_RE.finditer(text):
        step = int(match.group("step"))
        entries[step] = {
            "step": step,
            "latent_loss": float(match.group("latent")),
            "action_loss": float(match.group("action")),
            "grad_norm": float(match.group("grad")),
            "packs": int(match.group("packs")),
            "step_s": float(match.group("step_s")),
            "lr": float(match.group("lr")),
        }
    return entries


def _compare_progress(off_log: Path, on_log: Path) -> dict[str, Any]:
    off = _parse_progress(off_log)
    on = _parse_progress(on_log)
    common_steps = sorted(set(off) & set(on))

    def max_diff(key: str) -> float | None:
        if not common_steps:
            return None
        return max(abs(float(off[step][key]) - float(on[step][key])) for step in common_steps)

    pack_mismatches = [
        {
            "step": step,
            "off_packs": off[step]["packs"],
            "on_packs": on[step]["packs"],
        }
        for step in common_steps
        if off[step]["packs"] != on[step]["packs"]
    ]
    lr_mismatches = [
        {
            "step": step,
            "off_lr": off[step]["lr"],
            "on_lr": on[step]["lr"],
        }
        for step in common_steps
        if off[step]["lr"] != on[step]["lr"]
    ]
    return {
        "off_logged_steps": len(off),
        "on_logged_steps": len(on),
        "common_steps": len(common_steps),
        "missing_in_on": sorted(set(off) - set(on)),
        "missing_in_off": sorted(set(on) - set(off)),
        "max_abs_latent_loss_diff_logged_precision": max_diff("latent_loss"),
        "max_abs_action_loss_diff_logged_precision": max_diff("action_loss"),
        "max_abs_grad_norm_diff_logged_precision": max_diff("grad_norm"),
        "pack_mismatch_count": len(pack_mismatches),
        "pack_mismatches_head": pack_mismatches[:10],
        "lr_mismatch_count": len(lr_mismatches),
        "lr_mismatches_head": lr_mismatches[:10],
        "off_mean_step_s_excluding_step0": (
            sum(off[step]["step_s"] for step in off if step > 0) / max(len([s for s in off if s > 0]), 1)
        ),
        "on_mean_step_s_excluding_step0": (
            sum(on[step]["step_s"] for step in on if step > 0) / max(len([s for s in on if s > 0]), 1)
        ),
    }


def _checkpoint_diff(off_ckpt: Path, on_ckpt: Path) -> dict[str, Any]:
    off_sha = _sha256_file(off_ckpt)
    on_sha = _sha256_file(on_ckpt)
    out: dict[str, Any] = {
        "off_checkpoint": str(off_ckpt),
        "on_checkpoint": str(on_ckpt),
        "off_sha256": off_sha,
        "on_sha256": on_sha,
        "bitwise_identical": off_sha == on_sha,
    }
    if off_sha == on_sha:
        out["max_abs_parameter_diff"] = 0.0
        out["max_abs_parameter_diff_tensor"] = None
        return out

    try:
        from safetensors import safe_open
    except Exception as error:
        out["max_abs_parameter_diff"] = None
        out["max_abs_parameter_diff_error"] = f"safetensors unavailable: {error}"
        return out

    max_diff = -math.inf
    max_name = None
    compared = 0
    with safe_open(off_ckpt, framework="pt", device="cpu") as off_file, safe_open(
        on_ckpt, framework="pt", device="cpu"
    ) as on_file:
        off_keys = set(off_file.keys())
        on_keys = set(on_file.keys())
        if off_keys != on_keys:
            out["key_mismatch"] = {
                "only_off": sorted(off_keys - on_keys)[:50],
                "only_on": sorted(on_keys - off_keys)[:50],
            }
        for key in sorted(off_keys & on_keys):
            lhs = off_file.get_tensor(key)
            rhs = on_file.get_tensor(key)
            diff = float((lhs.float() - rhs.float()).abs().max().item())
            compared += 1
            if diff > max_diff:
                max_diff = diff
                max_name = key
    out["max_abs_parameter_diff"] = max_diff
    out["max_abs_parameter_diff_tensor"] = max_name
    out["parameter_tensor_count_compared"] = compared
    return out


def _load_repo_modules(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "wan_va"))


def _make_replay_signatures(
    *,
    repo_root: Path,
    bundle_dir: Path,
    config_name: str,
    steps: int,
    seed: int,
    packing_seed: int,
    global_episodes_per_update: int,
    max_self_tokens: int,
    max_episodes_per_pack: int,
    world_size: int,
) -> dict[str, Any]:
    _load_repo_modules(repo_root)
    from dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
    from dataset.packing import TokenBudgetPackingBatchSampler, packed_episode_collate
    from lineage_audit import stable_hash
    from wan_va.configs import VA_CONFIGS

    result: dict[str, Any] = {
        "source": "sampler/datapath replay with LineageTracer disabled",
        "rank_results": {},
    }
    for rank in range(world_size):
        config = copy.deepcopy(VA_CONFIGS[config_name])
        config.rank = rank
        config.local_rank = rank
        config.world_size = world_size
        config.global_episodes_per_update = global_episodes_per_update
        config.max_self_tokens = max_self_tokens
        config.max_episodes_per_pack = max_episodes_per_pack
        config.num_steps = steps
        config.seed = seed
        config.packing_seed = packing_seed
        config.load_worker = 0
        config.defer_cfg_dropout = True
        config.defer_text_embedding = True

        dataset = MultiLatentLeRobotDataset(config=config, num_init_worker=0)
        text_cache_path = Path(
            getattr(
                config,
                "text_embeddings_path",
                Path(config.dataset_path) / "text_embeddings.pt",
            )
        )
        text_cache = torch.load(
            text_cache_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        dataset.set_text_lookup(sorted(text_cache))
        del text_cache

        packing_manifest = dataset.get_packing_manifest()
        sampler = TokenBudgetPackingBatchSampler(
            packing_manifest,
            global_episodes_per_update=global_episodes_per_update,
            max_self_tokens=max_self_tokens,
            max_episodes_per_pack=max_episodes_per_pack,
            num_replicas=world_size,
            rank=rank,
            seed=packing_seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=packed_episode_collate,
            num_workers=0,
            pin_memory=False,
        )
        rows: list[dict[str, Any]] = []
        microbatch_id = 0
        completed_updates = set()
        for batch in loader:
            update_id = int(batch["update_id"].item())
            micro_idx = int(batch["micro_idx"].item())
            micro_count = int(batch["micro_count"].item())
            if update_id >= steps:
                break
            frame_offsets = [int(value) for value in batch["frame_offsets"].tolist()]
            action_tokens_per_frame = int(batch["actions_mask"].shape[3])
            action_mask_any = (
                batch["actions_mask"][0, :, :, :, 0]
                .bool()
                .any(dim=0)
                .flatten()
                .tolist()
            )
            signature_items = []
            for sample_slot, metadata in enumerate(batch["lineage_records"]):
                start = frame_offsets[sample_slot]
                end = frame_offsets[sample_slot + 1]
                flat_start = start * action_tokens_per_frame
                flat_end = end * action_tokens_per_frame
                target_timestamps = [
                    int(value)
                    for value in metadata["action_target_timestamps"][: flat_end - flat_start]
                ]
                valid_mask = [bool(value) for value in action_mask_any[flat_start:flat_end]]
                signature_items.append(
                    {
                        "sample_slot": int(sample_slot),
                        "dataset_index": int(metadata.get("dataset_index", -1)),
                        "sample_uid": int(metadata.get("sample_uid", -1)),
                        "dataset": str(metadata["dataset"]),
                        "episode": int(metadata["episode"]),
                        "requested_anchor_t": int(metadata["requested_anchor_t"]),
                        "remapped_anchor_t": int(metadata["remapped_anchor_t"]),
                        "observation_timestamps": [
                            int(value) for value in metadata["observation_timestamps"]
                        ],
                        "action_target_timestamps": target_timestamps,
                        "valid_mask": valid_mask,
                    }
                )
            rows.append(
                {
                    "run_id": "sampler_replay_tracer_off",
                    "optimizer_step": update_id,
                    "rank": rank,
                    "microbatch_id": microbatch_id,
                    "gradient_accumulation_index": micro_idx,
                    "signature_sha256": stable_hash(signature_items),
                    "items": signature_items,
                }
            )
            microbatch_id += 1
            if micro_idx + 1 == micro_count:
                completed_updates.add(update_id)
            if len(completed_updates) >= steps:
                break
        path = bundle_dir / f"batch_signatures_replay_rank{rank}.jsonl"
        _write_jsonl(path, rows)
        result["rank_results"][str(rank)] = {
            "path": str(path),
            "signature_rows": len(rows),
            "completed_updates": len(completed_updates),
            "sampler_manifest_hash": sampler.manifest_hash,
        }
    return result


def _compare_signatures(bundle_dir: Path, world_size: int) -> dict[str, Any]:
    out = {"ranks": {}, "all_exact": True}
    for rank in range(world_size):
        replay_path = bundle_dir / f"batch_signatures_replay_rank{rank}.jsonl"
        on_path = bundle_dir / f"batch_signatures_rank{rank}.jsonl"
        replay = _read_jsonl(replay_path)
        on = _read_jsonl(on_path)
        mismatch_examples = []
        for index, (lhs, rhs) in enumerate(zip(replay, on)):
            if (
                lhs["optimizer_step"] != rhs["optimizer_step"]
                or lhs["rank"] != rhs["rank"]
                or lhs["microbatch_id"] != rhs["microbatch_id"]
                or lhs["gradient_accumulation_index"] != rhs["gradient_accumulation_index"]
                or lhs["signature_sha256"] != rhs["signature_sha256"]
                or lhs["items"] != rhs["items"]
            ):
                mismatch_examples.append(
                    {
                        "row": index,
                        "replay_signature": lhs.get("signature_sha256"),
                        "on_signature": rhs.get("signature_sha256"),
                        "replay_step": lhs.get("optimizer_step"),
                        "on_step": rhs.get("optimizer_step"),
                    }
                )
                if len(mismatch_examples) >= 5:
                    break
        exact = len(replay) == len(on) and not mismatch_examples
        out["ranks"][str(rank)] = {
            "replay_rows": len(replay),
            "on_rows": len(on),
            "exact_match": exact,
            "mismatch_examples": mismatch_examples,
        }
        out["all_exact"] = out["all_exact"] and exact
    return out


def _combine_examples(bundle_dir: Path) -> dict[str, Any]:
    actual_path = bundle_dir / "human_readable_examples_actual.jsonl"
    synthetic_path = bundle_dir / "human_readable_examples_synthetic.jsonl"
    rows: list[dict[str, Any]] = []
    if actual_path.is_file():
        rows.extend(_read_jsonl(actual_path)[:30])
    if synthetic_path.is_file():
        rows.extend(_read_jsonl(synthetic_path)[: max(0, 50 - len(rows))])
    out_path = bundle_dir / "human_readable_examples.jsonl"
    _write_jsonl(out_path, rows[:50])
    return {
        "path": str(out_path),
        "total_examples": min(len(rows), 50),
        "actual_examples": min(len(_read_jsonl(actual_path)), 30) if actual_path.is_file() else 0,
        "synthetic_examples": min(
            len(_read_jsonl(synthetic_path)),
            max(0, 50 - (min(len(_read_jsonl(actual_path)), 30) if actual_path.is_file() else 0)),
        )
        if synthetic_path.is_file()
        else 0,
    }


def _augment_manifest(bundle_dir: Path, off_log: Path, on_log: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    manifest = _read_json(bundle_dir / "dry_run_manifest.json")
    support = _read_json(bundle_dir / "support_summary.json")
    manifest["bundle_dir"] = str(bundle_dir)
    manifest["train_off_log"] = str(off_log)
    manifest["train_on_log"] = str(on_log)
    manifest["training_pair"] = {
        "off_run": "train_off_full100",
        "on_run": "train_on_full100",
        "only_intended_difference": "LineageTracer enabled for ON",
    }
    manifest["full_retained_support_sha256"] = support.get("retained_support_sha256")
    manifest["subset_manifest_note"] = (
        "Full run uses subset_manifest_id='full'; no external subset manifest file "
        "exists. The analytical full retained support SHA256 is recorded as "
        "full_retained_support_sha256."
    )
    manifest["checkpoint_comparison"] = checkpoint
    out_path = bundle_dir / "dry_run_manifest_resolved.json"
    _write_json(out_path, manifest)
    return {"path": str(out_path)}


def _write_invariance_markdown(path: Path, report: dict[str, Any]) -> None:
    progress = report["progress_comparison"]
    checkpoint = report["checkpoint_comparison"]
    signatures = report["signature_comparison"]
    overhead = report["overhead"]
    lines = [
        "# Invariance Report",
        "",
        "## Runs",
        "",
        f"- tracer OFF log: `{report['off_log']}`",
        f"- tracer ON log: `{report['on_log']}`",
        f"- steps: `{report['steps']}`",
        f"- evidence note: OFF training has no lineage file by construction; batch signatures are compared using an OFF sampler/datapath replay and ON actual tracer signatures.",
        "",
        "## Batch Lineage Signature",
        "",
        f"- rank count: `{report['world_size']}`",
        f"- all ranks exact match: `{signatures['all_exact']}`",
    ]
    for rank, item in signatures["ranks"].items():
        lines.append(
            f"- rank {rank}: replay_rows=`{item['replay_rows']}`, on_rows=`{item['on_rows']}`, exact_match=`{item['exact_match']}`"
        )
    lines.extend(
        [
            "",
            "## Loss And Optimizer Trace",
            "",
            f"- logged common steps: `{progress['common_steps']}`",
            f"- max latent loss diff at logged precision: `{progress['max_abs_latent_loss_diff_logged_precision']}`",
            f"- max action loss diff at logged precision: `{progress['max_abs_action_loss_diff_logged_precision']}`",
            f"- max grad norm diff at logged precision: `{progress['max_abs_grad_norm_diff_logged_precision']}`",
            f"- pack mismatch count: `{progress['pack_mismatch_count']}`",
            f"- lr mismatch count: `{progress['lr_mismatch_count']}`",
            "",
            "## Parameters",
            "",
            f"- bitwise identical checkpoint: `{checkpoint['bitwise_identical']}`",
            f"- OFF sha256: `{checkpoint['off_sha256']}`",
            f"- ON sha256: `{checkpoint['on_sha256']}`",
            f"- max abs parameter diff: `{checkpoint.get('max_abs_parameter_diff')}`",
            f"- max diff tensor: `{checkpoint.get('max_abs_parameter_diff_tensor')}`",
            "",
            "## Dataloader Iteration",
            "",
            f"- OFF training lineage file: `none`",
            f"- ON lineage sample_draws: `{report['on_sample_draws']}`",
            f"- expected sample draws from optimizer steps x global batch: `{report['expected_sample_draws']}`",
            f"- extra dataloader iteration detected by lineage: `{report['extra_dataloader_iteration_detected']}`",
            "",
            "## Overhead",
            "",
            f"- OFF mean step_s excluding step 0: `{overhead['off_mean_step_s_excluding_step0']:.4f}`",
            f"- ON mean step_s excluding step 0: `{overhead['on_mean_step_s_excluding_step0']:.4f}`",
            f"- relative step-time overhead excluding step 0: `{overhead['relative_step_time_overhead']:.6f}`",
            f"- peak VRAM bytes from ON Resource Card: `{overhead['on_peak_vram_bytes']}`",
            f"- OFF peak VRAM: `unavailable from current logs`",
            "",
            "## Limitations",
            "",
            "- Per-step loss comparison uses rank0 tqdm values rounded in the existing logs.",
            "- OFF batch signatures are generated by deterministic sampler/datapath replay because a true tracer-OFF training process cannot emit lineage signatures.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_bundle_index(bundle_dir: Path, report: dict[str, Any], examples: dict[str, Any]) -> None:
    resource = _read_json(bundle_dir / "resource_card.json")
    whole = resource["whole_run"]
    lines = [
        "# First Lineage Audit Bundle",
        "",
        f"- bundle_dir: `{bundle_dir}`",
        f"- run_id: `{resource['run_id']}`",
        "",
        "## Files",
        "",
        "- `lineage_spec.md`",
        "- `lineage_rank0.parquet`",
        "- `lineage_rank1.parquet`",
        "- `human_readable_examples.jsonl`",
        "- `human_readable_examples_actual.jsonl`",
        "- `human_readable_examples_synthetic.jsonl`",
        "- `resource_card.json`",
        "- `resource_card.md`",
        "- `synthetic_tests.json`",
        "- `synthetic_tests.md`",
        "- `invariance_report.json`",
        "- `invariance_report.md`",
        "- `dry_run_manifest.json`",
        "- `dry_run_manifest_resolved.json`",
        "- `support_summary.json`",
        "- `batch_signatures_rank{rank}.jsonl`",
        "- `batch_signatures_replay_rank{rank}.jsonl`",
        "",
        "## Resource Card Snapshot",
        "",
        f"- sample_draws: `{whole['sample_draws']['value']}`",
        f"- U_ret: `{whole['U_ret']['value']}`",
        f"- U_eligible: `{whole['U_eligible']['value']}`",
        f"- U_seen: `{whole['U_seen']['value']}`",
        f"- E_action_unweighted: `{whole['E_action_unweighted']['value']}`",
        f"- E_action_weighted: `{whole['E_action_weighted']['value']}`",
        f"- mean_replay: `{whole['mean_replay']['value']}`",
        f"- N_eff: `{whole['N_eff']['value']}`",
        "",
        "## Invariance Snapshot",
        "",
        f"- signature all-rank exact match: `{report['signature_comparison']['all_exact']}`",
        f"- logged loss max diff: latent `{report['progress_comparison']['max_abs_latent_loss_diff_logged_precision']}`, action `{report['progress_comparison']['max_abs_action_loss_diff_logged_precision']}`",
        f"- checkpoint bitwise identical: `{report['checkpoint_comparison']['bitwise_identical']}`",
        f"- extra dataloader iteration detected: `{report['extra_dataloader_iteration_detected']}`",
        "",
        "## Example Count",
        "",
        f"- combined examples: `{examples['total_examples']}`",
        f"- actual examples: `{examples['actual_examples']}`",
        f"- synthetic examples: `{examples['synthetic_examples']}`",
    ]
    (bundle_dir / "bundle_index.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--off-log", required=True, type=Path)
    parser.add_argument("--on-log", required=True, type=Path)
    parser.add_argument("--off-ckpt", required=True, type=Path)
    parser.add_argument("--on-ckpt", required=True, type=Path)
    parser.add_argument("--config-name", default="libero_all_train")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--packing-seed", type=int, default=42)
    parser.add_argument("--global-episodes-per-update", type=int, default=32)
    parser.add_argument("--max-self-tokens", type=int, default=24576)
    parser.add_argument("--max-episodes-per-pack", type=int, default=10)
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()

    started = time.time()
    replay = _make_replay_signatures(
        repo_root=args.repo_root,
        bundle_dir=args.bundle_dir,
        config_name=args.config_name,
        steps=args.steps,
        seed=args.seed,
        packing_seed=args.packing_seed,
        global_episodes_per_update=args.global_episodes_per_update,
        max_self_tokens=args.max_self_tokens,
        max_episodes_per_pack=args.max_episodes_per_pack,
        world_size=args.world_size,
    )
    signature_comparison = _compare_signatures(args.bundle_dir, args.world_size)
    progress = _compare_progress(args.off_log, args.on_log)
    checkpoint = _checkpoint_diff(args.off_ckpt, args.on_ckpt)
    resource = _read_json(args.bundle_dir / "resource_card.json")
    on_sample_draws = int(resource["whole_run"]["sample_draws"]["value"])
    expected_sample_draws = args.steps * args.global_episodes_per_update
    extra_iteration = on_sample_draws != expected_sample_draws
    off_step = progress["off_mean_step_s_excluding_step0"]
    on_step = progress["on_mean_step_s_excluding_step0"]
    overhead = {
        "off_mean_step_s_excluding_step0": off_step,
        "on_mean_step_s_excluding_step0": on_step,
        "relative_step_time_overhead": (on_step - off_step) / off_step if off_step else None,
        "on_peak_vram_bytes": resource["whole_run"]["peak_VRAM"]["value"],
        "off_peak_vram_bytes": None,
    }
    report = {
        "format_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle_dir": str(args.bundle_dir),
        "repo_root": str(args.repo_root),
        "off_log": str(args.off_log),
        "on_log": str(args.on_log),
        "steps": args.steps,
        "world_size": args.world_size,
        "replay_generation": replay,
        "signature_comparison": signature_comparison,
        "progress_comparison": progress,
        "checkpoint_comparison": checkpoint,
        "on_sample_draws": on_sample_draws,
        "expected_sample_draws": expected_sample_draws,
        "extra_dataloader_iteration_detected": extra_iteration,
        "overhead": overhead,
        "build_wall_clock_s": time.time() - started,
    }
    _write_json(args.bundle_dir / "invariance_report.json", report)
    _write_invariance_markdown(args.bundle_dir / "invariance_report.md", report)
    examples = _combine_examples(args.bundle_dir)
    _augment_manifest(args.bundle_dir, args.off_log, args.on_log, checkpoint)
    _write_bundle_index(args.bundle_dir, report, examples)


if __name__ == "__main__":
    main()
