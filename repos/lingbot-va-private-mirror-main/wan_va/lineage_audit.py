from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


LINEAGE_FORMAT_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_action_id(record: dict[str, Any], target_t: int) -> str:
    return "\t".join(
        (
            str(record["dataset"]),
            str(record["task"]),
            str(int(record["episode"])),
            str(int(target_t)),
        )
    )


def _field(value: Any, source: str, unit: str | None = None, note: str | None = None):
    out = {"value": value, "source": source}
    if unit is not None:
        out["unit"] = unit
    if note is not None:
        out["note"] = note
    return out


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def _effective_support(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    numerator = float(arr.sum() ** 2)
    denominator = float(np.square(arr).sum())
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _iter_effective_targets(record: dict[str, Any]):
    timestamps = record.get("action_target_timestamps") or []
    valid_mask = record.get("valid_mask") or []
    loss_weights = record.get("loss_weights") or []
    limit = min(len(timestamps), len(valid_mask), len(loss_weights))
    for index in range(limit):
        target_t = int(timestamps[index])
        valid = bool(valid_mask[index])
        weight = float(loss_weights[index])
        if target_t >= 0 and valid and weight > 0.0:
            yield canonical_action_id(record, target_t), weight


def support_ids_from_dataset(dataset: Any) -> set[str]:
    ids: set[str] = set()
    for index in range(len(dataset)):
        metadata = dataset.get_lineage_metadata(index)
        for target_t in metadata["action_target_timestamps"]:
            if int(target_t) >= 0:
                ids.add(canonical_action_id(metadata, int(target_t)))
    return ids


def _support_hash(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(ids):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def compute_resource_summary(
    records: list[dict[str, Any]],
    *,
    retained_ids: set[str] | None,
    eligible_ids: set[str] | None,
    topology: dict[str, Any],
    training_wall_clock_s: float | None,
    peak_vram_bytes: int | None,
    group_name: str = "whole_run",
) -> dict[str, Any]:
    committed = [
        record for record in records if bool(record.get("optimizer_update_applied"))
    ]
    weighted_counts: defaultdict[str, float] = defaultdict(float)
    unweighted_counts: Counter[str] = Counter()
    padding_targets = 0
    invalid_or_zero_weight_targets = 0
    for record in committed:
        timestamps = record.get("action_target_timestamps") or []
        valid_mask = record.get("valid_mask") or []
        loss_weights = record.get("loss_weights") or []
        for target_t, valid, weight in zip(timestamps, valid_mask, loss_weights):
            target_i = int(target_t)
            valid_b = bool(valid)
            weight_f = float(weight)
            if target_i < 0:
                padding_targets += 1
                continue
            if not valid_b or weight_f <= 0.0:
                invalid_or_zero_weight_targets += 1
                continue
            action_id = canonical_action_id(record, target_i)
            weighted_counts[action_id] += weight_f
            unweighted_counts[action_id] += 1

    u_seen = len(weighted_counts)
    e_unweighted = int(sum(unweighted_counts.values()))
    e_weighted = float(sum(weighted_counts.values()))
    exposure_values = list(weighted_counts.values())
    n_eff = _effective_support(exposure_values)
    optimizer_steps = len({int(record["optimizer_step"]) for record in committed})
    sample_draws = len(committed)
    wall = float(training_wall_clock_s) if training_wall_clock_s is not None else None
    vps = None
    if wall is not None and wall > 0:
        vps = e_unweighted / wall

    def _zero_fraction(support: set[str] | None):
        if support is None or len(support) == 0:
            return None
        return (len(support) - len(set(weighted_counts) & support)) / len(support)

    u_ret = len(retained_ids) if retained_ids is not None else None
    u_eligible = len(eligible_ids) if eligible_ids is not None else None
    q = _quantiles(exposure_values)

    summary = {
        "format_version": LINEAGE_FORMAT_VERSION,
        "group_name": group_name,
        "sample_draws": _field(sample_draws, "exact", "sample occurrences"),
        "U_ret": _field(
            u_ret,
            "analytical" if retained_ids is not None else "unavailable",
            "canonical action targets",
        ),
        "U_eligible": _field(
            u_eligible,
            "analytical" if eligible_ids is not None else "unavailable",
            "canonical action targets",
        ),
        "U_seen": _field(u_seen, "exact", "canonical action targets"),
        "E_action_unweighted": _field(e_unweighted, "exact", "target exposures"),
        "E_action_weighted": _field(e_weighted, "exact", "loss-weighted target exposures"),
        "mean_replay": _field(
            (e_weighted / u_seen) if u_seen else 0.0,
            "exact",
            "weighted exposures per seen target",
            "R_bar = E_action_weighted / U_seen",
        ),
        "mean_replay_unweighted": _field(
            (e_unweighted / u_seen) if u_seen else 0.0,
            "exact",
            "unweighted exposures per seen target",
        ),
        "exposure_p10": _field(q["p10"], "exact", "weighted exposures"),
        "exposure_p50": _field(q["p50"], "exact", "weighted exposures"),
        "exposure_p90": _field(q["p90"], "exact", "weighted exposures"),
        "exposure_p99": _field(q["p99"], "exact", "weighted exposures"),
        "N_eff": _field(n_eff, "exact", "canonical action targets"),
        "N_eff/U_seen": _field((n_eff / u_seen) if u_seen else 0.0, "exact"),
        "N_eff/U_eligible": _field(
            (n_eff / u_eligible) if u_eligible else None,
            "exact" if u_eligible else "unavailable",
        ),
        "zero_exposure_fraction_eligible": _field(
            _zero_fraction(eligible_ids),
            "exact" if eligible_ids is not None else "unavailable",
        ),
        "zero_exposure_fraction_retained": _field(
            _zero_fraction(retained_ids),
            "exact" if retained_ids is not None else "unavailable",
        ),
        "optimizer_steps": _field(optimizer_steps, "exact", "optimizer updates"),
        "global_batch_size": _field(
            topology.get("global_batch_size"),
            "analytical",
            "sample occurrences per optimizer update",
        ),
        "world_size": _field(topology.get("world_size"), "analytical", "ranks"),
        "gradient_accumulation": _field(
            topology.get("gradient_accumulation"),
            "analytical",
            "configured gradient accumulation steps",
            "Packed mode also records actual micro_count per update in lineage.",
        ),
        "valid_targets_per_second": _field(
            vps,
            "exact" if vps is not None else "unavailable",
            "unweighted valid real targets/s",
        ),
        "training_wall_clock": _field(
            wall,
            "exact" if wall is not None else "unavailable",
            "seconds",
        ),
        "training_GPU_hours": _field(
            (wall * float(topology.get("world_size", 0)) / 3600.0)
            if wall is not None
            else None,
            "analytical" if wall is not None else "unavailable",
            "GPU-hours",
            "wall-clock seconds multiplied by world_size",
        ),
        "peak_VRAM": _field(
            peak_vram_bytes,
            "exact" if peak_vram_bytes is not None else "unavailable",
            "bytes",
        ),
        "padding_target_slots_ignored_for_U_seen_E_A": _field(
            padding_targets,
            "exact",
            "target slots",
        ),
        "invalid_or_zero_weight_real_target_slots": _field(
            invalid_or_zero_weight_targets,
            "exact",
            "target slots",
        ),
    }
    return summary


def write_resource_markdown(card: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Resource Card",
        "",
        f"- run_id: `{card['run_id']}`",
        f"- generated_utc: `{card['generated_utc']}`",
        "",
        "## Whole Run",
        "",
        "| field | value | source | unit | note |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for key, item in card["whole_run"].items():
        if not isinstance(item, dict) or "source" not in item:
            continue
        value = item.get("value")
        if isinstance(value, float):
            value_s = f"{value:.10g}"
        else:
            value_s = str(value)
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                key,
                value_s,
                item.get("source", ""),
                item.get("unit", ""),
                str(item.get("note", "")).replace("|", "/"),
            )
        )

    lines.extend(["", "## Phases", ""])
    for phase_name, summary in card.get("phases", {}).items():
        lines.extend(
            [
                f"### `{phase_name}`",
                "",
                "| field | value | source | unit |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for key, item in summary.items():
            if not isinstance(item, dict) or "source" not in item:
                continue
            value = item.get("value")
            if isinstance(value, float):
                value_s = f"{value:.10g}"
            else:
                value_s = str(value)
            lines.append(
                f"| `{key}` | {value_s} | {item.get('source', '')} | {item.get('unit', '')} |"
            )
        lines.append("")

    lines.extend(["", "## Source Views", ""])
    for view_name, summary in card.get("source_views", {}).items():
        lines.extend(
            [
                f"### `{view_name}`",
                "",
                "| field | value | source | unit |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for key, item in summary.items():
            if not isinstance(item, dict) or "source" not in item:
                continue
            value = item.get("value")
            if isinstance(value, float):
                value_s = f"{value:.10g}"
            else:
                value_s = str(value)
            lines.append(
                f"| `{key}` | {value_s} | {item.get('source', '')} | {item.get('unit', '')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- `U_seen` and `E_action_*` count only targets with "
            "`target_t >= 0`, `valid_mask=True`, `loss_weight > 0`, and a "
            "committed optimizer update.",
            "- Padding target slots with timestamp `-1` are kept in lineage but "
            "excluded from canonical support and exposure.",
            "- `mean_replay` follows `E_action_weighted / U_seen`; "
            "`mean_replay_unweighted` is reported for draw-count interpretation.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        return {"commit": commit, "dirty_status": dirty}
    except Exception as error:  # pragma: no cover - diagnostic only
        return {"commit": None, "dirty_status": None, "error": str(error)}


def _config_snapshot(config: Any) -> dict[str, Any]:
    keys = [
        "dataset_path",
        "empty_emb_path",
        "wan22_pretrained_model_name_or_path",
        "training_mode",
        "packing_enabled",
        "global_episodes_per_update",
        "max_self_tokens",
        "max_episodes_per_pack",
        "packing_seed",
        "batch_size",
        "gradient_accumulation_steps",
        "num_steps",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "load_worker",
        "param_dtype",
        "obs_cam_keys",
        "used_action_channel_ids",
        "action_per_frame",
        "action_dim",
        "norm_stat",
        "save_interval",
    ]
    return {key: getattr(config, key, None) for key in keys}


class LineageTracer:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        run_id: str,
        config: Any,
        dataset: Any,
        sampler: Any,
        source_view: str,
        subset_manifest_id: str,
        sampler_probability: float | None,
        compute_support: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.config = config
        self.rank = int(config.rank)
        self.world_size = int(config.world_size)
        self.source_view = source_view
        self.subset_manifest_id = subset_manifest_id
        self.sampler_probability = sampler_probability
        self.records: list[dict[str, Any]] = []
        self.pending: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        self.microbatch_signatures: list[dict[str, Any]] = []
        self.started_wall_time = time.time()
        self.started_perf = time.perf_counter()
        self.repo_root = Path(__file__).resolve().parents[1]

        self.retained_ids: set[str] | None = None
        self.eligible_ids: set[str] | None = None
        if compute_support:
            support_ids = support_ids_from_dataset(dataset)
            self.retained_ids = set(support_ids)
            self.eligible_ids = set(support_ids)
            write_json(
                self.output_dir / "support_summary.json",
                {
                    "format_version": LINEAGE_FORMAT_VERSION,
                    "support_mode": "full_dataset_analytical",
                    "U_ret": len(self.retained_ids),
                    "U_eligible": len(self.eligible_ids),
                    "retained_support_sha256": _support_hash(self.retained_ids),
                    "eligible_support_sha256": _support_hash(self.eligible_ids),
                    "source": "dataset.get_lineage_metadata over all dataset indices",
                },
            )

        manifest_hash = getattr(dataset, "packing_manifest_hash", None)
        if manifest_hash is None and hasattr(sampler, "manifest_hash"):
            manifest_hash = sampler.manifest_hash
        dry_run_manifest = {
            "format_version": LINEAGE_FORMAT_VERSION,
            "run_id": self.run_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "repo": str(self.repo_root),
            "git": _git_state(self.repo_root),
            "resolved_config": _config_snapshot(config),
            "dataset_fingerprint": getattr(dataset, "dataset_fingerprint", None),
            "packing_manifest_hash": manifest_hash,
            "subset_manifest_id": subset_manifest_id,
            "subset_manifest_sha256": (
                sha256_file(subset_manifest_id)
                if subset_manifest_id and Path(subset_manifest_id).is_file()
                else None
            ),
            "random_seeds": {
                "seed": getattr(config, "seed", None),
                "packing_seed": getattr(config, "packing_seed", None),
            },
            "hardware": {
                "world_size": self.world_size,
                "rank": self.rank,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": (
                    torch.cuda.get_device_name(torch.cuda.current_device())
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "precision": str(getattr(config, "param_dtype", None)),
            "model_checkpoint_initialization": getattr(
                config, "wan22_pretrained_model_name_or_path", None
            ),
            "batch_size": getattr(config, "batch_size", None),
            "world_size": self.world_size,
            "gradient_accumulation": getattr(
                config, "gradient_accumulation_steps", None
            ),
            "number_of_optimizer_steps": getattr(config, "num_steps", None),
            "resume_status": getattr(config, "resume_from", None) or "fresh",
        }
        write_json(
            self.output_dir / f"dry_run_manifest_rank{self.rank}.json",
            dry_run_manifest,
        )
        if self.rank == 0:
            write_json(self.output_dir / "dry_run_manifest.json", dry_run_manifest)

    def capture_microbatch(
        self,
        *,
        batch: dict[str, Any],
        input_dict: dict[str, Any],
        optimizer_step: int,
        microbatch_id: int,
        gradient_accumulation_index: int,
    ) -> None:
        lineage_records = batch.get("lineage_records")
        if not lineage_records:
            return
        frame_offsets = [int(value) for value in input_dict["frame_offsets"].tolist()]
        frame_lengths = [int(value) for value in input_dict["frame_lengths"].tolist()]
        action_mask = input_dict["action_dict"]["actions_mask"]
        action_mask_any = (
            action_mask[0, :, :, :, 0]
            .detach()
            .bool()
            .any(dim=0)
            .flatten()
            .cpu()
            .tolist()
        )
        action_frame_weights = (
            input_dict["action_frame_loss_weights"].detach().float().cpu().tolist()
        )
        action_tokens_per_frame = int(action_mask.shape[3])
        signature_items = []

        for sample_slot, metadata in enumerate(lineage_records):
            start = frame_offsets[sample_slot]
            end = frame_offsets[sample_slot + 1]
            flat_start = start * action_tokens_per_frame
            flat_end = end * action_tokens_per_frame
            target_timestamps = [
                int(value)
                for value in metadata["action_target_timestamps"][: flat_end - flat_start]
            ]
            valid_mask = [bool(value) for value in action_mask_any[flat_start:flat_end]]
            frame_weights = action_frame_weights[start:end]
            loss_weights = [
                float(weight)
                for weight in frame_weights
                for _ in range(action_tokens_per_frame)
            ][: len(target_timestamps)]
            if not (
                len(target_timestamps) == len(valid_mask) == len(loss_weights)
            ):
                raise ValueError(
                    "Lineage target/mask/weight length mismatch: "
                    f"targets={len(target_timestamps)}, mask={len(valid_mask)}, "
                    f"weights={len(loss_weights)}"
                )
            record = {
                "format_version": LINEAGE_FORMAT_VERSION,
                "run_id": self.run_id,
                "phase": "train",
                "optimizer_step": int(optimizer_step),
                "optimizer_update_applied": False,
                "rank": self.rank,
                "world_size": self.world_size,
                "microbatch_id": int(microbatch_id),
                "gradient_accumulation_index": int(gradient_accumulation_index),
                "sample_slot": int(sample_slot),
                "source_view": self.source_view,
                "dataset": str(metadata["dataset"]),
                "task": str(metadata["task"]),
                "episode": int(metadata["episode"]),
                "dataset_index": int(metadata.get("dataset_index", -1)),
                "local_dataset_index": int(metadata.get("local_dataset_index", -1)),
                "sample_uid": int(metadata.get("sample_uid", -1)),
                "requested_anchor_t": int(metadata["requested_anchor_t"]),
                "remapped_anchor_t": int(metadata["remapped_anchor_t"]),
                "segment_start_t": int(metadata.get("segment_start_t", metadata["requested_anchor_t"])),
                "segment_end_t": int(metadata.get("segment_end_t", metadata["remapped_anchor_t"])),
                "observation_timestamps": [
                    int(value) for value in metadata["observation_timestamps"]
                ],
                "action_target_timestamps": target_timestamps,
                "valid_mask": valid_mask,
                "loss_weights": loss_weights,
                "subset_manifest_id": self.subset_manifest_id,
                "sampler_probability": (
                    None
                    if self.sampler_probability is None
                    else float(self.sampler_probability)
                ),
                "frame_count": int(frame_lengths[sample_slot]),
                "action_tokens_per_frame": action_tokens_per_frame,
                "padding_target_count": int(sum(t < 0 for t in target_timestamps)),
                "real_target_count": int(sum(t >= 0 for t in target_timestamps)),
                "micro_count": int(batch["micro_count"].item())
                if "micro_count" in batch
                else None,
                "planned_tokens": int(batch["planned_tokens"].item())
                if "planned_tokens" in batch
                else None,
            }
            self.pending[int(optimizer_step)].append(record)
            signature_items.append(
                {
                    "sample_slot": int(sample_slot),
                    "dataset_index": record["dataset_index"],
                    "sample_uid": record["sample_uid"],
                    "dataset": record["dataset"],
                    "episode": record["episode"],
                    "requested_anchor_t": record["requested_anchor_t"],
                    "remapped_anchor_t": record["remapped_anchor_t"],
                    "observation_timestamps": record["observation_timestamps"],
                    "action_target_timestamps": record["action_target_timestamps"],
                    "valid_mask": record["valid_mask"],
                }
            )

        self.microbatch_signatures.append(
            {
                "run_id": self.run_id,
                "optimizer_step": int(optimizer_step),
                "rank": self.rank,
                "microbatch_id": int(microbatch_id),
                "gradient_accumulation_index": int(gradient_accumulation_index),
                "signature_sha256": stable_hash(signature_items),
                "items": signature_items,
            }
        )

    def commit_update(self, optimizer_step: int) -> None:
        pending = self.pending.pop(int(optimizer_step), [])
        for record in pending:
            record["optimizer_update_applied"] = True
        self.records.extend(pending)

    def finalize_rank(self, *, peak_vram_bytes: int | None) -> None:
        for pending in self.pending.values():
            self.records.extend(pending)
        self.pending.clear()
        ended_perf = time.perf_counter()
        ended_wall_time = time.time()
        table = pa.Table.from_pylist(self.records)
        pq.write_table(table, self.output_dir / f"lineage_rank{self.rank}.parquet")
        with (self.output_dir / f"batch_signatures_rank{self.rank}.jsonl").open(
            "w"
        ) as file:
            for item in self.microbatch_signatures:
                file.write(json.dumps(item, sort_keys=True, default=_json_default) + "\n")
        write_json(
            self.output_dir / f"lineage_rank{self.rank}_summary.json",
            {
                "format_version": LINEAGE_FORMAT_VERSION,
                "run_id": self.run_id,
                "rank": self.rank,
                "world_size": self.world_size,
                "record_count": len(self.records),
                "started_wall_time": self.started_wall_time,
                "ended_wall_time": ended_wall_time,
                "training_wall_clock_s": ended_perf - self.started_perf,
                "peak_vram_bytes": peak_vram_bytes,
            },
        )

    def finalize_global(self) -> None:
        records: list[dict[str, Any]] = []
        peak_vram = 0
        wall_clock = 0.0
        for rank in range(self.world_size):
            rank_path = self.output_dir / f"lineage_rank{rank}.parquet"
            if not rank_path.is_file():
                raise FileNotFoundError(f"Missing rank lineage file: {rank_path}")
            records.extend(pq.read_table(rank_path).to_pylist())
            summary_path = self.output_dir / f"lineage_rank{rank}_summary.json"
            with summary_path.open() as file:
                rank_summary = json.load(file)
            peak_vram = max(peak_vram, int(rank_summary.get("peak_vram_bytes") or 0))
            wall_clock = max(
                wall_clock, float(rank_summary.get("training_wall_clock_s") or 0.0)
            )

        support_path = self.output_dir / "support_summary.json"
        retained_ids = self.retained_ids
        eligible_ids = self.eligible_ids
        if retained_ids is None or eligible_ids is None:
            support_summary = {
                "U_ret": None,
                "U_eligible": None,
                "retained_support_sha256": None,
                "eligible_support_sha256": None,
            }
        else:
            with support_path.open() as file:
                support_summary = json.load(file)

        topology = {
            "global_batch_size": getattr(self.config, "global_episodes_per_update", None),
            "world_size": self.world_size,
            "gradient_accumulation": getattr(
                self.config, "gradient_accumulation_steps", None
            ),
        }
        whole = compute_resource_summary(
            records,
            retained_ids=retained_ids,
            eligible_ids=eligible_ids,
            topology=topology,
            training_wall_clock_s=wall_clock,
            peak_vram_bytes=peak_vram,
            group_name="whole_run",
        )
        source_views = {}
        for source_view in sorted({str(record["source_view"]) for record in records}):
            view_records = [
                record for record in records if str(record["source_view"]) == source_view
            ]
            source_views[source_view] = compute_resource_summary(
                view_records,
                retained_ids=retained_ids,
                eligible_ids=eligible_ids,
                topology=topology,
                training_wall_clock_s=wall_clock,
                peak_vram_bytes=peak_vram,
                group_name=f"source_view:{source_view}",
            )
        phases = {}
        for phase in sorted({str(record["phase"]) for record in records}):
            phase_records = [
                record for record in records if str(record["phase"]) == phase
            ]
            phases[phase] = compute_resource_summary(
                phase_records,
                retained_ids=retained_ids,
                eligible_ids=eligible_ids,
                topology=topology,
                training_wall_clock_s=wall_clock,
                peak_vram_bytes=peak_vram,
                group_name=f"phase:{phase}",
            )
        card = {
            "format_version": LINEAGE_FORMAT_VERSION,
            "run_id": self.run_id,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "support_summary": support_summary,
            "whole_run": whole,
            "phases": phases,
            "source_views": source_views,
        }
        write_json(self.output_dir / "resource_card.json", card)
        write_resource_markdown(card, self.output_dir / "resource_card.md")
        self._write_human_examples(records)

    def _write_human_examples(self, records: list[dict[str, Any]]) -> None:
        examples = []
        seen_kinds = set()

        def add(kind: str, record: dict[str, Any]) -> None:
            if len(examples) >= 30:
                return
            key = (kind, record["rank"], record["microbatch_id"], record["sample_slot"])
            if key in seen_kinds:
                return
            seen_kinds.add(key)
            timestamps = record["action_target_timestamps"]
            valid = record["valid_mask"]
            weights = record["loss_weights"]
            examples.append(
                {
                    "example_kind": kind,
                    "example_source": "actual_lineage",
                    "run_id": record["run_id"],
                    "rank": record["rank"],
                    "optimizer_step": record["optimizer_step"],
                    "microbatch_id": record["microbatch_id"],
                    "sample_slot": record["sample_slot"],
                    "source_view": record["source_view"],
                    "dataset": record["dataset"],
                    "task": record["task"],
                    "episode": record["episode"],
                    "requested_anchor_t": record["requested_anchor_t"],
                    "remapped_anchor_t": record["remapped_anchor_t"],
                    "observation_timestamps_head": record["observation_timestamps"][:8],
                    "observation_timestamps_tail": record["observation_timestamps"][-8:],
                    "target_head": timestamps[:16],
                    "target_tail": timestamps[-16:],
                    "valid_head": valid[:16],
                    "loss_weight_head": weights[:16],
                    "padding_target_count": record["padding_target_count"],
                    "real_target_count": record["real_target_count"],
                }
            )

        for record in records:
            if record["padding_target_count"] > 0:
                add("padding_prefix", record)
                break
        for record in records:
            if int(record["requested_anchor_t"]) == 0:
                add("episode_start", record)
                break
        for record in sorted(records, key=lambda item: max(item["action_target_timestamps"])):
            if max(record["action_target_timestamps"]) > 0:
                endpoint = max(record["action_target_timestamps"])
                if endpoint >= int(record["segment_end_t"]) - 8:
                    add("episode_end_or_tail", record)
                    break
        for record in records[:20]:
            add("ordinary_full_sample", record)

        path = self.output_dir / "human_readable_examples_actual.jsonl"
        with path.open("w") as file:
            for item in examples:
                file.write(json.dumps(item, sort_keys=True, default=_json_default) + "\n")
