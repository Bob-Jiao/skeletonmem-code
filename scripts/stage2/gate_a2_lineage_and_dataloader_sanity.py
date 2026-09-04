#!/usr/bin/env python3
"""Gate A2 lineage and dataloader sanity for LingBot-VA LIBERO-LONG.

Read-only checks only. This script must not train or save checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any


CAMERA_KEYS = [
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def shape_dtype_stats(tensor: Any) -> dict[str, Any]:
    import torch

    out = {
        "type": type(tensor).__name__,
        "shape": list(tensor.shape) if hasattr(tensor, "shape") else None,
        "dtype": str(getattr(tensor, "dtype", None)),
    }
    if torch.is_tensor(tensor):
        finite = torch.isfinite(tensor.float()) if tensor.numel() else torch.tensor([True])
        out.update(
            {
                "numel": int(tensor.numel()),
                "finite": bool(finite.all().item()),
                "min": float(tensor.float().min().item()) if tensor.numel() else None,
                "max": float(tensor.float().max().item()) if tensor.numel() else None,
                "mean": float(tensor.float().mean().item()) if tensor.numel() else None,
            }
        )
    return out


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
    return value


def write_lineage_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Gate A2 Single-Episode Lineage",
        "",
        f"Status: {report['lineage']['status']}",
        "",
        "## Episode",
        "",
        f"- dataset: `{report['paths']['dataset']}`",
        f"- episode_index: `{report['lineage']['episode_index']}`",
        f"- task: `{report['lineage']['task']}`",
        f"- episode_length: `{report['lineage']['episode_length']}`",
        f"- action_config: `{report['lineage']['action_config']}`",
        "",
        "## Parquet",
        "",
        f"- path: `{report['lineage']['parquet']['path']}`",
        f"- rows: `{report['lineage']['parquet'].get('rows')}`",
        f"- columns: `{report['lineage']['parquet'].get('columns')}`",
        f"- action_shape_first: `{report['lineage']['parquet'].get('action_shape_first')}`",
        f"- observation_state_shape_first: `{report['lineage']['parquet'].get('observation_state_shape_first')}`",
        "",
        "## Latents",
        "",
    ]
    for cam, item in report["lineage"]["latents"].items():
        lines.extend(
            [
                f"### {cam}",
                "",
                f"- path: `{item['path']}`",
                f"- exists: `{item['exists']}`",
                f"- latent_shape: `{item.get('latent_shape')}`",
                f"- latent_dtype: `{item.get('latent_dtype')}`",
                f"- text_emb_shape: `{item.get('text_emb_shape')}`",
                f"- text_emb_dtype: `{item.get('text_emb_dtype')}`",
                f"- latent_num_frames: `{item.get('latent_num_frames')}`",
                f"- latent_height: `{item.get('latent_height')}`",
                f"- latent_width: `{item.get('latent_width')}`",
                f"- frame_ids_head: `{item.get('frame_ids_head')}`",
                f"- frame_ids_tail: `{item.get('frame_ids_tail')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataloader_md(report: dict[str, Any], path: Path) -> None:
    dl = report["dataloader"]
    lines = [
        "# Gate A2 Dataloader Sanity",
        "",
        f"Status: {dl['status']}",
        "",
        "## Dataset construction",
        "",
        f"- official_direct_status: `{dl.get('official_direct_status')}`",
        f"- official_direct_error: `{dl.get('official_direct_error')}`",
        f"- runtime_compat_patch: `{dl.get('runtime_compat_patch')}`",
        f"- dataset_len: `{dl.get('dataset_len')}`",
        f"- cfg_prob_used: `{dl.get('cfg_prob_used')}`",
        f"- empty_emb: `{dl.get('empty_emb')}`",
        "",
    ]
    if "error" in dl:
        lines.extend(["## Error", "", f"`{dl['error']}`", ""])
    if "sample" in dl:
        lines.extend(["## Sample tensors", ""])
        for key, item in dl["sample"].items():
            lines.append(f"- `{key}`: `{item}`")
        lines.append("")
    if "batch" in dl:
        lines.extend(["## Batch tensors", ""])
        for key, item in dl["batch"].items():
            lines.append(f"- `{key}`: `{item}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data/jiaoguanbo/LIBERO/libero-long-lerobot"))
    parser.add_argument("--lingbot-root", type=Path, default=Path("/data/jiaoguanbo/skeletonmem/repos/lingbot-va-gatea-official"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--lineage-md", type=Path, required=True)
    parser.add_argument("--dataloader-md", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    lingbot_root = args.lingbot_root.resolve()
    sys.path.insert(0, str(lingbot_root))

    report: dict[str, Any] = {
        "status": "unknown",
        "paths": {
            "dataset": str(repo),
            "lingbot_root": str(lingbot_root),
        },
        "lineage": {},
        "dataloader": {},
    }

    episodes = read_jsonl(repo / "meta" / "episodes.jsonl")
    episode = next(ep for ep in episodes if int(ep["episode_index"]) == args.episode_index)
    cfg = episode["action_config"][0]
    start, end = int(cfg["start_frame"]), int(cfg["end_frame"])
    task = episode["tasks"][0] if episode.get("tasks") else ""
    parquet_path = repo / "data" / "chunk-000" / f"episode_{args.episode_index:06d}.parquet"

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    frame = pd.read_parquet(parquet_path)
    lineage: dict[str, Any] = {
        "status": "pass",
        "episode_index": args.episode_index,
        "task": task,
        "episode_length": int(episode["length"]),
        "action_config": cfg,
        "parquet": {
            "path": str(parquet_path),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "action_shape_first": list(frame["action"].iloc[0].shape),
            "action_first": frame["action"].iloc[0].tolist(),
            "action_mid": frame["action"].iloc[len(frame) // 2].tolist(),
            "action_last": frame["action"].iloc[-1].tolist(),
            "observation_state_shape_first": list(frame["observation.state"].iloc[0].shape),
        },
        "latents": {},
    }
    for cam in CAMERA_KEYS:
        latent_path = repo / "latents" / "chunk-000" / cam / f"episode_{args.episode_index:06d}_{start}_{end}.pth"
        item: dict[str, Any] = {"path": str(latent_path), "exists": latent_path.exists()}
        data = torch.load(latent_path, map_location="cpu", weights_only=False)
        item.update(
            {
                "keys": sorted(data.keys()),
                "latent_shape": list(data["latent"].shape),
                "latent_dtype": str(data["latent"].dtype),
                "text_emb_shape": list(data["text_emb"].shape),
                "text_emb_dtype": str(data["text_emb"].dtype),
                "latent_num_frames": int(data["latent_num_frames"]),
                "latent_height": int(data["latent_height"]),
                "latent_width": int(data["latent_width"]),
                "video_num_frames": int(data["video_num_frames"]),
                "video_height": int(data["video_height"]),
                "video_width": int(data["video_width"]),
                "frame_ids_len": len(data["frame_ids"]),
                "frame_ids_head": list(data["frame_ids"][:8]),
                "frame_ids_tail": list(data["frame_ids"][-8:]),
                "start_frame": int(data["start_frame"]),
                "end_frame": int(data["end_frame"]),
                "fps": int(data["fps"]),
                "ori_fps": int(data["ori_fps"]),
                "text": str(data["text"]),
            }
        )
        lineage["latents"][cam] = item
    report["lineage"] = lineage

    empty_emb_path = repo / "empty_emb.pt"
    empty_emb = torch.load(empty_emb_path, map_location="cpu", weights_only=False)

    try:
        dataset_module_path = lingbot_root / "wan_va" / "dataset" / "lerobot_latent_dataset.py"
        spec = importlib.util.spec_from_file_location("gate_a2_lerobot_latent_dataset", dataset_module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load dataset module from {dataset_module_path}")
        dataset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_module)
        LatentLeRobotDataset = dataset_module.LatentLeRobotDataset

        used_action_channel_ids = list(range(0, 7))
        inverse_used_action_channel_ids = [len(used_action_channel_ids)] * 30
        for i, j in enumerate(used_action_channel_ids):
            inverse_used_action_channel_ids[j] = i
        config = SimpleNamespace(
            dataset_path=str(repo),
            empty_emb_path=str(empty_emb_path),
            wan22_pretrained_model_name_or_path="/data/jiaoguanbo/models/lingbot-va-posttrain-libero-long",
            cfg_prob=0.0,
            obs_cam_keys=CAMERA_KEYS,
            env_type="none",
            inverse_used_action_channel_ids=inverse_used_action_channel_ids,
            norm_stat={
                "q01": [
                    -0.6589285731315613,
                    -0.84375,
                    -0.9375,
                    -0.12107142806053162,
                    -0.15964286029338837,
                    -0.26571428775787354,
                    -1.0,
                ]
                + [0.0] * 23,
                "q99": [
                    0.8999999761581421,
                    0.8544642925262451,
                    0.9375,
                    0.17142857611179352,
                    0.1842857152223587,
                    0.34392857551574707,
                    1.0,
                ]
                + [0.0] * 23,
            },
        )

        official_direct_error = None
        runtime_compat_patch = None
        try:
            dataset = LatentLeRobotDataset(repo_id=str(repo), config=config)
        except TypeError as exc:
            official_direct_error = f"{type(exc).__name__}: {exc}"
            if "dataclass" not in str(exc):
                raise
            runtime_compat_patch = "explicit_parquet_features_for_datasets_2_21_metadata_List"

            def load_hf_dataset_with_explicit_features(self):  # noqa: ANN001
                from datasets import Features, Sequence, Value, load_dataset

                features = Features(
                    {
                        "episode_index": Value("int64"),
                        "index": Value("int64"),
                        "frame_index": Value("int64"),
                        "task_index": Value("int64"),
                        "timestamp": Value("float32"),
                        "action": Sequence(Value("float32"), length=7),
                        "observation.state": Sequence(Value("float32"), length=8),
                    }
                )
                return load_dataset("parquet", data_dir=str(self.root / "data"), split="train", features=features)

            LatentLeRobotDataset.load_hf_dataset = load_hf_dataset_with_explicit_features
            dataset = LatentLeRobotDataset(repo_id=str(repo), config=config)
        sample = dataset[0]
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        report["dataloader"] = {
            "status": "pass",
            "official_direct_status": "failed" if official_direct_error else "pass",
            "official_direct_error": official_direct_error,
            "runtime_compat_patch": runtime_compat_patch,
            "dataset_len": len(dataset),
            "cfg_prob_used": float(config.cfg_prob),
            "empty_emb": {
                "path": str(empty_emb_path),
                "sha256": sha256(empty_emb_path),
                **shape_dtype_stats(empty_emb),
            },
            "sample": {k: shape_dtype_stats(v) for k, v in sample.items()},
            "batch": {k: shape_dtype_stats(v) for k, v in batch.items()},
        }
        report["status"] = "pass"
    except Exception as exc:  # noqa: BLE001
        report["dataloader"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "empty_emb": {
                "path": str(empty_emb_path),
                "sha256": sha256(empty_emb_path),
                **shape_dtype_stats(empty_emb),
            },
        }
        report["status"] = "partial_lineage_pass_dataloader_failed"

    report = jsonable(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_lineage_md(report, args.lineage_md)
    write_dataloader_md(report, args.dataloader_md)
    print(json.dumps({"status": report["status"], "dataloader": report["dataloader"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
