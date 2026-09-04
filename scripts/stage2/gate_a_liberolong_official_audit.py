#!/usr/bin/env python3
"""Detailed read-only audit for LingBot-VA Gate A LIBERO-LONG data.

This script does not generate latents and does not modify the dataset. It checks
whether the downloaded LeRobot dataset matches the assumptions in LingBot-VA's
`libero_train` dataloader.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


OFFICIAL_CAMERA_KEYS = [
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def safe_shape(value: Any) -> list[int] | str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(x) for x in shape]
    try:
        return [len(value)]
    except Exception:  # noqa: BLE001
        return type(value).__name__


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
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


def sample_parquet(repo: Path, episode_index: int) -> dict[str, Any]:
    parquet_path = repo / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    result: dict[str, Any] = {
        "path": str(parquet_path),
        "exists": parquet_path.exists(),
    }
    if not parquet_path.exists():
        return result
    try:
        import pandas as pd

        frame = pd.read_parquet(parquet_path)
        result["rows"] = int(len(frame))
        result["columns"] = list(frame.columns)
        if "action" in frame.columns and len(frame):
            result["action_shape_first"] = safe_shape(frame["action"].iloc[0])
            result["action_first"] = frame["action"].iloc[0].tolist()[:10]
            result["action_last"] = frame["action"].iloc[-1].tolist()[:10]
        for column in frame.columns:
            if column.startswith("observation."):
                result[f"{column}_shape_first"] = safe_shape(frame[column].iloc[0])
        return result
    except Exception as exc:  # noqa: BLE001
        result["read_error"] = f"{type(exc).__name__}: {exc}"
        return result


def sample_latent(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
        result["keys"] = sorted(data.keys())
        for key in [
            "latent",
            "text_emb",
            "latent_num_frames",
            "latent_height",
            "latent_width",
            "video_num_frames",
            "video_height",
            "video_width",
            "frame_ids",
            "start_frame",
            "end_frame",
            "fps",
            "ori_fps",
            "text",
        ]:
            if key not in data:
                continue
            value = data[key]
            if key in {"latent", "text_emb"}:
                result[f"{key}_shape"] = safe_shape(value)
                result[f"{key}_dtype"] = str(getattr(value, "dtype", "unknown"))
            elif key == "frame_ids":
                result["frame_ids_len"] = len(value)
                result["frame_ids_head"] = list(value[:8])
                result["frame_ids_tail"] = list(value[-8:])
            elif key == "text":
                result["text"] = str(value)
            else:
                result[key] = int(value) if isinstance(value, int) else value
        return result
    except Exception as exc:  # noqa: BLE001
        result["read_error"] = f"{type(exc).__name__}: {exc}"
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-segments", type=int, default=200)
    args = parser.parse_args()

    repo = args.repo.resolve()
    meta_dir = repo / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    info_path = meta_dir / "info.json"
    tasks_path = meta_dir / "tasks.jsonl"

    report: dict[str, Any] = {
        "repo": str(repo),
        "status": "unknown",
        "required_paths": {
            "meta_info": str(info_path),
            "meta_episodes": str(episodes_path),
            "meta_tasks": str(tasks_path),
            "data_dir": str(repo / "data"),
            "latents_dir": str(repo / "latents"),
            "empty_emb": str(repo / "empty_emb.pt"),
        },
        "exists": {
            "meta_info": info_path.exists(),
            "meta_episodes": episodes_path.exists(),
            "meta_tasks": tasks_path.exists(),
            "data_dir": (repo / "data").exists(),
            "latents_dir": (repo / "latents").exists(),
            "empty_emb": (repo / "empty_emb.pt").exists(),
        },
    }

    if info_path.exists():
        report["info"] = read_json(info_path)
    if tasks_path.exists():
        report["tasks_sample"] = read_jsonl(tasks_path, limit=5)

    if not episodes_path.exists():
        report["status"] = "missing_meta_episodes"
    else:
        episodes = read_jsonl(episodes_path)
        report["episodes"] = len(episodes)
        report["frames_total_from_meta"] = sum(int(ep.get("length", 0)) for ep in episodes)
        report["episode_sample"] = episodes[0] if episodes else None
        segment_rows: list[dict[str, Any]] = []
        segment_count = 0
        missing = Counter()
        present = Counter()
        camera_dirs = []
        for cam in OFFICIAL_CAMERA_KEYS:
            cam_dirs = sorted((repo / "latents").glob(f"chunk-*/*{cam}*")) if (repo / "latents").exists() else []
            camera_dirs.extend(str(path.relative_to(repo / "latents")) for path in cam_dirs if path.is_dir())
        report["latent_camera_dirs_observed"] = sorted(set(camera_dirs))

        for ep in episodes:
            episode_index = int(ep["episode_index"])
            chunk = episode_index // 1000
            for action_cfg in ep.get("action_config", []):
                segment_count += 1
                start = int(action_cfg["start_frame"])
                end = int(action_cfg["end_frame"])
                row = {
                    "episode_index": episode_index,
                    "start_frame": start,
                    "end_frame": end,
                    "text": action_cfg.get("action_text", ""),
                    "camera_files": {},
                }
                for cam in OFFICIAL_CAMERA_KEYS:
                    latent_path = repo / "latents" / f"chunk-{chunk:03d}" / cam / f"episode_{episode_index:06d}_{start}_{end}.pth"
                    ok = latent_path.exists()
                    row["camera_files"][cam] = {"path": str(latent_path), "exists": ok}
                    present[cam] += int(ok)
                    missing[cam] += int(not ok)
                if len(segment_rows) < args.max_segments:
                    segment_rows.append(row)

        report["segments_total_from_action_config"] = segment_count
        report["latent_present_by_camera"] = dict(present)
        report["latent_missing_by_camera"] = dict(missing)
        report["segment_sample_checked"] = segment_rows[:5]

        first_ep = episodes[0] if episodes else None
        if first_ep is not None:
            first_episode_index = int(first_ep["episode_index"])
            report["parquet_sample"] = sample_parquet(repo, first_episode_index)
            first_cfg = first_ep.get("action_config", [{}])[0]
            if first_cfg:
                start = int(first_cfg["start_frame"])
                end = int(first_cfg["end_frame"])
                report["latent_sample"] = {}
                for cam in OFFICIAL_CAMERA_KEYS:
                    latent_path = repo / "latents" / "chunk-000" / cam / f"episode_{first_episode_index:06d}_{start}_{end}.pth"
                    report["latent_sample"][cam] = sample_latent(latent_path)

        if not report["exists"]["latents_dir"]:
            report["status"] = "missing_latents_dir"
        elif any(missing[cam] for cam in OFFICIAL_CAMERA_KEYS):
            report["status"] = "missing_required_latents"
        elif not report["exists"]["empty_emb"]:
            report["status"] = "missing_empty_emb"
        else:
            report["status"] = "audit_pass_candidate"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    report = to_jsonable(report)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# LIBERO-LONG Official Data Audit Result",
        "",
        f"- repo: `{report['repo']}`",
        f"- status: `{report['status']}`",
        f"- has meta/info.json: `{report['exists']['meta_info']}`",
        f"- has meta/episodes.jsonl: `{report['exists']['meta_episodes']}`",
        f"- has data dir: `{report['exists']['data_dir']}`",
        f"- has latents dir: `{report['exists']['latents_dir']}`",
        f"- has empty_emb.pt: `{report['exists']['empty_emb']}`",
        f"- episodes: `{report.get('episodes', 0)}`",
        f"- frames from meta: `{report.get('frames_total_from_meta', 0)}`",
        f"- segments from action_config: `{report.get('segments_total_from_action_config', 0)}`",
        f"- latent present by camera: `{report.get('latent_present_by_camera', {})}`",
        f"- latent missing by camera: `{report.get('latent_missing_by_camera', {})}`",
        "",
        "Detailed JSON:",
        "",
        f"`{args.output_json}`",
        "",
    ]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
