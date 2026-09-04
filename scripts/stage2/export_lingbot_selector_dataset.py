#!/usr/bin/env python3
"""Export selector manifests as LingBot-compatible LeRobot dataset roots.

The exporter writes only under the requested output root. Large source files
(`data/`, `latents/`, `videos/`) are linked read-only by symlink. The selected
supervision budget is represented by filtering each episode's `action_config`
in `meta/episodes.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_link_or_copy(src: Path, dst: Path, copy_files: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def segment_signature(item: dict) -> tuple[int, int, int]:
    return (
        int(item["episode_index"]),
        int(item["start_frame"]),
        int(item["end_frame"]),
    )


def filter_episodes(source_episodes: list[dict], manifest_rows: list[dict]) -> tuple[list[dict], dict]:
    selected = {segment_signature(item): item for item in manifest_rows}
    kept_segments = 0
    out = []
    for episode in source_episodes:
        episode_index = int(episode["episode_index"])
        new_episode = dict(episode)
        filtered = []
        for fallback_id, config in enumerate(episode.get("action_config", [])):
            start = int(config["start_frame"])
            end = int(config["end_frame"])
            key = (episode_index, start, end)
            if key not in selected:
                continue
            new_config = dict(config)
            manifest_item = selected[key]
            new_config["skeleton_selector_manifest"] = {
                "segment_id": int(manifest_item.get("segment_id", fallback_id)),
                "action_loss_first": int(manifest_item.get("action_loss_first", start)),
                "action_loss_last": int(manifest_item.get("action_loss_last", end - 1)),
                "motion_score": float(manifest_item.get("motion_score", 0.0)),
            }
            filtered.append(new_config)
        if filtered:
            new_episode["action_config"] = filtered
            kept_segments += len(filtered)
            out.append(new_episode)
    summary = {
        "source_episodes": len(source_episodes),
        "selected_episodes": len(out),
        "manifest_segments": len(manifest_rows),
        "kept_segments": kept_segments,
        "dropped_manifest_segments": len(manifest_rows) - kept_segments,
    }
    return out, summary


def export_one(source_repo: Path, source_collection_root: Path, manifest_path: Path, output_collection_root: Path) -> dict:
    selector_name = manifest_path.stem
    repo_name = f"{source_repo.name}__{selector_name}"
    output_repo = output_collection_root / repo_name
    output_meta = output_repo / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)

    source_episodes = read_jsonl(source_repo / "meta" / "episodes.jsonl")
    manifest_rows = read_jsonl(manifest_path)
    filtered_episodes, summary = filter_episodes(source_episodes, manifest_rows)
    if not filtered_episodes:
        raise ValueError(f"{manifest_path} selected no valid episode configs from {source_repo}")

    for meta_name in ("info.json", "tasks.jsonl", "episodes_stats.jsonl", "episodes_ori.jsonl"):
        src = source_repo / "meta" / meta_name
        if src.exists() and not (output_meta / meta_name).exists():
            shutil.copy2(src, output_meta / meta_name)
    write_jsonl(output_meta / "episodes.jsonl", filtered_episodes)

    for dirname in ("data", "latents", "videos"):
        src = source_repo / dirname
        if src.exists():
            safe_link_or_copy(src, output_repo / dirname)

    empty_src = source_collection_root / "empty_emb.pt"
    if empty_src.exists():
        safe_link_or_copy(empty_src, output_collection_root / "empty_emb.pt")
        safe_link_or_copy(empty_src, output_repo / "empty_emb.pt")

    summary.update({
        "selector_manifest": str(manifest_path),
        "output_collection_root": str(output_collection_root),
        "output_repo": str(output_repo),
        "repo_name": repo_name,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--source-collection-root", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source_repo = args.source_repo.resolve()
    source_collection_root = (
        args.source_collection_root.resolve()
        if args.source_collection_root
        else source_repo.parent.resolve()
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        export_one(source_repo, source_collection_root, manifest_path.resolve(), output_root)
        for manifest_path in args.manifest
    ]
    (output_root / "export_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n")
    for summary in summaries:
        print(json.dumps(summary, ensure_ascii=False))
    print(f"wrote {output_root / 'export_summary.json'}")


if __name__ == "__main__":
    main()
