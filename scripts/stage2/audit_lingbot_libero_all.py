#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_suite(path: Path) -> dict:
    meta = path / "meta"
    info = load_json(meta / "info.json")
    episodes_path = meta / "episodes.jsonl"
    tasks_path = meta / "tasks.jsonl"
    latents_root = path / "latents"
    data_root = path / "data"
    video_root = path / "videos"
    parquet_files = sorted(data_root.rglob("*.parquet")) if data_root.exists() else []
    video_files = sorted(video_root.rglob("*.mp4")) if video_root.exists() else []
    latent_files = sorted(latents_root.rglob("*.pth")) if latents_root.exists() else []
    latent_by_camera: dict[str, int] = {}
    for latent in latent_files:
        rel = latent.relative_to(latents_root)
        camera = str(rel.parts[1]) if len(rel.parts) >= 3 and rel.parts[0].startswith("chunk-") else str(rel.parts[0])
        latent_by_camera[camera] = latent_by_camera.get(camera, 0) + 1

    top_files = {
        "empty_emb.pt": path / "empty_emb.pt",
        "text_embeddings.pt": path / "text_embeddings.pt",
        "preprocess_manifest.json": path / "preprocess_manifest.json",
        "validation_summary.json": path / "validation_summary.json",
    }
    hashes = {}
    for name, file_path in top_files.items():
        if file_path.exists():
            hashes[name] = {
                "sha256": sha256(file_path),
                "bytes": file_path.stat().st_size,
            }

    return {
        "path": str(path),
        "exists": path.exists(),
        "info": {
            "codebase_version": info.get("codebase_version"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "total_tasks": info.get("total_tasks"),
            "features": sorted((info.get("features") or {}).keys()),
        },
        "meta_files": {
            "info_json": (meta / "info.json").exists(),
            "episodes_jsonl": episodes_path.exists(),
            "tasks_jsonl": tasks_path.exists(),
            "episodes_stats_jsonl": (meta / "episodes_stats.jsonl").exists(),
        },
        "jsonl_counts": {
            "episodes": jsonl_count(episodes_path),
            "tasks": jsonl_count(tasks_path),
        },
        "latents": {
            "root_exists": latents_root.exists(),
            "files": len(latent_files),
            "by_camera": latent_by_camera,
        },
        "data": {
            "root_exists": data_root.exists(),
            "parquet_files": len(parquet_files),
        },
        "videos": {
            "root_exists": video_root.exists(),
            "mp4_files": len(video_files),
        },
        "top_files": {name: file_path.exists() for name, file_path in top_files.items()},
        "hashes": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/data/zouyude/data/lingbot-va/libero_all"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    suites = sorted([p for p in args.source.iterdir() if p.is_dir() and p.name.startswith("libero_")])
    result = {
        "source": str(args.source),
        "source_exists": args.source.exists(),
        "suites": [audit_suite(p) for p in suites],
        "root_files": {
            name: (args.source / name).exists()
            for name in ["empty_emb.pt", "text_embeddings.pt", "preprocess_manifest.json", "validation_summary.json"]
        },
    }
    root_hashes = {}
    for name, exists in result["root_files"].items():
        if exists:
            p = args.source / name
            root_hashes[name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    result["root_hashes"] = root_hashes

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# LingBot LIBERO All Source Audit",
        "",
        f"- source: `{args.source}`",
        f"- source_exists: `{result['source_exists']}`",
        f"- suite_count: `{len(result['suites'])}`",
        "",
        "## Root Files",
    ]
    for name, exists in result["root_files"].items():
        lines.append(f"- {name}: `{exists}`")
    lines.extend(["", "## Suites"])
    for suite in result["suites"]:
        lines.extend([
            "",
            f"### {Path(suite['path']).name}",
            f"- path: `{suite['path']}`",
            f"- total_episodes(info): `{suite['info']['total_episodes']}`",
            f"- total_frames(info): `{suite['info']['total_frames']}`",
            f"- total_tasks(info): `{suite['info']['total_tasks']}`",
            f"- episodes_jsonl_count: `{suite['jsonl_counts']['episodes']}`",
            f"- tasks_jsonl_count: `{suite['jsonl_counts']['tasks']}`",
            f"- features: `{suite['info']['features']}`",
            f"- latent_files: `{suite['latents']['files']}`",
            f"- latent_by_camera: `{suite['latents']['by_camera']}`",
            f"- parquet_files: `{suite['data']['parquet_files']}`",
            f"- video_files: `{suite['videos']['mp4_files']}`",
            f"- top_files: `{suite['top_files']}`",
        ])
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
