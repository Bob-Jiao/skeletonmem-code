#!/usr/bin/env python3
"""Build two-episode, self-contained Stage-1 datasets under skeletonmem.

The RoboTwin variant is a strict subset of locally downloaded official data.
The LIBERO variant only exercises the LIBERO tensor/data contract: it remaps
two real camera latents and slices actions to seven dimensions. It is not a
substitute for the unavailable official LIBERO-LONG training distribution.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch


SOURCE = Path(
    "/data/jiaoguanbo/datasets/official-robotwin-adjust-bottle-audit"
)
SOURCE_REPO = SOURCE / (
    "lerobot_robotwin_eef_clean_50/"
    "adjust_bottle-demo_clean_collect_200-50"
)
DEST_ROOT = Path("/data/jiaoguanbo/skeletonmem/data/stage1")
ROBOTWIN_ROOT = DEST_ROOT / "robotwin_official_2ep"
LIBERO_ROOT = DEST_ROOT / "libero_contract_proxy_2ep"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )


def copy_two_episode_subset(destination: Path, repo_name: str) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    repo_dest = destination / repo_name
    shutil.copytree(SOURCE_REPO, repo_dest)
    shutil.copy2(SOURCE / "empty_emb.pt", destination / "empty_emb.pt")

    episodes = read_jsonl(repo_dest / "meta/episodes.jsonl")[:2]
    write_jsonl(repo_dest / "meta/episodes.jsonl", episodes)

    original = read_jsonl(repo_dest / "meta/episodes_ori.jsonl")[:2]
    write_jsonl(repo_dest / "meta/episodes_ori.jsonl", original)

    stats = read_jsonl(repo_dest / "meta/episodes_stats.jsonl")[:2]
    write_jsonl(repo_dest / "meta/episodes_stats.jsonl", stats)

    tasks = read_jsonl(repo_dest / "meta/tasks.jsonl")[:2]
    write_jsonl(repo_dest / "meta/tasks.jsonl", tasks)

    info_path = repo_dest / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = 2
    info["total_frames"] = sum(record["length"] for record in episodes)
    info["total_tasks"] = 2
    info["total_videos"] = 6
    info["splits"] = {"train": "0:2"}
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return repo_dest


def convert_to_libero_contract(robotwin_repo: Path, libero_repo: Path) -> None:
    latent_root = libero_repo / "latents/chunk-000"
    video_root = libero_repo / "videos/chunk-000"
    camera_map = {
        "observation.images.cam_left_wrist": "observation.images.agentview_rgb",
        "observation.images.cam_right_wrist": "observation.images.eye_in_hand_rgb",
    }
    for root in (latent_root, video_root):
        shutil.rmtree(root / "observation.images.cam_high")
        for source_name, target_name in camera_map.items():
            (root / source_name).rename(root / target_name)

    # Official LIBERO uses four controls per latent frame. The source RoboTwin
    # samples use a temporal stride of four (16 controls per latent), so the
    # contract proxy normalizes only the frame-id metadata to stride one.
    for latent_path in sorted(latent_root.glob("*/*.pth")):
        payload = torch.load(latent_path, weights_only=False)
        payload["frame_ids"] = list(range(len(payload["frame_ids"])))
        torch.save(payload, latent_path)

    info_path = libero_repo / "meta/info.json"
    info = json.loads(info_path.read_text())
    features = info["features"]
    features["observation.images.agentview_rgb"] = features.pop(
        "observation.images.cam_left_wrist"
    )
    features["observation.images.eye_in_hand_rgb"] = features.pop(
        "observation.images.cam_right_wrist"
    )
    features.pop("observation.images.cam_high")
    for name in ("action", "observation.state"):
        features[name]["shape"] = [7]
        feature_names = features[name].get("names")
        if feature_names:
            features[name]["names"] = [feature_names[0][:7]]
    info["total_videos"] = 4
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    for parquet_path in sorted((libero_repo / "data/chunk-000").glob("*.parquet")):
        table = pq.read_table(parquet_path)
        for name in ("action", "observation.state"):
            index = table.schema.get_field_index(name)
            values = [value.as_py()[:7] for value in table[name]]
            table = table.set_column(index, name, pa.array(values, type=pa.list_(pa.float32())))
        pq.write_table(table, parquet_path)

    stats_path = libero_repo / "meta/episodes_stats.jsonl"
    stats_records = read_jsonl(stats_path)
    for record in stats_records:
        stats = record["stats"]
        for name in ("action", "observation.state"):
            for key in ("min", "max", "mean", "std"):
                stats[name][key] = stats[name][key][:7]
        stats["observation.images.agentview_rgb"] = stats.pop(
            "observation.images.cam_left_wrist"
        )
        stats["observation.images.eye_in_hand_rgb"] = stats.pop(
            "observation.images.cam_right_wrist"
        )
        stats.pop("observation.images.cam_high")
    write_jsonl(stats_path, stats_records)


def main() -> None:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    robotwin_repo = copy_two_episode_subset(
        ROBOTWIN_ROOT, "adjust_bottle-demo_clean-2ep"
    )
    libero_repo = copy_two_episode_subset(
        LIBERO_ROOT, "libero-contract-proxy-2ep"
    )
    convert_to_libero_contract(robotwin_repo, libero_repo)
    print(ROBOTWIN_ROOT)
    print(LIBERO_ROOT)


if __name__ == "__main__":
    main()
