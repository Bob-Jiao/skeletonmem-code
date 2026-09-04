#!/usr/bin/env python3
"""Build Stage2 skeleton-supervision manifests and budget audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass(frozen=True)
class Segment:
    episode_index: int
    segment_id: int
    start_frame: int
    end_frame: int
    action_first: int
    action_last: int
    frame_ids: tuple[int, ...]
    real_action_count: int
    contains_pick3_completion: bool
    contains_success_tail: bool
    motion_score: float = 0.0

    @property
    def key(self) -> tuple[int, int]:
        return self.episode_index, self.segment_id

    @property
    def action_timesteps(self) -> range:
        return range(self.action_first, self.action_last + 1)

    @property
    def observation_timesteps(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.frame_ids)))


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_keyframes(keyframe_root: Path | None) -> dict[int, list[int]]:
    if keyframe_root is None:
        return {}
    path = keyframe_root / "meta" / "episodes.jsonl"
    if not path.exists():
        return {}
    return {
        item["episode_index"]: item.get("keyframe_steps", [])
        for item in read_jsonl(path)
    }


def segment_from_config(episode: dict, config: dict, fallback_id: int) -> Segment:
    start = int(config["start_frame"])
    end = int(config["end_frame"])
    action_first = int(config.get("action_loss_first", start))
    action_last = int(config.get("action_loss_last", min(end, episode["length"]) - 1))
    frame_ids = tuple(int(x) for x in config.get("frame_ids", range(start, end, 4)))
    return Segment(
        episode_index=int(episode["episode_index"]),
        segment_id=int(config.get("segment_id", fallback_id)),
        start_frame=start,
        end_frame=end,
        action_first=action_first,
        action_last=action_last,
        frame_ids=frame_ids,
        real_action_count=int(config.get("real_action_count", action_last - action_first + 1)),
        contains_pick3_completion=bool(config.get("contains_pick3_completion", False)),
        contains_success_tail=bool(config.get("contains_success_tail", False)),
    )


def load_segments(dataset_root: Path) -> tuple[list[dict], list[Segment]]:
    episodes = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    segments: list[Segment] = []
    for episode in episodes:
        for fallback_id, config in enumerate(episode.get("action_config", [])):
            segments.append(segment_from_config(episode, config, fallback_id))
    return episodes, segments


def load_actions(dataset_root: Path, episode_index: int) -> list[list[float]]:
    parquet_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    frame = pd.read_parquet(parquet_path, columns=["action"])
    return [list(row) for row in frame["action"]]


def attach_motion_scores(dataset_root: Path, segments: list[Segment]) -> list[Segment]:
    by_episode: dict[int, list[Segment]] = {}
    for segment in segments:
        by_episode.setdefault(segment.episode_index, []).append(segment)

    scored: list[Segment] = []
    for episode_index, episode_segments in by_episode.items():
        actions = load_actions(dataset_root, episode_index)
        diffs = [0.0]
        for prev, current in zip(actions, actions[1:]):
            arm_delta = sum((float(a) - float(b)) ** 2 for a, b in zip(current, prev)) ** 0.5
            gripper_delta = abs(float(current[7]) - float(prev[7])) + abs(float(current[15]) - float(prev[15]))
            diffs.append(arm_delta + 2.0 * gripper_delta)
        for segment in episode_segments:
            score = sum(diffs[i] for i in segment.action_timesteps if i < len(diffs))
            scored.append(Segment(**{**segment.__dict__, "motion_score": score}))
    return scored


def unique_action_count(segments: Iterable[Segment]) -> int:
    return len({(s.episode_index, t) for s in segments for t in s.action_timesteps})


def target_action_count(full_segments: list[Segment], budget: float) -> int:
    return math.ceil(unique_action_count(full_segments) * budget)


def greedy_to_budget(candidates: list[Segment], target: int) -> list[Segment]:
    selected: list[Segment] = []
    covered: set[tuple[int, int]] = set()
    for segment in candidates:
        selected.append(segment)
        covered.update((segment.episode_index, t) for t in segment.action_timesteps)
        if len(covered) >= target:
            break
    return selected


def by_episode_round_robin(groups: dict[int, list[Segment]]) -> list[Segment]:
    ordered: list[Segment] = []
    max_len = max(len(items) for items in groups.values())
    for offset in range(max_len):
        for episode_index in sorted(groups):
            if offset < len(groups[episode_index]):
                ordered.append(groups[episode_index][offset])
    return ordered


def select_segments(
    selector: str,
    budget: float,
    segments: list[Segment],
    episodes: list[dict],
    keyframes: dict[int, list[int]],
    seed: int,
) -> list[Segment]:
    if budget >= 0.999 or selector == "full":
        return list(segments)

    target = target_action_count(segments, budget)
    groups: dict[int, list[Segment]] = {}
    for segment in segments:
        groups.setdefault(segment.episode_index, []).append(segment)

    if selector == "random":
        rng = random.Random(seed)
        shuffled_groups = {}
        for episode_index, items in groups.items():
            items = list(items)
            rng.shuffle(items)
            shuffled_groups[episode_index] = items
        return greedy_to_budget(by_episode_round_robin(shuffled_groups), target)

    if selector == "uniform":
        uniform_groups = {}
        for episode_index, items in groups.items():
            count = max(1, math.ceil(len(items) * budget))
            if count == 1:
                chosen = [items[len(items) // 2]]
            else:
                chosen = [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]
            uniform_groups[episode_index] = chosen
        return greedy_to_budget(by_episode_round_robin(uniform_groups), target)

    if selector == "motion_gripper":
        ranked = sorted(segments, key=lambda item: (-item.motion_score, item.episode_index, item.segment_id))
        return greedy_to_budget(ranked, target)

    if selector == "event_stage":
        scored = []
        for segment in segments:
            episode_len = episodes[segment.episode_index]["length"]
            anchors = [0, episode_len // 3, 2 * episode_len // 3, episode_len - 1]
            event_hit = any(segment.start_frame <= step < segment.end_frame for step in keyframes.get(segment.episode_index, []))
            stage_hit = any(segment.start_frame <= step < segment.end_frame for step in anchors)
            score = (
                1000.0 * event_hit
                + 500.0 * segment.contains_success_tail
                + 250.0 * segment.contains_pick3_completion
                + 100.0 * stage_hit
                + segment.motion_score
            )
            scored.append((score, segment.episode_index, segment.segment_id, segment))
        ranked = [item[-1] for item in sorted(scored, reverse=True)]
        return greedy_to_budget(ranked, target)

    raise ValueError(f"unknown selector: {selector}")


def latent_bytes(dataset_root: Path, segments: Iterable[Segment]) -> int:
    total = 0
    for segment in segments:
        for camera in CAMERAS:
            path = (
                dataset_root
                / "latents"
                / "chunk-000"
                / camera
                / f"episode_{segment.episode_index:06d}_{segment.start_frame}_{segment.end_frame}.pth"
            )
            if path.exists():
                total += path.stat().st_size
    return total


def budget_row(dataset_root: Path, selector: str, budget: float, selected: list[Segment], full_unique: int) -> dict:
    unique_actions = {(s.episode_index, t) for s in selected for t in s.action_timesteps}
    unique_observations = {(s.episode_index, t) for s in selected for t in s.observation_timesteps}
    action_exposures = sum(s.real_action_count for s in selected)
    return {
        "selector": selector,
        "budget": budget,
        "training_samples": len(selected),
        "nominal_segments": len(selected),
        "unique_action_labels": len(unique_actions),
        "unique_action_ratio": len(unique_actions) / full_unique,
        "action_label_exposures": action_exposures,
        "action_overlap_ratio": action_exposures / max(1, len(unique_actions)),
        "unique_observation_timesteps": len(unique_observations),
        "latent_files": len(selected) * len(CAMERAS),
        "latent_bytes": latent_bytes(dataset_root, selected),
        "oracle_selector": selector == "event_stage",
        "deployable_selector": selector in {"full", "random", "uniform", "motion_gripper"},
    }


def write_manifest(path: Path, selected: list[Segment]) -> None:
    with path.open("w") as handle:
        for segment in sorted(selected, key=lambda item: item.key):
            handle.write(json.dumps({
                "episode_index": segment.episode_index,
                "segment_id": segment.segment_id,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "action_loss_first": segment.action_first,
                "action_loss_last": segment.action_last,
                "real_action_count": segment.real_action_count,
                "contains_pick3_completion": segment.contains_pick3_completion,
                "contains_success_tail": segment.contains_success_tail,
                "motion_score": segment.motion_score,
            }) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--keyframe-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes, raw_segments = load_segments(args.dataset_root)
    segments = attach_motion_scores(args.dataset_root, raw_segments)
    keyframes = load_keyframes(args.keyframe_root)
    full_unique = unique_action_count(segments)

    rows = []
    selectors = ("full", "random", "uniform", "motion_gripper", "event_stage")
    for selector in selectors:
        for budget in args.budgets:
            if selector == "full" and budget < 0.999:
                continue
            selected = select_segments(selector, budget, segments, episodes, keyframes, args.seed)
            rows.append(budget_row(args.dataset_root, selector, budget, selected, full_unique))
            manifest_path = args.output_dir / f"{selector}_b{int(round(budget * 100)):03d}.jsonl"
            write_manifest(manifest_path, selected)

    csv_path = args.output_dir / "budget_table.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": str(args.dataset_root),
        "keyframe_root": str(args.keyframe_root) if args.keyframe_root else None,
        "episodes": len(episodes),
        "raw_frames": sum(item["length"] for item in episodes),
        "full_segments": len(segments),
        "full_unique_action_labels": full_unique,
        "keyframe_episodes": len(keyframes),
        "selectors": list(selectors),
        "budgets": args.budgets,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
