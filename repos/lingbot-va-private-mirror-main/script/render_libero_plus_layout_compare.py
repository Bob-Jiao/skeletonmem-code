#!/usr/bin/env python3
"""Prepare and finalize a fixed LIBERO-Plus Level-5 layout comparison.

Preparation is run in the LIBERO-Plus environment.  Video generation is run
through render_posttrain_video_compare.py so it uses the exact same LingBot-VA
inference path as the existing GT/full reference.  The final comparison reuses
the existing GT and original-layout full videos byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


TASK_PREFIX = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
SELECTED_TASK = f"{TASK_PREFIX}_level5_sample1"
PROMPT = "pick up the alphabet soup and place it in the basket"
MILK_PROMPT = "pick up the milk and place it in the basket"
MILK_SAMPLE = f"{SELECTED_TASK}_instruction_milk"
SEED = 4202
NUM_CHUNKS = 6
FRAME_CHUNK_SIZE = 4
EXPECTED_FRAMES = 93
FPS = 10

PLUS_ROOT = Path("/root/zouyude/LIBERO-plus")
BASE_BDDL = (
    PLUS_ROOT
    / "libero/libero/bddl_files/libero_object"
    / f"{TASK_PREFIX}.bddl"
)
BASE_MODEL_ROOT = Path("/root/shared/zouyude/ckpts/lingbot-va/lingbot-va-base")
FULL_TRANSFORMER = Path(
    "/root/shared/zouyude/train/lingbot-va/libero-all-full-4gpu/"
    "checkpoints/checkpoint_step_20000/transformer"
)
REFERENCE_SAMPLE = Path(
    "/root/shared/zouyude/eval/lingbot-va/"
    "video_posttrain_compare_20260804_052758/libero/"
    "libero_object_ep000000"
)
REFERENCE_GT = REFERENCE_SAMPLE / "gt.mp4"
REFERENCE_FULL = REFERENCE_SAMPLE / "full/generated.mp4"
FONT_FILE = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, with_hash: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if with_hash:
        result["sha256"] = sha256(path)
    return result


def ffprobe(path: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,"
            "nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return {
        **file_record(path, with_hash=True),
        **json.loads(output)["streams"][0],
    }


def read_region_centers(path: Path) -> dict[str, list[float]]:
    text = path.read_text()
    pattern = re.compile(
        r"\((bin_region|target_object_region|other_object_region_[0-9]+)"
        r"\s+.*?\(:ranges\s*\(\s*\(([-+0-9.eE\s]+)\)",
        re.DOTALL,
    )
    centers = {}
    for name, values_text in pattern.findall(text):
        values = [float(value) for value in values_text.split()]
        if len(values) != 4:
            raise ValueError(f"Unexpected range in {path}: {name}={values}")
        centers[name] = [(values[0] + values[2]) / 2, (values[1] + values[3]) / 2]
    if len(centers) != 7:
        raise ValueError(f"Expected 7 movable regions in {path}, found {centers}")
    return centers


def displacement(base: dict[str, list[float]], changed: dict[str, list[float]]) -> dict[str, Any]:
    per_region = {
        name: math.dist(base[name], changed[name])
        for name in sorted(base)
    }
    return {
        "sum_center_displacement_m": sum(per_region.values()),
        "max_center_displacement_m": max(per_region.values()),
        "target_center_displacement_m": per_region["target_object_region"],
        "basket_center_displacement_m": per_region["bin_region"],
        "per_region_center_displacement_m": per_region,
        "base_region_centers_m": base,
        "changed_region_centers_m": changed,
    }


def prepare(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    import torch
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    root = args.output_root.resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(manifest_path)
    for required in (BASE_BDDL, REFERENCE_GT, REFERENCE_FULL, FULL_TRANSFORMER):
        if not required.exists():
            raise FileNotFoundError(required)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task_names = suite.get_task_names()
    candidate_names = [f"{TASK_PREFIX}_level5_sample{i}" for i in range(1, 5)]
    missing = sorted(set(candidate_names) - set(task_names))
    if missing:
        raise ValueError(f"Missing LIBERO-Plus tasks: {missing}")

    base_centers = read_region_centers(BASE_BDDL)
    candidates_root = root / "candidate_initial_views"
    selected_input = root / "libero" / SELECTED_TASK / "input"
    candidate_records = []
    for task_name in candidate_names:
        task_id = task_names.index(task_name)
        task = suite.get_task(task_id)
        bddl_path = Path(suite.get_task_bddl_file_path(task_id)).resolve()
        init_states = suite.get_task_init_states(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=512,
            camera_widths=512,
        )
        try:
            env.seed(SEED)
            env.reset()
            obs = env.set_init_state(init_states[0])
            settling_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            for _ in range(5):
                obs, _, _, _ = env.step(settling_action)
        finally:
            env.close()

        # FastWAM/LeRobot LIBERO inputs are rotated 180 degrees relative to the
        # raw robosuite observations.  Match the existing reference exactly.
        agent = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        candidate_dir = candidates_root / task_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        agent_path = candidate_dir / "observation.images.agentview_rgb.png"
        wrist_path = candidate_dir / "observation.images.eye_in_hand_rgb.png"
        cv2.imwrite(str(agent_path), cv2.cvtColor(agent, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
        composite_path = candidates_root / f"{task_name}.png"
        composite = np.concatenate([agent, wrist], axis=1)
        cv2.imwrite(str(composite_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        if task_name == SELECTED_TASK:
            selected_input.mkdir(parents=True, exist_ok=True)
            shutil.copy2(agent_path, selected_input / agent_path.name)
            shutil.copy2(wrist_path, selected_input / wrist_path.name)

        candidate_records.append(
            {
                "task_name": task_name,
                "task_id": task_id,
                "bddl_path": str(bddl_path),
                "init_state_shape": list(init_states.shape),
                "initial_view": file_record(composite_path, with_hash=True),
                "layout": displacement(base_centers, read_region_centers(bddl_path)),
            }
        )

    selected_record = next(
        record for record in candidate_records if record["task_name"] == SELECTED_TASK
    )
    sample = {
        "domain": "libero",
        "dataset": "libero_plus/libero_object",
        "sample_id": SELECTED_TASK,
        "episode_index": 0,
        "prompt": PROMPT,
        "seed": SEED,
        "num_chunks": NUM_CHUNKS,
        "frame_chunk_size": FRAME_CHUNK_SIZE,
        "expected_output_frames": EXPECTED_FRAMES,
        "panel_width": 256,
        "panel_height": 128,
        "input_dir": str(selected_input.resolve()),
        "gt_path": str(REFERENCE_GT.resolve()),
        "reference_full_path": str(REFERENCE_FULL.resolve()),
        "selected_plus_task": selected_record,
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "original-layout versus LIBERO-Plus Level-5 layout OOD generation",
        "selection": (
            "Level-5 sample1 has the largest summed 2-D region-center displacement "
            "among the four Level-5 variants"
        ),
        "output_fps": FPS,
        "references_reused_without_regeneration": {
            "gt": ffprobe(REFERENCE_GT),
            "full_original_layout": ffprobe(REFERENCE_FULL),
        },
        "base_model_root": file_record(BASE_MODEL_ROOT / "transformer/config.json"),
        "models": {
            "full": {
                "transformer_path": str(FULL_TRANSFORMER.resolve()),
                "config": file_record(FULL_TRANSFORMER / "config.json"),
                "checkpoint_step": 20000,
                "training_run": "libero-all-full-4gpu",
            }
        },
        "candidate_level5_layouts": candidate_records,
        "samples": [sample],
    }
    atomic_json(root / "libero" / SELECTED_TASK / "sample.json", sample)
    atomic_json(manifest_path, manifest)
    print(f"Prepared 4 Level-5 candidates; selected {SELECTED_TASK}")
    print(f"Manifest: {manifest_path}")


def finalize(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    sample = manifest["samples"][0]
    changed_full = root / "libero" / sample["sample_id"] / "full/generated.mp4"
    output = root / "comparison_gt_full_original_full_level5.mp4"
    for path in (REFERENCE_GT, REFERENCE_FULL, changed_full):
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = ("GT", "FULL ORIGINAL LAYOUT", "FULL LEVEL-5 LAYOUT")
    streams = []
    for index, label in enumerate(labels):
        streams.append(
            f"[{index}:v]trim=start_frame=0:end_frame={EXPECTED_FRAMES},"
            f"setpts=N/({FPS}*TB),fps={FPS},scale=256:128:flags=lanczos,"
            "pad=iw:ih+36:0:36:black,"
            f"drawtext=fontfile={FONT_FILE}:text='{label}':fontcolor=white:"
            "fontsize=18:x=(w-text_w)/2:y=9"
            f"[v{index}]"
        )
    filter_graph = ";".join(streams) + ";[v0][v1][v2]hstack=inputs=3[out]"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(REFERENCE_GT),
        "-i", str(REFERENCE_FULL),
        "-i", str(changed_full),
        "-filter_complex", filter_graph,
        "-map", "[out]", "-frames:v", str(EXPECTED_FRAMES),
        "-r", str(FPS), "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", "-y", str(output),
    ]
    subprocess.run(command, check=True)
    result = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": PROMPT,
        "seed": SEED,
        "gt_reused": ffprobe(REFERENCE_GT),
        "full_original_layout_reused": ffprobe(REFERENCE_FULL),
        "full_level5_layout_generated": ffprobe(changed_full),
        "comparison": ffprobe(output),
        "selected_plus_task": sample["selected_plus_task"],
    }
    atomic_json(root / "results.json", result)
    (root / "COMPLETE").touch()
    print(f"Comparison: {output}")


def prepare_milk(args: argparse.Namespace) -> None:
    """Add a matched-noise milk-instruction sample using the same input images."""
    root = args.output_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = next(
        sample for sample in manifest["samples"] if sample["sample_id"] == SELECTED_TASK
    )
    milk_sample = {
        **source,
        "sample_id": MILK_SAMPLE,
        "prompt": MILK_PROMPT,
        "instruction_control": {
            "source_sample_id": SELECTED_TASK,
            "changed_field": "prompt",
            "unchanged_input_dir": source["input_dir"],
            "unchanged_seed": source["seed"],
        },
    }
    existing = {sample["sample_id"]: sample for sample in manifest["samples"]}
    if MILK_SAMPLE in existing and not args.force:
        print(f"Milk sample already present: {MILK_SAMPLE}")
        return
    manifest["samples"] = [
        sample for sample in manifest["samples"] if sample["sample_id"] != MILK_SAMPLE
    ] + [milk_sample]
    manifest["instruction_control"] = {
        "layout": SELECTED_TASK,
        "fixed_seed": SEED,
        "fixed_num_chunks": NUM_CHUNKS,
        "prompts": [PROMPT, MILK_PROMPT],
    }
    atomic_json(root / "libero" / MILK_SAMPLE / "sample.json", milk_sample)
    atomic_json(manifest_path, manifest)
    print(f"Added matched-layout milk sample: {MILK_SAMPLE}")


def finalize_milk(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    soup_full = root / "libero" / SELECTED_TASK / "full/generated.mp4"
    milk_full = root / "libero" / MILK_SAMPLE / "full/generated.mp4"
    output = (
        root
        / "comparison_gt_full_original_full_level5_soup_full_level5_milk.mp4"
    )
    inputs = (REFERENCE_GT, REFERENCE_FULL, soup_full, milk_full)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = (
        "GT",
        "FULL ORIGINAL LAYOUT",
        "FULL LEVEL-5 SOUP",
        "FULL LEVEL-5 MILK",
    )
    streams = []
    for index, label in enumerate(labels):
        streams.append(
            f"[{index}:v]trim=start_frame=0:end_frame={EXPECTED_FRAMES},"
            f"setpts=N/({FPS}*TB),fps={FPS},scale=256:128:flags=lanczos,"
            "pad=iw:ih+36:0:36:black,"
            f"drawtext=fontfile={FONT_FILE}:text='{label}':fontcolor=white:"
            "fontsize=18:x=(w-text_w)/2:y=9"
            f"[v{index}]"
        )
    filter_graph = (
        ";".join(streams) + ";[v0][v1][v2][v3]hstack=inputs=4[out]"
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    for path in inputs:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex", filter_graph,
            "-map", "[out]", "-frames:v", str(EXPECTED_FRAMES),
            "-r", str(FPS), "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an", "-y", str(output),
        ]
    )
    subprocess.run(command, check=True)
    result = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixed_layout": SELECTED_TASK,
        "fixed_seed": SEED,
        "soup_prompt": PROMPT,
        "milk_prompt": MILK_PROMPT,
        "gt_reused": ffprobe(REFERENCE_GT),
        "full_original_layout_reused": ffprobe(REFERENCE_FULL),
        "full_level5_soup_reused": ffprobe(soup_full),
        "full_level5_milk_generated": ffprobe(milk_full),
        "comparison": ffprobe(output),
    }
    atomic_json(root / "results_with_milk_instruction.json", result)
    (root / "MILK_INSTRUCTION_COMPLETE").touch()
    print(f"Four-way comparison: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(func=prepare)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.set_defaults(func=finalize)
    prepare_milk_parser = subparsers.add_parser("prepare-milk")
    prepare_milk_parser.add_argument("--output-root", type=Path, required=True)
    prepare_milk_parser.add_argument("--force", action="store_true")
    prepare_milk_parser.set_defaults(func=prepare_milk)
    finalize_milk_parser = subparsers.add_parser("finalize-milk")
    finalize_milk_parser.add_argument("--output-root", type=Path, required=True)
    finalize_milk_parser.set_defaults(func=finalize_milk)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
