#!/usr/bin/env python3
"""Measure LingBot-VA planner sensitivity to observation-layout changes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAUSAL_TEST_DIR = PROJECT_ROOT / "isolated_tests" / "causal_plan_interventions"
for path in (PROJECT_ROOT, CAUSAL_TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run as causal  # noqa: E402
from wan_va.configs import VA_CONFIGS  # noqa: E402
from wan_va.distributed.util import init_distributed  # noqa: E402
from wan_va.wan_va_server import VA_Server  # noqa: E402


DEFAULT_BASE = Path("/root/shared/zouyude/ckpts/lingbot-va/lingbot-va-base")
DEFAULT_TRANSFORMER = Path(
    "/root/shared/zouyude/train/lingbot-va/libero-all-full/"
    "checkpoints/checkpoint_step_20000/transformer"
)
DEFAULT_ORIGINAL = Path(
    "/root/shared/zouyude/eval/lingbot-va/"
    "video_posttrain_compare_20260804_052758/libero/"
    "libero_object_ep000000/input"
)
DEFAULT_PLUS_ROOT = Path(
    "/root/shared/zouyude/eval/lingbot-va/"
    "libero_plus_layout_ood_compare_20260805_131630/candidate_initial_views"
)
DEFAULT_PLUS_MANIFEST = DEFAULT_PLUS_ROOT.parent / "manifest.json"
DEFAULT_PROMPT = "pick up the alphabet soup and place it in the basket"


def layout_inputs(original: Path, plus_root: Path) -> dict[str, Path]:
    result = {"original": original}
    directories = sorted(
        path
        for path in plus_root.glob(
            "pick_up_the_alphabet_soup_and_place_it_in_the_basket_level5_sample*"
        )
        if path.is_dir()
    )
    for path in directories:
        result[path.name.rsplit("_", 1)[-1]] = path
    if len(result) != 5:
        raise RuntimeError(f"Expected original plus four level-5 layouts, got {result}")
    return result


def to_uint8_frames(frames: Any) -> np.ndarray:
    array = np.asarray([np.asarray(frame) for frame in frames])
    if np.issubdtype(array.dtype, np.floating):
        if float(array.max()) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def response_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Frame shape mismatch: {reference.shape} vs {candidate.shape}")
    width = reference.shape[2]
    slices = {"all": slice(0, width), "agentview": slice(0, width // 2), "wrist": slice(width // 2, width)}
    output: dict[str, Any] = {}
    for view, xs in slices.items():
        initial = (candidate[0, :, xs].astype(np.float64) - reference[0, :, xs].astype(np.float64)).reshape(-1)
        denom = float(np.dot(initial, initial))
        if denom == 0:
            raise RuntimeError(f"Zero initial cross-layout difference for {view}")
        per_frame = []
        initial_norm = float(np.sqrt(denom))
        for frame_index in range(reference.shape[0]):
            delta = (candidate[frame_index, :, xs].astype(np.float64) - reference[frame_index, :, xs].astype(np.float64)).reshape(-1)
            delta_norm = float(np.linalg.norm(delta))
            cosine = float(np.dot(delta, initial) / (delta_norm * initial_norm)) if delta_norm else 0.0
            per_frame.append(
                {
                    "frame": frame_index,
                    "gain": delta_norm / initial_norm,
                    "directional_cosine": cosine,
                    "retained_projection": float(np.dot(delta, initial) / denom),
                    "cross_layout_rmse": float(np.sqrt(np.mean(delta**2))),
                }
            )
        future = per_frame[1:]
        output[view] = {
            "initial_cross_layout_rmse": per_frame[0]["cross_layout_rmse"],
            "future_mean_gain": float(np.mean([x["gain"] for x in future])),
            "future_mean_directional_cosine": float(np.mean([x["directional_cosine"] for x in future])),
            "future_mean_retained_projection": float(np.mean([x["retained_projection"] for x in future])),
            "last": per_frame[-1],
            "per_frame": per_frame,
        }
    return output


def save_comparison(frame_sets: dict[str, np.ndarray], path: Path, fps: int = 10) -> None:
    names = list(frame_sets)
    count = min(len(frame_sets[name]) for name in names)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for index in range(count):
            panels = []
            for name in names:
                panel = Image.fromarray(frame_sets[name][index]).convert("RGB")
                canvas = Image.new("RGB", (panel.width, panel.height + 22), "black")
                canvas.paste(panel, (0, 22))
                ImageDraw.Draw(canvas).text((5, 5), name, fill="white")
                panels.append(np.asarray(canvas))
            writer.append_data(np.concatenate(panels, axis=1))
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--transformer", type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument("--original-input", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--plus-input-root", type=Path, default=DEFAULT_PLUS_ROOT)
    parser.add_argument("--plus-manifest", type=Path, default=DEFAULT_PLUS_MANIFEST)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--video-seed", type=int, default=4202)
    parser.add_argument("--video-steps", type=int, default=20)
    parser.add_argument("--master-port", type=int, default=29651)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES.")
    inputs = layout_inputs(args.original_input, args.plus_input_root)
    for required in (args.base_model, args.transformer, args.plus_manifest, *inputs.values()):
        if not required.exists():
            raise FileNotFoundError(required)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "plans").mkdir()
    (args.output_dir / "inputs").mkdir()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    init_distributed(1, 0, 0)

    config = copy.deepcopy(VA_CONFIGS["libero_all"])
    config.wan22_pretrained_model_name_or_path = str(args.base_model.resolve())
    config.transformer_model_name_or_path = str(args.transformer.resolve())
    config.input_img_path = str(args.original_input.resolve())
    config.enable_offload = False
    config.save_debug_data = False
    config.save_root = str(args.output_dir)
    config.rank = config.local_rank = 0
    config.world_size = 1

    started = time.time()
    server = VA_Server(config)
    server.video_processor = VideoProcessor(vae_scale_factor=1)
    context = causal.encode_prompt(server, args.prompt)
    plans: dict[str, torch.Tensor] = {}
    frame_sets: dict[str, np.ndarray] = {}
    first_latents: dict[str, torch.Tensor] = {}

    for name, input_dir in inputs.items():
        obs_images = {
            key: np.asarray(Image.open(input_dir / f"{key}.png").convert("RGB"))
            for key in config.obs_cam_keys
        }
        obs = {"obs": [obs_images]}
        plan, first = causal.generate_plan(server, obs, context, args.video_seed, args.video_steps)
        plans[name] = plan
        first_latents[name] = first
        torch.save(plan.cpu(), args.output_dir / "plans" / f"{name}.pt")
        with torch.no_grad():
            decoded = server.decode_one_video(plan, "np")[0]
        frames = to_uint8_frames(decoded)
        frame_sets[name] = frames
        export_to_video(list(frames), str(args.output_dir / "plans" / f"{name}.mp4"), fps=10)
        Image.fromarray(frames[0]).save(args.output_dir / "inputs" / f"{name}.png")

    obs_images = {
        key: np.asarray(Image.open(args.original_input / f"{key}.png").convert("RGB"))
        for key in config.obs_cam_keys
    }
    repeated, repeated_first = causal.generate_plan(
        server, {"obs": [obs_images]}, context, args.video_seed, args.video_steps
    )
    repeat_exact = torch.equal(plans["original"], repeated) and torch.equal(first_latents["original"], repeated_first)
    if not repeat_exact:
        raise RuntimeError("Fixed-seed original plan replay is not deterministic.")

    reference = plans["original"]
    reference_frames = frame_sets["original"]
    metrics = {}
    for name in list(inputs)[1:]:
        metrics[name] = {
            "latent_vs_original": causal.plan_metrics(reference, plans[name]),
            "rgb_layout_response": response_stats(reference_frames, frame_sets[name]),
        }
    save_comparison(frame_sets, args.output_dir / "plans" / "comparison.mp4")

    plus_manifest = json.loads(args.plus_manifest.read_text())
    layout_metadata = {
        item["task_name"].rsplit("_", 1)[-1]: item["layout"]
        for item in plus_manifest["candidate_level5_layouts"]
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "model": "lingbot-va",
        "experiment": "M1-E003 planner observation/layout causal sensitivity",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(args.transformer.resolve()),
        "base_model": str(args.base_model.resolve()),
        "prompt": args.prompt,
        "video_seed": args.video_seed,
        "video_steps": args.video_steps,
        "inputs": {name: str(path.resolve()) for name, path in inputs.items()},
        "layout_metadata": layout_metadata,
        "metrics": metrics,
        "determinism": {"original_plan_repeat_exact": repeat_exact},
        "git": causal.git_record(),
        "metric_note": "RGB retained_projection compares each future cross-layout difference vector with its frame-zero cross-layout difference; 1 retains it, 0 loses it, negative reverses it.",
    }
    causal.atomic_json(args.output_dir / "results.json", result)
    (args.output_dir / "COMPLETE").touch()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    server.transformer.clear_cache(server.cache_name)
    del server
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
