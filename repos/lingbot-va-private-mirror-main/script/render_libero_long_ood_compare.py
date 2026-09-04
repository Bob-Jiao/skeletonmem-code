#!/usr/bin/env python3
"""Render the official LIBERO-LONG checkpoint on three LIBERO OOD suites.

The input observations, prompts, seeds, video lengths, and GT videos come from
the fixed 2026-08-04 comparison.  A base control is rendered with the same
official ``libero`` inference config and the same auxiliary components, so the
transformer weights are the only generation-time difference.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from script.render_posttrain_video_compare import (
    atomic_json,
    comparison_filter,
    encode_video,
    ffprobe,
    file_record,
    seed_generation,
    sha256,
)


REFERENCE_ROOT = Path(
    "/root/shared/zouyude/eval/lingbot-va/"
    "video_posttrain_compare_20260804_052758"
)
BASE_ROOT = Path("/root/shared/zouyude/ckpts/lingbot-va/lingbot-va-base")
LONG_ROOT = Path(
    "/root/shared/zouyude/ckpts/lingbot-va/"
    "lingbot-va-posttrain-libero-long"
)
TARGET_SAMPLE_IDS = (
    "libero_goal_ep000000",
    "libero_object_ep000000",
    "libero_spatial_ep000000",
)
OUTPUT_FPS = 10


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def prepare(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(manifest_path)

    reference_manifest_path = REFERENCE_ROOT / "manifest.json"
    reference_manifest = _read_json(reference_manifest_path)
    reference_by_id = {
        sample["sample_id"]: sample
        for sample in reference_manifest["samples"]
        if sample["domain"] == "libero"
    }
    samples: list[dict[str, Any]] = []
    for sample_id in TARGET_SAMPLE_IDS:
        source = copy.deepcopy(reference_by_id[sample_id])
        reference_sample_root = REFERENCE_ROOT / "libero" / sample_id
        source.update(
            reference_sample_root=str(reference_sample_root.resolve()),
            reference_base_path=str(
                (reference_sample_root / "base" / "generated.mp4").resolve()
            ),
            reference_full_path=str(
                (reference_sample_root / "full" / "generated.mp4").resolve()
            ),
            ood_reason=(
                "LIBERO goal/object/spatial suite evaluated with the "
                "LIBERO-LONG (LIBERO-10) post-trained checkpoint"
            ),
        )
        sample_root = output_root / "libero" / sample_id
        sample_root.mkdir(parents=True, exist_ok=True)
        atomic_json(sample_root / "sample.json", source)
        samples.append(source)

    manifest = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "LIBERO-LONG checkpoint OOD video generation",
        "reference_manifest": {
            **file_record(reference_manifest_path),
            "sha256": sha256(reference_manifest_path),
        },
        "controlled_variables": {
            "config_name": "libero",
            "component_root": str(LONG_ROOT.resolve()),
            "component_equivalence": (
                "LONG and base tokenizer, text encoder, and VAE were "
                "byte-for-byte verified equal"
            ),
            "model_mode": "i2va",
            "frame_chunk_size": 4,
            "num_chunks": 6,
            "expected_output_frames": 93,
            "fps": OUTPUT_FPS,
            "guidance_scale": 5,
            "video_inference_steps": 20,
            "action_inference_steps": 50,
        },
        "models": {
            "base_control": {
                "label": "BASE CONTROL",
                "component_root": str(LONG_ROOT.resolve()),
                "transformer_path": str((BASE_ROOT / "transformer").resolve()),
                "config_name": "libero",
                "weights": "base full video-action transformer",
            },
            "libero_long": {
                "label": "LIBERO-LONG",
                "component_root": str(LONG_ROOT.resolve()),
                "transformer_path": str((LONG_ROOT / "transformer").resolve()),
                "config_name": "libero",
                "weights": "official full post-trained video-action transformer",
            },
        },
        "samples": samples,
        "overview_video": False,
    }
    atomic_json(manifest_path, manifest)
    print(f"Prepared {len(samples)} OOD samples: {manifest_path}", flush=True)


def render_worker(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from diffusers.utils import export_to_video
    from diffusers.video_processor import VideoProcessor

    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.wan_va_server import VA_Server

    output_root = args.output_root.resolve()
    manifest = _read_json(output_root / "manifest.json")
    model_spec = manifest["models"][args.model]
    transformer_path = Path(model_spec["transformer_path"]).resolve()
    component_root = Path(model_spec["component_root"]).resolve()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "render-worker expects one visible GPU; set CUDA_VISIBLE_DEVICES"
        )
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    init_distributed(1, 0, 0)

    config = copy.deepcopy(VA_CONFIGS[model_spec["config_name"]])
    config.wan22_pretrained_model_name_or_path = str(component_root)
    config.transformer_model_name_or_path = str(transformer_path)
    config.lora_adapter_path = None
    config.infer_mode = "i2va"
    config.enable_offload = False
    config.save_debug_data = False
    config.save_root = str(output_root / "libero" / f"_{args.model}_worker")
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1

    worker_started = time.time()
    print(
        f"Loading {args.model}: root={component_root} "
        f"transformer={transformer_path}",
        flush=True,
    )
    server = VA_Server(config)
    server.video_processor = VideoProcessor(vae_scale_factor=1)
    worker_results = []
    try:
        for sample in manifest["samples"]:
            sample_root = output_root / "libero" / sample["sample_id"]
            model_output_root = sample_root / args.model
            model_output_root.mkdir(parents=True, exist_ok=True)
            output_video = model_output_root / "generated.mp4"
            result_path = model_output_root / "render.json"
            if output_video.is_file() and result_path.is_file() and not args.force:
                existing = _read_json(result_path)
                if existing.get("status") == "complete":
                    print(f"Skipping complete output: {output_video}", flush=True)
                    worker_results.append(existing)
                    continue

            server.job_config.input_img_path = sample["input_dir"]
            server.job_config.num_chunks_to_infer = int(sample["num_chunks"])
            server.job_config.prompt = sample["prompt"]
            server.save_root = str(model_output_root)
            started = time.time()
            result = {
                "status": "running",
                "model": args.model,
                "sample_id": sample["sample_id"],
                "prompt": sample["prompt"],
                "seed": int(sample["seed"]),
                "num_chunks": int(sample["num_chunks"]),
                "config_name": model_spec["config_name"],
                "component_root": str(component_root),
                "transformer_path": str(transformer_path),
                "lora_adapter_path": None,
                "started_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)
                ),
            }
            atomic_json(result_path, result)
            print(
                f"Rendering {args.model}/{sample['sample_id']} "
                f"seed={sample['seed']}",
                flush=True,
            )

            server._reset(sample["prompt"])
            seed_generation(int(sample["seed"]))
            init_obs = server.load_init_obs()
            latent_chunks = []
            for chunk_id in range(int(sample["num_chunks"])):
                _, latents = server._infer(
                    init_obs,
                    frame_st_id=chunk_id * int(sample["frame_chunk_size"]),
                )
                latent_chunks.append(latents)
            pred_latent = torch.cat(latent_chunks, dim=2)
            with torch.no_grad():
                decoded_video = server.decode_one_video(pred_latent, "np")[0]
            export_to_video(decoded_video, str(output_video), fps=OUTPUT_FPS)
            expected_frames = int(sample["expected_output_frames"])
            if len(decoded_video) != expected_frames:
                raise RuntimeError(
                    f"{sample['sample_id']} decoded {len(decoded_video)} frames, "
                    f"expected {expected_frames}"
                )

            server.transformer.clear_cache(server.cache_name)
            server.streaming_vae.clear_cache()
            del pred_latent, decoded_video, latent_chunks
            torch.cuda.empty_cache()

            probe = ffprobe(output_video)
            if int(probe["nb_read_frames"]) != expected_frames:
                raise RuntimeError(f"Unexpected video probe: {probe}")
            result.update(
                status="complete",
                elapsed_seconds=time.time() - started,
                completed_at_utc=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                video=probe,
            )
            atomic_json(result_path, result)
            worker_results.append(result)
    finally:
        del server
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()

    atomic_json(
        output_root / f"worker_{args.model}.json",
        {
            "status": "complete",
            "model": args.model,
            "elapsed_seconds": time.time() - worker_started,
            "results": worker_results,
        },
    )


def concat(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    manifest = _read_json(output_root / "manifest.json")
    results = []
    for sample in manifest["samples"]:
        sample_root = output_root / "libero" / sample["sample_id"]
        inputs = [
            Path(sample["gt_path"]),
            sample_root / "base_control" / "generated.mp4",
            sample_root / "libero_long" / "generated.mp4",
        ]
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
        output = sample_root / "comparison_gt_base_control_libero_long.mp4"
        expected_frames = int(sample["expected_output_frames"])
        encode_video(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(inputs[0]),
                "-i",
                str(inputs[1]),
                "-i",
                str(inputs[2]),
                "-filter_complex",
                comparison_filter(
                    int(sample["panel_width"]),
                    int(sample["panel_height"]),
                    expected_frames,
                    labels=("GT", "BASE CONTROL", "LIBERO-LONG"),
                ),
            ],
            output,
            expected_frames,
        )
        record = {
            "sample_id": sample["sample_id"],
            "prompt": sample["prompt"],
            "seed": sample["seed"],
            "gt": ffprobe(inputs[0]),
            "base_control": ffprobe(inputs[1]),
            "libero_long": ffprobe(inputs[2]),
            "comparison": ffprobe(output),
        }
        atomic_json(sample_root / "comparison.json", record)
        results.append(record)
        print(f"Created comparison: {sample['sample_id']}", flush=True)
    atomic_json(
        output_root / "results.json",
        {
            "status": "complete",
            "overview_video": None,
            "comparisons": results,
        },
    )


def validate(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    manifest = _read_json(output_root / "manifest.json")
    checks = []
    for sample in manifest["samples"]:
        sample_root = output_root / "libero" / sample["sample_id"]
        paths = {
            "gt": Path(sample["gt_path"]),
            "base_control": sample_root / "base_control" / "generated.mp4",
            "libero_long": sample_root / "libero_long" / "generated.mp4",
            "comparison": (
                sample_root / "comparison_gt_base_control_libero_long.mp4"
            ),
        }
        probes = {name: ffprobe(path) for name, path in paths.items()}
        expected_frames = int(sample["expected_output_frames"])
        for name, probe in probes.items():
            if int(probe["nb_read_frames"]) != expected_frames:
                raise RuntimeError(f"{sample['sample_id']} {name}: {probe}")
            if probe["avg_frame_rate"] != f"{OUTPUT_FPS}/1":
                raise RuntimeError(f"{sample['sample_id']} {name}: {probe}")
            expected_size = (
                (int(sample["panel_width"]) * 3, int(sample["panel_height"]) + 36)
                if name == "comparison"
                else (int(sample["panel_width"]), int(sample["panel_height"]))
            )
            if (int(probe["width"]), int(probe["height"])) != expected_size:
                raise RuntimeError(f"{sample['sample_id']} {name}: {probe}")
        if probes["base_control"]["sha256"] == probes["libero_long"]["sha256"]:
            raise RuntimeError(f"Identical model videos: {sample['sample_id']}")
        checks.append(
            {
                "sample_id": sample["sample_id"],
                "expected_frames": expected_frames,
                "videos": probes,
            }
        )
    report = {"valid": True, "sample_count": len(checks), "samples": checks}
    atomic_json(output_root / "validation.json", report)
    (output_root / "COMPLETE").touch()
    print(json.dumps({"valid": True, "samples": len(checks)}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    render_parser = subparsers.add_parser("render-worker")
    render_parser.add_argument("--output-root", type=Path, required=True)
    render_parser.add_argument(
        "--model", choices=("base_control", "libero_long"), required=True
    )
    render_parser.add_argument("--master-port", type=int, required=True)
    render_parser.add_argument("--force", action="store_true")
    render_parser.set_defaults(func=render_worker)

    concat_parser = subparsers.add_parser("concat")
    concat_parser.add_argument("--output-root", type=Path, required=True)
    concat_parser.set_defaults(func=concat)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output-root", type=Path, required=True)
    validate_parser.set_defaults(func=validate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    args.func(args)


if __name__ == "__main__":
    main()
