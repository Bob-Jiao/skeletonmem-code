#!/usr/bin/env python3
"""Render base/full/LoRA I2VA videos and build per-sample GT comparisons.

The experiment intentionally fixes every factor except the transformer weights:
the base pipeline supplies the tokenizer, text encoder and VAE, while the base
transformer, full-finetune transformer, or a LoRA adapter over base is selected.
Generation seeds are reset *after* model loading and prompt encoding so that
all model workers start from identical video and action noise.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


BASE_MODEL_ROOT = Path(
    "/root/shared/zouyude/ckpts/lingbot-va/lingbot-va-base"
)
BASE_TRANSFORMER = BASE_MODEL_ROOT / "transformer"
FULL_TRANSFORMER = Path(
    "/root/shared/zouyude/train/lingbot-va/libero-all-full-4gpu/"
    "checkpoints/checkpoint_step_20000/transformer"
)
LORA_STEP16000_ROOT = Path(
    "/root/shared/zouyude/train/lingbot-va/"
    "libero-all-lora-r64-action-4gpu/checkpoints/checkpoint_step_16000"
)
LORA_STEP16000_ADAPTER = LORA_STEP16000_ROOT / "adapter"
LIBERO_ROOT = Path("/root/shared/zouyude/data/lingbot-va/libero_all")
ROBOTWIN_ROOT = Path(
    "/root/shared/zouyude/data/robotwin_clean_50_raw/"
    "lerobot_robotwin_eef_clean_50"
)
OUTPUT_FPS = 10
FONT_FILE = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


SAMPLE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "domain": "libero",
        "sample_id": "libero_10_ep000000",
        "dataset": "libero_10_no_noops_lingbot",
        "episode_index": 0,
        "num_chunks": 6,
        "seed": 4200,
    },
    {
        "domain": "libero",
        "sample_id": "libero_goal_ep000000",
        "dataset": "libero_goal_no_noops_lingbot",
        "episode_index": 0,
        "num_chunks": 6,
        "seed": 4201,
    },
    {
        "domain": "libero",
        "sample_id": "libero_object_ep000000",
        "dataset": "libero_object_no_noops_lingbot",
        "episode_index": 0,
        "num_chunks": 6,
        "seed": 4202,
    },
    {
        "domain": "libero",
        "sample_id": "libero_spatial_ep000000",
        "dataset": "libero_spatial_no_noops_lingbot",
        "episode_index": 0,
        "num_chunks": 6,
        "seed": 4203,
    },
    {
        "domain": "robotwin",
        "sample_id": "robotwin_hanging_mug_ep000000",
        "dataset": "hanging_mug-demo_clean_collect_200-50",
        "episode_index": 0,
        "num_chunks": 4,
        "seed": 4300,
    },
    {
        "domain": "robotwin",
        "sample_id": "robotwin_open_laptop_ep000000",
        "dataset": "open_laptop-demo_clean_collect_200-50",
        "episode_index": 0,
        "num_chunks": 4,
        "seed": 4301,
    },
    {
        "domain": "robotwin",
        "sample_id": "robotwin_handover_block_ep000000",
        "dataset": "handover_block-demo_clean_collect_200-50",
        "episode_index": 0,
        "num_chunks": 4,
        "seed": 4302,
    },
    {
        "domain": "robotwin",
        "sample_id": "robotwin_beat_block_hammer_ep000000",
        "dataset": "beat_block_hammer-demo_clean_collect_200-50",
        "episode_index": 0,
        "num_chunks": 4,
        "seed": 4303,
    },
)


def run_command(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    stream = json.loads(output)["streams"][0]
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256(path),
        **stream,
    }


def read_libero_episode(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    with episodes_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["episode_index"]) == episode_index:
                return record
    raise ValueError(f"Episode {episode_index} not found in {episodes_path}")


def read_robotwin_episode(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Preparing RoboTwin samples requires pyarrow") from error

    path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    columns = [
        "episode_index",
        "length",
        "tasks",
        "videos/observation.images.cam_high/from_timestamp",
        "videos/observation.images.cam_high/to_timestamp",
    ]
    table = pq.read_table(path, columns=columns).to_pylist()
    for record in table:
        if int(record["episode_index"]) == episode_index:
            record["tasks"] = list(record["tasks"])
            return record
    raise ValueError(f"Episode {episode_index} not found in {path}")


def extract_first_frame(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            "select=eq(n\\,0)",
            "-frames:v",
            "1",
            str(destination),
        ]
    )


def encode_video(command: list[str], output: Path, expected_frames: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        command
        + [
            "-map",
            "[out]",
            "-frames:v",
            str(expected_frames),
            "-r",
            str(OUTPUT_FPS),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-y",
            str(output),
        ]
    )


def prepare_libero_sample(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    dataset_root = LIBERO_ROOT / spec["dataset"]
    episode = read_libero_episode(dataset_root, int(spec["episode_index"]))
    prompt = str(episode["tasks"][0])
    length = int(episode["length"])
    frame_chunk_size = 4
    expected_frames = 1 + (spec["num_chunks"] * frame_chunk_size - 1) * 4
    if length < expected_frames:
        raise ValueError(
            f"{spec['sample_id']} has {length} frames, needs {expected_frames}"
        )

    source_dir = dataset_root / "videos" / "chunk-000"
    source_videos = {
        "observation.images.agentview_rgb": (
            source_dir / "observation.images.image" / "episode_000000.mp4"
        ),
        "observation.images.eye_in_hand_rgb": (
            source_dir / "observation.images.wrist_image" / "episode_000000.mp4"
        ),
    }
    sample_root = root / "libero" / spec["sample_id"]
    input_root = sample_root / "input"
    for target_key, source in source_videos.items():
        extract_first_frame(source, input_root / f"{target_key}.png")

    gt_path = sample_root / "gt.mp4"
    filter_graph = (
        f"[0:v]trim=start_frame=0:end_frame={expected_frames},"
        f"setpts=N/({OUTPUT_FPS}*TB),scale=128:128:flags=lanczos[a];"
        f"[1:v]trim=start_frame=0:end_frame={expected_frames},"
        f"setpts=N/({OUTPUT_FPS}*TB),scale=128:128:flags=lanczos[b];"
        "[a][b]hstack=inputs=2[out]"
    )
    encode_video(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_videos["observation.images.agentview_rgb"]),
            "-i",
            str(source_videos["observation.images.eye_in_hand_rgb"]),
            "-filter_complex",
            filter_graph,
        ],
        gt_path,
        expected_frames,
    )
    return {
        **spec,
        "prompt": prompt,
        "episode_length": length,
        "source_fps": 20,
        "source_frame_stride": 1,
        "frame_chunk_size": frame_chunk_size,
        "expected_output_frames": expected_frames,
        "panel_width": 256,
        "panel_height": 128,
        "input_dir": str(input_root.resolve()),
        "gt_path": str(gt_path.resolve()),
        "gt": ffprobe(gt_path),
        "source_videos": {
            key: file_record(path) for key, path in source_videos.items()
        },
    }


def prepare_robotwin_sample(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    dataset_root = ROBOTWIN_ROOT / spec["dataset"]
    episode = read_robotwin_episode(dataset_root, int(spec["episode_index"]))
    prompt = str(episode["tasks"][0])
    length = int(episode["length"])
    frame_chunk_size = 2
    expected_frames = 1 + (spec["num_chunks"] * frame_chunk_size - 1) * 4
    source_stride = 4
    required_source_frames = 1 + (expected_frames - 1) * source_stride
    if length < required_source_frames:
        raise ValueError(
            f"{spec['sample_id']} has {length} source frames, needs "
            f"{required_source_frames} at stride {source_stride}"
        )

    video_root = dataset_root / "videos"
    camera_keys = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    source_videos = {
        key: video_root / key / "chunk-000" / "file-000.mp4"
        for key in camera_keys
    }
    sample_root = root / "robotwin" / spec["sample_id"]
    input_root = sample_root / "input"
    for key, source in source_videos.items():
        extract_first_frame(source, input_root / f"{key}.png")

    gt_path = sample_root / "gt.mp4"
    select = f"select='not(mod(n\\,{source_stride}))',trim=end_frame={expected_frames}"
    filter_graph = (
        f"[0:v]{select},setpts=N/({OUTPUT_FPS}*TB),"
        "scale=320:256:flags=lanczos[high];"
        f"[1:v]{select},setpts=N/({OUTPUT_FPS}*TB),"
        "scale=160:128:flags=lanczos[left];"
        f"[2:v]{select},setpts=N/({OUTPUT_FPS}*TB),"
        "scale=160:128:flags=lanczos[right];"
        "[left][right]hstack=inputs=2[wrists];"
        "[wrists][high]vstack=inputs=2[out]"
    )
    encode_video(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_videos[camera_keys[0]]),
            "-i",
            str(source_videos[camera_keys[1]]),
            "-i",
            str(source_videos[camera_keys[2]]),
            "-filter_complex",
            filter_graph,
        ],
        gt_path,
        expected_frames,
    )
    return {
        **spec,
        "prompt": prompt,
        "episode_length": length,
        "source_fps": 50,
        "source_frame_stride": source_stride,
        "frame_chunk_size": frame_chunk_size,
        "expected_output_frames": expected_frames,
        "panel_width": 320,
        "panel_height": 384,
        "input_dir": str(input_root.resolve()),
        "gt_path": str(gt_path.resolve()),
        "gt": ffprobe(gt_path),
        "source_videos": {
            key: file_record(path) for key, path in source_videos.items()
        },
    }


def prepare(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}; pass --force to rebuild"
        )

    samples = []
    for spec in SAMPLE_SPECS:
        if spec["domain"] == "libero":
            sample = prepare_libero_sample(root, dict(spec))
        else:
            sample = prepare_robotwin_sample(root, dict(spec))
        atomic_json(
            root / sample["domain"] / sample["sample_id"] / "sample.json",
            sample,
        )
        samples.append(sample)

    manifest = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "paired in-domain/out-of-domain video generation before and after LIBERO post-training",
        "output_fps": OUTPUT_FPS,
        "base_model_root": file_record(BASE_MODEL_ROOT / "transformer" / "config.json"),
        "models": {
            "base": {
                "transformer_path": str(BASE_TRANSFORMER.resolve()),
                "config": file_record(BASE_TRANSFORMER / "config.json"),
            },
            "full": {
                "transformer_path": str(FULL_TRANSFORMER.resolve()),
                "config": file_record(FULL_TRANSFORMER / "config.json"),
                "checkpoint_step": 20000,
                "training_run": "libero-all-full-4gpu",
            },
            "lora16000": {
                "transformer_path": str(BASE_TRANSFORMER.resolve()),
                "lora_adapter_path": str(LORA_STEP16000_ADAPTER.resolve()),
                "adapter_config": file_record(
                    LORA_STEP16000_ADAPTER / "adapter_config.json"
                ),
                "adapter_weights": file_record(
                    LORA_STEP16000_ADAPTER / "pytorch_lora_weights.safetensors"
                ),
                "checkpoint_step": 16000,
                "training_run": "libero-all-lora-r64-action-4gpu",
            },
        },
        "samples": samples,
    }
    atomic_json(manifest_path, manifest)
    print(f"Prepared {len(samples)} samples: {manifest_path}")


def seed_generation(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError("Rendering requires the lingbot-va Python environment") from error
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def render_worker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist
    from diffusers.utils import export_to_video
    from diffusers.video_processor import VideoProcessor

    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.wan_va_server import VA_Server

    root = args.output_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    samples = [
        sample for sample in manifest["samples"] if sample["domain"] == args.domain
    ]
    if not samples:
        raise ValueError(f"No samples for domain {args.domain}")
    model_spec = manifest.get("models", {}).get(args.model)
    # Older prepared manifests predate the LoRA comparison.  Keep those exact
    # samples reusable instead of rebuilding their GT and input observations.
    if model_spec is None and args.model == "lora16000":
        model_spec = {
            "transformer_path": str(BASE_TRANSFORMER),
            "lora_adapter_path": str(LORA_STEP16000_ADAPTER),
        }
    if model_spec is None:
        raise KeyError(f"Model {args.model!r} is not present in the manifest")
    transformer_path = Path(model_spec["transformer_path"]).resolve()
    lora_adapter_path = model_spec.get("lora_adapter_path")
    if lora_adapter_path is not None:
        lora_adapter_path = str(Path(lora_adapter_path).resolve())

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "render-worker expects exactly one visible GPU; set CUDA_VISIBLE_DEVICES"
        )
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    init_distributed(1, 0, 0)

    config_name = "libero_all" if args.domain == "libero" else "robotwin"
    config = copy.deepcopy(VA_CONFIGS[config_name])
    config.wan22_pretrained_model_name_or_path = str(BASE_MODEL_ROOT)
    config.transformer_model_name_or_path = str(transformer_path)
    config.lora_adapter_path = lora_adapter_path
    config.infer_mode = "i2va"
    config.enable_offload = False
    config.save_debug_data = False
    config.save_root = str(root / args.domain / f"_{args.model}_worker")
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1

    worker_started = time.time()
    print(
        f"Loading {args.model} for {args.domain}: transformer={transformer_path} "
        f"adapter={lora_adapter_path}",
        flush=True,
    )
    server = VA_Server(config)
    server.video_processor = VideoProcessor(vae_scale_factor=1)
    worker_results = []

    for sample in samples:
        sample_root = root / args.domain / sample["sample_id"]
        output_dir = sample_root / args.model
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = output_dir / "generated.mp4"
        result_path = output_dir / "render.json"
        if output_video.is_file() and result_path.is_file() and not args.force:
            existing = json.loads(result_path.read_text())
            if existing.get("status") == "complete":
                print(f"Skipping complete output: {output_video}", flush=True)
                worker_results.append(existing)
                continue

        config.input_img_path = sample["input_dir"]
        config.num_chunks_to_infer = int(sample["num_chunks"])
        config.prompt = sample["prompt"]
        config.save_root = str(output_dir)
        server.save_root = str(output_dir)
        started = time.time()
        result = {
            "status": "running",
            "domain": args.domain,
            "model": args.model,
            "sample_id": sample["sample_id"],
            "prompt": sample["prompt"],
            "seed": int(sample["seed"]),
            "num_chunks": int(sample["num_chunks"]),
            "transformer_path": str(transformer_path),
            "lora_adapter_path": lora_adapter_path,
            "base_model_root": str(BASE_MODEL_ROOT),
            "started_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)
            ),
        }
        atomic_json(result_path, result)
        print(
            f"Rendering {args.model}/{args.domain}/{sample['sample_id']} "
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
        # ``generate`` normally supplies this guard, but this experiment invokes
        # the lower-level streaming helpers so one model load can serve several
        # fixed samples.  Decoding must therefore disable autograd explicitly.
        with torch.no_grad():
            decoded_video = server.decode_one_video(pred_latent, "np")[0]
        export_to_video(decoded_video, str(output_video), fps=OUTPUT_FPS)
        actual_frames = len(decoded_video)
        expected_frames = int(sample["expected_output_frames"])
        if actual_frames != expected_frames:
            raise RuntimeError(
                f"Decoded {actual_frames} frames for {sample['sample_id']}, "
                f"expected {expected_frames}"
            )

        server.transformer.clear_cache(server.cache_name)
        server.streaming_vae.clear_cache()
        if server.streaming_vae_half is not None:
            server.streaming_vae_half.clear_cache()
        del pred_latent, decoded_video, latent_chunks
        torch.cuda.empty_cache()

        result.update(
            status="complete",
            elapsed_seconds=time.time() - started,
            completed_at_utc=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            video=ffprobe(output_video),
        )
        atomic_json(result_path, result)
        worker_results.append(result)

    worker_summary = {
        "status": "complete",
        "domain": args.domain,
        "model": args.model,
        "elapsed_seconds": time.time() - worker_started,
        "results": worker_results,
    }
    atomic_json(root / f"worker_{args.domain}_{args.model}.json", worker_summary)
    del server
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()


def comparison_filter(
    width: int,
    height: int,
    expected_frames: int,
    labels: tuple[str, str, str] = ("GT", "BASE", "POST-TRAINED FULL"),
) -> str:
    streams = []
    for index, label in enumerate(labels):
        streams.append(
            f"[{index}:v]trim=start_frame=0:end_frame={expected_frames},"
            f"setpts=N/({OUTPUT_FPS}*TB),fps={OUTPUT_FPS},"
            f"scale={width}:{height}:flags=lanczos,"
            f"pad=iw:ih+36:0:36:black,"
            f"drawtext=fontfile={FONT_FILE}:text='{label}':"
            "fontcolor=white:fontsize=22:x=(w-text_w)/2:y=7"
            f"[v{index}]"
        )
    return ";".join(streams) + ";[v0][v1][v2]hstack=inputs=3[out]"


def concat(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    results = []
    for sample in manifest["samples"]:
        sample_root = root / sample["domain"] / sample["sample_id"]
        gt_path = Path(sample["gt_path"])
        base_path = sample_root / "base" / "generated.mp4"
        full_path = sample_root / "full" / "generated.mp4"
        for path in (gt_path, base_path, full_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        comparison_path = sample_root / "comparison_gt_base_full.mp4"
        expected_frames = int(sample["expected_output_frames"])
        encode_video(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(gt_path),
                "-i",
                str(base_path),
                "-i",
                str(full_path),
                "-filter_complex",
                comparison_filter(
                    int(sample["panel_width"]),
                    int(sample["panel_height"]),
                    expected_frames,
                ),
            ],
            comparison_path,
            expected_frames,
        )
        record = {
            "domain": sample["domain"],
            "sample_id": sample["sample_id"],
            "prompt": sample["prompt"],
            "seed": sample["seed"],
            "gt": ffprobe(gt_path),
            "base": ffprobe(base_path),
            "full": ffprobe(full_path),
            "comparison": ffprobe(comparison_path),
        }
        atomic_json(sample_root / "comparison.json", record)
        results.append(record)
    atomic_json(
        root / "results.json",
        {
            "status": "complete",
            "completed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "comparisons": results,
        },
    )
    (root / "COMPLETE").touch()
    print(f"Created {len(results)} per-sample comparison videos under {root}")


def concat_lora(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    source_manifest_path = root / "manifest.json"
    manifest = json.loads(source_manifest_path.read_text())
    results = []
    for sample in manifest["samples"]:
        sample_root = root / sample["domain"] / sample["sample_id"]
        gt_path = Path(sample["gt_path"])
        base_path = sample_root / "base" / "generated.mp4"
        lora_path = sample_root / "lora16000" / "generated.mp4"
        for path in (gt_path, base_path, lora_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        comparison_path = sample_root / "comparison_gt_base_lora16000.mp4"
        expected_frames = int(sample["expected_output_frames"])
        encode_video(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(gt_path),
                "-i",
                str(base_path),
                "-i",
                str(lora_path),
                "-filter_complex",
                comparison_filter(
                    int(sample["panel_width"]),
                    int(sample["panel_height"]),
                    expected_frames,
                    labels=("GT", "BASE", "LORA-16000"),
                ),
            ],
            comparison_path,
            expected_frames,
        )
        record = {
            "domain": sample["domain"],
            "sample_id": sample["sample_id"],
            "prompt": sample["prompt"],
            "seed": sample["seed"],
            "gt": ffprobe(gt_path),
            "base": ffprobe(base_path),
            "lora16000": ffprobe(lora_path),
            "comparison": ffprobe(comparison_path),
        }
        atomic_json(sample_root / "comparison_gt_base_lora16000.json", record)
        results.append(record)
    adapter_config_path = LORA_STEP16000_ADAPTER / "adapter_config.json"
    adapter_weights_path = (
        LORA_STEP16000_ADAPTER / "pytorch_lora_weights.safetensors"
    )
    checkpoint_manifest_path = LORA_STEP16000_ROOT / "manifest.json"
    atomic_json(
        root / "results_gt_base_lora16000.json",
        {
            "status": "complete",
            "completed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "lora_checkpoint": str(LORA_STEP16000_ROOT.resolve()),
            "lora_adapter": str(LORA_STEP16000_ADAPTER.resolve()),
            "source_manifest": {
                **file_record(source_manifest_path),
                "sha256": sha256(source_manifest_path),
            },
            "checkpoint_manifest": json.loads(
                checkpoint_manifest_path.read_text()
            ),
            "checkpoint_manifest_file": {
                **file_record(checkpoint_manifest_path),
                "sha256": sha256(checkpoint_manifest_path),
            },
            "adapter_config": json.loads(adapter_config_path.read_text()),
            "adapter_config_file": {
                **file_record(adapter_config_path),
                "sha256": sha256(adapter_config_path),
            },
            "adapter_weights_file": {
                **file_record(adapter_weights_path),
                "sha256": sha256(adapter_weights_path),
            },
            "comparisons": results,
        },
    )
    (root / "LORA16000_COMPLETE").touch()
    print(
        f"Created {len(results)} per-sample GT/base/LoRA comparisons under {root}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    render_parser = subparsers.add_parser("render-worker")
    render_parser.add_argument("--output-root", type=Path, required=True)
    render_parser.add_argument("--domain", choices=("libero", "robotwin"), required=True)
    render_parser.add_argument(
        "--model", choices=("base", "full", "lora16000"), required=True
    )
    render_parser.add_argument("--master-port", type=int, required=True)
    render_parser.add_argument("--force", action="store_true")
    render_parser.set_defaults(func=render_worker)

    concat_parser = subparsers.add_parser("concat")
    concat_parser.add_argument("--output-root", type=Path, required=True)
    concat_parser.set_defaults(func=concat)

    concat_lora_parser = subparsers.add_parser("concat-lora")
    concat_lora_parser.add_argument("--output-root", type=Path, required=True)
    concat_lora_parser.set_defaults(func=concat_lora)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    args.func(args)


if __name__ == "__main__":
    main()
