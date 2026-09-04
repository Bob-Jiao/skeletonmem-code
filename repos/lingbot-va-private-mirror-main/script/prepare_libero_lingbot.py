#!/usr/bin/env python3
"""Prepare LIBERO LeRobot videos for LingBot-VA latent training.

The script intentionally mirrors the encoding path used by ``VA_Server``:

* UMT5 embeddings are padded to 512 tokens after the valid tokens.
* Episodes are truncated to ``4k + 1`` video frames.
* The first frame is encoded alone, then frames are streamed in multiples of 4.
* VAE posterior means are normalized with the Wan VAE channel statistics.

Two subcommands are provided:

``prepare``
    Build metadata/action_config, encode text/empty embeddings, and encode video
    latents. ``--setup-only`` followed by two ``--encode-only`` processes can
    be used to split the work across two GPUs.

``validate``
    Re-encode official LingBot-VA LIBERO-Long videos and compare the generated
    text embeddings and video latents with the released ``.pth`` tensors.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.modules.utils import (  # noqa: E402
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)


SOURCE_CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
OUTPUT_CAMERA_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)
TEXT_CACHE_NAME = "text_embeddings.pt"
MANIFEST_NAME = "preprocess_manifest.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def safe_directory_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)


def release_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def encode_texts(
    model_root: Path,
    texts: list[str],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    tokenizer = load_tokenizer(str(model_root / "tokenizer"))
    text_encoder = load_text_encoder(
        str(model_root / "text_encoder"),
        torch_dtype=dtype,
        torch_device=device,
    ).eval()
    # The released checkpoint stores only ``shared.weight``. Transformers
    # 4.55 ties the encoder token embedding to it automatically, whereas newer
    # Transformers releases may materialize a missing, zero-initialized
    # ``encoder.embed_tokens`` parameter. Tie it explicitly so preprocessing
    # remains compatible with the LingBot-VA release environment.
    if (
        hasattr(text_encoder, "shared")
        and hasattr(text_encoder, "encoder")
        and text_encoder.encoder.embed_tokens is not text_encoder.shared
    ):
        text_encoder.encoder.embed_tokens = text_encoder.shared

    outputs: dict[str, torch.Tensor] = {}
    for text in texts:
        cleaned = prompt_clean(text)
        tokenized = tokenizer(
            [cleaned],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        sequence_length = int(attention_mask.gt(0).sum().item())
        hidden = text_encoder(input_ids, attention_mask).last_hidden_state
        hidden = hidden.to(dtype=dtype)[0, :sequence_length]
        hidden = torch.cat(
            [hidden, hidden.new_zeros(512 - sequence_length, hidden.shape[-1])],
            dim=0,
        )
        outputs[text] = hidden.cpu().contiguous()

    release_cuda(text_encoder, tokenizer)
    return outputs


def decode_rgb_video(path: Path) -> np.ndarray:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"No video frames decoded from {path}")
    return np.stack(frames, axis=0)


def resize_frames(
    frames: np.ndarray,
    height: int,
    width: int,
    device: torch.device,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Return uint8-like float frames as ``T,C,H,W`` on ``device``."""
    resized: list[torch.Tensor] = []
    for start in range(0, len(frames), chunk_size):
        chunk = torch.from_numpy(frames[start : start + chunk_size].copy())
        chunk = chunk.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        if chunk.shape[-2:] != (height, width):
            chunk = F.interpolate(
                chunk,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        resized.append(chunk)
    return torch.cat(resized, dim=0)


def usable_frame_count(frame_count: int) -> int:
    if frame_count < 1:
        raise ValueError("An episode must contain at least one frame")
    return 1 + ((frame_count - 1) // 4) * 4


@torch.inference_mode()
def encode_video_pair(
    vae: torch.nn.Module,
    streaming_vae: WanVAEStreamingWrapper,
    video_paths: tuple[Path, Path],
    device: torch.device,
    height: int,
    width: int,
    temporal_chunk_size: int,
) -> tuple[torch.Tensor, int, int]:
    if temporal_chunk_size < 4 or temporal_chunk_size % 4:
        raise ValueError("temporal_chunk_size must be a positive multiple of 4")

    decoded = [decode_rgb_video(path) for path in video_paths]
    source_frame_count = min(len(frames) for frames in decoded)
    frame_count = usable_frame_count(source_frame_count)
    resized = [
        resize_frames(frames[:frame_count], height, width, device)
        for frames in decoded
    ]
    video = torch.stack(
        [frames.permute(1, 0, 2, 3) for frames in resized], dim=0
    )
    video = (video / 255.0 * 2.0 - 1.0).to(dtype=torch.bfloat16)

    streaming_vae.clear_cache()
    encoded_chunks: list[torch.Tensor] = []

    def encode_chunk(chunk: torch.Tensor) -> None:
        encoded = streaming_vae.encode_chunk(chunk)
        mu, _ = torch.chunk(encoded, 2, dim=1)
        mean = torch.tensor(
            vae.config.latents_mean, device=mu.device, dtype=torch.float32
        ).view(1, -1, 1, 1, 1)
        inverse_std = (
            1.0
            / torch.tensor(
                vae.config.latents_std, device=mu.device, dtype=torch.float32
            ).view(1, -1, 1, 1, 1)
        )
        normalized = ((mu.float() - mean) * inverse_std).to(mu.dtype)
        encoded_chunks.append(normalized.cpu())

    encode_chunk(video[:, :, :1])
    for start in range(1, frame_count, temporal_chunk_size):
        encode_chunk(video[:, :, start : start + temporal_chunk_size])

    latents = torch.cat(encoded_chunks, dim=2).contiguous()
    expected_latent_frames = 1 + (frame_count - 1) // 4
    if latents.shape[2] != expected_latent_frames:
        raise RuntimeError(
            f"Unexpected latent frames {latents.shape[2]}, "
            f"expected {expected_latent_frames} for {frame_count} video frames"
        )
    return latents, frame_count, source_frame_count


def latent_record(
    latent: torch.Tensor,
    text_embedding: torch.Tensor,
    text: str,
    frame_count: int,
    start_frame: int,
    end_frame: int,
    fps: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    # Input is C,F,H,W; the released dataset stores (F*H*W),C.
    flattened = rearrange(latent, "c f h w -> (f h w) c").contiguous()
    return {
        "latent": flattened,
        "latent_num_frames": int(latent.shape[1]),
        "latent_height": int(latent.shape[2]),
        "latent_width": int(latent.shape[3]),
        "video_num_frames": int(frame_count),
        "video_height": int(height),
        "video_width": int(width),
        "text_emb": text_embedding,
        "text": text,
        "frame_ids": np.arange(start_frame, start_frame + frame_count),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "fps": int(fps),
        "ori_fps": int(fps),
    }


def setup_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_meta = source_root / "meta"
    output_meta = output_root / "meta"

    tasks = read_jsonl(source_meta / "tasks.jsonl")
    tasks.sort(key=lambda value: int(value["task_index"]))
    selected_tasks = [str(value["task"]) for value in tasks[: args.task_limit]]
    selected_task_set = set(selected_tasks)

    episodes = read_jsonl(source_meta / "episodes.jsonl")
    selected_episodes: list[dict[str, Any]] = []
    updated_episodes: list[dict[str, Any]] = []
    for original in episodes:
        episode = dict(original)
        task = str(episode["tasks"][0])
        if task in selected_task_set:
            episode["action_config"] = [
                {
                    "start_frame": 0,
                    "end_frame": int(episode["length"]),
                    "action_text": task,
                    "skill": "",
                }
            ]
            selected_episodes.append(
                {
                    "episode_index": int(episode["episode_index"]),
                    "length": int(episode["length"]),
                    "task": task,
                }
            )
        else:
            # The LingBot loader iterates this list. An empty list cleanly
            # excludes unselected episodes while keeping LeRobot indices intact.
            episode["action_config"] = []
        updated_episodes.append(episode)

    output_meta.mkdir(parents=True, exist_ok=True)
    for source_file in source_meta.iterdir():
        if source_file.name != "episodes.jsonl" and source_file.is_file():
            shutil.copy2(source_file, output_meta / source_file.name)
    write_jsonl(output_meta / "episodes.jsonl", updated_episodes)
    safe_directory_symlink(source_root / "data", output_root / "data")
    safe_directory_symlink(source_root / "videos", output_root / "videos")

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "model_root": str(args.model_root.resolve()),
        "source_camera_keys": list(SOURCE_CAMERA_KEYS),
        "output_camera_keys": list(OUTPUT_CAMERA_KEYS),
        "height": args.height,
        "width": args.width,
        "fps": int(json.loads((source_meta / "info.json").read_text())["fps"]),
        "selected_tasks": selected_tasks,
        "selected_episode_count": len(selected_episodes),
        "selected_episodes": selected_episodes,
    }
    write_json(output_root / MANIFEST_NAME, manifest)

    text_embeddings = encode_texts(
        args.model_root,
        [""] + selected_tasks,
        torch.device(args.device),
    )
    torch.save(text_embeddings, output_root / TEXT_CACHE_NAME)
    torch.save(text_embeddings[""], output_root / "empty_emb.pt")
    return manifest


def load_manifest(output_root: Path) -> dict[str, Any]:
    path = output_root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing setup manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def encode_prepared_dataset(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    output_root = args.output_root.resolve()
    source_root = Path(manifest["source_root"])
    text_embeddings: dict[str, torch.Tensor] = torch.load(
        output_root / TEXT_CACHE_NAME,
        map_location="cpu",
        weights_only=False,
    )

    device = torch.device(args.device)
    vae = load_vae(
        str(args.model_root / "vae"),
        torch_dtype=torch.bfloat16,
        torch_device=device,
    ).eval()
    streaming_vae = WanVAEStreamingWrapper(vae)

    selected = manifest["selected_episodes"]
    selected = [
        episode
        for position, episode in enumerate(selected)
        if position % args.num_shards == args.shard_index
    ]
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]

    for ordinal, episode in enumerate(selected, start=1):
        episode_index = int(episode["episode_index"])
        end_frame = int(episode["length"])
        task = str(episode["task"])
        source_paths = tuple(
            source_root
            / "videos"
            / "chunk-000"
            / camera_key
            / f"episode_{episode_index:06d}.mp4"
            for camera_key in SOURCE_CAMERA_KEYS
        )
        for source_path in source_paths:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)

        output_paths = tuple(
            output_root
            / "latents"
            / "chunk-000"
            / camera_key
            / f"episode_{episode_index:06d}_0_{end_frame}.pth"
            for camera_key in OUTPUT_CAMERA_KEYS
        )
        if not args.overwrite and all(path.is_file() for path in output_paths):
            print(f"[{ordinal}/{len(selected)}] skip episode {episode_index}: exists")
            continue

        latents, frame_count, source_frame_count = encode_video_pair(
            vae,
            streaming_vae,
            source_paths,
            device,
            int(manifest["height"]),
            int(manifest["width"]),
            args.temporal_chunk_size,
        )
        for camera_index, output_path in enumerate(output_paths):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            record = latent_record(
                latents[camera_index],
                text_embeddings[task],
                task,
                frame_count,
                0,
                end_frame,
                int(manifest["fps"]),
                int(manifest["height"]),
                int(manifest["width"]),
            )
            torch.save(record, output_path)
        print(
            f"[{ordinal}/{len(selected)}] episode {episode_index}: "
            f"source_frames={source_frame_count}, encoded_frames={frame_count}, "
            f"latent_frames={latents.shape[2]}"
        )

    release_cuda(vae, streaming_vae)


def find_reference_latent(
    reference_root: Path, camera_key: str, episode_index: int
) -> Path:
    matches = list(
        (
            reference_root
            / "latents"
            / "chunk-000"
            / camera_key
        ).glob(f"episode_{episode_index:06d}_0_*.pth")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one reference latent for episode {episode_index}, got {matches}"
        )
    return matches[0]


def tensor_metrics(generated: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "generated_shape": list(generated.shape),
        "reference_shape": list(reference.shape),
        "generated_dtype": str(generated.dtype),
        "reference_dtype": str(reference.dtype),
    }
    if generated.shape != reference.shape:
        metrics["shape_match"] = False
        return metrics
    generated_float = generated.float().flatten()
    reference_float = reference.float().flatten()
    difference = generated_float - reference_float
    metrics.update(
        {
            "shape_match": True,
            "mae": float(difference.abs().mean()),
            "rmse": float(torch.sqrt(torch.mean(difference.square()))),
            "max_abs": float(difference.abs().max()),
            "cosine_similarity": float(
                F.cosine_similarity(
                    generated_float.unsqueeze(0), reference_float.unsqueeze(0)
                ).item()
            ),
            "exact_equal": bool(torch.equal(generated, reference)),
        }
    )
    return metrics


def validate_reference(args: argparse.Namespace) -> None:
    reference_root = args.reference_root.resolve()
    device = torch.device(args.device)
    references: list[dict[str, Any]] = []
    prompts: list[str] = []
    for episode_index in args.episodes:
        reference_path = find_reference_latent(
            reference_root, OUTPUT_CAMERA_KEYS[0], episode_index
        )
        record = torch.load(reference_path, map_location="cpu", weights_only=False)
        references.append(
            {
                "episode_index": episode_index,
                "end_frame": int(record["end_frame"]),
                "text": str(record["text"]),
                "fps": int(record["fps"]),
                "reference_path": reference_path,
            }
        )
        prompts.append(str(record["text"]))

    text_embeddings = encode_texts(
        args.model_root,
        list(dict.fromkeys([""] + prompts)),
        device,
    )
    vae = load_vae(
        str(args.model_root / "vae"),
        torch_dtype=torch.bfloat16,
        torch_device=device,
    ).eval()
    streaming_vae = WanVAEStreamingWrapper(vae)

    report: dict[str, Any] = {
        "model_root": str(args.model_root.resolve()),
        "reference_root": str(reference_root),
        "height": args.height,
        "width": args.width,
        "temporal_chunk_size": args.temporal_chunk_size,
        "episodes": [],
    }
    reencoded_root = args.reencoded_root.resolve()

    for reference in references:
        episode_index = int(reference["episode_index"])
        end_frame = int(reference["end_frame"])
        text = str(reference["text"])
        video_paths = tuple(
            reference_root
            / "videos"
            / "chunk-000"
            / camera_key
            / f"episode_{episode_index:06d}.mp4"
            for camera_key in OUTPUT_CAMERA_KEYS
        )
        latents, frame_count, source_frame_count = encode_video_pair(
            vae,
            streaming_vae,
            video_paths,
            device,
            args.height,
            args.width,
            args.temporal_chunk_size,
        )

        episode_report: dict[str, Any] = {
            "episode_index": episode_index,
            "task": text,
            "source_video_frames": source_frame_count,
            "encoded_video_frames": frame_count,
            "cameras": {},
        }
        for camera_index, camera_key in enumerate(OUTPUT_CAMERA_KEYS):
            reference_path = find_reference_latent(
                reference_root, camera_key, episode_index
            )
            reference_record = torch.load(
                reference_path, map_location="cpu", weights_only=False
            )
            generated_record = latent_record(
                latents[camera_index],
                text_embeddings[text],
                text,
                frame_count,
                0,
                end_frame,
                int(reference["fps"]),
                args.height,
                args.width,
            )
            output_path = (
                reencoded_root
                / "latents"
                / "chunk-000"
                / camera_key
                / reference_path.name
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(generated_record, output_path)
            episode_report["cameras"][camera_key] = {
                "latent": tensor_metrics(
                    generated_record["latent"], reference_record["latent"]
                ),
                "text_emb": tensor_metrics(
                    generated_record["text_emb"], reference_record["text_emb"]
                ),
                "metadata_match": {
                    key: generated_record[key] == reference_record[key]
                    for key in (
                        "latent_num_frames",
                        "latent_height",
                        "latent_width",
                        "video_num_frames",
                        "video_height",
                        "video_width",
                        "text",
                        "start_frame",
                        "end_frame",
                        "fps",
                        "ori_fps",
                    )
                },
            }
        report["episodes"].append(episode_report)
        print(
            f"validated episode {episode_index}: "
            f"frames={frame_count}, latent_frames={latents.shape[2]}"
        )

    write_json(args.report_path.resolve(), report)
    torch.save(text_embeddings[""], reencoded_root / "empty_emb.pt")
    release_cuda(vae, streaming_vae)
    print(f"validation report: {args.report_path.resolve()}")


def audit_prepared_dataset(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    output_root = args.output_root.resolve()
    manifest = load_manifest(output_root)
    source_root = Path(manifest["source_root"])
    text_embeddings: dict[str, torch.Tensor] = torch.load(
        output_root / TEXT_CACHE_NAME,
        map_location="cpu",
        weights_only=False,
    )

    errors: list[str] = []
    action_arrays: list[np.ndarray] = []
    task_episode_counts: dict[str, int] = {}
    latent_value_count = 0
    latent_sum = 0.0
    latent_square_sum = 0.0
    latent_min = math.inf
    latent_max = -math.inf

    selected_episodes = manifest["selected_episodes"]
    for episode in selected_episodes:
        episode_index = int(episode["episode_index"])
        end_frame = int(episode["length"])
        task = str(episode["task"])
        task_episode_counts[task] = task_episode_counts.get(task, 0) + 1

        parquet_path = (
            source_root
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        actions = np.asarray(
            pq.read_table(parquet_path, columns=["action"])["action"].to_pylist(),
            dtype=np.float32,
        )
        if actions.shape != (end_frame, 7):
            errors.append(
                f"episode {episode_index}: action shape {actions.shape}, "
                f"expected {(end_frame, 7)}"
            )
        action_arrays.append(actions)

        expected_video_frames = usable_frame_count(end_frame)
        expected_latent_frames = 1 + (expected_video_frames - 1) // 4
        for camera_key in OUTPUT_CAMERA_KEYS:
            latent_path = (
                output_root
                / "latents"
                / "chunk-000"
                / camera_key
                / f"episode_{episode_index:06d}_0_{end_frame}.pth"
            )
            if not latent_path.is_file():
                errors.append(f"missing latent: {latent_path}")
                continue
            record = torch.load(latent_path, map_location="cpu", weights_only=False)
            expected_shape = (expected_latent_frames * 8 * 8, 48)
            if tuple(record["latent"].shape) != expected_shape:
                errors.append(
                    f"{latent_path.name}: latent shape {tuple(record['latent'].shape)}, "
                    f"expected {expected_shape}"
                )
            if record["latent"].dtype != torch.bfloat16:
                errors.append(f"{latent_path.name}: latent dtype {record['latent'].dtype}")
            if tuple(record["text_emb"].shape) != (512, 4096):
                errors.append(
                    f"{latent_path.name}: text shape {tuple(record['text_emb'].shape)}"
                )
            if record["text_emb"].dtype != torch.bfloat16:
                errors.append(f"{latent_path.name}: text dtype {record['text_emb'].dtype}")
            if not torch.equal(record["text_emb"], text_embeddings[task]):
                errors.append(f"{latent_path.name}: cached text embedding mismatch")
            if int(record["video_num_frames"]) != expected_video_frames:
                errors.append(
                    f"{latent_path.name}: video_num_frames="
                    f"{record['video_num_frames']}, expected {expected_video_frames}"
                )
            if int(record["latent_num_frames"]) != expected_latent_frames:
                errors.append(
                    f"{latent_path.name}: latent_num_frames="
                    f"{record['latent_num_frames']}, expected {expected_latent_frames}"
                )
            expected_frame_ids = np.arange(expected_video_frames)
            if not np.array_equal(record["frame_ids"], expected_frame_ids):
                errors.append(f"{latent_path.name}: frame_ids mismatch")
            for key, expected in (
                ("latent_height", 8),
                ("latent_width", 8),
                ("video_height", int(manifest["height"])),
                ("video_width", int(manifest["width"])),
                ("text", task),
                ("start_frame", 0),
                ("end_frame", end_frame),
                ("fps", int(manifest["fps"])),
                ("ori_fps", int(manifest["fps"])),
            ):
                if record[key] != expected:
                    errors.append(
                        f"{latent_path.name}: {key}={record[key]!r}, expected {expected!r}"
                    )

            latent_float = record["latent"].float()
            if not torch.isfinite(latent_float).all():
                errors.append(f"{latent_path.name}: non-finite latent values")
            latent_value_count += latent_float.numel()
            latent_sum += float(latent_float.sum())
            latent_square_sum += float(latent_float.square().sum())
            latent_min = min(latent_min, float(latent_float.min()))
            latent_max = max(latent_max, float(latent_float.max()))

    all_actions = np.concatenate(action_arrays, axis=0)
    q01 = np.quantile(all_actions, 0.01, axis=0).tolist()
    q99 = np.quantile(all_actions, 0.99, axis=0).tolist()
    latent_mean = latent_sum / latent_value_count
    latent_variance = max(
        latent_square_sum / latent_value_count - latent_mean * latent_mean, 0.0
    )
    actual_latent_files = len(list((output_root / "latents").rglob("*.pth")))
    expected_latent_files = len(selected_episodes) * len(OUTPUT_CAMERA_KEYS)
    if actual_latent_files != expected_latent_files:
        errors.append(
            f"latent file count {actual_latent_files}, expected {expected_latent_files}"
        )

    report = {
        "passed": not errors,
        "errors": errors,
        "selected_tasks": manifest["selected_tasks"],
        "task_episode_counts": task_episode_counts,
        "selected_episode_count": len(selected_episodes),
        "latent_files": {
            "actual": actual_latent_files,
            "expected": expected_latent_files,
        },
        "tensor_schema": {
            "latent_dtype": "torch.bfloat16",
            "latent_channels": 48,
            "latent_height": 8,
            "latent_width": 8,
            "text_emb_shape": [512, 4096],
            "text_emb_dtype": "torch.bfloat16",
        },
        "latent_statistics": {
            "count": latent_value_count,
            "mean": latent_mean,
            "std": math.sqrt(latent_variance),
            "min": latent_min,
            "max": latent_max,
        },
        "action_statistics": {
            "frame_count": int(all_actions.shape[0]),
            "q01_7d": q01,
            "q99_7d": q99,
            "norm_stat_q01_30d": q01 + [0.0] * 23,
            "norm_stat_q99_30d": q99 + [0.0] * 23,
        },
    }
    write_json(args.report_path.resolve(), report)
    print(
        f"audit {'passed' if report['passed'] else 'failed'}: "
        f"episodes={len(selected_episodes)}, latent_files={actual_latent_files}, "
        f"errors={len(errors)}"
    )
    print(f"audit report: {args.report_path.resolve()}")
    if errors:
        raise RuntimeError(f"Prepared dataset audit found {len(errors)} errors")


def audit_prepared_collection(args: argparse.Namespace) -> None:
    """Aggregate audited LeRobot roots into one LingBot training collection."""
    import pyarrow.parquet as pq

    output_root = args.output_root.resolve()
    manifest_paths = sorted(output_root.glob(f"*/{MANIFEST_NAME}"))
    if not manifest_paths:
        raise FileNotFoundError(f"No subset manifests found under {output_root}")

    errors: list[str] = []
    subsets: list[dict[str, Any]] = []
    selected_tasks: list[str] = []
    task_episode_counts: dict[str, int] = {}
    action_arrays: list[np.ndarray] = []
    text_embeddings: dict[str, torch.Tensor] = {}
    empty_embedding: torch.Tensor | None = None
    empty_embedding_path: Path | None = None
    tensor_schema: dict[str, Any] | None = None
    latent_count = 0
    latent_sum = 0.0
    latent_square_sum = 0.0
    latent_min = math.inf
    latent_max = -math.inf
    total_episodes = 0
    expected_latent_files = 0

    for manifest_path in manifest_paths:
        subset_root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report_path = subset_root / args.subset_report_name
        if not report_path.is_file():
            errors.append(f"missing subset audit report: {report_path}")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report["passed"]:
            errors.append(f"subset audit failed: {subset_root}")
        errors.extend(
            f"{subset_root.name}: {message}" for message in report["errors"]
        )

        current_schema = report["tensor_schema"]
        if tensor_schema is None:
            tensor_schema = current_schema
        elif current_schema != tensor_schema:
            errors.append(f"tensor schema mismatch: {subset_root}")

        current_latent = report["latent_statistics"]
        current_count = int(current_latent["count"])
        current_mean = float(current_latent["mean"])
        current_std = float(current_latent["std"])
        latent_count += current_count
        latent_sum += current_count * current_mean
        latent_square_sum += current_count * (
            current_std * current_std + current_mean * current_mean
        )
        latent_min = min(latent_min, float(current_latent["min"]))
        latent_max = max(latent_max, float(current_latent["max"]))

        subset_text: dict[str, torch.Tensor] = torch.load(
            subset_root / TEXT_CACHE_NAME,
            map_location="cpu",
            weights_only=False,
        )
        for text, embedding in subset_text.items():
            if text in text_embeddings and not torch.equal(
                text_embeddings[text], embedding
            ):
                errors.append(f"text embedding mismatch for {text!r}")
            text_embeddings[text] = embedding
        current_empty = subset_text[""]
        if empty_embedding is None:
            empty_embedding = current_empty
            empty_embedding_path = subset_root / "empty_emb.pt"
        elif not torch.equal(empty_embedding, current_empty):
            errors.append(f"empty embedding mismatch: {subset_root}")

        for episode in manifest["selected_episodes"]:
            parquet_path = (
                Path(manifest["source_root"])
                / "data"
                / "chunk-000"
                / f"episode_{int(episode['episode_index']):06d}.parquet"
            )
            actions = np.asarray(
                pq.read_table(parquet_path, columns=["action"])["action"].to_pylist(),
                dtype=np.float32,
            )
            action_arrays.append(actions)

        for task, count in report["task_episode_counts"].items():
            task_episode_counts[task] = task_episode_counts.get(task, 0) + int(count)
        selected_tasks.extend(manifest["selected_tasks"])
        total_episodes += int(report["selected_episode_count"])
        expected_latent_files += int(report["latent_files"]["expected"])
        subsets.append(
            {
                "name": subset_root.name,
                "path": str(subset_root),
                "task_count": len(manifest["selected_tasks"]),
                "episode_count": int(report["selected_episode_count"]),
                "action_frame_count": int(report["action_statistics"]["frame_count"]),
                "latent_files": int(report["latent_files"]["actual"]),
                "audit_report": str(report_path),
            }
        )

    actual_latent_files = len(list(output_root.glob("*/latents/**/*.pth")))
    if actual_latent_files != expected_latent_files:
        errors.append(
            f"latent file count {actual_latent_files}, expected {expected_latent_files}"
        )
    if tensor_schema is None or empty_embedding is None or empty_embedding_path is None:
        raise RuntimeError("Collection has no complete audited subset")

    all_actions = np.concatenate(action_arrays, axis=0)
    q01 = np.quantile(all_actions, 0.01, axis=0).tolist()
    q99 = np.quantile(all_actions, 0.99, axis=0).tolist()
    latent_mean = latent_sum / latent_count
    latent_variance = max(
        latent_square_sum / latent_count - latent_mean * latent_mean, 0.0
    )
    shutil.copy2(empty_embedding_path, output_root / "empty_emb.pt")
    torch.save(text_embeddings, output_root / TEXT_CACHE_NAME)

    collection_manifest = {
        "output_root": str(output_root),
        "model_roots": sorted(
            {
                json.loads(path.read_text(encoding="utf-8"))["model_root"]
                for path in manifest_paths
            }
        ),
        "subsets": subsets,
        "task_count": len(selected_tasks),
        "episode_count": total_episodes,
    }
    write_json(output_root / MANIFEST_NAME, collection_manifest)
    report = {
        "passed": not errors,
        "errors": errors,
        "subsets": subsets,
        "selected_tasks": selected_tasks,
        "task_episode_counts": task_episode_counts,
        "task_count": len(selected_tasks),
        "selected_episode_count": total_episodes,
        "latent_files": {
            "actual": actual_latent_files,
            "expected": expected_latent_files,
        },
        "tensor_schema": tensor_schema,
        "latent_statistics": {
            "count": latent_count,
            "mean": latent_mean,
            "std": math.sqrt(latent_variance),
            "min": latent_min,
            "max": latent_max,
        },
        "action_statistics": {
            "frame_count": int(all_actions.shape[0]),
            "q01_7d": q01,
            "q99_7d": q99,
            "norm_stat_q01_30d": q01 + [0.0] * 23,
            "norm_stat_q99_30d": q99 + [0.0] * 23,
        },
    }
    write_json(args.report_path.resolve(), report)
    print(
        f"collection audit {'passed' if report['passed'] else 'failed'}: "
        f"subsets={len(subsets)}, tasks={len(selected_tasks)}, "
        f"episodes={total_episodes}, latent_files={actual_latent_files}, "
        f"errors={len(errors)}"
    )
    print(f"collection audit report: {args.report_path.resolve()}")
    if errors:
        raise RuntimeError(f"Prepared collection audit found {len(errors)} errors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare and encode a dataset")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--model-root", type=Path, required=True)
    prepare.add_argument("--task-limit", type=int, default=5)
    prepare.add_argument("--height", type=int, default=128)
    prepare.add_argument("--width", type=int, default=128)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--temporal-chunk-size", type=int, default=64)
    prepare.add_argument("--num-shards", type=int, default=1)
    prepare.add_argument("--shard-index", type=int, default=0)
    prepare.add_argument("--max-episodes", type=int)
    prepare.add_argument("--setup-only", action="store_true")
    prepare.add_argument("--encode-only", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="Re-encode and compare official LingBot reference files"
    )
    validate.add_argument("--reference-root", type=Path, required=True)
    validate.add_argument("--reencoded-root", type=Path, required=True)
    validate.add_argument("--report-path", type=Path, required=True)
    validate.add_argument("--model-root", type=Path, required=True)
    validate.add_argument("--episodes", type=int, nargs="+", required=True)
    validate.add_argument("--height", type=int, default=128)
    validate.add_argument("--width", type=int, default=128)
    validate.add_argument("--device", default="cuda:0")
    validate.add_argument("--temporal-chunk-size", type=int, default=64)

    audit = subparsers.add_parser(
        "audit", help="Audit all generated tensors and calculate action quantiles"
    )
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--report-path", type=Path, required=True)

    audit_collection = subparsers.add_parser(
        "audit-collection", help="Aggregate audited dataset roots for training"
    )
    audit_collection.add_argument("--output-root", type=Path, required=True)
    audit_collection.add_argument("--report-path", type=Path, required=True)
    audit_collection.add_argument(
        "--subset-report-name", default="validation_summary.json"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        validate_reference(args)
        return
    if args.command == "audit":
        audit_prepared_dataset(args)
        return
    if args.command == "audit-collection":
        audit_prepared_collection(args)
        return

    if args.setup_only and args.encode_only:
        raise ValueError("--setup-only and --encode-only are mutually exclusive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    if args.encode_only:
        manifest = load_manifest(args.output_root.resolve())
    else:
        manifest = setup_dataset(args)
    if not args.setup_only:
        encode_prepared_dataset(args, manifest)


if __name__ == "__main__":
    main()
