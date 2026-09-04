#!/usr/bin/env python3
"""Screen inference-only first-frame anchors on LingBot-VA."""

from __future__ import annotations

import argparse
import copy
import importlib.util
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
import torch.nn.functional as F
from diffusers.video_processor import VideoProcessor
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_isolated_helper(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


causal = load_isolated_helper(
    "m1e001_causal_helpers",
    PROJECT_ROOT / "isolated_tests/causal_plan_interventions/run.py",
)

from wan_va.configs import VA_CONFIGS  # noqa: E402
from wan_va.distributed.util import init_distributed  # noqa: E402
from wan_va.utils import data_seq_to_patch  # noqa: E402
from wan_va.wan_va_server import VA_Server  # noqa: E402


DEFAULT_BASE = Path("/root/shared/zouyude/ckpts/lingbot-va/lingbot-va-base")
DEFAULT_TRANSFORMER = Path(
    "/root/shared/zouyude/train/lingbot-va/libero-all-full/"
    "checkpoints/checkpoint_step_20000/transformer"
)
DEFAULT_ORIGINAL_INPUT = Path(
    "/root/shared/zouyude/eval/lingbot-va/"
    "video_posttrain_compare_20260804_052758/libero/"
    "libero_object_ep000000/input"
)
DEFAULT_SAMPLE_INPUT = Path(
    "/root/shared/zouyude/eval/lingbot-va/"
    "libero_plus_layout_ood_compare_20260805_131630/candidate_initial_views/"
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket_level5_sample2"
)
DEFAULT_BASELINE_ROOT = Path(
    "/root/shared/zouyude/eval/ideas/1-causal-video-planner-faithful-idm/"
    "M1-E003_20260806_153832/lingbot"
)
DEFAULT_PROMPT = "pick up the alphabet soup and place it in the basket"
VALID_MODES = {"global_lowfreq", "agent_lowfreq", "motion_selective_lowfreq"}


def spatial_lowpass(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size == 1:
        return value
    if kernel_size % 2 != 1:
        raise ValueError("lowpass kernel must be odd")
    batch, channels, frames, height, width = value.shape
    flat = value.permute(0, 1, 2, 3, 4).reshape(batch * channels * frames, 1, height, width)
    padding = kernel_size // 2
    flat = F.pad(flat, (padding, padding, padding, padding), mode="replicate")
    flat = F.avg_pool2d(flat, kernel_size=kernel_size, stride=1)
    return flat.reshape(batch, channels, frames, height, width)


def baseline_motion_map(plans: dict[str, torch.Tensor]) -> torch.Tensor:
    maps = []
    for plan in plans.values():
        future_delta = (plan[:, :, 1:].float() - plan[:, :, :1].float()).abs()
        maps.append(future_delta.mean(dim=(0, 1, 2)))
    return torch.stack(maps).amax(dim=0)


def make_anchor_mask(
    mode: str,
    height: int,
    width: int,
    baseline_plans: dict[str, torch.Tensor],
    motion_free_fraction: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode}; choose from {sorted(VALID_MODES)}")
    mask = torch.zeros((1, 1, 1, height, width), dtype=torch.float32)
    if mode == "global_lowfreq":
        mask.fill_(1.0)
        return mask, {"anchored_fraction": 1.0}

    agent_width = width // 2
    mask[..., :agent_width] = 1.0
    metadata: dict[str, Any] = {"agent_width_latent": agent_width}
    if mode == "motion_selective_lowfreq":
        motion = baseline_motion_map(baseline_plans)[:, :agent_width]
        flat = motion.flatten()
        threshold = torch.quantile(flat, 1.0 - motion_free_fraction)
        free = (motion >= threshold).float()[None, None, None]
        # One-cell dilation protects a small neighborhood around moving cells.
        free = F.max_pool2d(free.reshape(1, 1, height, agent_width), 3, stride=1, padding=1)
        mask[..., :agent_width] *= 1.0 - free.reshape(1, 1, 1, height, agent_width)
        metadata.update(
            {
                "motion_free_fraction_requested": motion_free_fraction,
                "motion_threshold": float(threshold.item()),
                "free_fraction_after_dilation_agent": float(free.mean().item()),
            }
        )
    metadata["anchored_fraction"] = float(mask.mean().item())
    metadata["anchored_fraction_agent"] = float(mask[..., :agent_width].mean().item())
    return mask, metadata


@torch.no_grad()
def generate_anchored_plan(
    server: VA_Server,
    obs: dict[str, Any],
    context: tuple[torch.Tensor, torch.Tensor],
    seed: int,
    video_steps: int,
    total_strength: float,
    anchor_mask: torch.Tensor,
    lowpass_kernel: int,
    clean_anchor_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= total_strength < 1.0:
        raise ValueError("total strength must be in [0, 1)")
    causal.reset_with_context(server, context)
    init_latent = server._encode_obs(obs)
    server.init_latent = init_latent
    frame_chunk_size = int(server.job_config.frame_chunk_size)
    generator = torch.Generator(device=server.device).manual_seed(seed)
    initial_noise = torch.randn(
        1,
        48,
        frame_chunk_size,
        server.latent_height,
        server.latent_width,
        generator=generator,
        device=server.device,
        dtype=server.dtype,
    )
    latents = initial_noise.clone()
    if clean_anchor_override is None:
        clean_anchor = init_latent[:, :, :1].to(server.dtype).repeat(1, 1, frame_chunk_size, 1, 1)
    else:
        clean_anchor = clean_anchor_override.to(device=server.device, dtype=server.dtype)
        if clean_anchor.shape != latents.shape:
            raise ValueError(
                f"Static-video anchor shape {clean_anchor.shape} does not match plan {latents.shape}"
            )
    mask = anchor_mask.to(device=server.device, dtype=server.dtype)
    # Applying this alpha at every denoising update gives total_strength after
    # video_steps updates if the same target were used throughout.
    step_alpha = 1.0 - (1.0 - total_strength) ** (1.0 / video_steps)

    server.scheduler.set_timesteps(video_steps)
    timesteps = F.pad(server.scheduler.timesteps, (0, 1), mode="constant", value=0)
    for index, timestep in enumerate(timesteps):
        is_last = index == len(timesteps) - 1
        first_frame = init_latent[:, :, :1].to(server.dtype)
        input_dict = server._prepare_latent_input(
            latents, None, timestep, timestep, first_frame, None, frame_st_id=0
        )
        prediction = server.transformer(
            server._repeat_input_for_cfg(input_dict["latent_res_lst"]),
            update_cache=1 if is_last else 0,
            cache_name=server.cache_name,
            action_mode=False,
        )
        if not is_last:
            prediction = data_seq_to_patch(
                server.job_config.patch_size,
                prediction,
                frame_chunk_size,
                server.latent_height,
                server.latent_width,
                batch_size=2 if server.use_cfg else 1,
            )
            if server.job_config.guidance_scale > 1:
                prediction = prediction[1:] + server.job_config.guidance_scale * (prediction[:1] - prediction[1:])
            else:
                prediction = prediction[:1]
            latents = server.scheduler.step(prediction, timestep, latents, return_dict=False)

            if total_strength > 0:
                if index + 1 < len(server.scheduler.sigmas):
                    next_sigma = float(server.scheduler.sigmas[index + 1].item())
                else:
                    next_sigma = 0.0
                noised_anchor = (1.0 - next_sigma) * clean_anchor + next_sigma * initial_noise
                correction = spatial_lowpass(noised_anchor - latents, lowpass_kernel)
                # Frame zero is hard-conditioned below, so anchoring only the
                # future frames avoids changing the observation condition.
                latents[:, :, 1:] += step_alpha * mask * correction[:, :, 1:]
        latents[:, :, :1] = first_frame
    return latents.detach().clone(), init_latent.detach().clone()


@torch.no_grad()
def encode_static_video_anchor(
    server: VA_Server,
    obs: dict[str, Any],
    context: tuple[torch.Tensor, torch.Tensor],
    latent_frame_count: int,
    temporal_stride: int,
) -> torch.Tensor:
    """Encode a repeated RGB observation with the streaming VAE chunk contract."""
    causal.reset_with_context(server, context)
    if len(obs["obs"]) != 1:
        raise ValueError("Expected a single observation frame")
    # Wan's causal VAE maps one initial RGB frame to the first latent and each
    # following temporal_stride RGB frames to one additional latent.  Its
    # streaming wrapper expects those chunks sequentially; a bulk 13-frame
    # tensor is not a valid streaming call for this implementation.
    pieces = [server._encode_obs(obs)]
    repeated_chunk = {
        "obs": [obs["obs"][0] for _ in range(temporal_stride)]
    }
    for _ in range(latent_frame_count - 1):
        pieces.append(server._encode_obs(repeated_chunk))
    anchor = torch.cat(pieces, dim=2).detach().clone()
    server.streaming_vae.clear_cache()
    return anchor


def load_frames(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        return np.asarray([frame for frame in reader], dtype=np.uint8)
    finally:
        reader.close()


def decoded_frames(server: VA_Server, plan: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        frames = server.decode_one_video(plan, "np")[0]
    array = np.asarray([np.asarray(frame) for frame in frames])
    if np.issubdtype(array.dtype, np.floating):
        if float(array.max()) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def save_video(frames: np.ndarray, path: Path, fps: int = 10) -> None:
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def save_pair_video(
    original: np.ndarray,
    sample: np.ndarray,
    label: str,
    path: Path,
    fps: int = 10,
) -> None:
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for index in range(min(len(original), len(sample))):
            panels = []
            for name, frame in (("original", original[index]), ("level5_sample2", sample[index])):
                panel = Image.fromarray(frame).convert("RGB")
                canvas = Image.new("RGB", (panel.width, panel.height + 32), "black")
                canvas.paste(panel, (0, 32))
                ImageDraw.Draw(canvas).text((5, 5), f"{label} | {name}", fill="white")
                panels.append(np.asarray(canvas))
            writer.append_data(np.concatenate(panels, axis=1))
    finally:
        writer.close()


def layout_response(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    width = reference.shape[2] // 2
    initial = (candidate[0, :, :width].astype(np.float64) - reference[0, :, :width].astype(np.float64)).reshape(-1)
    denom = float(np.dot(initial, initial))
    initial_norm = float(np.sqrt(denom))
    per_frame = []
    for index in range(len(reference)):
        delta = (candidate[index, :, :width].astype(np.float64) - reference[index, :, :width].astype(np.float64)).reshape(-1)
        norm = float(np.linalg.norm(delta))
        per_frame.append(
            {
                "frame": index,
                "gain": norm / initial_norm,
                "directional_cosine": float(np.dot(delta, initial) / (norm * initial_norm)) if norm else 0.0,
                "retained_projection": float(np.dot(delta, initial) / denom),
            }
        )
    return {
        "future_mean_retained_projection": float(np.mean([x["retained_projection"] for x in per_frame[1:]])),
        "last_retained_projection": per_frame[-1]["retained_projection"],
        "per_frame": per_frame,
    }


def view_motion(frames: np.ndarray, xs: slice) -> tuple[np.ndarray, float]:
    values = frames[:, :, xs].astype(np.float64)
    temporal = np.abs(values[1:] - values[:1]).mean(axis=(0, 3))
    return temporal, float(temporal.mean())


def motion_preservation(
    frames: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    width = frames.shape[2]
    views = {"agentview": slice(0, width // 2), "wrist": slice(width // 2, width)}
    output = {}
    for name, xs in views.items():
        baseline_map, baseline_mean = view_motion(baseline, xs)
        candidate_map, candidate_mean = view_motion(frames, xs)
        threshold = float(np.quantile(baseline_map, 0.90))
        dynamic_mask = baseline_map >= threshold
        baseline_dynamic = float(baseline_map[dynamic_mask].mean())
        candidate_dynamic = float(candidate_map[dynamic_mask].mean())
        output[name] = {
            "mean_motion_l1": candidate_mean,
            "mean_motion_ratio_vs_baseline": candidate_mean / baseline_mean if baseline_mean else 0.0,
            "baseline_dynamic_threshold_q90": threshold,
            "dynamic_region_motion_l1": candidate_dynamic,
            "dynamic_region_ratio_vs_baseline": candidate_dynamic / baseline_dynamic if baseline_dynamic else 0.0,
        }
    return output


def parse_csv(value: str, cast) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--transformer", type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument("--original-input", type=Path, default=DEFAULT_ORIGINAL_INPUT)
    parser.add_argument("--sample-input", type=Path, default=DEFAULT_SAMPLE_INPUT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--video-seed", type=int, default=4202)
    parser.add_argument("--video-steps", type=int, default=20)
    parser.add_argument("--modes", default="global_lowfreq")
    parser.add_argument("--strengths", default="0.25,0.50,0.75")
    parser.add_argument("--lowpass-kernel", type=int, default=3)
    parser.add_argument(
        "--anchor-source",
        choices=("static_video", "repeated_first_latent"),
        default="static_video",
    )
    parser.add_argument("--motion-free-fraction", type=float, default=0.10)
    parser.add_argument("--verify-baseline", action="store_true")
    parser.add_argument("--master-port", type=int, default=29661)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES.")
    modes = parse_csv(args.modes, str)
    strengths = parse_csv(args.strengths, float)
    if not modes or not strengths:
        raise ValueError("At least one mode and strength are required")
    unknown = set(modes) - VALID_MODES
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}")
    required = [args.base_model, args.transformer, args.original_input, args.sample_input, args.baseline_root]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "conditions").mkdir()

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
    input_dirs = {"original": args.original_input, "sample2": args.sample_input}
    observations = {
        name: {
            "obs": [
                {
                    key: np.asarray(Image.open(path / f"{key}.png").convert("RGB"))
                    for key in config.obs_cam_keys
                }
            ]
        }
        for name, path in input_dirs.items()
    }
    baseline_plans = {
        name: torch.load(args.baseline_root / "plans" / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in input_dirs
    }
    baseline_frames = {
        name: load_frames(args.baseline_root / "plans" / f"{name}.mp4")
        for name in input_dirs
    }
    latent_height, latent_width = baseline_plans["original"].shape[-2:]
    static_anchors = None
    if args.anchor_source == "static_video":
        static_anchors = {
            name: encode_static_video_anchor(
                server,
                observations[name],
                context,
                int(config.frame_chunk_size),
                4,
            )
            for name in input_dirs
        }
        for name, anchor in static_anchors.items():
            if anchor.shape != baseline_plans[name].shape:
                raise RuntimeError(
                    f"Static anchor for {name} has {anchor.shape}, expected {baseline_plans[name].shape}"
                )

    baseline_verified = None
    if args.verify_baseline:
        baseline_verified = {}
        for name in input_dirs:
            replay, _ = generate_anchored_plan(
                server,
                observations[name],
                context,
                args.video_seed,
                args.video_steps,
                0.0,
                torch.ones((1, 1, 1, latent_height, latent_width)),
                args.lowpass_kernel,
                None if static_anchors is None else static_anchors[name],
            )
            exact = torch.equal(replay.cpu(), baseline_plans[name])
            baseline_verified[name] = exact
            if not exact:
                raise RuntimeError(f"Zero-strength replay differs from M1-E003 baseline for {name}")

    condition_results = {}
    mask_records = {}
    for mode in modes:
        mask, mask_metadata = make_anchor_mask(
            mode,
            latent_height,
            latent_width,
            baseline_plans,
            args.motion_free_fraction,
        )
        mask_records[mode] = mask_metadata
        for strength in strengths:
            strength_key = f"{strength:.2f}".replace(".", "p")
            condition_name = f"{mode}_s{strength_key}"
            condition_dir = args.output_dir / "conditions" / condition_name
            condition_dir.mkdir()
            plans = {}
            frames = {}
            for layout_name in input_dirs:
                plan, _ = generate_anchored_plan(
                    server,
                    observations[layout_name],
                    context,
                    args.video_seed,
                    args.video_steps,
                    strength,
                    mask,
                    args.lowpass_kernel,
                    None if static_anchors is None else static_anchors[layout_name],
                )
                plans[layout_name] = plan
                frames[layout_name] = decoded_frames(server, plan)
                torch.save(plan.cpu(), condition_dir / f"{layout_name}.pt")
                save_video(frames[layout_name], condition_dir / f"{layout_name}.mp4")
            save_pair_video(frames["original"], frames["sample2"], condition_name, condition_dir / "comparison.mp4")
            condition_results[condition_name] = {
                "mode": mode,
                "total_strength": strength,
                "layout_response_agentview": layout_response(frames["original"], frames["sample2"]),
                "motion_preservation": {
                    name: motion_preservation(frames[name], baseline_frames[name])
                    for name in input_dirs
                },
                "latent_motion_ratio_vs_baseline": {
                    name: float(
                        ((plans[name][:, :, 1:].float() - plans[name][:, :, :1].float()).square().mean().sqrt()
                        / (baseline_plans[name][:, :, 1:].float() - baseline_plans[name][:, :, :1].float()).square().mean().sqrt()).item()
                    )
                    for name in input_dirs
                },
            }

    result = {
        "schema_version": 1,
        "status": "complete",
        "model": "lingbot-va",
        "experiment": "M1-E005 inference-only first-frame anchor screen",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(args.transformer.resolve()),
        "base_model": str(args.base_model.resolve()),
        "baseline_root": str(args.baseline_root.resolve()),
        "prompt": args.prompt,
        "video_seed": args.video_seed,
        "video_steps": args.video_steps,
        "modes": modes,
        "strengths": strengths,
        "lowpass_kernel": args.lowpass_kernel,
        "anchor_source": args.anchor_source,
        "mask_metadata": mask_records,
        "zero_strength_baseline_exact": baseline_verified,
        "conditions": condition_results,
        "git": causal.git_record(),
        "notes": [
            "Primary layout metric uses fixed agentview only.",
            "Motion ratios compare each anchored plan with the exact M1-E003 baseline of the same layout.",
            "Dynamic-region motion uses the top 10% baseline temporal-change pixels and is a proxy, not object tracking.",
        ],
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
