#!/usr/bin/env python3
"""Run inference-only causal plan/text interventions on LingBot-VA.

All experimental logic lives in this file.  Existing model code is imported as
read-only and checkpoints are never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wan_va.configs import VA_CONFIGS  # noqa: E402
from wan_va.distributed.util import init_distributed  # noqa: E402
from wan_va.utils import data_seq_to_patch  # noqa: E402
from wan_va.wan_va_server import VA_Server  # noqa: E402


DEFAULT_BASE = Path("/root/shared/zouyude/ckpts/lingbot-va/lingbot-va-base")
DEFAULT_TRAIN_RUN = Path("/root/shared/zouyude/train/lingbot-va/libero-all-full")
DEFAULT_TRANSFORMER = (
    DEFAULT_TRAIN_RUN / "checkpoints/checkpoint_step_20000/transformer"
)
DEFAULT_INPUT = Path(
    "/root/shared/zouyude/eval/lingbot-va/"
    "video_posttrain_compare_20260804_052758/libero/"
    "libero_object_ep000000/input"
)
DEFAULT_PROMPT = "pick up the alphabet soup and place it in the basket"
DEFAULT_WRONG_PROMPT = "pick up the milk and place it in the basket"
DEFAULT_PARAPHRASE = "place the alphabet soup can inside the basket"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, path)


def git_record() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "status", "--short"], text=True
    ).splitlines()
    return {"commit": commit, "dirty": dirty}


def encode_prompt(server: VA_Server, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    positive, negative = server.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=server.job_config.guidance_scale > 1,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=server.device,
        dtype=server.dtype,
    )
    if negative is None:
        negative = positive.new_zeros(positive.shape)
    return positive.detach(), negative.detach()


def reset_with_context(
    server: VA_Server,
    context: tuple[torch.Tensor, torch.Tensor],
) -> None:
    server._reset(prompt=None)
    server.prompt_embeds, server.negative_prompt_embeds = context


@torch.no_grad()
def generate_plan(
    server: VA_Server,
    obs: dict[str, Any],
    context: tuple[torch.Tensor, torch.Tensor],
    seed: int,
    video_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reset_with_context(server, context)
    init_latent = server._encode_obs(obs)
    server.init_latent = init_latent
    frame_chunk_size = int(server.job_config.frame_chunk_size)
    generator = torch.Generator(device=server.device).manual_seed(seed)
    latents = torch.randn(
        1,
        48,
        frame_chunk_size,
        server.latent_height,
        server.latent_width,
        generator=generator,
        device=server.device,
        dtype=server.dtype,
    )

    server.scheduler.set_timesteps(video_steps)
    timesteps = F.pad(server.scheduler.timesteps, (0, 1), mode="constant", value=0)
    for index, timestep in enumerate(timesteps):
        is_last = index == len(timesteps) - 1
        first_frame = init_latent[:, :, 0:1].to(server.dtype)
        input_dict = server._prepare_latent_input(
            latents,
            None,
            timestep,
            timestep,
            first_frame,
            None,
            frame_st_id=0,
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
                prediction = prediction[1:] + server.job_config.guidance_scale * (
                    prediction[:1] - prediction[1:]
                )
            else:
                prediction = prediction[:1]
            latents = server.scheduler.step(
                prediction, timestep, latents, return_dict=False
            )
        latents[:, :, 0:1] = first_frame
    return latents.detach().clone(), init_latent.detach().clone()


@torch.no_grad()
def action_from_plan(
    server: VA_Server,
    plan: torch.Tensor,
    init_latent: torch.Tensor,
    plan_context: tuple[torch.Tensor, torch.Tensor],
    action_context: tuple[torch.Tensor, torch.Tensor],
    seed: int,
    action_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    reset_with_context(server, plan_context)
    server.init_latent = init_latent
    zero_t = torch.tensor(0.0, dtype=torch.float32, device=server.device)
    plan_input = server._prepare_latent_input(
        plan,
        None,
        zero_t,
        zero_t,
        init_latent[:, :, 0:1].to(server.dtype),
        None,
        frame_st_id=0,
    )
    server.transformer(
        server._repeat_input_for_cfg(plan_input["latent_res_lst"]),
        update_cache=1,
        cache_name=server.cache_name,
        action_mode=False,
    )

    # Only the action pass receives this context.  The cached plan remains the
    # one computed above with plan_context.
    server.prompt_embeds, server.negative_prompt_embeds = action_context
    frame_chunk_size = int(server.job_config.frame_chunk_size)
    generator = torch.Generator(device=server.device).manual_seed(seed)
    actions = torch.randn(
        1,
        server.job_config.action_dim,
        frame_chunk_size,
        server.action_per_frame,
        1,
        generator=generator,
        device=server.device,
        dtype=server.dtype,
    )
    action_cond = server._initial_action_condition(actions)
    server.action_scheduler.set_timesteps(action_steps)
    timesteps = F.pad(
        server.action_scheduler.timesteps, (0, 1), mode="constant", value=0
    )
    for index, timestep in enumerate(timesteps):
        is_last = index == len(timesteps) - 1
        input_dict = server._prepare_latent_input(
            None,
            actions,
            timestep,
            timestep,
            None,
            action_cond,
            frame_st_id=0,
        )
        prediction = server.transformer(
            server._repeat_input_for_cfg(input_dict["action_res_lst"]),
            update_cache=1 if is_last else 0,
            cache_name=server.cache_name,
            action_mode=True,
        )
        if not is_last:
            prediction = rearrange(
                prediction,
                "b (f n) c -> b c f n 1",
                f=frame_chunk_size,
            )
            if server.job_config.action_guidance_scale > 1:
                prediction = prediction[1:] + server.job_config.action_guidance_scale * (
                    prediction[:1] - prediction[1:]
                )
            else:
                prediction = prediction[:1]
            actions = server.action_scheduler.step(
                prediction, timestep, actions, return_dict=False
            )
        actions[:, :, 0:1] = action_cond

    actions[:, ~server.action_mask] *= 0
    selected = actions[
        0, server.job_config.used_action_channel_ids, :, :, 0
    ].float().cpu().numpy()
    physical = server.postprocess_action(actions.clone()).astype(np.float32)
    return selected, physical


def action_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(candidate, dtype=np.float64).reshape(-1)
    delta = b - a
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / denom) if denom > 0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "l2": float(np.linalg.norm(delta)),
        "reference_l2": float(np.linalg.norm(a)),
        "candidate_l2": float(np.linalg.norm(b)),
        "cosine": cosine,
    }


def plan_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    a = reference[:, :, 1:].float()
    b = candidate[:, :, 1:].float()
    delta = b - a
    return {
        "future_rmse": float(delta.square().mean().sqrt().item()),
        "future_mean_abs": float(delta.abs().mean().item()),
        "future_l2": float(delta.norm().item()),
        "candidate_future_l2": float(b.norm().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--transformer", type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--wrong-prompt", default=DEFAULT_WRONG_PROMPT)
    parser.add_argument("--paraphrase", default=DEFAULT_PARAPHRASE)
    parser.add_argument("--video-seed", type=int, default=4202)
    parser.add_argument("--action-seed", type=int, default=904202)
    parser.add_argument("--video-steps", type=int, default=20)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--master-port", type=int, default=29641)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES.")
    for required in (args.base_model, args.transformer, args.input_dir):
        if not required.exists():
            raise FileNotFoundError(required)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "plans").mkdir()
    (args.output_dir / "actions").mkdir()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    init_distributed(1, 0, 0)

    config = copy.deepcopy(VA_CONFIGS["libero_all"])
    config.wan22_pretrained_model_name_or_path = str(args.base_model.resolve())
    config.transformer_model_name_or_path = str(args.transformer.resolve())
    config.input_img_path = str(args.input_dir.resolve())
    config.enable_offload = False
    config.save_debug_data = False
    config.save_root = str(args.output_dir)
    config.rank = config.local_rank = 0
    config.world_size = 1

    started = time.time()
    server = VA_Server(config)
    server.video_processor = VideoProcessor(vae_scale_factor=1)
    original = encode_prompt(server, args.prompt)
    wrong = encode_prompt(server, args.wrong_prompt)
    empty = encode_prompt(server, "")
    paraphrase = encode_prompt(server, args.paraphrase)
    obs = {
        "obs": [
            {
                key: np.asarray(
                    Image.open(args.input_dir / f"{key}.png").convert("RGB")
                )
                for key in config.obs_cam_keys
            }
        ]
    }

    normal, init_latent = generate_plan(
        server, obs, original, args.video_seed, args.video_steps
    )
    wrong_plan, wrong_init = generate_plan(
        server, obs, wrong, args.video_seed, args.video_steps
    )
    if not torch.equal(init_latent, wrong_init):
        raise RuntimeError("Observation encoding changed across prompt-only plan runs.")
    static = init_latent[:, :, 0:1].repeat(
        1, 1, int(config.frame_chunk_size), 1, 1
    )
    zero_future = torch.zeros_like(normal)
    zero_future[:, :, 0:1] = init_latent[:, :, 0:1]
    plans = {
        "normal": (normal, original),
        "static": (static, original),
        "zero_future": (zero_future, original),
        "wrong_prompt": (wrong_plan, wrong),
    }

    for name, (plan, _) in plans.items():
        torch.save(plan.cpu(), args.output_dir / "plans" / f"{name}.pt")
        # VA_Server.generate() normally supplies this no-grad guard around VAE
        # decoding; this isolated runner calls the lower-level helper directly.
        with torch.no_grad():
            frames = server.decode_one_video(plan, "np")[0]
        export_to_video(frames, str(args.output_dir / "plans" / f"{name}.mp4"), fps=10)

    conditions = {
        "normal_original": ("normal", original),
        "normal_repeat": ("normal", original),
        "static_original": ("static", original),
        "zero_future_original": ("zero_future", original),
        "wrong_plan_original": ("wrong_prompt", original),
        "normal_empty_text": ("normal", empty),
        "normal_wrong_text": ("normal", wrong),
        "normal_paraphrase": ("normal", paraphrase),
    }
    normalized_actions: dict[str, np.ndarray] = {}
    physical_actions: dict[str, np.ndarray] = {}
    condition_records = {}
    for name, (plan_name, action_context) in conditions.items():
        plan, plan_context = plans[plan_name]
        normalized, physical = action_from_plan(
            server,
            plan,
            init_latent,
            plan_context,
            action_context,
            args.action_seed,
            args.action_steps,
        )
        normalized_actions[name] = normalized
        physical_actions[name] = physical
        np.save(args.output_dir / "actions" / f"{name}_normalized.npy", normalized)
        np.save(args.output_dir / "actions" / f"{name}_physical.npy", physical)
        condition_records[name] = {
            "plan": plan_name,
            "normalized_shape": list(normalized.shape),
            "physical_shape": list(physical.shape),
        }

    reference = physical_actions["normal_original"]
    metrics = {}
    for name, action in physical_actions.items():
        metrics[name] = {
            "all": action_metrics(reference, action),
            # Frame zero is a hard action condition in LingBot-VA.  The suffix
            # is the causally informative part of this first generated chunk.
            "free_suffix": action_metrics(reference[:, 1:], action[:, 1:]),
        }
    if not np.array_equal(
        physical_actions["normal_original"], physical_actions["normal_repeat"]
    ):
        raise RuntimeError("Fixed-seed baseline action replay is not deterministic.")

    result = {
        "schema_version": 1,
        "status": "complete",
        "model": "lingbot-va",
        "experiment": "M1-E001-E002 causal plan/text intervention",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(args.transformer.resolve()),
        "training_run": str(DEFAULT_TRAIN_RUN),
        "base_model": str(args.base_model.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "prompt": args.prompt,
        "wrong_prompt": args.wrong_prompt,
        "paraphrase": args.paraphrase,
        "video_seed": args.video_seed,
        "action_seed": args.action_seed,
        "video_steps": args.video_steps,
        "action_steps": args.action_steps,
        "git": git_record(),
        "plans_vs_normal": {
            name: plan_metrics(normal, plan) for name, (plan, _) in plans.items()
        },
        "conditions": condition_records,
        "action_metrics_vs_normal_original": metrics,
        "determinism": {"normal_repeat_exact": True},
    }
    atomic_json(args.output_dir / "results.json", result)
    (args.output_dir / "COMPLETE").touch()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    server.transformer.clear_cache(server.cache_name)
    del server
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
