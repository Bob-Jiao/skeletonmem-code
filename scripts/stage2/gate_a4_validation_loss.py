#!/usr/bin/env python3
"""A4 validation loss for LingBot-VA LIBERO-LONG held-out split.

This script runs distributed FSDP validation only. It does not train, does not
update weights, and does not run closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


def setup_imports(lingbot_root: Path) -> None:
    sys.path.insert(0, str(lingbot_root))
    sys.path.insert(0, str(lingbot_root / "wan_va"))


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lingbot-root", required=True)
    parser.add_argument("--config-name", default="libero_train")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--max-batches", type=int, default=10, help="Per-rank validation batches. Use 0 or negative to evaluate the full loader.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args()

    lingbot_root = Path(args.lingbot_root).resolve()
    setup_imports(lingbot_root)

    from configs import VA_CONFIGS  # type: ignore
    from dataset import MultiLatentLeRobotDataset  # type: ignore
    from distributed.fsdp import apply_ac, shard_model  # type: ignore
    from distributed.util import _configure_model, init_distributed  # type: ignore
    from modules.utils import load_transformer  # type: ignore
    from train import Trainer  # type: ignore
    from utils import FlowMatchScheduler  # type: ignore

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    init_distributed(world_size, local_rank, rank)

    config = VA_CONFIGS[args.config_name]
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.save_root = args.save_root
    config.resume_from = args.checkpoint
    config.enable_wandb = False
    config.cfg_prob = 0.0
    config.gradient_accumulation_steps = 1
    config.load_worker = int(os.environ.get("LINGBOT_VA_LOAD_WORKERS", 0))
    config.init_worker = int(os.environ.get("LINGBOT_VA_INIT_WORKERS", 1))

    device = torch.device(f"cuda:{local_rank}")
    checkpoint_transformer = Path(args.checkpoint) / "transformer"

    if rank == 0:
        print(f"Loading validation transformer from {checkpoint_transformer}", flush=True)
    transformer = load_transformer(
        str(checkpoint_transformer),
        torch_dtype=torch.float32,
        torch_device="cpu",
        attn_mode="flex",
    )
    apply_ac(transformer)
    transformer = _configure_model(
        model=transformer,
        shard_fn=shard_model,
        param_dtype=config.param_dtype,
        device=device,
        eval_mode=True,
    )

    # Reuse Trainer helper methods without calling Trainer.__init__.
    helper = object.__new__(Trainer)
    helper.config = config
    helper.device = device
    helper.dtype = config.param_dtype
    helper.patch_size = config.patch_size
    helper.gradient_accumulation_steps = 1
    helper.transformer = transformer
    helper.train_scheduler_latent = FlowMatchScheduler(
        shift=config.snr_shift, sigma_min=0.0, extra_one_step=True
    )
    helper.train_scheduler_latent.set_timesteps(1000, training=True)
    helper.train_scheduler_action = FlowMatchScheduler(
        shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True
    )
    helper.train_scheduler_action.set_timesteps(1000, training=True)

    if rank == 0:
        print("Constructing validation dataset", flush=True)
    val_dataset = MultiLatentLeRobotDataset(
        config=config,
        num_init_worker=getattr(config, "init_worker", 1),
    )
    sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=0,
        drop_last=False,
    )
    loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=config.load_worker,
    )

    metrics_path = Path(args.metrics_jsonl)
    report_path = Path(args.report_md)
    if rank == 0:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w", encoding="utf-8")
    else:
        metrics_file = None

    local_seen_batches = 0
    local_seen_labels = 0.0
    local_latent_loss_sum = 0.0
    local_action_loss_sum = 0.0
    local_total_loss_sum = 0.0
    start_time = time.time()

    max_batches = len(loader) if args.max_batches <= 0 else min(args.max_batches, len(loader))
    mode_name = "full" if args.max_batches <= 0 or max_batches == len(loader) else "smoke"

    pbar = tqdm(
        enumerate(loader),
        total=max_batches,
        disable=(rank != 0),
        desc="Validation",
    )
    transformer.eval()
    with torch.no_grad():
        for batch_idx, batch in pbar:
            if batch_idx >= max_batches:
                break
            batch = helper.convert_input_format(batch)
            input_dict = helper._prepare_input_dict(batch)
            output = transformer(input_dict, train_mode=True)
            latent_loss, action_loss = helper.compute_loss(input_dict, output)
            total_loss = latent_loss + action_loss
            seen_labels = batch["actions_mask"].sum().float()

            reduced = torch.tensor(
                [
                    latent_loss.detach().float().item(),
                    action_loss.detach().float().item(),
                    total_loss.detach().float().item(),
                    seen_labels.detach().float().item(),
                    1.0,
                ],
                device=device,
            )
            reduce_sum(reduced)
            global_latent, global_action, global_total, global_labels, global_count = (
                reduced.detach().cpu().tolist()
            )
            mean_latent = global_latent / global_count
            mean_action = global_action / global_count
            mean_total = global_total / global_count

            local_seen_batches += 1
            local_seen_labels += seen_labels.detach().float().item()
            local_latent_loss_sum += latent_loss.detach().float().item()
            local_action_loss_sum += action_loss.detach().float().item()
            local_total_loss_sum += total_loss.detach().float().item()

            if rank == 0:
                row = {
                    "batch": batch_idx,
                    "global_latent_loss": mean_latent,
                    "global_action_loss": mean_action,
                    "global_total_loss": mean_total,
                    "global_seen_labels": int(global_labels),
                }
                metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                metrics_file.flush()
                pbar.set_postfix(
                    {
                        "total": f"{mean_total:.4f}",
                        "action": f"{mean_action:.4f}",
                        "labels": f"{int(global_labels)}",
                    }
                )

    local_summary = torch.tensor(
        [
            local_latent_loss_sum,
            local_action_loss_sum,
            local_total_loss_sum,
            local_seen_labels,
            float(local_seen_batches),
        ],
        device=device,
    )
    reduce_sum(local_summary)
    elapsed = time.time() - start_time
    if rank == 0:
        metrics_file.close()
        latent_sum, action_sum, total_sum, label_sum, batch_sum = (
            local_summary.detach().cpu().tolist()
        )
        avg_latent = latent_sum / batch_sum
        avg_action = action_sum / batch_sum
        avg_total = total_sum / batch_sum
        report = [
            f"# Gate A4 Validation Loss {mode_name.title()}",
            "",
            "Status: pass",
            "",
            "Scope: held-out validation loss only. No training, no selector, no closed-loop evaluation.",
            "",
            f"- checkpoint: `{args.checkpoint}`",
            f"- seed: `{args.seed}`",
            f"- requested_max_batches_per_rank: `{args.max_batches}`",
            f"- evaluated_batches_per_rank: `{max_batches}`",
            f"- world_size: `{world_size}`",
            f"- validation_dataset_segments: `{len(val_dataset)}`",
            f"- global_batches_evaluated: `{int(batch_sum)}`",
            f"- global_seen_labels: `{int(label_sum)}`",
            f"- avg_latent_loss: `{avg_latent:.6f}`",
            f"- avg_action_loss: `{avg_action:.6f}`",
            f"- avg_total_loss: `{avg_total:.6f}`",
            f"- elapsed_seconds: `{elapsed:.2f}`",
            f"- metrics_jsonl: `{metrics_path}`",
            "",
            "Notes:",
            "",
            "- Diffusion validation loss depends on sampled timesteps/noise; this run fixes the torch seed per rank.",
            "- This is still an open-loop loss qualification, not a closed-loop success result.",
        ]
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        print(json.dumps({"status": "pass", "avg_total_loss": avg_total, "batches": batch_sum}, indent=2))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
