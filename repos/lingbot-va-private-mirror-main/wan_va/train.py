# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from lineage_audit import LineageTracer
from modules.lora import (
    LORA_ACTION_MODULES_TO_SAVE,
    add_lora_adapter,
    get_adapter_state_dict,
    load_lora_adapter,
    lora_optimizer_param_groups,
    save_lora_adapter,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import (
    MultiLatentLeRobotDataset,
    TokenBudgetPackingBatchSampler,
    episode_equal_loss,
    packed_episode_collate,
    stable_hash,
)
from dataset.resumable_sampler import ResumableDistributedSampler


def _transformer_fingerprint(transformer_path: str | Path) -> str:
    """Fingerprint model structure without reading multi-gigabyte tensor payloads."""
    transformer_path = Path(transformer_path)
    config_path = transformer_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Transformer config not found: {config_path}")
    with config_path.open() as file:
        config = json.load(file)
    for key in ("_name_or_path", "_diffusers_version", "attn_mode"):
        config.pop(key, None)

    weight_files = []
    for path in sorted(transformer_path.glob("*.safetensors")):
        weight_files.append({"name": path.name, "size": path.stat().st_size})
    return stable_hash({"config": config, "weight_files": weight_files})


def _sha256_file(path: str | Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(config, dataset) -> str:
    fingerprint = {
        "dataset_path": str(Path(config.dataset_path).resolve()),
        "length": len(dataset),
        "obs_cam_keys": list(config.obs_cam_keys),
        "env_type": config.env_type,
        "used_action_channel_ids": list(config.used_action_channel_ids),
        "norm_stat": getattr(config, "norm_stat", None),
        "packing_manifest_hash": getattr(dataset, "packing_manifest_hash", None),
    }
    # Preserve the legacy RoboTwin/LIBERO fingerprint byte-for-byte.  EBench
    # adds its audited stats identity, and targeted smoke selections add their
    # explicit episode IDs, without invalidating older non-EBench checkpoints.
    action_stats_sha256 = getattr(dataset, "action_stats_sha256", None)
    if action_stats_sha256 is not None:
        fingerprint["action_stats_sha256"] = action_stats_sha256
    segment_manifest_sha256 = getattr(dataset, "segment_manifest_sha256", None)
    if segment_manifest_sha256 is not None:
        fingerprint["segment_manifest_sha256"] = segment_manifest_sha256
    text_embeddings_sha256 = getattr(dataset, "text_embeddings_sha256", None)
    empty_embedding_sha256 = getattr(dataset, "empty_embedding_sha256", None)
    if text_embeddings_sha256 is not None:
        fingerprint["text_embeddings_sha256"] = text_embeddings_sha256
    if empty_embedding_sha256 is not None:
        fingerprint["empty_embedding_sha256"] = empty_embedding_sha256
    episode_indices = getattr(config, "ebench_episode_indices", None)
    if episode_indices is not None:
        fingerprint["ebench_episode_indices"] = list(episode_indices)
    return stable_hash(fingerprint)


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _sample_nonpacked_attention_layout(
    device: torch.device | str,
    *,
    sync_across_ranks: bool = False,
) -> tuple[int, int]:
    """Sample the random training mask layout, optionally once per global step.

    FlexAttention specializes work for the selected chunk/window layout.  For
    full-episode EBench batches, letting every rank select a different layout
    can leave one rank in local mask/attention work while the other ranks have
    already entered an FSDP collective.  Sampling on rank zero and broadcasting
    the two scalars preserves the existing random ranges while keeping every
    rank in a global optimizer step on the same attention schedule.
    """
    if (
        not sync_across_ranks
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return (
            int(torch.randint(1, 5, (1,)).item()),
            int(torch.randint(4, 65, (1,)).item()),
        )

    values = torch.empty(2, dtype=torch.long, device=device)
    if dist.get_rank() == 0:
        sampled = torch.tensor(
            [
                int(torch.randint(1, 5, (1,)).item()),
                int(torch.randint(4, 65, (1,)).item()),
            ],
            dtype=torch.long,
            device=device,
        )
        values.copy_(sampled)
    dist.broadcast(values, src=0)
    return int(values[0].item()), int(values[1].item())


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            wandb_mode = str(getattr(config, "wandb_mode", "online")).lower()
            if wandb_mode not in {"online", "offline", "disabled"}:
                raise ValueError(
                    f"Unsupported wandb_mode={wandb_mode!r}; expected online, offline, or disabled."
                )
            if wandb_mode != "online":
                # W&B validates connection-related environment variables even
                # in offline/disabled mode. Ignore any values inherited from a
                # cluster launcher because these modes never contact a server.
                os.environ.pop("WANDB_API_KEY", None)
                os.environ.pop("WANDB_BASE_URL", None)
                os.environ["WANDB_MODE"] = wandb_mode
            if wandb_mode == "online":
                wandb_api_key = os.getenv("WANDB_API_KEY")
                if not wandb_api_key:
                    raise RuntimeError("WANDB_API_KEY is required when wandb_mode='online'.")
                wandb.login(
                    host=os.getenv("WANDB_BASE_URL", "https://api.wandb.ai"),
                    key=wandb_api_key,
                )
            wandb_dir = Path(config.save_root).resolve()
            wandb_dir.mkdir(parents=True, exist_ok=True)
            self.wandb = wandb
            self.wandb.init(
                entity=(
                    getattr(config, "wandb_entity", None)
                    if wandb_mode != "online"
                    else getattr(config, "wandb_entity", None)
                    or os.getenv("WANDB_TEAM_NAME")
                ),
                project=getattr(
                    config,
                    "wandb_project",
                    os.getenv("WANDB_PROJECT", "va_robotwin"),
                ),
                dir=str(wandb_dir),
                config=config,
                mode=wandb_mode,
                name=getattr(config, "wandb_run_name", None),
            )
            logger.info(f"WandB logging enabled (mode={wandb_mode}, dir={wandb_dir})")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.training_mode = getattr(config, "training_mode", "full")
        if self.training_mode not in {"full", "lora"}:
            raise ValueError(f"Unsupported training mode: {self.training_mode}")
        self.resume_from = getattr(config, "resume_from", None)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.packing_enabled = bool(getattr(config, "packing_enabled", False))
        self.sync_packed_gradients_each_microbatch = bool(
            getattr(config, "sync_packed_gradients_each_microbatch", False)
        )
        self.global_episodes_per_update = int(
            getattr(
                config,
                "global_episodes_per_update",
                config.batch_size * config.world_size * self.gradient_accumulation_steps,
            )
        )
        if self.global_episodes_per_update % config.world_size != 0:
            raise ValueError(
                "global_episodes_per_update must be divisible by world_size: "
                f"{self.global_episodes_per_update} % {config.world_size} != 0"
            )
        self.local_episodes_per_update = (
            self.global_episodes_per_update // config.world_size
        )
        self.lora_rank = getattr(config, "lora_rank", 64)
        self.lora_alpha = getattr(config, "lora_alpha", self.lora_rank)
        self.lora_dropout = getattr(config, "lora_dropout", 0.0)
        self.lora_train_action_modules = bool(
            getattr(config, "lora_train_action_modules", False)
        )
        self.lora_action_modules = (
            tuple(LORA_ACTION_MODULES_TO_SAVE)
            if self.lora_train_action_modules
            else ()
        )
        self.action_learning_rate = float(
            getattr(config, "action_learning_rate", config.learning_rate)
        )
        self.action_weight_decay = float(
            getattr(config, "action_weight_decay", config.weight_decay)
        )
        self.lora_stats = None
        self.resume_manifest = None
        self.global_samples_seen = 0
        self.micro_batches_seen = 0
        self.batches_in_epoch = 0
        self.train_loader_iter = None
        self.interrupt_after_microbatches = getattr(
            config, "lineage_interrupt_after_microbatches", None
        )
        self.interrupt_exit_code = int(
            getattr(config, "lineage_interrupt_exit_code", 42)
        )

        if self.resume_from:
            manifest_path = Path(self.resume_from) / "manifest.json"
            if self.training_mode == "lora" and not manifest_path.is_file():
                raise FileNotFoundError(
                    f"LoRA resume manifest not found: {manifest_path}"
                )
            if manifest_path.is_file():
                with manifest_path.open() as file:
                    self.resume_manifest = json.load(file)
                if (
                    self.resume_manifest.get("format_version") != 1
                    or self.resume_manifest.get("training_mode") != self.training_mode
                ):
                    raise ValueError(
                        f"Unsupported {self.training_mode} checkpoint manifest: "
                        f"{manifest_path}"
                    )

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        base_transformer_path = os.path.join(
            config.wan22_pretrained_model_name_or_path, "transformer"
        )
        if self.training_mode == "full" and self.resume_from:
            transformer_path = os.path.join(self.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = base_transformer_path

        self.base_transformer_path = transformer_path
        self.base_transformer_fingerprint = _transformer_fingerprint(transformer_path)
        if self.resume_manifest is not None and self.training_mode == "lora":
            expected_fingerprint = self.resume_manifest.get("base_transformer_fingerprint")
            if expected_fingerprint != self.base_transformer_fingerprint:
                raise ValueError(
                    "The configured base transformer does not match the LoRA checkpoint."
                )
            saved_lora = self.resume_manifest.get("lora", {})
            requested_lora = {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
            }
            for key, value in requested_lora.items():
                if saved_lora.get(key) != value:
                    raise ValueError(
                        f"LoRA {key} mismatch: checkpoint={saved_lora.get(key)}, requested={value}."
                    )
            saved_action_modules = tuple(saved_lora.get("modules_to_save", ()))
            if saved_action_modules != self.lora_action_modules:
                raise ValueError(
                    "LoRA action modules mismatch: "
                    f"checkpoint={saved_action_modules}, "
                    f"requested={self.lora_action_modules}."
                )

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode="flex"
        )

        if self.training_mode == "lora":
            if self.resume_from:
                self.lora_stats = load_lora_adapter(
                    self.transformer,
                    Path(self.resume_from) / "adapter",
                    is_trainable=True,
                    validate_percentage=True,
                )
            else:
                self.lora_stats = add_lora_adapter(
                    self.transformer,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                    modules_to_save=self.lora_action_modules,
                    validate_percentage=True,
                )
            if config.rank == 0:
                logger.info(
                    "LoRA enabled: rank=%d alpha=%d targets=%d trainable=%d/%d (%.4f%%)",
                    self.lora_rank,
                    self.lora_alpha,
                    self.lora_stats.target_modules,
                    self.lora_stats.trainable_params,
                    self.lora_stats.total_params,
                    self.lora_stats.trainable_percent,
                )
                if self.lora_action_modules:
                    logger.info(
                        "Full action-module training enabled: modules=%s lr=%.2e weight_decay=%.2e",
                        ",".join(self.lora_action_modules),
                        self.action_learning_rate,
                        self.action_weight_decay,
                    )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(
            self.transformer,
            preserve_rng_state=(self.training_mode == "lora" and self.lora_dropout > 0),
        )

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        if self.training_mode == "full":
            self.transformer.requires_grad_(True)

        # Optimizer
        if self.training_mode == "lora":
            optimizer_parameters = lora_optimizer_param_groups(
                self.transformer,
                lora_learning_rate=config.learning_rate,
                lora_weight_decay=config.weight_decay,
                action_learning_rate=self.action_learning_rate,
                action_weight_decay=self.action_weight_decay,
            )
            has_action_group = any(
                group["group_name"] == "action" for group in optimizer_parameters
            )
            if has_action_group != self.lora_train_action_modules:
                raise RuntimeError(
                    "LoRA action optimizer group does not match "
                    f"lora_train_action_modules={self.lora_train_action_modules}"
                )
        else:
            optimizer_parameters = [
                parameter
                for parameter in self.transformer.parameters()
                if parameter.requires_grad
            ]

        self.optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        self.shared_text_embeddings = self.packing_enabled or bool(
            getattr(config, "shared_text_embeddings", False)
        )
        config.defer_cfg_dropout = (
            self.shared_text_embeddings or self.training_mode == "lora"
        )
        config.defer_text_embedding = self.shared_text_embeddings
        train_dataset = MultiLatentLeRobotDataset(
            config=config,
            num_init_worker=getattr(config, "load_worker", 8),
        )
        self.train_dataset = train_dataset
        self.text_embedding_cache = None
        self.empty_text_emb = None

        if self.shared_text_embeddings:
            text_cache_path = Path(
                getattr(
                    config,
                    "text_embeddings_path",
                    Path(config.dataset_path) / "text_embeddings.pt",
                )
            )
            audited_text_path = getattr(
                train_dataset, "text_embeddings_path", None
            )
            audited_text_sha256 = getattr(
                train_dataset, "text_embeddings_sha256", None
            )
            audited_empty_path = getattr(
                train_dataset, "empty_embedding_path", None
            )
            audited_empty_sha256 = getattr(
                train_dataset, "empty_embedding_sha256", None
            )
            audited_text = audited_text_path is not None
            if audited_text:
                if (
                    audited_text_sha256 is None
                    or audited_empty_path is None
                    or audited_empty_sha256 is None
                ):
                    raise ValueError("Audited shared text artifact metadata is incomplete")
                if text_cache_path.resolve() != Path(audited_text_path).resolve():
                    raise ValueError(
                        "Configured shared text cache differs from the audited artifact"
                    )
                empty_path = Path(config.empty_emb_path)
                if empty_path.resolve() != Path(audited_empty_path).resolve():
                    raise ValueError(
                        "Configured empty embedding differs from the audited artifact"
                    )
                if _sha256_file(text_cache_path) != audited_text_sha256:
                    raise ValueError("Audited shared text cache SHA256 mismatch")
                if _sha256_file(empty_path) != audited_empty_sha256:
                    raise ValueError("Audited empty embedding SHA256 mismatch")
            else:
                empty_path = None
            text_cache = torch.load(
                text_cache_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            if not isinstance(text_cache, dict) or "" not in text_cache:
                raise ValueError(
                    f"Shared text cache must be a dict containing the empty prompt: "
                    f"{text_cache_path}"
                )
            text_keys = sorted(text_cache)
            if not all(torch.is_tensor(text_cache[key]) for key in text_keys):
                raise ValueError("Shared text cache values must be tensors")
            text_shapes = {tuple(text_cache[key].shape) for key in text_keys}
            if text_shapes != {(512, 4096)}:
                raise ValueError(f"Unexpected shared text embedding shapes: {text_shapes}")
            if audited_text:
                if any(text_cache[key].dtype != torch.bfloat16 for key in text_keys):
                    raise ValueError("Audited shared text embeddings must use bfloat16")
                if any(
                    not torch.isfinite(text_cache[key]).all() for key in text_keys
                ):
                    raise ValueError("Audited shared text embeddings must be finite")
                empty_embedding = torch.load(
                    empty_path,
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
                if (
                    not torch.is_tensor(empty_embedding)
                    or empty_embedding.dtype != torch.bfloat16
                    or tuple(empty_embedding.shape) != (512, 4096)
                    or not torch.isfinite(empty_embedding).all()
                    or not torch.equal(text_cache[""], empty_embedding)
                ):
                    raise ValueError(
                        "Audited empty embedding does not match the shared text cache"
                    )
                del empty_embedding
            train_dataset.set_text_lookup(text_keys)
            self.text_embedding_cache = torch.stack(
                [text_cache[key] for key in text_keys]
            ).to(self.device, non_blocking=True)
            self.empty_text_id = text_keys.index("")
            del text_cache

        if self.packing_enabled:
            packing_manifest = train_dataset.get_packing_manifest()
            train_sampler = TokenBudgetPackingBatchSampler(
                packing_manifest,
                global_episodes_per_update=self.global_episodes_per_update,
                max_self_tokens=int(config.max_self_tokens),
                max_episodes_per_pack=int(config.max_episodes_per_pack),
                num_replicas=config.world_size,
                rank=config.rank,
                seed=int(getattr(config, "packing_seed", 42)),
            )
            train_dataset.packing_manifest_hash = train_sampler.manifest_hash
        elif self.training_mode == "lora":
            train_sampler = ResumableDistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                seed=42,
            )
        else:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=42
            ) if config.world_size > 1 else None
        self.train_sampler = train_sampler
        self.action_stats_path = getattr(train_dataset, "action_stats_path", None)
        self.action_stats_sha256 = getattr(
            train_dataset, "action_stats_sha256", None
        )
        self.segment_manifest_path = getattr(
            train_dataset, "segment_manifest_path", None
        )
        self.segment_manifest_sha256 = getattr(
            train_dataset, "segment_manifest_sha256", None
        )
        self.segment_audit_path = getattr(
            train_dataset, "segment_audit_path", None
        )
        self.dataset_fingerprint = _dataset_fingerprint(config, train_dataset)
        train_dataset.dataset_fingerprint = self.dataset_fingerprint
        if self.packing_enabled and dist.is_initialized():
            fingerprints = [None] * config.world_size
            dist.all_gather_object(fingerprints, self.dataset_fingerprint)
            if len(set(fingerprints)) != 1:
                raise RuntimeError(
                    f"Dataset/packing manifests differ across ranks: {fingerprints}"
                )
        if self.packing_enabled and config.rank == 0:
            initial_plan = train_sampler.plan_update(0)
            micro_count = len(initial_plan[0])
            rank_tokens = [
                sum(item.info.token_count for pack in packs for item in pack)
                for packs in initial_plan
            ]
            logger.info(
                "True packing enabled: global_episodes_per_update=%d, "
                "local_episodes_per_update=%d, initial_micro_batches=%d, "
                "rank_token_imbalance=%.3f, sync_gradients_each_microbatch=%s",
                self.global_episodes_per_update,
                self.local_episodes_per_update,
                micro_count,
                max(rank_tokens) / max(min(rank_tokens), 1),
                self.sync_packed_gradients_each_microbatch,
            )

        self.lineage_tracer = None
        lineage_audit_dir = getattr(config, "lineage_audit_dir", None)
        if lineage_audit_dir:
            sampler_probability = (
                1.0 / len(train_dataset) if len(train_dataset) > 0 else None
            )
            self.lineage_tracer = LineageTracer(
                output_dir=lineage_audit_dir,
                run_id=getattr(
                    config,
                    "lineage_run_id",
                    Path(config.save_root).resolve().name,
                ),
                config=config,
                dataset=train_dataset,
                sampler=train_sampler,
                source_view=getattr(config, "lineage_source_view", "full"),
                subset_manifest_id=getattr(
                    config, "lineage_subset_manifest_id", "full"
                ),
                sampler_probability=sampler_probability,
                compute_support=(config.rank == 0),
            )
            _barrier()

        loader_kwargs = {
            "num_workers": int(config.load_worker),
            "pin_memory": bool(getattr(config, "pin_memory", True)),
        }
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = bool(
                getattr(config, "persistent_workers", True)
            )
            loader_kwargs["prefetch_factor"] = int(
                getattr(config, "prefetch_factor", 2)
            )
        if self.packing_enabled:
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                collate_fn=packed_episode_collate,
                **loader_kwargs,
            )
        else:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                **loader_kwargs,
            )
        if (
            self.training_mode == "lora"
            and not self.shared_text_embeddings
        ):
            self.empty_text_emb = torch.load(config.empty_emb_path, weights_only=False)

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.resume_from and self.resume_manifest is not None:
            self._load_training_state(self.resume_from)
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            if self.packing_enabled:
                raise RuntimeError("Packed DataLoader unexpectedly exhausted")
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.batches_in_epoch = 0
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)

        self.batches_in_epoch += 1
        self.micro_batches_seen += 1
        if not self.packing_enabled:
            local_batch_size = next(
                value.shape[0]
                for value in batch.values()
                if isinstance(value, torch.Tensor) and value.ndim > 0
            )
            self.global_samples_seen += local_batch_size * self.config.world_size
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    def _stateless_seed(self, sample_uid: int, tag: str) -> int:
        payload = (
            f"{int(getattr(self.config, 'packing_seed', 42))}:"
            f"{sample_uid}:{tag}"
        ).encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "little") % (2**63 - 1)

    def _stateless_uniform(self, sample_uid: int, tag: str) -> float:
        return self._stateless_seed(sample_uid, tag) / float(2**63 - 1)

    def _episode_generator(self, sample_uid: int, tag: str):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self._stateless_seed(sample_uid, tag))
        return generator

    @torch.no_grad()
    def _add_noise_packed(
        self,
        latent,
        train_scheduler,
        frame_lengths,
        sample_uids,
        *,
        tag_prefix,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
    ):
        """Generate independent diffusion inputs for each packed occurrence."""
        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        scheduler_timesteps = train_scheduler.timesteps.to(self.device)
        noisy_parts = []
        target_parts = []
        clean_parts = []
        timestep_parts = []
        cond_timestep_parts = []
        grid_parts = []
        frame_offset = 0

        for frame_count, sample_uid in zip(frame_lengths, sample_uids):
            segment = latent[:, :, frame_offset : frame_offset + frame_count]
            timestep_u = torch.rand(
                (frame_count,),
                generator=self._episode_generator(sample_uid, f"{tag_prefix}:timestep"),
                device=self.device,
            )
            timestep_ids = (timestep_u * train_scheduler.num_train_timesteps).clamp(
                min=0, max=train_scheduler.num_train_timesteps - 1
            ).long()
            timesteps = scheduler_timesteps[timestep_ids]
            noise = torch.randn(
                segment.shape,
                dtype=segment.dtype,
                device=segment.device,
                generator=self._episode_generator(sample_uid, f"{tag_prefix}:noise"),
            )
            noisy_segment = train_scheduler.add_noise(
                segment, noise, timesteps, t_dim=2
            )
            target_segment = train_scheduler.training_target(segment, noise, timesteps)
            clean_segment = segment

            if self._stateless_uniform(sample_uid, f"{tag_prefix}:cond_gate") < noisy_cond_prob:
                cond_u = torch.rand(
                    (frame_count,),
                    generator=self._episode_generator(
                        sample_uid, f"{tag_prefix}:cond_timestep"
                    ),
                    device=self.device,
                )
                cond_ids = (
                    (cond_u * 0.5 + 0.5) * train_scheduler.num_train_timesteps
                ).clamp(
                    min=0, max=train_scheduler.num_train_timesteps - 1
                ).long()
                cond_timesteps = scheduler_timesteps[cond_ids]
                cond_noise = torch.randn(
                    segment.shape,
                    dtype=segment.dtype,
                    device=segment.device,
                    generator=self._episode_generator(
                        sample_uid, f"{tag_prefix}:cond_noise"
                    ),
                )
                clean_segment = train_scheduler.add_noise(
                    segment, cond_noise, cond_timesteps, t_dim=2
                )
            else:
                cond_timesteps = torch.zeros_like(timesteps)

            if action_mask is not None:
                mask_segment = action_mask[
                    :, :, frame_offset : frame_offset + frame_count
                ].float()
                noisy_segment = noisy_segment * mask_segment
                target_segment = target_segment * mask_segment
                clean_segment = clean_segment * mask_segment

            grid_parts.append(
                get_mesh_id(
                    frame_count // patch_f,
                    segment.shape[-2] // patch_h,
                    segment.shape[-1] // patch_w,
                    t=1 if action_mode else 0,
                    f_w=1,
                    f_shift=0,
                    action=action_mode,
                ).to(self.device)
            )
            noisy_parts.append(noisy_segment)
            target_parts.append(target_segment)
            clean_parts.append(clean_segment)
            timestep_parts.append(timesteps)
            cond_timestep_parts.append(cond_timesteps)
            frame_offset += frame_count

        return {
            "timesteps": torch.cat(timestep_parts)[None],
            "noisy_latents": torch.cat(noisy_parts, dim=2),
            "targets": torch.cat(target_parts, dim=2),
            "latent": torch.cat(clean_parts, dim=2),
            "cond_timesteps": torch.cat(cond_timestep_parts)[None],
            "grid_id": torch.cat(grid_parts, dim=1)[None],
        }

    @torch.no_grad()
    def _prepare_packed_input_dict(self, batch_dict):
        frame_lengths = [int(value) for value in batch_dict["frame_lengths"].tolist()]
        sample_uids = [int(value) for value in batch_dict["sample_uid"].tolist()]
        if (
            not frame_lengths
            or len(frame_lengths) != len(sample_uids)
            or any(frame_count <= 0 for frame_count in frame_lengths)
        ):
            raise ValueError("Packed episode metadata is empty or inconsistent")
        text_ids = batch_dict["text_id"].to(self.device, non_blocking=True)
        text_emb = self.text_embedding_cache.index_select(0, text_ids).clone()
        for episode_idx, sample_uid in enumerate(sample_uids):
            if self._stateless_uniform(sample_uid, "cfg_dropout") < self.config.cfg_prob:
                text_emb[episode_idx] = self.text_embedding_cache[self.empty_text_id]

        latent_dict = self._add_noise_packed(
            batch_dict["latents"],
            self.train_scheduler_latent,
            frame_lengths,
            sample_uids,
            tag_prefix="video",
            noisy_cond_prob=0.5,
        )
        action_dict = self._add_noise_packed(
            batch_dict["actions"],
            self.train_scheduler_action,
            frame_lengths,
            sample_uids,
            tag_prefix="action",
            action_mask=batch_dict["actions_mask"],
            action_mode=True,
        )
        action_dict["frame_loss_weights"] = self.train_scheduler_action.training_weight(
            action_dict["timesteps"].flatten()
        ).reshape(1, -1)
        latent_dict["text_emb"] = text_emb
        action_dict["text_emb"] = text_emb
        action_dict["actions_mask"] = batch_dict["actions_mask"]
        return {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "action_frame_loss_weights": action_dict["frame_loss_weights"].flatten(),
            "frame_lengths": batch_dict["frame_lengths"],
            "frame_offsets": batch_dict["frame_offsets"],
            "chunk_sizes": torch.tensor(
                [1 + self._stateless_seed(uid, "chunk_size") % 4 for uid in sample_uids],
                dtype=torch.long,
            ),
            "window_sizes": torch.tensor(
                [4 + self._stateless_seed(uid, "window_size") % 61 for uid in sample_uids],
                dtype=torch.long,
            ),
            "planned_tokens": int(batch_dict["planned_tokens"]),
        }

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        if self.packing_enabled:
            return self._prepare_packed_input_dict(batch_dict)
        if self.shared_text_embeddings:
            text_ids = batch_dict['text_id'].to(self.device, non_blocking=True)
            text_emb = self.text_embedding_cache.index_select(0, text_ids).clone()
            if torch.rand(1, device=self.device).item() < self.config.cfg_prob:
                text_emb[:] = self.text_embedding_cache[self.empty_text_id]
        else:
            text_emb = batch_dict['text_emb']
            if (
                self.training_mode == "lora"
                and torch.rand(1, device=self.device).item() < self.config.cfg_prob
            ):
                empty_text_emb = self.empty_text_emb.to(text_emb)
                while empty_text_emb.ndim < text_emb.ndim:
                    empty_text_emb = empty_text_emb.unsqueeze(0)
                text_emb = empty_text_emb.expand_as(text_emb)

        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)
        action_dict["frame_loss_weights"] = self.train_scheduler_action.training_weight(
            action_dict["timesteps"].flatten()
        ).reshape(1, -1)

        latent_dict['text_emb'] = text_emb
        action_dict['text_emb'] = text_emb
        action_dict['actions_mask'] = batch_dict['actions_mask']

        chunk_size, window_size = _sample_nonpacked_attention_layout(
            self.device,
            sync_across_ranks=bool(
                getattr(
                    self.config,
                    "sync_nonpacked_attention_layout_across_ranks",
                    False,
                )
            ),
        )
        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            "action_frame_loss_weights": action_dict["frame_loss_weights"].flatten(),
            'chunk_size': chunk_size,
            'window_size': window_size,
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        cpu_metadata = {
            "frame_lengths",
            "frame_offsets",
            "sample_uid",
            "update_id",
            "micro_idx",
            "micro_count",
            "planned_tokens",
            "lineage_records",
        }
        for key, value in input_dict.items():
            if key not in cpu_metadata:
                input_dict[key] = value.to(self.device, non_blocking=True)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss_per_frame = latent_loss_per_frame / (
            latent_mask_per_frame + 1e-6
        )

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss_per_frame = action_loss_per_frame / (
            action_mask_per_frame + 1e-6
        )

        if "frame_lengths" in input_dict:
            frame_lengths = [int(value) for value in input_dict["frame_lengths"].tolist()]
            # FSDP averages gradients across ranks. Each rank divides by the
            # configured number of local episodes, so the final gradient is an
            # equal-weight mean over global_episodes_per_update.
            latent_loss = episode_equal_loss(
                latent_loss_per_frame,
                frame_lengths,
                self.local_episodes_per_update,
            )
            action_loss = episode_equal_loss(
                action_loss_per_frame,
                frame_lengths,
                self.local_episodes_per_update,
            )
            return latent_loss, action_loss

        latent_loss = latent_loss_per_frame.mean()
        action_loss = action_loss_per_frame.mean()
        return (
            latent_loss / self.gradient_accumulation_steps,
            action_loss / self.gradient_accumulation_steps,
        )

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        if self.packing_enabled:
            micro_idx = int(batch["micro_idx"])
            micro_count = int(batch["micro_count"])
            update_id = int(batch["update_id"])
            if not 0 <= micro_idx < micro_count:
                raise ValueError(
                    f"Invalid packed microstep {micro_idx}/{micro_count}"
                )
            should_sync = micro_idx + 1 == micro_count
        else:
            micro_idx = batch_idx
            micro_count = self.gradient_accumulation_steps
            update_id = None
            should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0

        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        if self.lineage_tracer is not None:
            self.lineage_tracer.capture_microbatch(
                batch=batch,
                input_dict=input_dict,
                optimizer_step=self.step,
                microbatch_id=self.micro_batches_seen - 1,
                gradient_accumulation_index=micro_idx,
            )
            if (
                self.interrupt_after_microbatches is not None
                and self.micro_batches_seen >= int(self.interrupt_after_microbatches)
            ):
                logger.error(
                    "Forced lineage interruption after microbatch %d before optimizer update.",
                    self.micro_batches_seen - 1,
                )
                os._exit(self.interrupt_exit_code)
        
        requires_gradient_sync = should_sync or (
            self.packing_enabled and self.sync_packed_gradients_each_microbatch
        )
        self.transformer.set_requires_gradient_sync(requires_gradient_sync)

        fb_started = time.perf_counter()
        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss

        loss.backward()
        if self.lineage_tracer is not None:
            self.lineage_tracer.add_timing(
                "forward_backward_wall_clock_s", time.perf_counter() - fb_started
            )

        losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}
        
        # Only update weights after accumulating gradients
        if should_sync:
            opt_started = time.perf_counter()
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            if self.packing_enabled:
                self.train_sampler.mark_update_completed(update_id)
                self.global_samples_seen += self.global_episodes_per_update
            if self.lineage_tracer is not None:
                self.lineage_tracer.commit_update(self.step)
                self.lineage_tracer.add_timing(
                    "optimizer_update_wall_clock_s", time.perf_counter() - opt_started
                )
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        losses["micro_idx"] = micro_idx
        losses["micro_count"] = micro_count
        if self.packing_enabled:
            losses["planned_tokens"] = int(batch["planned_tokens"])
            losses["episode_count"] = int(batch["frame_lengths"].numel())
        return losses

    def save_checkpoint(self):
        """Save a legacy full checkpoint or an adapter-only LoRA checkpoint."""
        try:
            if self.training_mode == "lora":
                self._save_lora_checkpoint()
            else:
                self._save_full_checkpoint()
            if self.config.rank == 0:
                logger.info(f"Checkpoint saved successfully at step {self.step}")
            _barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            _barrier()

    def _save_full_checkpoint(self):
        """Save a full fine-tuning checkpoint with resumable trainer state."""
        checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
        trainer_state_dir = checkpoint_dir / "trainer_state"
        if self.config.rank == 0:
            trainer_state_dir.mkdir(parents=True, exist_ok=True)
        _barrier()

        state_dict = get_model_state_dict(
            self.transformer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        optimizer_state = get_optimizer_state_dict(
            self.transformer,
            self.optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if self.config.rank != 0:
            torch.save(
                self._rng_state(),
                trainer_state_dir / f"rng_rank_{self.config.rank}.pt",
            )
            return

        transformer_dir = checkpoint_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving transformer to {transformer_dir}")
        state_dict_bf16 = {key: value.to(torch.bfloat16) for key, value in state_dict.items()}
        save_file(
            state_dict_bf16,
            transformer_dir / "diffusion_pytorch_model.safetensors",
        )
        config_dict = dict(self.transformer.config)
        config_dict.pop('_name_or_path', None)
        with (transformer_dir / "config.json").open('w') as file:
            json.dump(config_dict, file, indent=2)
        torch.save(optimizer_state, trainer_state_dir / "optimizer.pt")
        torch.save(self.lr_scheduler.state_dict(), trainer_state_dir / "scheduler.pt")
        with (trainer_state_dir / "sampler.json").open("w") as file:
            json.dump(self._sampler_state(), file, indent=2)
        torch.save(
            self._rng_state(),
            trainer_state_dir / f"rng_rank_{self.config.rank}.pt",
        )
        with (checkpoint_dir / "manifest.json").open("w") as file:
            json.dump(self._full_manifest(), file, indent=2)
        self._save_action_stats_artifact(checkpoint_dir)
        (checkpoint_dir / "_SUCCESS").touch()

    def _save_action_stats_artifact(self, checkpoint_dir: Path) -> None:
        """Copy the exact versioned EBench stats used by this training run."""
        if self.action_stats_path is None:
            return
        source = Path(self.action_stats_path)
        if not source.is_file():
            raise FileNotFoundError(f"Action stats artifact disappeared: {source}")
        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if (
            self.action_stats_sha256 is not None
            and actual_sha256 != self.action_stats_sha256
        ):
            raise RuntimeError(
                "Action stats artifact changed after dataset initialization: "
                f"expected={self.action_stats_sha256}, actual={actual_sha256}"
            )
        destination = checkpoint_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        artifact_manifest = {
            "filename": destination.name,
            "sha256": actual_sha256,
            "source": str(source.resolve()),
        }
        segment_manifest_path = getattr(self, "segment_manifest_path", None)
        if segment_manifest_path is not None:
            segment_source = Path(segment_manifest_path)
            segment_sha256 = _sha256_file(segment_source)
            if segment_sha256 != getattr(self, "segment_manifest_sha256", None):
                raise RuntimeError("EBench segment manifest changed during training")
            segment_destination = checkpoint_dir / segment_source.name
            if segment_source.resolve() != segment_destination.resolve():
                shutil.copy2(segment_source, segment_destination)
            artifact_manifest["segment_manifest"] = {
                "filename": segment_destination.name,
                "sha256": segment_sha256,
                "source": str(segment_source.resolve()),
            }
        segment_audit_path = getattr(self, "segment_audit_path", None)
        if segment_audit_path is not None:
            audit_source = Path(segment_audit_path)
            audit_destination = checkpoint_dir / audit_source.name
            if audit_source.resolve() != audit_destination.resolve():
                shutil.copy2(audit_source, audit_destination)
            artifact_manifest["segment_audit"] = {
                "filename": audit_destination.name,
                "sha256": _sha256_file(audit_destination),
                "source": str(audit_source.resolve()),
            }
        with (checkpoint_dir / "dataset_artifacts.json").open("w") as file:
            json.dump(artifact_manifest, file, indent=2)

    def _rng_state(self) -> dict:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state(self.device)
        return state

    def _sampler_state(self) -> dict:
        state = {
            "epoch": getattr(self.train_sampler, "epoch", 0),
            "batches_in_epoch": self.batches_in_epoch,
            "micro_batches_seen": self.micro_batches_seen,
            "global_samples_seen": self.global_samples_seen,
        }
        if self.packing_enabled:
            state["packing"] = self.train_sampler.state_dict()
        return state

    def _lora_manifest(self) -> dict:
        return {
            "format_version": 1,
            "training_mode": "lora",
            "step": self.step,
            "base_transformer_path": str(Path(self.base_transformer_path).resolve()),
            "base_transformer_fingerprint": self.base_transformer_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "action_stats": (
                {
                    "filename": Path(self.action_stats_path).name,
                    "sha256": self.action_stats_sha256,
                }
                if self.action_stats_path is not None
                else None
            ),
            "lora": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "modules_to_save": list(self.lora_action_modules),
                **self.lora_stats.to_dict(),
            },
            "optimizer": {
                "lora_learning_rate": self.config.learning_rate,
                "lora_weight_decay": self.config.weight_decay,
                "action_learning_rate": self.action_learning_rate,
                "action_weight_decay": self.action_weight_decay,
            },
            "topology": {
                "world_size": self.config.world_size,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "packing_enabled": self.packing_enabled,
                "global_episodes_per_update": self.global_episodes_per_update,
            },
            "sampler": self._sampler_state(),
        }

    def _full_manifest(self) -> dict:
        return {
            "format_version": 1,
            "training_mode": "full",
            "step": self.step,
            "base_transformer_path": str(Path(self.base_transformer_path).resolve()),
            "base_transformer_fingerprint": self.base_transformer_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "action_stats": (
                {
                    "filename": Path(self.action_stats_path).name,
                    "sha256": self.action_stats_sha256,
                }
                if self.action_stats_path is not None
                else None
            ),
            "optimizer": {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            },
            "topology": {
                "world_size": self.config.world_size,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "packing_enabled": self.packing_enabled,
                "global_episodes_per_update": self.global_episodes_per_update,
            },
            "sampler": self._sampler_state(),
        }

    def _save_lora_checkpoint(self):
        checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
        trainer_state_dir = checkpoint_dir / "trainer_state"
        if self.config.rank == 0:
            trainer_state_dir.mkdir(parents=True, exist_ok=True)
        _barrier()

        model_state = get_model_state_dict(
            self.transformer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=True,
            ),
        )
        optimizer_state = get_optimizer_state_dict(
            self.transformer,
            self.optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        if self.config.rank == 0:
            adapter_state = get_adapter_state_dict(self.transformer, model_state)
            save_lora_adapter(
                self.transformer,
                adapter_state,
                checkpoint_dir / "adapter",
            )
            torch.save(optimizer_state, trainer_state_dir / "optimizer.pt")
            torch.save(self.lr_scheduler.state_dict(), trainer_state_dir / "scheduler.pt")
            with (trainer_state_dir / "sampler.json").open("w") as file:
                json.dump(self._sampler_state(), file, indent=2)
            with (checkpoint_dir / "manifest.json").open("w") as file:
                json.dump(self._lora_manifest(), file, indent=2)
            self._save_action_stats_artifact(checkpoint_dir)

        torch.save(
            self._rng_state(),
            trainer_state_dir / f"rng_rank_{self.config.rank}.pt",
        )
        _barrier()
        if self.config.rank == 0:
            (checkpoint_dir / "_SUCCESS").touch()

    def _load_training_state(self, checkpoint_path):
        """Restore optimizer, scheduler, sampler, RNG, and step after FSDP setup."""
        checkpoint_dir = Path(checkpoint_path)
        trainer_state_dir = checkpoint_dir / "trainer_state"
        required_paths = [
            checkpoint_dir / "_SUCCESS",
            trainer_state_dir / "optimizer.pt",
            trainer_state_dir / "scheduler.pt",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete {self.training_mode} checkpoint; missing: "
                + ", ".join(missing)
            )
        if self.resume_manifest.get("dataset_fingerprint") != self.dataset_fingerprint:
            raise ValueError(
                f"The dataset does not match the {self.training_mode} resume checkpoint."
            )

        saved_topology = self.resume_manifest["topology"]
        current_topology = {
            "world_size": self.config.world_size,
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "packing_enabled": self.packing_enabled,
            "global_episodes_per_update": self.global_episodes_per_update,
        }
        topology_changed = saved_topology != current_topology
        if topology_changed and self.config.rank == 0:
            logger.warning(
                "Resuming with a changed topology. Model, optimizer, scheduler, and step will be "
                "restored, but data order and RNG trajectory will not be bit-exact. "
                f"saved={saved_topology}, current={current_topology}"
            )

        optimizer_state = torch.load(
            trainer_state_dir / "optimizer.pt",
            map_location="cpu",
            weights_only=False,
        )
        if self.training_mode == "full":
            state = optimizer_state.setdefault("state", {})
            missing_optimizer_state_keys = []
            for group in optimizer_state.get("param_groups", []):
                for fqn in group.get("params", []):
                    if fqn not in state:
                        state[fqn] = {}
                        missing_optimizer_state_keys.append(fqn)
            if missing_optimizer_state_keys and self.config.rank == 0:
                logger.warning(
                    "Full checkpoint optimizer state had %d param group entries with no slot state; "
                    "restored them as empty AdamW states. First examples: %s",
                    len(missing_optimizer_state_keys),
                    missing_optimizer_state_keys[:5],
                )
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=optimizer_state,
            options=StateDictOptions(
                full_state_dict=True,
                strict=(not topology_changed and self.training_mode == "lora"),
            ),
        )
        scheduler_state = torch.load(
            trainer_state_dir / "scheduler.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.lr_scheduler.load_state_dict(scheduler_state)
        self.step = self.resume_manifest["step"]

        saved_sampler = self.resume_manifest["sampler"]
        self.micro_batches_seen = saved_sampler["micro_batches_seen"]
        self.global_samples_seen = saved_sampler["global_samples_seen"]
        if self.packing_enabled:
            if "packing" not in saved_sampler:
                raise ValueError("Packed resume checkpoint has no packing cursor")
            self.train_sampler.load_state_dict(saved_sampler["packing"])
            expected_samples = self.train_sampler.committed_global_cursor
            if self.global_samples_seen != expected_samples:
                raise ValueError(
                    "Packed checkpoint sample cursor mismatch: "
                    f"trainer={self.global_samples_seen}, sampler={expected_samples}"
                )
            self.batches_in_epoch = saved_sampler["batches_in_epoch"]
            self.train_loader_iter = iter(self.train_loader)
        elif topology_changed:
            samples_per_epoch = len(self.train_dataset)
            sampler_epoch, samples_in_epoch = divmod(
                self.global_samples_seen,
                max(samples_per_epoch, 1),
            )
            global_batch_size = self.config.batch_size * self.config.world_size
            self.batches_in_epoch = samples_in_epoch // max(global_batch_size, 1)
        else:
            sampler_epoch = saved_sampler["epoch"]
            self.batches_in_epoch = saved_sampler["batches_in_epoch"]
        if not self.packing_enabled:
            self.train_sampler.set_epoch(sampler_epoch)
            start_index = min(
                self.batches_in_epoch * self.config.batch_size,
                self.train_sampler.num_samples,
            )
            self.train_sampler.set_start_index(start_index)
            self.train_loader_iter = iter(self.train_loader)

        rng_path = trainer_state_dir / f"rng_rank_{self.config.rank}.pt"
        if not topology_changed and rng_path.is_file():
            rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch_cpu"])
            if "torch_cuda" in rng_state:
                torch.cuda.set_rng_state(rng_state["torch_cuda"], self.device)
        else:
            resume_seed = 42 + self.step * 100_003 + self.config.rank
            random.seed(resume_seed)
            np.random.seed(resume_seed % (2**32))
            torch.manual_seed(resume_seed)
            torch.cuda.manual_seed(resume_seed)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")
        _barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_planned_tokens = 0
        accumulated_episode_count = 0
        step_in_accumulation = 0
        update_started_at = time.perf_counter()
        end_to_end_started_at = time.perf_counter()

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            dataloader_started = time.perf_counter()
            batch = self._get_next_batch()
            if self.lineage_tracer is not None:
                self.lineage_tracer.add_timing(
                    "dataloader_wait_wall_clock_s",
                    time.perf_counter() - dataloader_started,
                )
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            if self.packing_enabled:
                accumulated_planned_tokens += losses["planned_tokens"]
                accumulated_episode_count += losses["episode_count"]
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                update_seconds = time.perf_counter() - update_started_at
                current_lrs = self.lr_scheduler.get_last_lr()
                lr = current_lrs[0]
                action_lr = current_lrs[1] if len(current_lrs) > 1 else None

                # Average accumulated losses
                local_loss_sums = torch.stack(
                    [
                        torch.stack(accumulated_latent_losses).sum(),
                        torch.stack(accumulated_action_losses).sum(),
                    ]
                )
                mean_loss_sums = dist_mean(local_loss_sums.clone()).detach().cpu()
                max_loss_sums = dist_max(local_loss_sums).detach().cpu()
                latent_loss_show, action_loss_show = mean_loss_sums.tolist()
                max_latent_loss_show, max_action_loss_show = max_loss_sums.tolist()
                if self.packing_enabled:
                    if accumulated_episode_count != self.local_episodes_per_update:
                        raise RuntimeError(
                            "Packed update consumed "
                            f"{accumulated_episode_count} local episodes, expected "
                            f"{self.local_episodes_per_update}."
                        )
                    packing_fill = accumulated_planned_tokens / (
                        losses["micro_count"] * self.config.max_self_tokens
                    )
                else:
                    packing_fill = 0.0

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_planned_tokens = 0
                accumulated_episode_count = 0
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += 1
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'packs': losses['micro_count'],
                        'step_s': f'{update_seconds:.1f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        wandb_metrics = {
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'packing/micro_batches_per_update': losses['micro_count'],
                            'packing/token_fill': packing_fill,
                            'packing/global_episodes_per_update': self.global_episodes_per_update,
                            'performance/update_seconds': update_seconds,
                            'performance/episodes_per_second': (
                                self.global_episodes_per_update / update_seconds
                            ),
                            'lr': lr,
                        }
                        if action_lr is not None:
                            wandb_metrics['action_lr'] = action_lr
                        self.wandb.log(wandb_metrics, step=self.step)
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    checkpoint_started = time.perf_counter()
                    self.save_checkpoint()
                    if self.lineage_tracer is not None:
                        self.lineage_tracer.add_timing(
                            "checkpoint_IO_wall_clock_s",
                            time.perf_counter() - checkpoint_started,
                        )
                update_started_at = time.perf_counter()
        progress_bar.close()
        logger.info("Training completed!")
        if self.lineage_tracer is not None:
            torch.cuda.synchronize()
            self.lineage_tracer.add_timing(
                "end_to_end_wall_clock_s", time.perf_counter() - end_to_end_started_at
            )
            self.lineage_tracer.finalize_rank(
                peak_vram_bytes=int(torch.cuda.max_memory_allocated(self.device))
            )
            _barrier()
            if self.config.rank == 0:
                self.lineage_tracer.finalize_global()
            _barrier()


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.training_mode is not None:
        config.training_mode = args.training_mode
    resolved_training_mode = getattr(config, "training_mode", "full")
    if resolved_training_mode == "lora":
        if args.lora_rank is not None:
            config.lora_rank = args.lora_rank
        elif not hasattr(config, "lora_rank"):
            config.lora_rank = 64
        if args.lora_alpha is not None:
            config.lora_alpha = args.lora_alpha
        elif not hasattr(config, "lora_alpha"):
            config.lora_alpha = config.lora_rank
        if args.lora_dropout is not None:
            config.lora_dropout = args.lora_dropout
        elif not hasattr(config, "lora_dropout"):
            config.lora_dropout = 0.0
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.global_episodes_per_update is not None:
        config.global_episodes_per_update = args.global_episodes_per_update
    if args.max_self_tokens is not None:
        config.max_self_tokens = args.max_self_tokens
    if args.max_episodes_per_pack is not None:
        config.max_episodes_per_pack = args.max_episodes_per_pack
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.save_interval is not None:
        if args.save_interval <= 0:
            raise ValueError("--save-interval must be positive")
        config.save_interval = args.save_interval
    if args.seed is not None:
        config.seed = int(args.seed)
        seed = int(args.seed) + rank
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    if args.packing_seed is not None:
        config.packing_seed = int(args.packing_seed)
    if args.lineage_audit_dir is not None:
        config.lineage_audit_dir = args.lineage_audit_dir
        config.lineage_run_id = args.lineage_run_id or (
            Path(args.save_root).resolve().name if args.save_root else "lineage_audit"
        )
        config.lineage_source_view = args.lineage_source_view
        config.lineage_subset_manifest_id = args.lineage_subset_manifest_id
        config.lineage_logical_run_id = (
            args.lineage_logical_run_id or config.lineage_run_id
        )
        config.lineage_training_attempt_id = (
            args.lineage_training_attempt_id
            or f"{config.lineage_run_id}_rank{rank}_attempt0"
        )
        config.lineage_resume_generation = int(args.lineage_resume_generation)
        config.lineage_checkpoint_base_step = int(args.lineage_checkpoint_base_step)
        config.lineage_shard_steps = int(args.lineage_shard_steps)
        if args.lineage_interrupt_after_microbatches is not None:
            config.lineage_interrupt_after_microbatches = int(
                args.lineage_interrupt_after_microbatches
            )
            config.lineage_interrupt_exit_code = int(args.lineage_interrupt_exit_code)
    if args.load_worker is not None:
        if args.load_worker < 0:
            raise ValueError("--load-worker must be non-negative")
        config.load_worker = args.load_worker
    if args.ebench_episode_indices is not None:
        if getattr(config, "dataset_type", None) != "ebench":
            raise ValueError(
                "--ebench-episode-indices is only valid with an EBench config"
            )
        config.ebench_episode_indices = tuple(args.ebench_episode_indices)
    if args.disable_wandb:
        config.enable_wandb = False

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Training mode: {getattr(config, 'training_mode', 'full')}")

    try:
        trainer = Trainer(config)
        trainer.train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--training-mode",
        choices=("full", "lora"),
        default=None,
        help="Fine-tuning mode. Defaults to the config value or legacy full fine-tuning.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank (default: 64 in LoRA mode).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha; it must equal the selected rank.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="LoRA dropout probability (default: 0).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint root to resume from.",
    )
    parser.add_argument(
        "--global-episodes-per-update",
        type=int,
        default=None,
        help=(
            "Configurable global episode batch for each optimizer update; "
            "it must be divisible by world size."
        ),
    )
    parser.add_argument(
        "--max-self-tokens",
        type=int,
        default=None,
        help="Override the per-pack self-attention token budget.",
    )
    parser.add_argument(
        "--max-episodes-per-pack",
        type=int,
        default=None,
        help="Override the maximum number of episodes in one pack.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override optimizer steps (useful for smoke tests and benchmarks).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Override checkpoint save interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set Python, NumPy, and Torch seeds for reproducibility probes.",
    )
    parser.add_argument(
        "--packing-seed",
        type=int,
        default=None,
        help="Override token-packing sampler seed.",
    )
    parser.add_argument(
        "--lineage-audit-dir",
        type=str,
        default=None,
        help="Write lineage parquet and Resource Card artifacts to this directory.",
    )
    parser.add_argument(
        "--lineage-run-id",
        type=str,
        default=None,
        help="Run id stored in lineage artifacts.",
    )
    parser.add_argument(
        "--lineage-logical-run-id",
        type=str,
        default=None,
        help="Stable logical run id used for occurrence_id hashing across attempts.",
    )
    parser.add_argument(
        "--lineage-training-attempt-id",
        type=str,
        default=None,
        help="Attempt id stored in lineage records for resume auditing.",
    )
    parser.add_argument(
        "--lineage-resume-generation",
        type=int,
        default=0,
        help="Monotonic resume generation used to accept the final training trajectory.",
    )
    parser.add_argument(
        "--lineage-checkpoint-base-step",
        type=int,
        default=0,
        help="Optimizer step of the checkpoint used to start this attempt.",
    )
    parser.add_argument(
        "--lineage-shard-steps",
        type=int,
        default=50,
        help="Flush rank-local lineage parquet after this many committed optimizer steps.",
    )
    parser.add_argument(
        "--lineage-interrupt-after-microbatches",
        type=int,
        default=None,
        help="Force process exit after capturing this absolute microbatch count.",
    )
    parser.add_argument(
        "--lineage-interrupt-exit-code",
        type=int,
        default=42,
        help="Exit code used by the forced lineage interruption hook.",
    )
    parser.add_argument(
        "--lineage-source-view",
        type=str,
        default="full",
        help="Source-view label stored in lineage records.",
    )
    parser.add_argument(
        "--lineage-subset-manifest-id",
        type=str,
        default="full",
        help="Subset manifest identifier or path stored in lineage records.",
    )
    parser.add_argument(
        "--load-worker",
        type=int,
        default=None,
        help="Override DataLoader worker processes (use 0 for deterministic smoke tests).",
    )
    parser.add_argument(
        "--ebench-episode-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Restrict an EBench run to audited episode IDs (for targeted smoke "
            "tests only; omitted for full training)."
        ),
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B for local smoke tests and benchmarks.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
