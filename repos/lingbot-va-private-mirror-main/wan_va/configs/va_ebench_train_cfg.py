# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_ebench_cfg import va_ebench_cfg

va_ebench_train_cfg = EasyDict(__name__="Config: VA EBench train")
va_ebench_train_cfg.update(va_ebench_cfg)

va_ebench_train_cfg.dataset_path = (
    "/root/shared/yaoyifei/dataset/openpi_lerobot/ebench/generalist_lingbot_va"
)
va_ebench_train_cfg.action_stats_path = os.path.join(
    va_ebench_train_cfg.dataset_path, "action_stats_segment_81920_v1.json"
)
va_ebench_train_cfg.segment_manifest_path = os.path.join(
    va_ebench_train_cfg.dataset_path, "segment_manifest_81920_v1.json"
)
va_ebench_train_cfg.segment_audit_path = os.path.join(
    va_ebench_train_cfg.dataset_path, "segment_audit_success_81920.json"
)
va_ebench_train_cfg.text_embeddings_path = os.path.join(
    va_ebench_train_cfg.dataset_path, "text_embeddings.pt"
)
va_ebench_train_cfg.empty_emb_path = os.path.join(
    va_ebench_train_cfg.dataset_path, "empty_emb.pt"
)
va_ebench_train_cfg.shared_text_embeddings = True

va_ebench_train_cfg.training_mode = "full"
va_ebench_train_cfg.enable_wandb = True
va_ebench_train_cfg.wandb_mode = "offline"
va_ebench_train_cfg.wandb_project = "lingbot-va-ebench"
va_ebench_train_cfg.wandb_run_name = "ebench-generalist-full"
va_ebench_train_cfg.load_worker = 16
va_ebench_train_cfg.save_interval = 2000
va_ebench_train_cfg.gc_interval = 50
va_ebench_train_cfg.cfg_prob = 0.1
va_ebench_train_cfg.pin_memory = True
va_ebench_train_cfg.persistent_workers = True
va_ebench_train_cfg.prefetch_factor = 2

va_ebench_train_cfg.learning_rate = 1e-5
va_ebench_train_cfg.beta1 = 0.9
va_ebench_train_cfg.beta2 = 0.95
va_ebench_train_cfg.weight_decay = 0.1
va_ebench_train_cfg.warmup_steps = 10

# Packing preserves complete variable-length segment action-configs.  The
# effective global batch remains 128; on four GPUs each rank contributes 32
# segments to one optimizer update.  The 520 split parents contribute two
# independently anchored samples, matching RoboTwin action_config semantics.
va_ebench_train_cfg.packing_enabled = True
va_ebench_train_cfg.global_episodes_per_update = 128
# Complete source episodes longer than this budget are exposed as independent
# segment action-configs by the EBench loader.
va_ebench_train_cfg.max_self_tokens = 81920
# This is a per-micro-pack cap, not the per-rank episode batch.  The token
# budget is normally the tighter constraint for EBench.
va_ebench_train_cfg.max_episodes_per_pack = 4
va_ebench_train_cfg.packing_seed = 42
# Reduce-scatter after every packed micro-batch so FSDP accumulates sharded
# gradients.  Deferring gradient synchronization across EBench's large packs
# retains full FP32 gradients on each rank and exceeds A100 80 GB memory.
va_ebench_train_cfg.sync_packed_gradients_each_microbatch = True

# This only affects the legacy non-packed path; packed layouts are derived
# deterministically per episode and do not use rank-broadcast layout sampling.
va_ebench_train_cfg.sync_nonpacked_attention_layout_across_ranks = True
va_ebench_train_cfg.batch_size = 1
va_ebench_train_cfg.gradient_accumulation_steps = 1
va_ebench_train_cfg.num_steps = 50000
