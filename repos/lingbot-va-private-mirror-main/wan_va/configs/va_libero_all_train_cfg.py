# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_libero_all_cfg import va_libero_all_cfg
import os

va_libero_all_train_cfg = EasyDict(__name__='Config: VA LIBERO-All train')
va_libero_all_train_cfg.update(va_libero_all_cfg)

va_libero_all_train_cfg.dataset_path = '/data/jiaoguanbo/LIBERO/libero_all'
va_libero_all_train_cfg.empty_emb_path = os.path.join(va_libero_all_train_cfg.dataset_path, 'empty_emb.pt')
va_libero_all_train_cfg.enable_wandb = True
va_libero_all_train_cfg.wandb_mode = 'offline'
va_libero_all_train_cfg.wandb_project = 'lingbot-va-libero-all'
va_libero_all_train_cfg.wandb_run_name = 'libero-all-full'
va_libero_all_train_cfg.load_worker = 16
va_libero_all_train_cfg.save_interval = 5000
va_libero_all_train_cfg.gc_interval = 50
va_libero_all_train_cfg.cfg_prob = 0.1
va_libero_all_train_cfg.pin_memory = True
va_libero_all_train_cfg.persistent_workers = True
va_libero_all_train_cfg.prefetch_factor = 2

# Training parameters
va_libero_all_train_cfg.learning_rate = 1e-5
va_libero_all_train_cfg.beta1 = 0.9
va_libero_all_train_cfg.beta2 = 0.95
va_libero_all_train_cfg.weight_decay = 1e-1
va_libero_all_train_cfg.warmup_steps = 10
# True packing keeps episode boundaries in the attention mask while combining
# variable-length episodes into token-budgeted forwards. The global batch is a
# configurable optimization parameter and must be divisible by world_size.
va_libero_all_train_cfg.packing_enabled = True
va_libero_all_train_cfg.global_episodes_per_update = 128
va_libero_all_train_cfg.max_self_tokens = 98304 # 4 gpus: 98304; 8 gpus: 49152
va_libero_all_train_cfg.max_episodes_per_pack = 32 # global_episodes_per_update // NGPU
va_libero_all_train_cfg.packing_seed = 42

# Retained for legacy non-packed training.
va_libero_all_train_cfg.batch_size = 1
va_libero_all_train_cfg.gradient_accumulation_steps = 16
va_libero_all_train_cfg.num_steps = 20000
