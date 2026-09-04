# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_libero_all_train_cfg import va_libero_all_train_cfg


va_libero_all_lora_train_cfg = EasyDict(
    __name__='Config: VA LIBERO-All LoRA train'
)
va_libero_all_lora_train_cfg.update(va_libero_all_train_cfg)

# LoRA fine-tuning
va_libero_all_lora_train_cfg.training_mode = 'lora'
va_libero_all_lora_train_cfg.lora_rank = 64
va_libero_all_lora_train_cfg.lora_alpha = 64
va_libero_all_lora_train_cfg.lora_dropout = 0.0
va_libero_all_lora_train_cfg.lora_train_action_modules = True

# LoRA can use a higher learning rate and less weight decay than full-model
# fine-tuning. The full action modules retain the full-model optimizer scale.
va_libero_all_lora_train_cfg.action_learning_rate = (
    va_libero_all_train_cfg.learning_rate
)
va_libero_all_lora_train_cfg.action_weight_decay = (
    va_libero_all_train_cfg.weight_decay
)
va_libero_all_lora_train_cfg.learning_rate = 1e-4
va_libero_all_lora_train_cfg.weight_decay = 1e-2

va_libero_all_lora_train_cfg.wandb_run_name = 'libero-all-lora-r64'
