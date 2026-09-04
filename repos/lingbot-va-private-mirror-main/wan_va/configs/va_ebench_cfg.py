# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_ebench_cfg = EasyDict(__name__="Config: VA EBench")
va_ebench_cfg.update(va_shared_cfg)

# EBench post-training starts from the released LingBot-VA base checkpoint.
va_ebench_cfg.wan22_pretrained_model_name_or_path = (
    "/root/shared/zouyude/ckpts/lingbot-va/lingbot-va-base"
)

va_ebench_cfg.attn_window = 72
va_ebench_cfg.frame_chunk_size = 2
va_ebench_cfg.env_type = "ebench_grid"
va_ebench_cfg.dataset_type = "ebench"

# These are the per-camera preprocessing dimensions.  The loader assembles the
# four [48,F,14,14] VAE records into one [48,F,28,28] latent grid.
va_ebench_cfg.height = 224
va_ebench_cfg.width = 224
va_ebench_cfg.action_dim = 30
va_ebench_cfg.action_per_frame = 16
va_ebench_cfg.obs_cam_keys = [
    "video.top_camera_view",
    "video.overlook_camera_view",
    "video.left_camera_view",
    "video.right_camera_view",
]

va_ebench_cfg.guidance_scale = 5
va_ebench_cfg.action_guidance_scale = 1
va_ebench_cfg.num_inference_steps = 25
va_ebench_cfg.video_exec_step = -1
va_ebench_cfg.action_num_inference_steps = 50
va_ebench_cfg.snr_shift = 5.0
va_ebench_cfg.action_snr_shift = 1.0

# Physical 19-D order -> pretrained 30-D action-head channels:
# left joints, left gripper, right joints, right gripper, base xyz.
va_ebench_cfg.used_action_channel_ids = (
    list(range(14, 20)) + [28, 20] + list(range(21, 27)) + [29, 27] + [0, 1, 2]
)
inverse_used_action_channel_ids = [
    len(va_ebench_cfg.used_action_channel_ids)
] * va_ebench_cfg.action_dim
for physical_channel, model_channel in enumerate(va_ebench_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[model_channel] = physical_channel
va_ebench_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_ebench_cfg.action_norm_method = "quantiles"
# The generated, versioned action_stats_v1.json is authoritative.  Keeping a
# sentinel here avoids baking dataset-specific values into source control.
va_ebench_cfg.norm_stat = None
