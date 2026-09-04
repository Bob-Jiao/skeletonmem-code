# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_libero_all_cfg = EasyDict(__name__='Config: VA LIBERO-All')
va_libero_all_cfg.update(va_shared_cfg)
va_libero_all_cfg.infer_mode = 'server'

va_libero_all_cfg.wan22_pretrained_model_name_or_path = "/data/jiaoguanbo/models/lingbot-va-base"

va_libero_all_cfg.attn_window = 30
va_libero_all_cfg.frame_chunk_size = 4
va_libero_all_cfg.env_type = 'none'

va_libero_all_cfg.height = 128
va_libero_all_cfg.width = 128
va_libero_all_cfg.action_dim = 30
va_libero_all_cfg.action_per_frame = 4
va_libero_all_cfg.obs_cam_keys = [
    'observation.images.agentview_rgb', 'observation.images.eye_in_hand_rgb'
]
va_libero_all_cfg.guidance_scale = 5
va_libero_all_cfg.action_guidance_scale = 1

va_libero_all_cfg.num_inference_steps = 20
va_libero_all_cfg.video_exec_step = -1
va_libero_all_cfg.action_num_inference_steps = 50

va_libero_all_cfg.snr_shift = 5.0
va_libero_all_cfg.action_snr_shift = 0.05

va_libero_all_cfg.used_action_channel_ids = list(range(0, 7))
inverse_used_action_channel_ids = [len(va_libero_all_cfg.used_action_channel_ids)
                                   ] * va_libero_all_cfg.action_dim
for i, j in enumerate(va_libero_all_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_libero_all_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_libero_all_cfg.action_norm_method = 'quantiles'
va_libero_all_cfg.norm_stat = {
    "q01": [
        -0.7071428298950195,
        -0.8008928298950195,
        -0.9375,
        -0.11464285850524902,
        -0.16285714507102966,
        -0.22285714745521545,
        0.0
    ] + [0.] * 23,
    "q99": [
        0.9375,
        0.8732143044471741,
        0.9375,
        0.13178572058677673,
        0.1907142847776413,
        0.33964285254478455,
        1.0
    ] + [0.] * 23,
}
