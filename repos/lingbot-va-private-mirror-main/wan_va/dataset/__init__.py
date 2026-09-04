# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .packing import (
    EpisodePackingInfo,
    PackedSampleRef,
    TokenBudgetPackingBatchSampler,
    episode_equal_loss,
    packed_episode_collate,
    stable_hash,
)
from .ebench_latent_dataset import (
    EBENCH_CAMERA_KEYS,
    EBENCH_MODEL_CHANNEL_IDS,
    EBenchLatentDataset,
    assemble_ebench_camera_grid,
    map_and_normalize_ebench_actions,
)

__all__ = [
    'MultiLatentLeRobotDataset',
    'EpisodePackingInfo',
    'PackedSampleRef',
    'TokenBudgetPackingBatchSampler',
    'episode_equal_loss',
    'packed_episode_collate',
    'stable_hash',
    'EBENCH_CAMERA_KEYS',
    'EBENCH_MODEL_CHANNEL_IDS',
    'EBenchLatentDataset',
    'assemble_ebench_camera_grid',
    'map_and_normalize_ebench_actions',
]


def __getattr__(name):
    # Keep planner/loss unit tests independent from the optional LeRobot stack.
    if name == "MultiLatentLeRobotDataset":
        from .lerobot_latent_dataset import MultiLatentLeRobotDataset

        return MultiLatentLeRobotDataset
    raise AttributeError(name)
