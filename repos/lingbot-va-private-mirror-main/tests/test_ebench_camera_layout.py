from types import SimpleNamespace

import numpy as np
import torch

from script.prepare_ebench_lingbot import (
    encode_episode_camera_batch,
    expected_latent_frame_count,
    mosaic_camera_latents,
    sampled_rgb_frame_ids,
    usable_sampled_frame_count,
)


class _FakeStreamingVAE:
    def __init__(self):
        self.clear_count = 0
        self.input_shapes = []

    def clear_cache(self):
        self.clear_count += 1

    def encode_chunk(self, video):
        self.input_shapes.append(tuple(video.shape))
        batch, _, sampled_frames, _, _ = video.shape
        latent_frames = 1 if sampled_frames == 1 else sampled_frames // 4
        mean = torch.stack(
            [
                torch.full((48, latent_frames, 14, 14), float(index + 1))
                for index in range(batch)
            ]
        ).to(video)
        return torch.cat((mean, torch.zeros_like(mean)), dim=1)


def test_four_camera_latents_form_the_declared_two_by_two_layout():
    cameras = {
        "top": torch.full((48, 2, 14, 14), 1.0),
        "overlook": torch.full((48, 2, 14, 14), 2.0),
        "left": torch.full((48, 2, 14, 14), 3.0),
        "right": torch.full((48, 2, 14, 14), 4.0),
    }

    mosaic = mosaic_camera_latents(cameras)

    assert mosaic.shape == (48, 2, 28, 28)
    torch.testing.assert_close(mosaic[:, :, :14, :14], cameras["top"])
    torch.testing.assert_close(mosaic[:, :, :14, 14:], cameras["overlook"])
    torch.testing.assert_close(mosaic[:, :, 14:, :14], cameras["left"])
    torch.testing.assert_close(mosaic[:, :, 14:, 14:], cameras["right"])


def test_sampling_thirteen_rgb_frames_has_four_temporal_latents_and_synced_ids():
    # 49 source action/video frames sampled at stride four produce 13 RGB
    # frames. Wan's 4x temporal compression produces 1 + 12/4 = 4 latents.
    frame_ids_by_camera = {
        camera: np.asarray(sampled_rgb_frame_ids(49))
        for camera in ("top", "overlook", "left", "right")
    }

    expected_ids = np.arange(0, 49, 4)
    assert usable_sampled_frame_count(49) == 13
    for frame_ids in frame_ids_by_camera.values():
        np.testing.assert_array_equal(frame_ids, expected_ids)
    assert all(
        np.array_equal(frame_ids, frame_ids_by_camera["top"])
        for frame_ids in frame_ids_by_camera.values()
    )
    assert expected_latent_frame_count(len(frame_ids_by_camera["top"])) == 4


def test_sampling_crops_to_one_plus_a_multiple_of_four_rgb_frames():
    # Length 48 would naively yield 12 sampled frames; only the first nine are
    # usable as a complete Wan temporal sequence, matching 32 retained actions.
    np.testing.assert_array_equal(sampled_rgb_frame_ids(48), np.arange(0, 33, 4))
    assert usable_sampled_frame_count(48) == 9


def test_fake_vae_encodes_four_independent_thirteen_frame_streams_in_one_batch():
    frames = {
        camera: np.full((13, 2, 2, 3), value, dtype=np.uint8)
        for value, camera in enumerate(("top", "overlook", "left", "right"), 1)
    }
    vae = SimpleNamespace(
        config=SimpleNamespace(latents_mean=[0.0] * 48, latents_std=[1.0] * 48)
    )
    streaming = _FakeStreamingVAE()

    latents = encode_episode_camera_batch(
        vae,
        streaming,
        frames,
        device="cpu",
        temporal_chunk_size=4,
    )

    assert streaming.clear_count == 1
    assert streaming.input_shapes == [
        (4, 3, 1, 2, 2),
        (4, 3, 4, 2, 2),
        (4, 3, 4, 2, 2),
        (4, 3, 4, 2, 2),
    ]
    assert set(latents) == {"top", "overlook", "left", "right"}
    for value, camera in enumerate(("top", "overlook", "left", "right"), 1):
        assert latents[camera].shape == (48, 4, 14, 14)
        torch.testing.assert_close(
            latents[camera], latents[camera].new_full((48, 4, 14, 14), value)
        )
