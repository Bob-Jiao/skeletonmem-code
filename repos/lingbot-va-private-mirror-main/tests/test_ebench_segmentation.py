import hashlib

import numpy as np

from wan_va.dataset.ebench_latent_dataset import (
    EBENCH_ZERO_PREFIX_ROWS,
    EBenchSegmentSpec,
    ebench_segment_action_sha256,
    ebench_self_token_count,
    plan_ebench_segments,
    reanchor_ebench_segment_actions,
)


MAX_SELF_TOKENS = 81_920


def test_f193_fits_in_one_segment_at_the_token_cap():
    segments = plan_ebench_segments(193, MAX_SELF_TOKENS)

    assert segments == (EBenchSegmentSpec(0, 0, 192),)
    assert segments[0].frame_count == 193
    assert (segments[0].latent_start, segments[0].latent_stop) == (0, 193)
    assert ebench_self_token_count(segments[0].frame_count) == MAX_SELF_TOKENS


def test_f194_splits_into_two_segments_with_one_latent_anchor_overlap():
    segments = plan_ebench_segments(194, MAX_SELF_TOKENS)

    assert segments == (
        EBenchSegmentSpec(0, 0, 192),
        EBenchSegmentSpec(1, 192, 193),
    )
    assert [segment.frame_count for segment in segments] == [193, 2]
    assert segments[0].latent_stop - 1 == segments[1].latent_start
    assert all(
        ebench_self_token_count(segment.frame_count) <= MAX_SELF_TOKENS
        for segment in segments
    )


def test_f270_splits_193_plus_78_and_reuses_the_boundary_latent():
    segments = plan_ebench_segments(270, MAX_SELF_TOKENS)

    assert [segment.frame_count for segment in segments] == [193, 78]
    assert [
        (segment.real_block_start, segment.real_block_end)
        for segment in segments
    ] == [(0, 192), (192, 269)]

    latent_ids = np.arange(270)
    first_latents = latent_ids[
        segments[0].latent_start : segments[0].latent_stop
    ]
    second_latents = latent_ids[
        segments[1].latent_start : segments[1].latent_stop
    ]
    assert first_latents[-1] == second_latents[0] == 192
    assert len(first_latents) + len(second_latents) == 271
    assert all(
        ebench_self_token_count(segment.frame_count) <= MAX_SELF_TOKENS
        for segment in segments
    )


def test_segment_actions_reanchor_joints_and_inclusive_base_without_touching_grippers():
    real_rows = 3 * EBENCH_ZERO_PREFIX_ROWS
    row = np.arange(real_rows, dtype=np.float32)
    parent_real = np.zeros((real_rows, 19), dtype=np.float32)
    parent_real[:, 0:6] = row[:, None] + 10 * np.arange(6, dtype=np.float32)
    parent_real[:, 6:8] = 0.01 * row[:, None] + np.array(
        [0.2, 0.7], dtype=np.float32
    )
    parent_real[:, 8:14] = 2 * row[:, None] + 20 * np.arange(
        6, dtype=np.float32
    )
    parent_real[:, 14:16] = 0.02 * row[:, None] + np.array(
        [0.3, 0.8], dtype=np.float32
    )
    base_delta = np.stack(
        (0.25 + row / 100, -0.5 - row / 200, 1.0 + row / 50), axis=1
    ).astype(np.float32)
    parent_real[:, 16:19] = np.cumsum(base_delta, axis=0)

    parent_action = np.concatenate(
        (
            np.zeros((EBENCH_ZERO_PREFIX_ROWS, 19), dtype=np.float32),
            parent_real,
        ),
        axis=0,
    )
    parent_mask = np.zeros_like(parent_action, dtype=bool)
    mask_pattern = (
        np.arange(real_rows)[:, None] + np.arange(19)[None, :]
    ) % 3 != 0
    parent_mask[EBENCH_ZERO_PREFIX_ROWS:] = mask_pattern

    spec = EBenchSegmentSpec(ordinal=1, real_block_start=1, real_block_end=3)
    segment_action, segment_mask, joint_anchor, base_anchor = (
        reanchor_ebench_segment_actions(parent_action, parent_mask, spec)
    )

    start = EBENCH_ZERO_PREFIX_ROWS
    retained = parent_real[start:]
    expected_joint_anchor = np.concatenate(
        (parent_real[start, 0:6], parent_real[start, 8:14])
    )
    np.testing.assert_array_equal(joint_anchor, expected_joint_anchor)
    np.testing.assert_array_equal(base_anchor, parent_real[start - 1, 16:19])

    assert segment_action.shape == (3 * EBENCH_ZERO_PREFIX_ROWS, 19)
    np.testing.assert_array_equal(
        segment_action[:EBENCH_ZERO_PREFIX_ROWS],
        np.zeros((EBENCH_ZERO_PREFIX_ROWS, 19), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        segment_mask[:EBENCH_ZERO_PREFIX_ROWS],
        np.ones((EBENCH_ZERO_PREFIX_ROWS, 19), dtype=bool),
    )

    segment_real = segment_action[EBENCH_ZERO_PREFIX_ROWS:]
    np.testing.assert_allclose(
        segment_real[:, 0:6], retained[:, 0:6] - retained[0, 0:6]
    )
    np.testing.assert_allclose(
        segment_real[:, 8:14], retained[:, 8:14] - retained[0, 8:14]
    )
    np.testing.assert_array_equal(segment_real[:, 6:8], retained[:, 6:8])
    np.testing.assert_array_equal(segment_real[:, 14:16], retained[:, 14:16])
    np.testing.assert_allclose(
        segment_real[:, 16:18],
        retained[:, 16:18] - parent_real[start - 1, 16:18],
    )
    np.testing.assert_allclose(
        segment_real[0, 16:18], base_delta[start, 0:2], atol=1e-6
    )
    np.testing.assert_array_equal(
        segment_real[:, 18], np.zeros(len(segment_real), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        segment_mask[EBENCH_ZERO_PREFIX_ROWS:], mask_pattern[start:]
    )

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(segment_action, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(segment_mask, dtype=np.uint8).tobytes())
    expected_sha256 = digest.hexdigest()
    assert ebench_segment_action_sha256(segment_action, segment_mask) == expected_sha256

    changed_action = segment_action.copy()
    changed_action[-1, 0] += 1
    assert (
        ebench_segment_action_sha256(changed_action, segment_mask)
        != expected_sha256
    )
