import numpy as np
import pytest

from script.prepare_ebench_lingbot import (
    aligned_real_action_count,
    compute_action_quantiles,
    map_action_19d_to_30d,
    map_action_30d_to_19d,
    model_action_mask_30d,
    prepare_episode_actions,
    transform_ebench_actions,
)


MODEL_CHANNELS = np.array(
    [
        14,
        15,
        16,
        17,
        18,
        19,
        28,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        29,
        27,
        0,
        1,
        2,
    ]
)


def test_action_transform_uses_one_episode_anchor_and_absolute_grippers():
    # The discontinuity at t=2 catches accidental chunk-local re-anchoring.
    joints = np.array(
        [
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 122, 133],
            [50, 60, 70, 80, 90, 100, 200, 210, 220, 230, 240, 250],
            [51, 61, 71, 81, 91, 101, 201, 211, 221, 231, 241, 251],
        ],
        dtype=np.float32,
    )
    gripper = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.11, 0.21, 0.31, 0.41],
            [0.12, 0.22, 0.32, 0.42],
            [0.13, 0.23, 0.33, 0.43],
        ],
        dtype=np.float32,
    )
    base_delta = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [-0.5, -5.0, -300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=np.float32,
    )

    actual = transform_ebench_actions(joints, gripper, base_delta)

    assert actual.shape == (4, 19)
    np.testing.assert_allclose(actual[:, :6], joints[:, :6] - joints[0, :6])
    np.testing.assert_allclose(actual[:, 6:8], gripper[:, :2])
    np.testing.assert_allclose(actual[:, 8:14], joints[:, 6:12] - joints[0, 6:12])
    np.testing.assert_allclose(actual[:, 14:16], gripper[:, 2:4])
    # Inclusive cumsum means the first delta is already present at t=0.
    np.testing.assert_allclose(actual[:, 16:18], np.cumsum(base_delta[:, :2], axis=0))
    np.testing.assert_array_equal(actual[:, 18], np.zeros(4, dtype=np.float32))
    # A later discontinuity remains relative to the episode's first command.
    assert actual[2, 0] == 40
    assert actual[2, 8] == 130


def test_19d_model_mapping_is_exact_and_round_trips():
    action_19d = np.arange(3 * 19, dtype=np.float32).reshape(3, 19) + 0.25

    action_30d = map_action_19d_to_30d(action_19d)

    assert action_30d.shape == (3, 30)
    np.testing.assert_array_equal(action_30d[:, MODEL_CHANNELS], action_19d)
    unused = np.setdiff1d(np.arange(30), MODEL_CHANNELS)
    np.testing.assert_array_equal(action_30d[:, unused], np.zeros((3, 11)))
    np.testing.assert_array_equal(map_action_30d_to_19d(action_30d), action_19d)

    mask = np.asarray(model_action_mask_30d())
    assert mask.shape == (30,)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.sort(MODEL_CHANNELS))


def test_aligned_real_actions_leave_one_anchor_frame_and_drop_tail():
    expected = {
        1: 0,
        16: 0,
        17: 16,
        32: 16,
        33: 32,
        49: 48,
        3300: 3296,
    }
    assert {
        length: aligned_real_action_count(length) for length in expected
    } == expected


def test_zero_prefix_occurs_once_and_prefix_and_tail_are_excluded_from_stats():
    full_episode = np.full((33, 19), 10.0, dtype=np.float32)
    full_episode[-1] = 999.0

    stats_actions, training_actions = prepare_episode_actions(full_episode)

    assert stats_actions.shape == (32, 19)
    assert training_actions.shape == (48, 19)
    np.testing.assert_array_equal(training_actions[:16], np.zeros((16, 19)))
    np.testing.assert_array_equal(training_actions[16:], stats_actions)
    assert np.count_nonzero(np.all(training_actions == 0, axis=1)) == 16
    assert not np.any(stats_actions == 0)
    assert not np.any(stats_actions == 999)
    # Both quantiles staying at ten catches inclusion of either synthetic zero
    # rows or the deliberately extreme incomplete tail.
    q01, q99 = compute_action_quantiles(stats_actions)
    np.testing.assert_array_equal(q01, np.full(19, 10.0))
    np.testing.assert_array_equal(q99, np.full(19, 10.0))


def test_quantiles_reject_non_finite_retained_actions():
    actions = np.zeros((16, 19), dtype=np.float32)
    actions[5, 3] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        compute_action_quantiles(actions)
