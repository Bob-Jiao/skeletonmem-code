import pytest
import torch

from wan_va.dataset.packing import (
    EpisodePackingInfo,
    PackedSampleRef,
    TokenBudgetPackingBatchSampler,
    episode_equal_loss,
    packed_episode_collate,
)


def make_manifest(count=37):
    return [
        EpisodePackingInfo(
            index=index,
            sample_key=f"sample-{index}",
            frame_count=8 + index % 23,
            token_count=128 * (4 + index % 20),
            shape_signature=(16, 8, 16, 30, 4, 1),
        )
        for index in range(count)
    ]


@pytest.mark.parametrize("global_size", [80, 128, 256])
def test_plan_preserves_configurable_global_batch_and_rank_sync(global_size):
    sampler = TokenBudgetPackingBatchSampler(
        make_manifest(),
        global_episodes_per_update=global_size,
        max_self_tokens=4096,
        max_episodes_per_pack=8,
        num_replicas=8,
        rank=3,
        seed=9,
    )
    plan = sampler.plan_update(5)

    assert {len(rank_packs) for rank_packs in plan} == {len(plan[0])}
    assert all(
        sum(map(len, rank_packs)) == global_size // 8
        for rank_packs in plan
    )
    assert all(
        sum(item.info.token_count for item in pack) <= 4096
        for rank_packs in plan
        for pack in rank_packs
    )
    sample_uids = [
        item.sample_uid
        for rank_packs in plan
        for pack in rank_packs
        for item in pack
    ]
    assert sorted(sample_uids) == list(range(5 * global_size, 6 * global_size))


def test_occurrence_stream_crosses_epochs_without_padding_duplicates():
    sampler = TokenBudgetPackingBatchSampler(
        make_manifest(10),
        global_episodes_per_update=16,
        max_self_tokens=4096,
        max_episodes_per_pack=8,
        num_replicas=4,
        rank=0,
        seed=42,
    )
    plan = sampler.plan_update(1)
    uids = [
        item.sample_uid
        for rank_packs in plan
        for pack in rank_packs
        for item in pack
    ]
    assert sorted(uids) == list(range(16, 32))


def test_sampler_cursor_advances_only_after_optimizer_commit():
    kwargs = dict(
        manifest=make_manifest(),
        global_episodes_per_update=32,
        max_self_tokens=4096,
        max_episodes_per_pack=8,
        num_replicas=8,
        rank=0,
        seed=11,
    )
    sampler = TokenBudgetPackingBatchSampler(**kwargs)
    iterator = iter(sampler)
    first_pack = next(iterator)
    assert first_pack[0].update_id == 0
    assert sampler.committed_global_cursor == 0

    sampler.mark_update_completed(0)
    state = sampler.state_dict()
    resumed = TokenBudgetPackingBatchSampler(**kwargs)
    resumed.load_state_dict(state)
    assert resumed.committed_global_cursor == 32
    assert next(iter(resumed))[0].update_id == 1


def test_invalid_global_batch_and_oversized_episode_fail_fast():
    with pytest.raises(ValueError, match="divisible"):
        TokenBudgetPackingBatchSampler(
            make_manifest(),
            global_episodes_per_update=65,
            max_self_tokens=4096,
            max_episodes_per_pack=8,
            num_replicas=8,
            rank=0,
        )
    with pytest.raises(ValueError, match="exceed"):
        TokenBudgetPackingBatchSampler(
            make_manifest(),
            global_episodes_per_update=64,
            max_self_tokens=512,
            max_episodes_per_pack=8,
            num_replicas=8,
            rank=0,
        )


def _sample(frames, uid, micro_idx=0, micro_count=1):
    ref = PackedSampleRef(
        index=uid,
        sample_uid=uid,
        update_id=7,
        micro_idx=micro_idx,
        micro_count=micro_count,
        planned_tokens=9984,
    )
    return {
        "latents": torch.full((16, frames, 8, 16), float(uid)),
        "actions": torch.full((30, frames, 4, 1), float(uid)),
        "actions_mask": torch.ones((30, frames, 4, 1), dtype=torch.bool),
        "text_id": uid,
        **ref.__dict__,
    }


def test_collator_concatenates_time_without_temporal_padding():
    batch = packed_episode_collate([_sample(3, 1), _sample(5, 2)])
    assert batch["latents"].shape == (1, 16, 8, 8, 16)
    assert batch["actions"].shape == (1, 30, 8, 4, 1)
    assert batch["frame_lengths"].tolist() == [3, 5]
    assert batch["frame_offsets"].tolist() == [0, 3, 8]
    assert batch["sample_uid"].tolist() == [1, 2]
    assert torch.equal(batch["latents"][0, :, :3], torch.ones(16, 3, 8, 16))
    assert torch.equal(
        batch["latents"][0, :, 3:], torch.full((16, 5, 8, 16), 2.0)
    )


def test_episode_equal_loss_matches_sequential_b1_accumulation():
    first = torch.tensor([1.0, 3.0])
    second = torch.tensor([10.0, 20.0, 30.0, 40.0])
    packed = episode_equal_loss(
        torch.cat([first, second]),
        frame_lengths=[len(first), len(second)],
        local_episodes_per_update=4,
    )
    sequential = first.mean() / 4 + second.mean() / 4
    assert torch.equal(packed, sequential)
