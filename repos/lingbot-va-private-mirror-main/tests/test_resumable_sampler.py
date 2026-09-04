from torch.utils.data import TensorDataset
import torch

from wan_va.dataset.resumable_sampler import ResumableDistributedSampler


def test_resume_cursor_returns_the_unconsumed_suffix():
    dataset = TensorDataset(torch.arange(12))
    sampler = ResumableDistributedSampler(
        dataset,
        num_replicas=2,
        rank=0,
        seed=42,
    )
    sampler.set_epoch(3)
    full_epoch = list(sampler)

    sampler.set_start_index(4)
    assert list(sampler) == full_epoch[4:]
    assert len(sampler) == len(full_epoch) - 4


def test_new_epoch_resets_resume_cursor():
    dataset = TensorDataset(torch.arange(8))
    sampler = ResumableDistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        seed=42,
    )
    sampler.set_start_index(3)
    sampler.set_epoch(1)
    assert sampler.start_index == 0
    assert len(list(sampler)) == len(dataset)
