"""Deterministic distributed sampler with an optimizer-checkpoint cursor."""

from __future__ import annotations

from collections.abc import Iterator

from torch.utils.data import Dataset, DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler that can start partway through the current epoch.

    The cursor counts samples on each rank. DataLoader prefetch does not affect
    it because the trainer advances the cursor only after yielding a batch.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        seed: int = 42,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        self.start_index = 0

    def set_start_index(self, start_index: int) -> None:
        if not 0 <= start_index <= self.num_samples:
            raise ValueError(
                f"Sampler start index must be in [0, {self.num_samples}], got {start_index}."
            )
        self.start_index = start_index

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.start_index = 0

    def __iter__(self) -> Iterator[int]:
        indices = list(super().__iter__())
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return self.num_samples - self.start_index
