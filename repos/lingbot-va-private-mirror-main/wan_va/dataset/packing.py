"""Deterministic, token-budgeted episode packing for distributed training."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import Sampler


PACKING_FORMAT_VERSION = 1


def round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class EpisodePackingInfo:
    """Shape and cost information needed before loading an episode."""

    index: int
    sample_key: str
    frame_count: int
    token_count: int
    shape_signature: tuple[int, ...]


@dataclass(frozen=True)
class PackedSampleRef:
    """A decorated dataset index emitted as part of a packed micro-batch."""

    index: int
    sample_uid: int
    update_id: int
    micro_idx: int
    micro_count: int
    planned_tokens: int


@dataclass(frozen=True)
class _Occurrence:
    info: EpisodePackingInfo
    sample_uid: int


def stable_hash(payload: Any) -> str:
    """Deterministic sha256 hash of a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def manifest_fingerprint(manifest: Sequence[EpisodePackingInfo]) -> str:
    payload = [
        {
            "index": item.index,
            "sample_key": item.sample_key,
            "frame_count": item.frame_count,
            "token_count": item.token_count,
            "shape_signature": item.shape_signature,
        }
        for item in manifest
    ]
    return stable_hash(payload)


def episode_equal_loss(
    frame_losses: torch.Tensor,
    frame_lengths: Sequence[int],
    local_episodes_per_update: int,
) -> torch.Tensor:
    """Reduce frame losses exactly like equal-weight B=1 accumulation."""
    if local_episodes_per_update <= 0:
        raise ValueError("local_episodes_per_update must be positive")
    if sum(frame_lengths) != frame_losses.numel():
        raise ValueError(
            f"Frame loss length {frame_losses.numel()} does not match segments "
            f"{list(frame_lengths)}"
        )
    episode_losses = []
    offset = 0
    for frame_count in frame_lengths:
        if frame_count <= 0:
            raise ValueError(f"Episode frame count must be positive, got {frame_count}")
        end = offset + frame_count
        episode_losses.append(frame_losses[offset:end].mean())
        offset = end
    return torch.stack(episode_losses).sum() / local_episodes_per_update


class TokenBudgetPackingBatchSampler(Sampler[list[PackedSampleRef]]):
    """Yield rank-local packs while preserving a configurable global batch.

    The sampler plans from an infinite stream of deterministic epoch
    permutations. DataLoader prefetch may ask for future plans, but the durable
    cursor advances only when ``mark_update_completed`` is called after a
    successful optimizer step.
    """

    def __init__(
        self,
        manifest: Sequence[EpisodePackingInfo],
        *,
        global_episodes_per_update: int,
        max_self_tokens: int,
        max_episodes_per_pack: int,
        num_replicas: int,
        rank: int,
        seed: int = 42,
        start_global_cursor: int = 0,
    ) -> None:
        if not manifest:
            raise ValueError("packing manifest must not be empty")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError(
                f"invalid distributed topology: rank={rank}, replicas={num_replicas}"
            )
        if global_episodes_per_update <= 0:
            raise ValueError("global_episodes_per_update must be positive")
        if global_episodes_per_update % num_replicas != 0:
            raise ValueError(
                "global_episodes_per_update must be divisible by world_size: "
                f"{global_episodes_per_update} % {num_replicas} != 0"
            )
        if max_self_tokens <= 0 or max_episodes_per_pack <= 0:
            raise ValueError("packing limits must be positive")
        if start_global_cursor % global_episodes_per_update != 0:
            raise ValueError(
                "Packed training resumes only at optimizer boundaries; "
                f"cursor {start_global_cursor} is not divisible by "
                f"global batch {global_episodes_per_update}."
            )

        self.manifest = tuple(manifest)
        self.global_episodes_per_update = global_episodes_per_update
        self.local_episodes_per_update = global_episodes_per_update // num_replicas
        self.max_self_tokens = max_self_tokens
        self.max_episodes_per_pack = max_episodes_per_pack
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.committed_global_cursor = start_global_cursor
        self.manifest_hash = manifest_fingerprint(self.manifest)
        self._permutation_cache: dict[int, tuple[int, ...]] = {}

        oversized = [item for item in self.manifest if item.token_count > max_self_tokens]
        if oversized:
            examples = ", ".join(
                f"{item.sample_key} ({item.token_count})" for item in oversized[:5]
            )
            raise ValueError(
                f"{len(oversized)} episode(s) exceed max_self_tokens="
                f"{max_self_tokens}: {examples}"
            )

    @property
    def committed_update_id(self) -> int:
        return self.committed_global_cursor // self.global_episodes_per_update

    def _epoch_permutation(self, epoch: int) -> tuple[int, ...]:
        permutation = self._permutation_cache.get(epoch)
        if permutation is None:
            values = list(range(len(self.manifest)))
            random.Random(self.seed + epoch).shuffle(values)
            permutation = tuple(values)
            self._permutation_cache[epoch] = permutation
            if len(self._permutation_cache) > 4:
                oldest = min(self._permutation_cache)
                del self._permutation_cache[oldest]
        return permutation

    def _occurrences_for_update(self, update_id: int) -> list[_Occurrence]:
        start = update_id * self.global_episodes_per_update
        occurrences = []
        for sample_uid in range(start, start + self.global_episodes_per_update):
            epoch, offset = divmod(sample_uid, len(self.manifest))
            index = self._epoch_permutation(epoch)[offset]
            occurrences.append(_Occurrence(self.manifest[index], sample_uid))
        return occurrences

    def _assign_ranks(
        self, occurrences: Sequence[_Occurrence]
    ) -> list[list[_Occurrence]]:
        rank_items: list[list[_Occurrence]] = [[] for _ in range(self.num_replicas)]
        rank_tokens = [0] * self.num_replicas
        for occurrence in sorted(
            occurrences,
            key=lambda item: (-item.info.token_count, item.sample_uid),
        ):
            candidates = [
                rank
                for rank in range(self.num_replicas)
                if len(rank_items[rank]) < self.local_episodes_per_update
            ]
            selected = min(
                candidates,
                key=lambda rank: (rank_tokens[rank], len(rank_items[rank]), rank),
            )
            rank_items[selected].append(occurrence)
            rank_tokens[selected] += occurrence.info.token_count
        return rank_items

    def _pack_rank(self, occurrences: Sequence[_Occurrence]) -> list[list[_Occurrence]]:
        packs: list[list[_Occurrence]] = []
        pack_tokens: list[int] = []
        for occurrence in sorted(
            occurrences,
            key=lambda item: (-item.info.token_count, item.sample_uid),
        ):
            compatible = []
            for pack_idx, pack in enumerate(packs):
                same_shape = pack[0].info.shape_signature == occurrence.info.shape_signature
                fits_tokens = (
                    pack_tokens[pack_idx] + occurrence.info.token_count
                    <= self.max_self_tokens
                )
                fits_count = len(pack) < self.max_episodes_per_pack
                if same_shape and fits_tokens and fits_count:
                    remaining = self.max_self_tokens - (
                        pack_tokens[pack_idx] + occurrence.info.token_count
                    )
                    compatible.append((remaining, pack_idx))
            if compatible:
                _, pack_idx = min(compatible)
                packs[pack_idx].append(occurrence)
                pack_tokens[pack_idx] += occurrence.info.token_count
            else:
                packs.append([occurrence])
                pack_tokens.append(occurrence.info.token_count)
        return packs

    @staticmethod
    def _pack_tokens(pack: Sequence[_Occurrence]) -> int:
        return sum(item.info.token_count for item in pack)

    def plan_update(self, update_id: int) -> list[list[list[_Occurrence]]]:
        rank_items = self._assign_ranks(self._occurrences_for_update(update_id))
        rank_packs = [self._pack_rank(items) for items in rank_items]
        target_micro_count = max(len(packs) for packs in rank_packs)

        for packs in rank_packs:
            while len(packs) < target_micro_count:
                splittable = [
                    (self._pack_tokens(pack), idx)
                    for idx, pack in enumerate(packs)
                    if len(pack) > 1
                ]
                if not splittable:
                    raise RuntimeError(
                        "Cannot equalize rank micro-batch counts without a dummy pack."
                    )
                _, pack_idx = max(splittable)
                pack = packs[pack_idx]
                moved = min(pack, key=lambda item: (item.info.token_count, item.sample_uid))
                pack.remove(moved)
                packs.append([moved])
            packs.sort(key=lambda pack: (-self._pack_tokens(pack), pack[0].sample_uid))

        self._validate_plan(rank_packs)
        return rank_packs

    def _validate_plan(self, rank_packs: Sequence[Sequence[Sequence[_Occurrence]]]) -> None:
        micro_counts = {len(packs) for packs in rank_packs}
        if len(micro_counts) != 1:
            raise RuntimeError(f"rank micro-batch counts differ: {micro_counts}")
        seen = []
        for packs in rank_packs:
            items = [item for pack in packs for item in pack]
            if len(items) != self.local_episodes_per_update:
                raise RuntimeError(
                    f"rank received {len(items)} episodes, expected "
                    f"{self.local_episodes_per_update}"
                )
            for pack in packs:
                if self._pack_tokens(pack) > self.max_self_tokens:
                    raise RuntimeError("planner emitted an over-budget pack")
                if len(pack) > self.max_episodes_per_pack:
                    raise RuntimeError("planner emitted an overfull pack")
                if len({item.info.shape_signature for item in pack}) != 1:
                    raise RuntimeError("planner mixed incompatible tensor shapes")
            seen.extend(item.sample_uid for item in items)
        if len(seen) != self.global_episodes_per_update or len(set(seen)) != len(seen):
            raise RuntimeError("planner dropped or duplicated an occurrence")

    def __iter__(self) -> Iterator[list[PackedSampleRef]]:
        update_id = self.committed_update_id
        while True:
            all_rank_packs = self.plan_update(update_id)
            local_packs = all_rank_packs[self.rank]
            micro_count = len(local_packs)
            for micro_idx, pack in enumerate(local_packs):
                planned_tokens = self._pack_tokens(pack)
                yield [
                    PackedSampleRef(
                        index=item.info.index,
                        sample_uid=item.sample_uid,
                        update_id=update_id,
                        micro_idx=micro_idx,
                        micro_count=micro_count,
                        planned_tokens=planned_tokens,
                    )
                    for item in pack
                ]
            update_id += 1

    def __len__(self) -> int:
        # Training is step-based and the occurrence stream is intentionally infinite.
        return 2**31

    def mark_update_completed(self, update_id: int) -> None:
        if update_id != self.committed_update_id:
            raise RuntimeError(
                f"Completing packed update {update_id}, but durable cursor expects "
                f"update {self.committed_update_id}."
            )
        self.committed_global_cursor += self.global_episodes_per_update

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": PACKING_FORMAT_VERSION,
            "global_cursor": self.committed_global_cursor,
            "manifest_hash": self.manifest_hash,
            "seed": self.seed,
            "global_episodes_per_update": self.global_episodes_per_update,
            "max_self_tokens": self.max_self_tokens,
            "max_episodes_per_pack": self.max_episodes_per_pack,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "format_version": PACKING_FORMAT_VERSION,
            "manifest_hash": self.manifest_hash,
            "seed": self.seed,
            "global_episodes_per_update": self.global_episodes_per_update,
            "max_self_tokens": self.max_self_tokens,
            "max_episodes_per_pack": self.max_episodes_per_pack,
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Packed sampler checkpoint mismatch: {mismatches}")
        cursor = int(state["global_cursor"])
        if cursor % self.global_episodes_per_update != 0:
            raise ValueError(f"Invalid packed sampler cursor: {cursor}")
        self.committed_global_cursor = cursor


def packed_episode_collate(samples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Concatenate compatible episodes along time without temporal padding."""
    if not samples:
        raise ValueError("cannot collate an empty pack")

    tensor_shapes = {
        (
            tuple(sample["latents"].shape[:1] + sample["latents"].shape[2:]),
            tuple(sample["actions"].shape[:1] + sample["actions"].shape[2:]),
        )
        for sample in samples
    }
    if len(tensor_shapes) != 1:
        raise ValueError(f"pack contains incompatible tensor shapes: {tensor_shapes}")

    frame_lengths = torch.tensor(
        [sample["latents"].shape[1] for sample in samples], dtype=torch.long
    )
    action_lengths = [sample["actions"].shape[1] for sample in samples]
    if action_lengths != frame_lengths.tolist():
        raise ValueError(
            f"video/action frame mismatch: video={frame_lengths.tolist()}, "
            f"action={action_lengths}"
        )

    update_ids = {int(sample["update_id"]) for sample in samples}
    micro_indices = {int(sample["micro_idx"]) for sample in samples}
    micro_counts = {int(sample["micro_count"]) for sample in samples}
    planned_tokens = {int(sample["planned_tokens"]) for sample in samples}
    if any(len(values) != 1 for values in (update_ids, micro_indices, micro_counts, planned_tokens)):
        raise ValueError("samples in one pack disagree on planner metadata")

    return {
        "latents": torch.cat([sample["latents"] for sample in samples], dim=1).unsqueeze(0),
        "actions": torch.cat([sample["actions"] for sample in samples], dim=1).unsqueeze(0),
        "actions_mask": torch.cat(
            [sample["actions_mask"] for sample in samples], dim=1
        ).unsqueeze(0),
        "text_id": torch.tensor([sample["text_id"] for sample in samples], dtype=torch.long),
        "frame_lengths": frame_lengths,
        "frame_offsets": torch.cat(
            [torch.zeros(1, dtype=torch.long), frame_lengths.cumsum(0)]
        ),
        "sample_uid": torch.tensor(
            [sample["sample_uid"] for sample in samples], dtype=torch.long
        ),
        "update_id": torch.tensor(next(iter(update_ids)), dtype=torch.long),
        "micro_idx": torch.tensor(next(iter(micro_indices)), dtype=torch.long),
        "micro_count": torch.tensor(next(iter(micro_counts)), dtype=torch.long),
        "planned_tokens": torch.tensor(next(iter(planned_tokens)), dtype=torch.long),
        "lineage_records": [sample.get("lineage", {}) for sample in samples],
    }
