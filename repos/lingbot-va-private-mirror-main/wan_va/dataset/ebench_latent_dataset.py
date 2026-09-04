"""EBench episode loader for the LingBot-VA training representation.

The EBench preprocessing pipeline deliberately keeps the four camera VAE
records separate and stores one already anchored 19-D action sidecar per
episode.  This module only performs the cheap, deterministic assembly needed
at training time; it never reads the source parquet actions or videos.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index as integer_index
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .packing import round_up

EBENCH_CAMERA_KEYS = (
    "video.top_camera_view",
    "video.overlook_camera_view",
    "video.left_camera_view",
    "video.right_camera_view",
)

# The entries are in 19-D physical-action order.  Their values are the
# corresponding channels of the pretrained 30-D LingBot-VA action head.
EBENCH_MODEL_CHANNEL_IDS = (
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
)

# Physical channel 18 is synthetic base yaw.  EBench has no commanded yaw,
# so preprocessing fixes it to raw zero and gives it the neutral [-1, 1]
# normalization range.  The shared LingBot formula adds 1e-6 to every
# denominator, which would otherwise turn that exact zero into a small
# negative value instead of normalized zero.
EBENCH_FIXED_ZERO_PHYSICAL_CHANNEL = 18

EBENCH_ACTION_STATS_VERSION = "ebench_lingbot_action_stats_v1"
EBENCH_DATA_MANIFEST_VERSION = "ebench_lingbot_data_v1"
EBENCH_LATENT_MANIFEST_VERSION = "ebench_lingbot_latents_v1"
EBENCH_PREPROCESS_MANIFEST_VERSION = "ebench_lingbot_preprocess_v1"
EBENCH_AUDIT_SUCCESS_MARKER_NAME = "audit_success.json"
EBENCH_AUDIT_SUCCESS_MARKER_VERSION = "ebench_lingbot_audit_success_v2"
EBENCH_SEGMENT_MANIFEST_VERSION = "ebench_lingbot_segments_v1"
EBENCH_SEGMENTED_ACTION_STATS_VERSION = (
    "ebench_lingbot_action_stats_segmented_v1"
)
EBENCH_SEGMENT_AUDIT_VERSION = "ebench_lingbot_segment_audit_v1"
EBENCH_ZERO_PREFIX_ROWS = 16
EBENCH_SELF_TOKENS_PER_FRAME = 424


@dataclass(frozen=True)
class EBenchSegmentSpec:
    """A virtual, independently anchored slice of one source episode."""

    ordinal: int
    real_block_start: int
    real_block_end: int

    @property
    def frame_count(self) -> int:
        return 1 + self.real_block_end - self.real_block_start

    @property
    def latent_start(self) -> int:
        return self.real_block_start

    @property
    def latent_stop(self) -> int:
        return self.real_block_end + 1


def ebench_self_token_count(frame_count: int) -> int:
    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("EBench frame_count must be positive")
    return round_up(EBENCH_SELF_TOKENS_PER_FRAME * frame_count, 128)


def plan_ebench_segments(
    frame_count: int, max_self_tokens: int
) -> tuple[EBenchSegmentSpec, ...]:
    """Partition real action blocks and overlap one latent anchor at boundaries."""
    frame_count = int(frame_count)
    max_self_tokens = int(max_self_tokens)
    if frame_count < 2:
        raise ValueError("An EBench training segment needs an anchor and real actions")
    max_segment_frames = max_self_tokens // EBENCH_SELF_TOKENS_PER_FRAME
    while (
        max_segment_frames > 0
        and ebench_self_token_count(max_segment_frames) > max_self_tokens
    ):
        max_segment_frames -= 1
    if max_segment_frames < 2:
        raise ValueError(
            f"max_self_tokens={max_self_tokens} cannot fit one EBench action frame"
        )
    max_real_blocks = max_segment_frames - 1
    specs = []
    real_block_start = 0
    real_block_count = frame_count - 1
    while real_block_start < real_block_count:
        real_block_end = min(
            real_block_start + max_real_blocks, real_block_count
        )
        specs.append(
            EBenchSegmentSpec(
                ordinal=len(specs),
                real_block_start=real_block_start,
                real_block_end=real_block_end,
            )
        )
        real_block_start = real_block_end
    return tuple(specs)


def reanchor_ebench_segment_actions(
    parent_action_19d: np.ndarray,
    parent_loss_mask_19d: np.ndarray,
    spec: EBenchSegmentSpec,
    *,
    action_per_frame: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one zero-prefixed segment with local joint/base anchors.

    Returns ``(action, mask, joint_anchor_delta_12d, base_anchor_3d)``.
    The base anchor is the cumulative pose immediately before the segment's
    first real action, preserving inclusive-cumsum semantics.
    """
    action = np.asarray(parent_action_19d, dtype=np.float32)
    loss_mask = np.asarray(parent_loss_mask_19d, dtype=bool)
    action_per_frame = int(action_per_frame)
    if action_per_frame != EBENCH_ZERO_PREFIX_ROWS:
        raise ValueError("EBench action_per_frame and zero prefix must both be 16")
    if action.ndim != 2 or action.shape[1] != 19 or loss_mask.shape != action.shape:
        raise ValueError("Parent EBench action and mask must both have shape [N,19]")
    if action.shape[0] < EBENCH_ZERO_PREFIX_ROWS * 2:
        raise ValueError("Parent EBench action sidecar is too short")
    if not np.all(action[:EBENCH_ZERO_PREFIX_ROWS] == 0):
        raise ValueError("Parent EBench action sidecar has no raw-zero16 prefix")

    parent_real = action[EBENCH_ZERO_PREFIX_ROWS:]
    parent_real_mask = loss_mask[EBENCH_ZERO_PREFIX_ROWS:]
    start = int(spec.real_block_start) * action_per_frame
    stop = int(spec.real_block_end) * action_per_frame
    if not 0 <= start < stop <= len(parent_real):
        raise ValueError(
            f"Segment action rows [{start},{stop}) exceed parent rows {len(parent_real)}"
        )
    segment_real = np.ascontiguousarray(parent_real[start:stop]).copy()
    segment_real_mask = np.ascontiguousarray(parent_real_mask[start:stop]).copy()

    joint_anchor_delta = np.concatenate(
        (segment_real[0, 0:6], segment_real[0, 8:14])
    ).astype(np.float32, copy=True)
    segment_real[:, 0:6] -= segment_real[0, 0:6].copy()
    segment_real[:, 8:14] -= segment_real[0, 8:14].copy()

    base_anchor = (
        np.zeros(3, dtype=np.float32)
        if start == 0
        else np.asarray(parent_real[start - 1, 16:19], dtype=np.float32).copy()
    )
    segment_real[:, 16:19] -= base_anchor
    segment_real[:, 18] = 0.0

    prefix = np.zeros((EBENCH_ZERO_PREFIX_ROWS, 19), dtype=np.float32)
    prefix_mask = np.ones_like(prefix, dtype=bool)
    segment_action = np.concatenate((prefix, segment_real), axis=0)
    segment_mask = np.concatenate((prefix_mask, segment_real_mask), axis=0)
    expected_rows = spec.frame_count * action_per_frame
    if segment_action.shape != (expected_rows, 19):
        raise RuntimeError("Segment action construction produced the wrong shape")
    return (
        np.ascontiguousarray(segment_action),
        np.ascontiguousarray(segment_mask),
        joint_anchor_delta,
        base_anchor,
    )


def ebench_segment_action_sha256(
    action_19d: np.ndarray, loss_mask_19d: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(action_19d, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(loss_mask_19d, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    return value.lower()


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_camera_records(
    records: Mapping[str, Mapping[str, Any]],
    camera_keys: Sequence[str],
) -> tuple[list[torch.Tensor], np.ndarray, int, int, int, int]:
    if tuple(camera_keys) != EBENCH_CAMERA_KEYS:
        raise ValueError(
            "EBench camera order must be top, overlook, left, right; got "
            f"{tuple(camera_keys)}"
        )

    reshaped: list[torch.Tensor] = []
    common_shape: tuple[int, int, int, int] | None = None
    common_frame_ids: np.ndarray | None = None
    for camera_key in camera_keys:
        if camera_key not in records:
            raise KeyError(f"Missing EBench camera record: {camera_key}")
        record = records[camera_key]
        latent = record.get("latent")
        if not torch.is_tensor(latent) or latent.ndim != 2:
            raise ValueError(f"{camera_key}: latent must be a rank-2 tensor")
        if latent.dtype != torch.bfloat16:
            raise ValueError(
                f"{camera_key}: latent dtype must be torch.bfloat16, got {latent.dtype}"
            )
        if not torch.isfinite(latent).all():
            raise ValueError(f"{camera_key}: latent contains non-finite values")

        frame_count = int(record["latent_num_frames"])
        height = int(record["latent_height"])
        width = int(record["latent_width"])
        channels = int(latent.shape[1])
        shape = (frame_count, height, width, channels)
        if frame_count <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"{camera_key}: invalid latent shape {shape}")
        if latent.shape[0] != frame_count * height * width:
            raise ValueError(
                f"{camera_key}: flattened latent has {latent.shape[0]} rows, "
                f"expected {frame_count * height * width}"
            )
        if common_shape is None:
            common_shape = shape
        elif shape != common_shape:
            raise ValueError(
                f"EBench camera latent mismatch: {camera_key} has {shape}, "
                f"expected {common_shape}"
            )

        frame_ids = _as_numpy(record["frame_ids"]).astype(np.int64, copy=False)
        if frame_ids.ndim != 1:
            raise ValueError(f"{camera_key}: frame_ids must be one-dimensional")
        expected_frame_ids = np.arange(0, 16 * (frame_count - 1) + 1, 4, dtype=np.int64)
        if not np.array_equal(frame_ids, expected_frame_ids):
            raise ValueError(
                f"{camera_key}: expected sampled frame_ids 0,4,... with "
                f"{len(expected_frame_ids)} entries for {frame_count} latent frames"
            )
        if "video_num_frames" in record and int(record["video_num_frames"]) != len(
            frame_ids
        ):
            raise ValueError(f"{camera_key}: video_num_frames does not match frame_ids")
        if common_frame_ids is None:
            common_frame_ids = frame_ids
        elif not np.array_equal(frame_ids, common_frame_ids):
            raise ValueError("EBench camera frame_ids are not identical")

        reshaped.append(latent.reshape(frame_count, height, width, channels))

    assert common_shape is not None and common_frame_ids is not None
    return reshaped, common_frame_ids, *common_shape


def assemble_ebench_camera_grid(
    records: Mapping[str, Mapping[str, Any]],
    camera_keys: Sequence[str] = EBENCH_CAMERA_KEYS,
    *,
    latent_start: int = 0,
    latent_stop: int | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Assemble top|overlook / left|right into ``[C,F,2H,2W]``."""
    latents, frame_ids, frame_count, _, _, _ = _validate_camera_records(
        records, camera_keys
    )
    latent_start = int(latent_start)
    latent_stop = frame_count if latent_stop is None else int(latent_stop)
    if not 0 <= latent_start < latent_stop <= frame_count:
        raise ValueError(
            f"Invalid EBench latent slice [{latent_start},{latent_stop}) for F={frame_count}"
        )
    latents = [latent[latent_start:latent_stop] for latent in latents]
    top_row = torch.cat((latents[0], latents[1]), dim=2)
    bottom_row = torch.cat((latents[2], latents[3]), dim=2)
    grid = torch.cat((top_row, bottom_row), dim=1)
    rgb_start = 4 * latent_start
    rgb_stop = 4 * (latent_stop - 1) + 1
    return (
        grid.permute(3, 0, 1, 2).contiguous(),
        frame_ids[rgb_start:rgb_stop],
    )


def map_and_normalize_ebench_actions(
    action_19d: np.ndarray | torch.Tensor,
    loss_mask_19d: np.ndarray | torch.Tensor,
    q01: np.ndarray | torch.Tensor,
    q99: np.ndarray | torch.Tensor,
    model_channel_ids: Sequence[int] = EBENCH_MODEL_CHANNEL_IDS,
    *,
    frame_count: int,
    action_per_frame: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map an episode sidecar to normalized 30-D action-head channels."""
    action = _as_numpy(action_19d).astype(np.float32, copy=False)
    loss_mask = _as_numpy(loss_mask_19d).astype(np.bool_, copy=False)
    q01_array = _as_numpy(q01).astype(np.float32, copy=False).reshape(-1)
    q99_array = _as_numpy(q99).astype(np.float32, copy=False).reshape(-1)
    channel_ids = tuple(int(value) for value in model_channel_ids)

    required_rows = int(frame_count) * int(action_per_frame)
    if action.shape != (required_rows, 19):
        raise ValueError(
            f"EBench action_19d has shape {action.shape}, expected "
            f"{(required_rows, 19)}"
        )
    if loss_mask.shape != action.shape:
        raise ValueError(
            f"EBench loss_mask_19d has shape {loss_mask.shape}, expected {action.shape}"
        )
    if len(channel_ids) != 19 or len(set(channel_ids)) != 19:
        raise ValueError("EBench model_channel_ids must contain 19 unique channels")
    if q01_array.shape != q99_array.shape or q01_array.shape != (30,):
        raise ValueError("EBench q01/q99 must each contain 30 model-head channels")
    if min(channel_ids) < 0 or max(channel_ids) >= len(q01_array):
        raise ValueError(f"EBench model channel is outside [0,{len(q01_array)})")
    if np.any(action[:, EBENCH_FIXED_ZERO_PHYSICAL_CHANNEL] != 0):
        raise ValueError("EBench base yaw physical channel must be fixed raw zero")

    selected_q01 = q01_array[list(channel_ids)]
    selected_q99 = q99_array[list(channel_ids)]
    ranges = selected_q99 - selected_q01
    if not np.all(np.isfinite(selected_q01)) or not np.all(np.isfinite(selected_q99)):
        raise ValueError("EBench normalization quantiles must be finite")
    if np.any(ranges <= 0):
        invalid = [channel_ids[index] for index in np.flatnonzero(ranges <= 0)]
        raise ValueError(f"EBench normalization ranges are not positive: {invalid}")

    normalized_19d = (action - selected_q01) / (ranges + 1e-6) * 2.0 - 1.0
    normalized_19d = np.clip(normalized_19d, -1.5, 1.5)
    normalized_19d[:, EBENCH_FIXED_ZERO_PHYSICAL_CHANNEL] = 0.0
    aligned = np.zeros((required_rows, 30), dtype=np.float32)
    aligned_mask = np.zeros((required_rows, 30), dtype=np.bool_)
    aligned[:, channel_ids] = normalized_19d
    aligned_mask[:, channel_ids] = loss_mask
    # Masked channels must remain exactly zero even if a future sidecar masks a
    # subset of its physical channels.
    aligned *= aligned_mask

    actions = (
        torch.from_numpy(aligned)
        .reshape(frame_count, action_per_frame, 30)
        .permute(2, 0, 1)
        .unsqueeze(-1)
    )
    actions_mask = (
        torch.from_numpy(aligned_mask)
        .reshape(frame_count, action_per_frame, 30)
        .permute(2, 0, 1)
        .unsqueeze(-1)
    )
    return actions.contiguous(), actions_mask.contiguous()


class EBenchLatentDataset(torch.utils.data.Dataset):
    """Load complete EBench episodes from precomputed LingBot-VA artifacts."""

    def __init__(self, repo_id: str | Path, config, root: str | Path | None = None):
        self.repo_id = str(repo_id)
        self.root = Path(root) if root is not None else Path(repo_id)
        self.config = config
        self.used_video_keys = tuple(config.obs_cam_keys)
        if self.used_video_keys != EBENCH_CAMERA_KEYS:
            raise ValueError(
                f"EBench obs_cam_keys must equal {EBENCH_CAMERA_KEYS}, got "
                f"{self.used_video_keys}"
            )
        self.model_channel_ids = tuple(config.used_action_channel_ids)
        if self.model_channel_ids != EBENCH_MODEL_CHANNEL_IDS:
            raise ValueError(
                "EBench used_action_channel_ids does not match the fixed 19D->30D mapping"
            )
        if int(config.action_dim) != 30 or int(config.action_per_frame) != 16:
            raise ValueError("EBench requires action_dim=30 and action_per_frame=16")

        info_path = self.root / "meta" / "info.json"
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not info_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(
                f"EBench metadata is incomplete under {self.root / 'meta'}"
            )
        with info_path.open() as file:
            self.info = json.load(file)
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        if self.chunks_size <= 0:
            raise ValueError(f"Invalid EBench chunks_size={self.chunks_size}")

        configured_segment_manifest = getattr(
            config, "segment_manifest_path", None
        )
        configured_segment_audit = getattr(config, "segment_audit_path", None)
        if (configured_segment_manifest is None) != (
            configured_segment_audit is None
        ):
            raise ValueError(
                "segment_manifest_path and segment_audit_path must be configured together"
            )
        self.segmented = configured_segment_manifest is not None
        self.segment_manifest_path = (
            Path(configured_segment_manifest)
            if configured_segment_manifest is not None
            else None
        )
        self.segment_audit_path = (
            Path(configured_segment_audit)
            if configured_segment_audit is not None
            else None
        )
        if self.segment_manifest_path is not None and not self.segment_manifest_path.is_absolute():
            self.segment_manifest_path = self.root / self.segment_manifest_path
        if self.segment_audit_path is not None and not self.segment_audit_path.is_absolute():
            self.segment_audit_path = self.root / self.segment_audit_path

        configured_stats_path = getattr(config, "action_stats_path", None)
        self.action_stats_path = (
            Path(configured_stats_path)
            if configured_stats_path is not None
            else self.root / "action_stats_v1.json"
        )
        if not self.action_stats_path.is_absolute():
            self.action_stats_path = self.root / self.action_stats_path
        self.source_action_stats_path = self.root / "action_stats_v1.json"
        if not self.source_action_stats_path.is_file():
            raise FileNotFoundError(self.source_action_stats_path)
        self.source_action_stats_sha256 = _sha256_file(
            self.source_action_stats_path
        )
        with self.action_stats_path.open() as file:
            self.action_stats_artifact = json.load(file)
        self.action_stats_sha256 = _sha256_file(self.action_stats_path)
        self._validate_action_stats_contract()
        try:
            norm_stat = self.action_stats_artifact["norm_stat"]
            self.q01 = np.asarray(norm_stat["q01"], dtype=np.float32)
            self.q99 = np.asarray(norm_stat["q99"], dtype=np.float32)
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Malformed EBench stats artifact: {self.action_stats_path}"
            ) from error
        if self.q01.shape != (30,) or self.q99.shape != (30,):
            raise ValueError("EBench stats norm_stat q01/q99 must each have length 30")
        if (
            not np.all(np.isfinite(self.q01))
            or not np.all(np.isfinite(self.q99))
            or np.any(self.q99 <= self.q01)
        ):
            raise ValueError("EBench stats normalization ranges must be finite and positive")
        yaw_model_channel = self.model_channel_ids[
            EBENCH_FIXED_ZERO_PHYSICAL_CHANNEL
        ]
        if self.q01[yaw_model_channel] != -1.0 or self.q99[yaw_model_channel] != 1.0:
            raise ValueError("EBench fixed-zero yaw must use the neutral [-1,1] range")
        episodes: list[dict[str, Any]] = []
        with episodes_path.open() as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                episode = json.loads(line)
                episode_index = int(episode["episode_index"])
                length = int(episode["length"])
                if length <= 0:
                    raise ValueError(
                        f"EBench episode {episode_index} has invalid length={length}"
                    )
                action_configs = episode.get("action_config", ())
                if len(action_configs) != 1:
                    raise ValueError(
                        f"EBench episode {episode_index} must have exactly one action_config"
                    )
                action_config = action_configs[0]
                if (
                    int(action_config["start_frame"]) != 0
                    or int(action_config["end_frame"]) != length
                ):
                    raise ValueError(
                        f"EBench episode {episode_index} action_config must cover [0,{length})"
                    )
                tasks = episode.get("tasks", ())
                if not tasks:
                    raise ValueError(
                        f"EBench episode {episode_index} has no task at line {line_number}"
                    )
                episodes.append(
                    {
                        "episode_index": episode_index,
                        "length": length,
                        "start_frame": 0,
                        "end_frame": length,
                        "tasks": list(tasks),
                    }
                )
        episode_by_index: dict[int, dict[str, Any]] = {}
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            if episode_index in episode_by_index:
                raise ValueError(
                    f"Duplicate EBench episode_index in metadata: {episode_index}"
                )
            episode_by_index[episode_index] = episode
        if not episode_by_index:
            raise ValueError(f"No EBench episodes found in {episodes_path}")
        expected_latent_keys = {
            (episode_index, camera_key)
            for episode_index in episode_by_index
            for camera_key in EBENCH_CAMERA_KEYS
        }
        if set(self.latent_manifest_entries) != expected_latent_keys:
            missing = len(expected_latent_keys - set(self.latent_manifest_entries))
            extra = len(set(self.latent_manifest_entries) - expected_latent_keys)
            raise ValueError(
                "EBench latent manifest does not cover every episode/camera exactly once: "
                f"missing={missing}, extra={extra}"
            )

        requested_indices = getattr(config, "ebench_episode_indices", None)
        if requested_indices is None:
            selected_indices = sorted(episode_by_index)
        else:
            if isinstance(requested_indices, (str, bytes)):
                raise ValueError("ebench_episode_indices must be a sequence of integers")
            try:
                selected_indices = [integer_index(value) for value in requested_indices]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "ebench_episode_indices must be a sequence of integers"
                ) from error
            if not selected_indices:
                raise ValueError("ebench_episode_indices must not be empty")
            if len(selected_indices) != len(set(selected_indices)):
                raise ValueError("ebench_episode_indices contains duplicate indices")
            unknown = sorted(set(selected_indices) - set(episode_by_index))
            if unknown:
                raise ValueError(
                    f"ebench_episode_indices contains unknown indices: {unknown}"
                )
        self.episode_indices = tuple(selected_indices)
        if self.segmented:
            segments_by_episode: dict[int, list[dict[str, Any]]] = {}
            for segment in self.segment_entries:
                segments_by_episode.setdefault(
                    int(segment["episode_index"]), []
                ).append(segment)
            segment_metas = []
            for episode_index in selected_indices:
                parent = episode_by_index[episode_index]
                for segment in segments_by_episode[episode_index]:
                    if segment["task"] != str(parent["tasks"][0]):
                        raise ValueError(
                            f"EBench segment task mismatch for episode {episode_index}"
                        )
                    segment_metas.append(
                        {
                            **parent,
                            **segment,
                            "source_start_frame": int(parent["start_frame"]),
                            "source_end_frame": int(parent["end_frame"]),
                        }
                    )
            if not segment_metas:
                raise ValueError("Selected EBench episodes contain no segments")
            self.new_metas = segment_metas
        else:
            self.new_metas = [episode_by_index[index] for index in selected_indices]

    def _read_verified_manifest(
        self, filename: str, expected_sha256: Any
    ) -> dict[str, Any]:
        path = self.root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing EBench artifact manifest: {path}")
        expected = _validated_sha256(
            expected_sha256, f"EBench {filename} reference"
        )
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"EBench {filename} SHA256 mismatch: expected={expected}, actual={actual}"
            )
        with path.open() as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError(f"EBench {filename} must contain a JSON object")
        return value

    def _validate_action_stats_contract(self) -> None:
        artifact = self.action_stats_artifact
        expected_stats_version = (
            EBENCH_SEGMENTED_ACTION_STATS_VERSION
            if self.segmented
            else EBENCH_ACTION_STATS_VERSION
        )
        if artifact.get("artifact_version") != expected_stats_version:
            raise ValueError(
                "Unsupported EBench action stats artifact_version: "
                f"{artifact.get('artifact_version')!r}"
            )

        representation = artifact.get("representation")
        expected_representation = {
            "physical_action_dim": 19,
            "model_action_dim": 30,
            "joint_anchor": (
                "segment first retained action.joints"
                if self.segmented
                else "episode action.joints[0]"
            ),
            "gripper_representation": "absolute action.gripper",
            "base_representation": (
                "segment-local inclusive cumsum(action.base_delta); yaw=0"
                if self.segmented
                else "inclusive cumsum(action.base_delta); yaw=0"
            ),
            "zero_prefix_rows": 16,
            "tail_alignment": "16 * floor((length - 1) / 16)",
        }
        if not isinstance(representation, Mapping) or any(
            representation.get(key) != expected
            for key, expected in expected_representation.items()
        ):
            raise ValueError("EBench action stats representation contract mismatch")

        channel_mapping = artifact.get("channel_mapping")
        expected_used = tuple(sorted(self.model_channel_ids))
        expected_unused = tuple(
            channel for channel in range(30) if channel not in expected_used
        )
        try:
            physical_to_model = tuple(
                int(value)
                for value in channel_mapping["physical_19d_to_model_30d"]
            )
            used_model_channels = tuple(
                int(value) for value in channel_mapping["used_model_channels"]
            )
            unused_model_channels = tuple(
                int(value) for value in channel_mapping["unused_model_channels"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed EBench action stats channel_mapping") from error
        if (
            physical_to_model != self.model_channel_ids
            or used_model_channels != expected_used
            or unused_model_channels != expected_unused
        ):
            raise ValueError("EBench action stats channel mapping mismatch")

        normalization = artifact.get("normalization")
        if (
            not isinstance(normalization, Mapping)
            or normalization.get("clip") != [-1.5, 1.5]
            or normalization.get("fixed_zero_neutral_range") != [-1.0, 1.0]
        ):
            raise ValueError("EBench action stats normalization contract mismatch")

        manifests = artifact.get("manifests")
        if not isinstance(manifests, Mapping):
            raise ValueError("EBench action stats must reference data/latent manifests")
        self.data_manifest = self._read_verified_manifest(
            "data_manifest.json", manifests.get("data_manifest_sha256")
        )
        self.latent_manifest = self._read_verified_manifest(
            "latent_manifest.json", manifests.get("latent_manifest_sha256")
        )
        if self.data_manifest.get("manifest_version") != EBENCH_DATA_MANIFEST_VERSION:
            raise ValueError("Unsupported EBench data manifest version")
        if (
            self.latent_manifest.get("manifest_version")
            != EBENCH_LATENT_MANIFEST_VERSION
            or self.latent_manifest.get("complete_dataset") is not True
        ):
            raise ValueError("EBench latent manifest is not a complete full-data audit")
        if tuple(self.latent_manifest.get("camera_keys", ())) != EBENCH_CAMERA_KEYS:
            raise ValueError("EBench latent manifest camera order mismatch")

        latent_entries: dict[tuple[int, str], dict[str, Any]] = {}
        latent_paths: set[str] = set()
        try:
            for raw_entry in self.latent_manifest["files"]:
                if not isinstance(raw_entry, Mapping):
                    raise TypeError("latent file entry must be an object")
                episode_index = integer_index(raw_entry["episode_index"])
                camera_key = str(raw_entry["camera_key"])
                relative_path = str(raw_entry["path"])
                relative = Path(relative_path)
                if (
                    camera_key not in EBENCH_CAMERA_KEYS
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or relative_path != relative.as_posix()
                ):
                    raise ValueError("invalid latent camera/path")
                size = integer_index(raw_entry["size"])
                latent_num_frames = integer_index(raw_entry["latent_num_frames"])
                if size <= 0 or latent_num_frames <= 0:
                    raise ValueError("invalid latent size/frame count")
                sha256 = _validated_sha256(
                    raw_entry["sha256"],
                    f"EBench episode {episode_index} {camera_key} latent",
                )
                key = (episode_index, camera_key)
                if key in latent_entries or relative_path in latent_paths:
                    raise ValueError("duplicate latent manifest entry")
                latent_entries[key] = {
                    "episode_index": episode_index,
                    "camera_key": camera_key,
                    "path": relative_path,
                    "size": size,
                    "sha256": sha256,
                    "latent_num_frames": latent_num_frames,
                }
                latent_paths.add(relative_path)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed EBench latent manifest file entries") from error
        self.latent_manifest_entries = latent_entries

        try:
            data_entries = {
                int(entry["episode_index"]): dict(entry)
                for entry in self.data_manifest["episodes"]
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed EBench data manifest episode entries") from error
        if len(data_entries) != len(self.data_manifest["episodes"]):
            raise ValueError("EBench data manifest contains duplicate episode indices")
        self.data_manifest_entries = data_entries

        expected_episodes = int(self.info.get("total_episodes", -1))
        expected_videos = int(self.info.get("total_videos", -1))
        if (
            len(data_entries) != expected_episodes
            or int(self.latent_manifest.get("episodes_audited", -1))
            != expected_episodes
            or len(self.latent_manifest.get("files", ())) != expected_videos
        ):
            raise ValueError(
                "EBench manifests do not cover the metadata-declared full dataset"
            )

        derived_hashes = self.data_manifest.get("derived_metadata_sha256")
        if not isinstance(derived_hashes, Mapping):
            raise ValueError("EBench data manifest lacks derived metadata hashes")
        for filename in ("info.json", "episodes.jsonl", "tasks.jsonl"):
            expected = _validated_sha256(
                derived_hashes.get(filename),
                f"EBench derived metadata {filename}",
            )
            path = self.root / "meta" / filename
            if not path.is_file() or _sha256_file(path) != expected:
                raise ValueError(f"EBench derived metadata SHA256 mismatch: {path}")

        preprocess_path = self.root / "preprocess_manifest.json"
        if not preprocess_path.is_file():
            raise FileNotFoundError(
                f"Missing EBench preprocess manifest: {preprocess_path}"
            )
        try:
            with preprocess_path.open() as file:
                preprocess_manifest = json.load(file)
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"Malformed EBench preprocess manifest: {preprocess_path}"
            ) from error
        if (
            not isinstance(preprocess_manifest, Mapping)
            or preprocess_manifest.get("manifest_version")
            != EBENCH_PREPROCESS_MANIFEST_VERSION
            or Path(str(preprocess_manifest.get("output_root", ""))).resolve()
            != self.root.resolve()
        ):
            raise ValueError("EBench preprocess manifest contract mismatch")
        preprocess_artifacts = preprocess_manifest.get("artifacts")
        if not isinstance(preprocess_artifacts, Mapping):
            raise ValueError("EBench preprocess manifest artifact map is missing")
        if (
            preprocess_artifacts.get("data_manifest") != "data_manifest.json"
            or preprocess_artifacts.get("data_manifest_sha256")
            != _sha256_file(self.root / "data_manifest.json")
            or preprocess_artifacts.get("action_stats") != "action_stats_v1.json"
            or preprocess_artifacts.get("action_stats_sha256")
            != self.source_action_stats_sha256
        ):
            raise ValueError("EBench preprocess data/stats artifact references mismatch")
        latent_artifact = preprocess_artifacts.get("latent_manifest")
        if (
            not isinstance(latent_artifact, Mapping)
            or latent_artifact.get("path") != "latent_manifest.json"
            or latent_artifact.get("sha256")
            != _sha256_file(self.root / "latent_manifest.json")
        ):
            raise ValueError("EBench preprocess latent artifact reference mismatch")
        text_artifact = preprocess_artifacts.get("text_cache")
        if not isinstance(text_artifact, Mapping):
            raise ValueError("EBench preprocess text artifact reference is missing")
        if (
            text_artifact.get("path") != "text_embeddings.pt"
            or text_artifact.get("empty_path") != "empty_emb.pt"
        ):
            raise ValueError("EBench preprocess text artifact paths mismatch")
        self.text_embeddings_path = self.root / "text_embeddings.pt"
        self.empty_embedding_path = self.root / "empty_emb.pt"
        if not self.text_embeddings_path.is_file() or not self.empty_embedding_path.is_file():
            raise FileNotFoundError("EBench shared text embedding artifacts are missing")
        self.text_embeddings_sha256 = _validated_sha256(
            text_artifact.get("sha256"), "EBench shared text cache"
        )
        self.empty_embedding_sha256 = _validated_sha256(
            text_artifact.get("empty_sha256"), "EBench empty text embedding"
        )
        if _sha256_file(self.text_embeddings_path) != self.text_embeddings_sha256:
            raise ValueError("EBench shared text cache SHA256 mismatch")
        if _sha256_file(self.empty_embedding_path) != self.empty_embedding_sha256:
            raise ValueError("EBench empty text embedding SHA256 mismatch")

        marker_path = self.root / EBENCH_AUDIT_SUCCESS_MARKER_NAME
        if not marker_path.is_file():
            raise FileNotFoundError(
                f"Missing EBench full-audit success marker: {marker_path}"
            )
        try:
            with marker_path.open() as file:
                marker = json.load(file)
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"Malformed EBench full-audit success marker: {marker_path}"
            ) from error
        if (
            not isinstance(marker, Mapping)
            or marker.get("marker_version")
            != EBENCH_AUDIT_SUCCESS_MARKER_VERSION
            or Path(str(marker.get("dataset_root", ""))).resolve()
            != self.root.resolve()
        ):
            raise ValueError("EBench full-audit success marker contract mismatch")
        marker_artifacts = marker.get("artifacts")
        expected_artifacts = {
            "action_stats_v1.json": self.source_action_stats_sha256,
            "data_manifest.json": _sha256_file(self.root / "data_manifest.json"),
            "latent_manifest.json": _sha256_file(self.root / "latent_manifest.json"),
            "preprocess_manifest.json": _sha256_file(preprocess_path),
            "text_embeddings.pt": self.text_embeddings_sha256,
            "empty_emb.pt": self.empty_embedding_sha256,
        }
        if not isinstance(marker_artifacts, Mapping) or dict(marker_artifacts) != expected_artifacts:
            raise ValueError(
                "EBench full-audit success marker does not match current artifacts"
            )
        if self.segmented:
            self._validate_segment_overlay()

    def _validate_segment_overlay(self) -> None:
        if self.segment_manifest_path is None or self.segment_audit_path is None:
            raise RuntimeError("Segment overlay paths are not configured")
        if not self.segment_manifest_path.is_file():
            raise FileNotFoundError(self.segment_manifest_path)
        if not self.segment_audit_path.is_file():
            raise FileNotFoundError(self.segment_audit_path)
        if self.segment_manifest_path.resolve().parent != self.root.resolve():
            raise ValueError("EBench segment manifest must live at the dataset root")
        if self.segment_audit_path.resolve().parent != self.root.resolve():
            raise ValueError("EBench segment audit marker must live at the dataset root")

        stats_manifests = self.action_stats_artifact.get("manifests")
        if not isinstance(stats_manifests, Mapping):
            raise ValueError("Segmented EBench stats lacks manifest references")
        expected_segment_sha = _validated_sha256(
            stats_manifests.get("segment_manifest_sha256"),
            "EBench segment manifest",
        )
        if stats_manifests.get("segment_manifest_path") != self.segment_manifest_path.name:
            raise ValueError("Segmented stats references a different segment manifest")
        if (
            _validated_sha256(
                stats_manifests.get("source_action_stats_sha256"),
                "EBench source action stats",
            )
            != self.source_action_stats_sha256
        ):
            raise ValueError("Segmented stats source action stats SHA256 mismatch")
        actual_segment_sha = _sha256_file(self.segment_manifest_path)
        if actual_segment_sha != expected_segment_sha:
            raise ValueError("EBench segment manifest SHA256 mismatch")

        with self.segment_manifest_path.open() as file:
            manifest = json.load(file)
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("manifest_version") != EBENCH_SEGMENT_MANIFEST_VERSION
            or Path(str(manifest.get("dataset_root", ""))).resolve()
            != self.root.resolve()
        ):
            raise ValueError("EBench segment manifest contract mismatch")
        max_self_tokens = int(manifest.get("max_self_tokens", -1))
        configured_max_tokens = int(getattr(self.config, "max_self_tokens", -1))
        if max_self_tokens <= 0 or max_self_tokens != configured_max_tokens:
            raise ValueError(
                "EBench segment manifest max_self_tokens differs from training config"
            )
        max_segment_frames = int(manifest.get("max_segment_frames", -1))
        max_real_blocks = int(manifest.get("max_real_blocks_per_segment", -1))
        if (
            max_segment_frames != max_self_tokens // EBENCH_SELF_TOKENS_PER_FRAME
            or max_real_blocks != max_segment_frames - 1
        ):
            raise ValueError("EBench segment manifest frame budget is inconsistent")

        source_paths = {
            "action_stats_v1.json": self.source_action_stats_path,
            "data_manifest.json": self.root / "data_manifest.json",
            "latent_manifest.json": self.root / "latent_manifest.json",
            "preprocess_manifest.json": self.root / "preprocess_manifest.json",
            EBENCH_AUDIT_SUCCESS_MARKER_NAME: (
                self.root / EBENCH_AUDIT_SUCCESS_MARKER_NAME
            ),
        }
        expected_source = {
            filename: _sha256_file(path) for filename, path in source_paths.items()
        }
        if dict(manifest.get("source_artifacts", {})) != expected_source:
            raise ValueError("EBench segment manifest source artifact SHA mismatch")

        raw_segments = manifest.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("EBench segment manifest has no segments")
        segments: list[dict[str, Any]] = []
        sample_keys: set[str] = set()
        grouped: dict[int, list[dict[str, Any]]] = {}
        for expected_index, raw in enumerate(raw_segments):
            if not isinstance(raw, Mapping):
                raise ValueError("EBench segment entry must be an object")
            try:
                entry = {
                    "segment_index": integer_index(raw["segment_index"]),
                    "episode_index": integer_index(raw["episode_index"]),
                    "segment_ordinal": integer_index(raw["segment_ordinal"]),
                    "real_block_start": integer_index(raw["real_block_start"]),
                    "real_block_end": integer_index(raw["real_block_end"]),
                    "real_action_row_start": integer_index(
                        raw["real_action_row_start"]
                    ),
                    "real_action_row_end": integer_index(
                        raw["real_action_row_end"]
                    ),
                    "latent_start": integer_index(raw["latent_start"]),
                    "latent_stop": integer_index(raw["latent_stop"]),
                    "frame_count": integer_index(raw["frame_count"]),
                    "token_count": integer_index(raw["token_count"]),
                    "task": str(raw["task"]),
                    "segment_joint_anchor_12d": list(
                        raw["segment_joint_anchor_12d"]
                    ),
                    "segment_base_anchor_3d": list(raw["segment_base_anchor_3d"]),
                    "segment_action_sha256": _validated_sha256(
                        raw["segment_action_sha256"],
                        f"EBench segment {expected_index} action",
                    ),
                    "sample_key": str(raw["sample_key"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Malformed EBench segment entry {expected_index}"
                ) from error
            start = entry["real_block_start"]
            end = entry["real_block_end"]
            frame_count = entry["frame_count"]
            if (
                entry["segment_index"] != expected_index
                or entry["segment_ordinal"] < 0
                or not 0 <= start < end
                or end - start > max_real_blocks
                or entry["latent_start"] != start
                or entry["latent_stop"] != end + 1
                or frame_count != 1 + end - start
                or entry["real_action_row_start"] != 16 * start
                or entry["real_action_row_end"] != 16 * end
                or entry["token_count"] != ebench_self_token_count(frame_count)
                or entry["token_count"] > max_self_tokens
            ):
                raise ValueError(f"Invalid EBench segment geometry at {expected_index}")
            joint_anchor = np.asarray(entry["segment_joint_anchor_12d"])
            base_anchor = np.asarray(entry["segment_base_anchor_3d"])
            if (
                joint_anchor.shape != (12,)
                or base_anchor.shape != (3,)
                or not np.isfinite(joint_anchor).all()
                or not np.isfinite(base_anchor).all()
            ):
                raise ValueError(f"Invalid EBench segment anchors at {expected_index}")
            expected_sample_key = "|".join(
                (
                    str(self.root.resolve()),
                    str(entry["episode_index"]),
                    "segment",
                    str(entry["segment_ordinal"]),
                    str(start),
                    str(end),
                    EBENCH_SEGMENT_MANIFEST_VERSION,
                )
            )
            if entry["sample_key"] != expected_sample_key:
                raise ValueError(f"Invalid EBench segment sample_key at {expected_index}")
            if entry["sample_key"] in sample_keys:
                raise ValueError("Duplicate EBench segment sample_key")
            sample_keys.add(entry["sample_key"])
            segments.append(entry)
            grouped.setdefault(entry["episode_index"], []).append(entry)

        expected_episode_indices = set(self.data_manifest_entries)
        if set(grouped) != expected_episode_indices:
            raise ValueError("EBench segment manifest does not cover every parent episode")
        split_episodes = 0
        virtual_frames = 0
        source_frames = 0
        real_action_rows = 0
        total_tokens = 0
        for episode_index in sorted(grouped):
            episode_segments = grouped[episode_index]
            latent_frame_counts = {
                int(self.latent_manifest_entries[(episode_index, camera)][
                    "latent_num_frames"
                ])
                for camera in EBENCH_CAMERA_KEYS
            }
            if len(latent_frame_counts) != 1:
                raise ValueError("Segment parent cameras have different frame counts")
            parent_frames = latent_frame_counts.pop()
            previous_end = 0
            for ordinal, entry in enumerate(episode_segments):
                if (
                    entry["segment_ordinal"] != ordinal
                    or entry["real_block_start"] != previous_end
                ):
                    raise ValueError("EBench segment blocks are not contiguous")
                previous_end = entry["real_block_end"]
            if previous_end != parent_frames - 1:
                raise ValueError("EBench segment blocks do not cover the parent episode")
            split_episodes += int(len(episode_segments) > 1)
            source_frames += parent_frames
            virtual_frames += sum(entry["frame_count"] for entry in episode_segments)
            real_action_rows += 16 * (parent_frames - 1)
            total_tokens += sum(entry["token_count"] for entry in episode_segments)

        counts = {
            "parent_episodes": len(grouped),
            "split_episodes": split_episodes,
            "segments": len(segments),
            "source_latent_frames": source_frames,
            "virtual_segment_frames": virtual_frames,
            "real_action_rows": real_action_rows,
            "zero_prefix_rows": EBENCH_ZERO_PREFIX_ROWS * len(segments),
            "training_action_rows": (
                real_action_rows + EBENCH_ZERO_PREFIX_ROWS * len(segments)
            ),
            "total_padded_self_tokens": total_tokens,
        }
        if dict(manifest.get("counts", {})) != counts:
            raise ValueError("EBench segment manifest counts are inconsistent")
        quantiles = self.action_stats_artifact.get("quantiles")
        segmentation = self.action_stats_artifact.get("segmentation")
        if (
            not isinstance(quantiles, Mapping)
            or int(quantiles.get("frame_count", -1)) != real_action_rows
            or not isinstance(segmentation, Mapping)
            or int(segmentation.get("max_self_tokens", -1)) != max_self_tokens
            or int(segmentation.get("max_segment_frames", -1))
            != max_segment_frames
            or dict(segmentation.get("counts", {})) != counts
        ):
            raise ValueError("Segmented EBench stats counts/policy mismatch")

        with self.segment_audit_path.open() as file:
            audit = json.load(file)
        expected_audit_artifacts = {
            "segment_manifest": {
                "path": self.segment_manifest_path.name,
                "sha256": actual_segment_sha,
            },
            "action_stats": {
                "path": self.action_stats_path.name,
                "sha256": self.action_stats_sha256,
            },
            "source": expected_source,
        }
        if (
            not isinstance(audit, Mapping)
            or audit.get("marker_version") != EBENCH_SEGMENT_AUDIT_VERSION
            or Path(str(audit.get("dataset_root", ""))).resolve()
            != self.root.resolve()
            or int(audit.get("max_self_tokens", -1)) != max_self_tokens
            or dict(audit.get("artifacts", {})) != expected_audit_artifacts
            or dict(audit.get("counts", {})) != counts
        ):
            raise ValueError("EBench segment audit marker contract mismatch")
        self.segment_manifest = manifest
        self.segment_entries = tuple(segments)
        self.segment_manifest_sha256 = actual_segment_sha

    def __len__(self) -> int:
        return len(self.new_metas)

    def _episode_chunk(self, episode_index: int) -> int:
        return int(episode_index) // self.chunks_size

    def _latent_file_path(self, meta: Mapping[str, Any], camera_key: str) -> Path:
        episode_index = int(meta["episode_index"])
        source_start = int(meta.get("source_start_frame", meta["start_frame"]))
        source_end = int(meta.get("source_end_frame", meta["end_frame"]))
        return (
            self.root
            / "latents"
            / f"chunk-{self._episode_chunk(episode_index):03d}"
            / camera_key
            / (
                f"episode_{episode_index:06d}_{source_start}_{source_end}.pth"
            )
        )

    def _action_file_path(self, meta: Mapping[str, Any]) -> Path:
        episode_index = int(meta["episode_index"])
        return (
            self.root
            / "actions"
            / f"chunk-{self._episode_chunk(episode_index):03d}"
            / f"episode_{episode_index:06d}.npz"
        )

    def _load_camera_records(self, meta: Mapping[str, Any]) -> dict[str, Any]:
        records = {}
        episode_index = int(meta["episode_index"])
        for camera_key in self.used_video_keys:
            path = self._latent_file_path(meta, camera_key)
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest_entry = self.latent_manifest_entries.get(
                (episode_index, camera_key)
            )
            if manifest_entry is None:
                raise ValueError(
                    f"EBench latent manifest has no entry for episode "
                    f"{episode_index} camera {camera_key}"
                )
            expected_relative_path = path.relative_to(self.root).as_posix()
            if manifest_entry["path"] != expected_relative_path:
                raise ValueError(
                    "EBench latent path differs from audited latent manifest: "
                    f"{path}"
                )
            if path.stat().st_size != manifest_entry["size"]:
                raise ValueError(
                    "EBench latent size differs from audited latent manifest: "
                    f"{path}"
                )
            actual_sha256 = _sha256_file(path)
            if actual_sha256 != manifest_entry["sha256"]:
                raise ValueError(
                    "EBench latent SHA256 differs from audited latent manifest: "
                    f"{path}"
                )
            record = torch.load(
                path, map_location="cpu", weights_only=False, mmap=True
            )
            if int(record.get("latent_num_frames", -1)) != manifest_entry[
                "latent_num_frames"
            ]:
                raise ValueError(
                    "EBench latent frame count differs from audited latent manifest: "
                    f"{path}"
                )
            records[camera_key] = record
        return records

    def _validate_episode_frame_count(
        self, meta: Mapping[str, Any], frame_count: int
    ) -> int:
        real_action_count = 16 * ((int(meta["length"]) - 1) // 16)
        expected_frame_count = 1 + real_action_count // 16
        if int(frame_count) != expected_frame_count:
            raise ValueError(
                f"EBench episode {meta['episode_index']} has {frame_count} latent "
                f"frames, expected {expected_frame_count} from length={meta['length']}"
            )
        return real_action_count

    def _load_action_sidecar(
        self, meta: Mapping[str, Any], frame_count: int
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        action_path = self._action_file_path(meta)
        if not action_path.is_file():
            raise FileNotFoundError(action_path)
        episode_index = int(meta["episode_index"])
        manifest_entry = self.data_manifest_entries.get(episode_index)
        if manifest_entry is None:
            raise ValueError(
                f"EBench data manifest has no action entry for episode {episode_index}"
            )
        expected_relative_path = str(action_path.relative_to(self.root))
        if manifest_entry.get("action_sidecar") != expected_relative_path:
            raise ValueError(
                f"EBench action sidecar path differs from data manifest: {action_path}"
            )
        real_action_count = self._validate_episode_frame_count(meta, frame_count)
        with np.load(action_path, allow_pickle=False) as sidecar:
            action_19d = sidecar["action_19d"]
            loss_mask_19d = sidecar["loss_mask_19d"]
            episode_anchor_12d = np.asarray(
                sidecar.get("episode_anchor_12d", np.zeros(12, np.float32)),
                dtype=np.float32,
            ).copy()
            expected_shape = (frame_count * int(self.config.action_per_frame), 19)
            if (
                action_19d.shape != expected_shape
                or loss_mask_19d.shape != expected_shape
            ):
                raise ValueError(
                    f"EBench action sidecar shape mismatch: action={action_19d.shape}, "
                    f"mask={loss_mask_19d.shape}, expected={expected_shape}"
                )
            if not np.isfinite(action_19d).all():
                raise ValueError("EBench action sidecar contains non-finite values")
            if not np.all(action_19d[:16] == 0):
                raise ValueError(
                    "EBench action sidecar must start with one raw-zero16 prefix"
                )
            if not np.all(loss_mask_19d):
                raise ValueError("All 19 physical EBench action channels must train")
            if (
                "real_action_count" in sidecar
                and int(sidecar["real_action_count"]) != real_action_count
            ):
                raise ValueError(
                    "EBench real_action_count does not match episode tail alignment"
                )
            if episode_anchor_12d.shape != (12,):
                raise ValueError("EBench episode_anchor_12d must have shape [12]")
            if "source_frame_ids" in sidecar:
                expected_source_ids = np.concatenate(
                    (
                        np.full(16, -1, dtype=np.int64),
                        np.arange(real_action_count, dtype=np.int64),
                    )
                )
                if not np.array_equal(sidecar["source_frame_ids"], expected_source_ids):
                    raise ValueError(
                        "EBench source_frame_ids does not describe one prefix plus "
                        "the aligned real-action range"
                    )
        if int(manifest_entry.get("action_sidecar_size", -1)) != action_path.stat().st_size:
            raise ValueError(
                f"EBench action sidecar size differs from audited data manifest: {action_path}"
            )
        expected_sha256 = _validated_sha256(
            manifest_entry.get("action_sidecar_sha256"),
            f"EBench episode {episode_index} action sidecar",
        )
        actual_sha256 = _sha256_file(action_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "EBench action sidecar SHA256 differs from audited data manifest: "
                f"{action_path}"
            )
        return (
            action_19d,
            loss_mask_19d,
            {"episode_anchor_12d": episode_anchor_12d},
        )

    def _validate_record_text(
        self, records: Mapping[str, Mapping[str, Any]], expected_text: str
    ) -> None:
        texts = {str(record.get("text", "")) for record in records.values()}
        if texts != {expected_text}:
            raise ValueError(
                f"EBench camera task text mismatch: expected {expected_text!r}, got {texts}"
            )

    def get_packing_metadata(self, idx: int) -> dict[str, Any]:
        meta = self.new_metas[int(idx)]
        if self.segmented:
            return {
                "sample_key": str(meta["sample_key"]),
                "frame_count": int(meta["frame_count"]),
                "token_count": int(meta["token_count"]),
                "shape_signature": (
                    48,
                    28,
                    28,
                    int(self.config.action_dim),
                    int(self.config.action_per_frame),
                    1,
                ),
            }
        records = self._load_camera_records(meta)
        _, _, frame_count, height, width, channels = _validate_camera_records(
            records, self.used_video_keys
        )
        self._validate_record_text(records, str(meta["tasks"][0]))
        if (channels, height, width) != (48, 14, 14):
            raise ValueError(
                "EBench per-camera latent shape must be [48,F,14,14], got "
                f"[{channels},{frame_count},{height},{width}]"
            )

        self._load_action_sidecar(meta, frame_count)

        video_height = height * 2
        video_width = width * 2
        patch_f, patch_h, patch_w = self.config.patch_size
        if patch_f != 1:
            raise ValueError("Episode packing requires patch_size[0] == 1")
        if video_height % patch_h or video_width % patch_w:
            raise ValueError(
                f"EBench latent grid {(video_height, video_width)} is not divisible "
                f"by patch {(patch_h, patch_w)}"
            )
        video_tokens = (
            frame_count * (video_height // patch_h) * (video_width // patch_w)
        )
        action_tokens = frame_count * int(self.config.action_per_frame)
        token_count = round_up(2 * (video_tokens + action_tokens), 128)
        sample_key = "|".join(
            (
                str(self.root.resolve()),
                str(meta["episode_index"]),
                str(meta["start_frame"]),
                str(meta["end_frame"]),
            )
        )
        return {
            "sample_key": sample_key,
            "frame_count": frame_count,
            "token_count": token_count,
            "shape_signature": (
                channels,
                video_height,
                video_width,
                int(self.config.action_dim),
                int(self.config.action_per_frame),
                1,
            ),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = int(idx)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        meta = self.new_metas[idx]
        records = self._load_camera_records(meta)
        text = str(meta["tasks"][0])
        self._validate_record_text(records, text)
        full_frame_count = int(
            records[self.used_video_keys[0]]["latent_num_frames"]
        )
        latent_start = int(meta.get("latent_start", 0))
        latent_stop = int(meta.get("latent_stop", full_frame_count))
        latents, _ = assemble_ebench_camera_grid(
            records,
            self.used_video_keys,
            latent_start=latent_start,
            latent_stop=latent_stop,
        )
        if tuple(latents.shape[:1] + latents.shape[2:]) != (48, 28, 28):
            raise ValueError(
                f"EBench assembled latent must be [48,F,28,28], got {tuple(latents.shape)}"
            )

        action_19d, loss_mask_19d, action_metadata = self._load_action_sidecar(
            meta, full_frame_count
        )
        if self.segmented:
            spec = EBenchSegmentSpec(
                ordinal=int(meta["segment_ordinal"]),
                real_block_start=int(meta["real_block_start"]),
                real_block_end=int(meta["real_block_end"]),
            )
            (
                action_19d,
                loss_mask_19d,
                joint_anchor_delta,
                base_anchor,
            ) = reanchor_ebench_segment_actions(
                action_19d,
                loss_mask_19d,
                spec,
                action_per_frame=int(self.config.action_per_frame),
            )
            parent_anchor = action_metadata["episode_anchor_12d"]
            segment_joint_anchor = parent_anchor.copy()
            segment_joint_anchor[:6] += joint_anchor_delta[:6]
            segment_joint_anchor[6:12] += joint_anchor_delta[6:12]
            if not np.array_equal(
                segment_joint_anchor,
                np.asarray(meta["segment_joint_anchor_12d"], dtype=np.float32),
            ):
                raise ValueError("EBench segment joint anchor differs from manifest")
            if not np.array_equal(
                base_anchor,
                np.asarray(meta["segment_base_anchor_3d"], dtype=np.float32),
            ):
                raise ValueError("EBench segment base anchor differs from manifest")
            if (
                ebench_segment_action_sha256(action_19d, loss_mask_19d)
                != meta["segment_action_sha256"]
            ):
                raise ValueError("EBench virtual segment action SHA256 mismatch")
        if int(latents.shape[1]) * int(self.config.action_per_frame) != len(
            action_19d
        ):
            raise ValueError("EBench segment latent/action frame count mismatch")
        actions, actions_mask = map_and_normalize_ebench_actions(
            action_19d,
            loss_mask_19d,
            self.q01,
            self.q99,
            self.model_channel_ids,
            frame_count=int(latents.shape[1]),
            action_per_frame=int(self.config.action_per_frame),
        )
        return {
            "latents": latents,
            "actions": actions,
            "actions_mask": actions_mask,
            "text": text,
        }


__all__ = [
    "EBENCH_CAMERA_KEYS",
    "EBENCH_MODEL_CHANNEL_IDS",
    "EBENCH_AUDIT_SUCCESS_MARKER_NAME",
    "EBENCH_SEGMENT_AUDIT_VERSION",
    "EBENCH_SEGMENT_MANIFEST_VERSION",
    "EBENCH_SEGMENTED_ACTION_STATS_VERSION",
    "EBenchSegmentSpec",
    "EBenchLatentDataset",
    "assemble_ebench_camera_grid",
    "ebench_segment_action_sha256",
    "ebench_self_token_count",
    "map_and_normalize_ebench_actions",
    "plan_ebench_segments",
    "reanchor_ebench_segment_actions",
]
