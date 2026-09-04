#!/usr/bin/env python3
"""Prepare EBench Generalist for LingBot-VA episode training.

The public pure helpers in this module intentionally have no CUDA/model
dependency so the action protocol and temporal layout can be tested without a
GPU.  The CLI provides ``setup``, ``encode-text``, ``encode`` and ``audit``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SOURCE_ROOT = Path(
    "/root/shared/yaoyifei/dataset/openpi_lerobot/ebench/generalist"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/root/shared/yaoyifei/dataset/openpi_lerobot/ebench/generalist_lingbot_va"
)
DEFAULT_MODEL_ROOT = Path(
    "/data/shared/zouyude/ckpts/lingbot-va/lingbot-va-base"
)

SOURCE_CAMERA_KEYS = (
    "video.top_camera_view",
    "video.overlook_camera_view",
    "video.left_camera_view",
    "video.right_camera_view",
)
CAMERA_SHORT_NAMES = ("top", "overlook", "left", "right")
CAMERA_LAYOUT = (("top", "overlook"), ("left", "right"))

IMAGE_STRIDE = 4
VAE_TEMPORAL_STRIDE = 4
ACTION_PER_LATENT_FRAME = IMAGE_STRIDE * VAE_TEMPORAL_STRIDE
ZERO_PREFIX_ROWS = 16
ACTION_DIM_PHYSICAL = 19
ACTION_DIM_MODEL = 30
MODEL_CHANNELS = np.asarray(
    [
        14, 15, 16, 17, 18, 19,  # left joints
        28, 20,                  # left gripper
        21, 22, 23, 24, 25, 26, # right joints
        29, 27,                  # right gripper
        0, 1, 2,                 # base x, y, yaw
    ],
    dtype=np.int64,
)

TEXT_CACHE_NAME = "text_embeddings.pt"
EMPTY_EMBEDDING_NAME = "empty_emb.pt"
STATS_NAME = "action_stats_v1.json"
PREPROCESS_MANIFEST_NAME = "preprocess_manifest.json"
DATA_MANIFEST_NAME = "data_manifest.json"
LATENT_MANIFEST_NAME = "latent_manifest.json"
AUDIT_REPORT_NAME = "audit_report.json"
AUDIT_SUCCESS_MARKER_NAME = "audit_success.json"
AUDIT_SUCCESS_MARKER_VERSION = "ebench_lingbot_audit_success_v2"


def aligned_real_action_count(length: int) -> int:
    """Real rows retained after reserving one 16-action prefix frame."""
    length = int(length)
    if length < 1:
        raise ValueError(f"episode length must be positive, got {length}")
    return ACTION_PER_LATENT_FRAME * ((length - 1) // ACTION_PER_LATENT_FRAME)


def sampled_rgb_frame_ids(length: int, stride: int = IMAGE_STRIDE) -> np.ndarray:
    """Raw frame IDs sampled at stride four and cropped to ``1 + 4k`` RGBs."""
    if int(stride) != IMAGE_STRIDE:
        raise ValueError(f"EBench image stride is fixed at {IMAGE_STRIDE}")
    retained = aligned_real_action_count(length)
    return np.arange(0, retained + 1, IMAGE_STRIDE, dtype=np.int64)


def usable_sampled_frame_count(length: int) -> int:
    return int(sampled_rgb_frame_ids(length).size)


def expected_latent_frame_count(sampled_count: int) -> int:
    sampled_count = int(sampled_count)
    if sampled_count < 1 or (sampled_count - 1) % VAE_TEMPORAL_STRIDE:
        raise ValueError("sampled RGB count must be 1 + 4k")
    return 1 + (sampled_count - 1) // VAE_TEMPORAL_STRIDE


def transform_ebench_actions(
    joints: np.ndarray,
    gripper: np.ndarray,
    base_delta: np.ndarray,
) -> np.ndarray:
    """Convert one complete EBench episode into the fixed physical 19D order."""
    joints = np.asarray(joints, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)
    base_delta = np.asarray(base_delta, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] != 12:
        raise ValueError(f"joints must have shape [L,12], got {joints.shape}")
    length = joints.shape[0]
    if gripper.shape != (length, 4):
        raise ValueError(f"gripper must have shape {(length, 4)}, got {gripper.shape}")
    if base_delta.shape != (length, 3):
        raise ValueError(
            f"base_delta must have shape {(length, 3)}, got {base_delta.shape}"
        )
    if length < 1:
        raise ValueError("an episode must contain at least one action")

    action = np.empty((length, ACTION_DIM_PHYSICAL), dtype=np.float32)
    action[:, 0:6] = joints[:, 0:6] - joints[0, 0:6]
    action[:, 6:8] = gripper[:, 0:2]
    action[:, 8:14] = joints[:, 6:12] - joints[0, 6:12]
    action[:, 14:16] = gripper[:, 2:4]
    action[:, 16:19] = np.cumsum(base_delta, axis=0, dtype=np.float32)
    action[:, 18] = 0.0
    return action


def prepare_episode_actions(
    action_19d: np.ndarray,
    raw_length: int | None = None,
    zero_prefix: int = ZERO_PREFIX_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(stats_real, training_prefix_plus_real)`` for one episode.

    The returned stats population excludes both the synthesized prefix and the
    unaligned tail.  The training array contains exactly one raw-zero prefix.
    """
    action_19d = np.asarray(action_19d, dtype=np.float32)
    if action_19d.ndim != 2 or action_19d.shape[1] != ACTION_DIM_PHYSICAL:
        raise ValueError(f"action_19d must have shape [L,19], got {action_19d.shape}")
    if raw_length is None:
        raw_length = action_19d.shape[0]
    raw_length = int(raw_length)
    if action_19d.shape[0] < raw_length:
        raise ValueError("action_19d has fewer rows than raw_length")
    if int(zero_prefix) != ZERO_PREFIX_ROWS:
        raise ValueError(f"EBench zero prefix is fixed at {ZERO_PREFIX_ROWS}")
    real_count = aligned_real_action_count(raw_length)
    stats_action = np.ascontiguousarray(action_19d[:real_count])
    training_action = np.concatenate(
        [np.zeros((ZERO_PREFIX_ROWS, ACTION_DIM_PHYSICAL), np.float32), stats_action],
        axis=0,
    )
    return stats_action, np.ascontiguousarray(training_action)


def _zeros_like_action(action: Any, last_dim: int) -> Any:
    try:
        import torch

        if torch.is_tensor(action):
            return action.new_zeros((*action.shape[:-1], last_dim))
    except ImportError:
        pass
    return np.zeros((*np.asarray(action).shape[:-1], last_dim), dtype=np.asarray(action).dtype)


def map_action_19d_to_30d(action_19d: Any) -> Any:
    if action_19d.shape[-1] != ACTION_DIM_PHYSICAL:
        raise ValueError(f"last dimension must be 19, got {action_19d.shape}")
    result = _zeros_like_action(action_19d, ACTION_DIM_MODEL)
    result[..., MODEL_CHANNELS.tolist()] = action_19d
    return result


def map_action_30d_to_19d(action_30d: Any) -> Any:
    if action_30d.shape[-1] != ACTION_DIM_MODEL:
        raise ValueError(f"last dimension must be 30, got {action_30d.shape}")
    return action_30d[..., MODEL_CHANNELS.tolist()]


def model_action_mask_30d() -> np.ndarray:
    mask = np.zeros(ACTION_DIM_MODEL, dtype=bool)
    mask[MODEL_CHANNELS] = True
    return mask


def mosaic_camera_latents(camera_latents: Mapping[str, Any]) -> Any:
    """Compose independent camera latents as top|overlook / left|right."""
    resolved: dict[str, Any] = {}
    for short, full in zip(CAMERA_SHORT_NAMES, SOURCE_CAMERA_KEYS):
        if short in camera_latents:
            resolved[short] = camera_latents[short]
        elif full in camera_latents:
            resolved[short] = camera_latents[full]
        else:
            raise KeyError(f"missing camera latent {short!r} ({full!r})")
    first = resolved["top"]
    expected_shape = tuple(first.shape)
    if len(expected_shape) != 4:
        raise ValueError(f"camera latents must be [C,F,H,W], got {expected_shape}")
    for key, value in resolved.items():
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"camera {key} shape {tuple(value.shape)} != {expected_shape}")
    try:
        import torch

        concatenate = torch.cat if torch.is_tensor(first) else np.concatenate
    except ImportError:
        concatenate = np.concatenate
    top_row = concatenate([resolved["top"], resolved["overlook"]], dim=-1) if hasattr(first, "dim") else concatenate([resolved["top"], resolved["overlook"]], axis=-1)
    bottom_row = concatenate([resolved["left"], resolved["right"]], dim=-1) if hasattr(first, "dim") else concatenate([resolved["left"], resolved["right"]], axis=-1)
    return concatenate([top_row, bottom_row], dim=-2) if hasattr(first, "dim") else concatenate([top_row, bottom_row], axis=-2)


def compute_action_quantiles(
    actions: np.ndarray | Iterable[np.ndarray],
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel physical quantiles from retained real rows only."""
    if not 0.0 <= float(lower) < float(upper) <= 1.0:
        raise ValueError(
            f"quantile probabilities must satisfy 0 <= lower < upper <= 1, "
            f"got lower={lower}, upper={upper}"
        )
    if isinstance(actions, np.ndarray):
        values = actions
    else:
        chunks = [np.asarray(chunk, dtype=np.float32) for chunk in actions]
        if not chunks:
            raise ValueError("quantile population is empty")
        values = np.concatenate(chunks, axis=0)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM_PHYSICAL or not len(values):
        raise ValueError(f"actions must be non-empty [N,19], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("quantile population contains non-finite action values")
    quantiles = np.quantile(values, [lower, upper], axis=0)
    return quantiles[0], quantiles[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    _atomic_write_bytes(path, payload)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    ).encode()
    _atomic_write_bytes(path, payload)


def _invalidate_audit_success(output_root: Path) -> None:
    """Atomically revoke the full-audit gate before changing training artifacts."""
    (output_root / AUDIT_SUCCESS_MARKER_NAME).unlink(missing_ok=True)


def _write_audit_success_marker(output_root: Path) -> None:
    """Publish the exact audited artifact set as the final atomic audit step."""
    artifacts = {}
    for filename in (
        STATS_NAME,
        DATA_MANIFEST_NAME,
        LATENT_MANIFEST_NAME,
        PREPROCESS_MANIFEST_NAME,
        TEXT_CACHE_NAME,
        EMPTY_EMBEDDING_NAME,
    ):
        path = output_root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"cannot publish EBench audit success without {path}"
            )
        artifacts[filename] = sha256_file(path)
    write_json(
        output_root / AUDIT_SUCCESS_MARKER_NAME,
        {
            "marker_version": AUDIT_SUCCESS_MARKER_VERSION,
            "dataset_root": str(output_root.resolve()),
            "artifacts": artifacts,
        },
    )


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _safe_directory_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""
    return first == second or first in second.parents or second in first.parents


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return int(episode_index) // int(chunks_size)


def _format_lerobot_path(info: Mapping[str, Any], key: str, episode_index: int) -> Path:
    episode_chunk = _episode_chunk(episode_index, int(info["chunks_size"]))
    return Path(str(info[key]).format(
        episode_chunk=episode_chunk,
        episode_index=int(episode_index),
    ))


def _parse_train_episode_indices(
    info: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]
) -> list[int]:
    split = str(info.get("splits", {}).get("train", ""))
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", split)
    if not match:
        raise ValueError(f"unsupported train split {split!r}; expected 'start:end'")
    start, end = map(int, match.groups())
    available = {int(episode["episode_index"]) for episode in episodes}
    selected = list(range(start, end))
    missing = [index for index in selected if index not in available]
    if missing:
        raise ValueError(f"train split references missing episodes: {missing[:10]}")
    return selected


def _read_action_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("setup/audit requires pyarrow (installed by lerobot)") from error
    table = pq.read_table(
        path,
        columns=["action.joints", "action.gripper", "action.base_delta"],
    )
    columns = tuple(
        np.asarray(table[name].to_pylist(), dtype=np.float32)
        for name in ("action.joints", "action.gripper", "action.base_delta")
    )
    return columns  # type: ignore[return-value]


def _write_action_sidecar(
    path: Path,
    training_action: np.ndarray,
    anchor: np.ndarray,
    raw_length: int,
    source_sha256: str,
) -> None:
    real_count = training_action.shape[0] - ZERO_PREFIX_ROWS
    source_frame_ids = np.concatenate(
        [
            np.full(ZERO_PREFIX_ROWS, -1, dtype=np.int64),
            np.arange(real_count, dtype=np.int64),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                action_19d=np.ascontiguousarray(training_action, dtype=np.float32),
                loss_mask_19d=np.ones_like(training_action, dtype=bool),
                source_frame_ids=source_frame_ids,
                episode_anchor_12d=np.asarray(anchor, dtype=np.float32),
                source_length=np.asarray(raw_length, dtype=np.int64),
                real_action_count=np.asarray(real_count, dtype=np.int64),
                zero_prefix_count=np.asarray(ZERO_PREFIX_ROWS, dtype=np.int64),
                tail_discarded_count=np.asarray(
                    raw_length - real_count, dtype=np.int64
                ),
                source_parquet_sha256=np.asarray(source_sha256),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_valid_action_sidecar(
    path: Path, raw_length: int, source_sha256: str
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as sidecar:
            action = np.asarray(sidecar["action_19d"], dtype=np.float32)
            mask = np.asarray(sidecar["loss_mask_19d"], dtype=bool)
            real_count = aligned_real_action_count(raw_length)
            expected_rows = ZERO_PREFIX_ROWS + real_count
            expected_source_ids = np.concatenate(
                [
                    np.full(ZERO_PREFIX_ROWS, -1, dtype=np.int64),
                    np.arange(real_count, dtype=np.int64),
                ]
            )
            if action.shape != (expected_rows, ACTION_DIM_PHYSICAL):
                return None
            if mask.shape != action.shape or not mask.all():
                return None
            if not np.isfinite(action).all():
                return None
            if np.asarray(sidecar["episode_anchor_12d"]).shape != (12,):
                return None
            if not np.isfinite(sidecar["episode_anchor_12d"]).all():
                return None
            if int(sidecar["source_length"]) != raw_length:
                return None
            if int(sidecar["real_action_count"]) != real_count:
                return None
            if int(sidecar["zero_prefix_count"]) != ZERO_PREFIX_ROWS:
                return None
            if int(sidecar["tail_discarded_count"]) != raw_length - real_count:
                return None
            if str(sidecar["source_parquet_sha256"]) != source_sha256:
                return None
            if not np.array_equal(sidecar["source_frame_ids"], expected_source_ids):
                return None
            if not np.all(action[:ZERO_PREFIX_ROWS] == 0):
                return None
            if not np.all(action[:, 18] == 0):
                return None
            return np.ascontiguousarray(action)
    except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None


def _trusted_previous_data_entries(
    output_root: Path,
    source_root: Path,
    previous_stats: Mapping[str, Any] | None,
) -> dict[int, dict[str, Any]]:
    """Load resume hashes only through the previous stats -> data SHA chain."""
    if previous_stats is None:
        return {}
    try:
        expected_sha = previous_stats["manifests"]["data_manifest_sha256"]
        manifest_path = output_root / DATA_MANIFEST_NAME
        if not isinstance(expected_sha, str) or not manifest_path.is_file():
            return {}
        if sha256_file(manifest_path) != expected_sha:
            return {}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("manifest_version") != "ebench_lingbot_data_v1":
            return {}
        if Path(str(manifest["source_root"])).resolve() != source_root:
            return {}
        if Path(str(manifest["output_root"])).resolve() != output_root:
            return {}
        entries = {
            int(entry["episode_index"]): dict(entry)
            for entry in manifest["episodes"]
        }
        if len(entries) != len(manifest["episodes"]):
            return {}
        return entries
    except (KeyError, OSError, TypeError, ValueError):
        return {}


def _effective_and_model_quantiles(
    observed_q01: np.ndarray, observed_q99: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    effective_q01 = np.asarray(observed_q01, dtype=np.float64).copy()
    effective_q99 = np.asarray(observed_q99, dtype=np.float64).copy()
    # Yaw is physically fixed zero.  A symmetric neutral range maps raw zero
    # to normalized zero instead of -1 under LingBot's quantile formula.
    effective_q01[18] = -1.0
    effective_q99[18] = 1.0
    model_q01 = np.full(ACTION_DIM_MODEL, -1.0, dtype=np.float64)
    model_q99 = np.full(ACTION_DIM_MODEL, 1.0, dtype=np.float64)
    model_q01[MODEL_CHANNELS] = effective_q01
    model_q99[MODEL_CHANNELS] = effective_q99
    return effective_q01, effective_q99, model_q01, model_q99


def _stats_artifact(
    observed_q01: np.ndarray,
    observed_q99: np.ndarray,
    frame_count: int,
    data_manifest_sha256: str,
    latent_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    effective_q01, effective_q99, model_q01, model_q99 = (
        _effective_and_model_quantiles(observed_q01, observed_q99)
    )
    unused = np.flatnonzero(~model_action_mask_30d()).tolist()
    return {
        "artifact_version": "ebench_lingbot_action_stats_v1",
        "representation": {
            "physical_action_dim": ACTION_DIM_PHYSICAL,
            "model_action_dim": ACTION_DIM_MODEL,
            "physical_channels": [
                "left_joint_0", "left_joint_1", "left_joint_2",
                "left_joint_3", "left_joint_4", "left_joint_5",
                "left_gripper_0", "left_gripper_1",
                "right_joint_0", "right_joint_1", "right_joint_2",
                "right_joint_3", "right_joint_4", "right_joint_5",
                "right_gripper_0", "right_gripper_1",
                "base_x_cumulative", "base_y_cumulative", "base_yaw_fixed_zero",
            ],
            "joint_anchor": "episode action.joints[0]",
            "gripper_representation": "absolute action.gripper",
            "base_representation": "inclusive cumsum(action.base_delta); yaw=0",
            "zero_prefix_rows": ZERO_PREFIX_ROWS,
            "tail_alignment": "16 * floor((length - 1) / 16)",
        },
        "channel_mapping": {
            "physical_19d_to_model_30d": MODEL_CHANNELS.tolist(),
            "used_model_channels": np.flatnonzero(model_action_mask_30d()).tolist(),
            "unused_model_channels": unused,
        },
        "quantiles": {
            "lower_probability": 0.01,
            "upper_probability": 0.99,
            "population": "train retained real actions; excludes zero16 and tail",
            "frame_count": int(frame_count),
            "observed_q01_19d": np.asarray(observed_q01).tolist(),
            "observed_q99_19d": np.asarray(observed_q99).tolist(),
            "effective_q01_19d": effective_q01.tolist(),
            "effective_q99_19d": effective_q99.tolist(),
        },
        "normalization": {
            "formula": "(raw - q01) / (q99 - q01 + 1e-6) * 2 - 1",
            "clip": [-1.5, 1.5],
            "fixed_zero_neutral_range": [-1.0, 1.0],
        },
        "norm_stat": {"q01": model_q01.tolist(), "q99": model_q99.tolist()},
        "manifests": {
            "data_manifest_sha256": data_manifest_sha256,
            "latent_manifest_sha256": latent_manifest_sha256,
        },
    }


def _validate_preprocess_inputs(
    manifest: Mapping[str, Any],
    output_root: Path,
    model_root: Path,
) -> None:
    """Reject stale or mismatched setup artifacts before GPU encoding."""
    try:
        prepared_output = Path(str(manifest["output_root"])).resolve()
        prepared_source = Path(str(manifest["source_root"])).resolve()
        prepared_model = Path(str(manifest["model_root"])).resolve()
        selected_episodes = manifest["selected_episodes"]
        tasks = manifest["tasks"]
        artifacts = manifest["artifacts"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("preprocess manifest is malformed") from error

    if prepared_output != output_root:
        raise ValueError(
            f"preprocess manifest belongs to output {prepared_output}, not {output_root}"
        )
    if _paths_overlap(prepared_source, output_root):
        raise ValueError("preprocess manifest has overlapping source/output roots")
    if prepared_model != model_root:
        raise ValueError(
            f"model-root differs from setup: requested={model_root}, "
            f"prepared={prepared_model}"
        )
    if int(manifest.get("selected_episode_count", -1)) != len(selected_episodes):
        raise ValueError("preprocess manifest selected episode count is inconsistent")
    if int(manifest.get("task_count", -1)) != len(tasks):
        raise ValueError("preprocess manifest task count is inconsistent")
    episode_indices = [int(record["episode_index"]) for record in selected_episodes]
    if len(episode_indices) != len(set(episode_indices)):
        raise ValueError("preprocess manifest contains duplicate episode indices")
    if len(tasks) != len(set(tasks)):
        raise ValueError("preprocess manifest contains duplicate tasks")

    for path_key, sha_key, expected_name in (
        ("data_manifest", "data_manifest_sha256", DATA_MANIFEST_NAME),
        ("action_stats", "action_stats_sha256", STATS_NAME),
    ):
        if artifacts.get(path_key) != expected_name:
            raise ValueError(
                f"preprocess manifest has unexpected {path_key}: "
                f"{artifacts.get(path_key)!r}"
            )
        artifact_path = output_root / expected_name
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        actual_sha = sha256_file(artifact_path)
        if artifacts.get(sha_key) != actual_sha:
            raise ValueError(
                f"preprocess manifest {sha_key} mismatch: "
                f"recorded={artifacts.get(sha_key)!r}, actual={actual_sha}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "encode-text", "encode", "audit"):
        command = subparsers.add_parser(name)
        command.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        if name in ("setup", "encode-text", "encode"):
            command.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
        if name == "setup":
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--skip-text", action="store_true")
            command.add_argument("--max-episodes", type=int)
            command.add_argument("--overwrite-actions", action="store_true")
            command.add_argument("--overwrite-text", action="store_true")
        elif name == "encode-text":
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--overwrite-text", action="store_true")
        elif name == "encode":
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--num-shards", type=int, default=1)
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--episode-indices", type=int, nargs="*")
            command.add_argument("--temporal-chunk-size", type=int, default=64)
            command.add_argument("--overwrite", action="store_true")
        elif name == "audit":
            command.add_argument("--report-path", type=Path)
            command.add_argument("--episode-indices", type=int, nargs="*")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    handlers = {
        "setup": setup_dataset,
        "encode-text": encode_text_cache,
        "encode": encode_dataset,
        "audit": audit_dataset,
    }
    handlers[args.command](args)


def setup_dataset(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if _paths_overlap(source_root, output_root):
        raise ValueError(
            "source-root and output-root must be separate, non-nested directories: "
            f"source={source_root}, output={output_root}"
        )
    source_meta = source_root / "meta"
    output_meta = output_root / "meta"
    for required_source in (source_meta, source_root / "data", source_root / "videos"):
        if not required_source.is_dir():
            raise FileNotFoundError(f"required source directory is missing: {required_source}")
    # Setup always rewrites derived metadata/manifests, so a prior full-audit
    # success must stop authorizing training before the first derived write.
    _invalidate_audit_success(output_root)
    previous_stats: dict[str, Any] | None = None
    previous_stats_path = output_root / STATS_NAME
    if previous_stats_path.is_file():
        try:
            loaded_stats = json.loads(
                previous_stats_path.read_text(encoding="utf-8")
            )
            if isinstance(loaded_stats, dict):
                previous_stats = loaded_stats
        except (OSError, ValueError):
            previous_stats = None
    previous_data_entries = _trusted_previous_data_entries(
        output_root, source_root, previous_stats
    )
    info = json.loads((source_meta / "info.json").read_text(encoding="utf-8"))
    episodes = read_jsonl(source_meta / "episodes.jsonl")
    tasks = read_jsonl(source_meta / "tasks.jsonl")
    episode_by_index = {int(item["episode_index"]): item for item in episodes}
    selected_indices = _parse_train_episode_indices(info, episodes)
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            raise ValueError("--max-episodes must be positive")
        selected_indices = selected_indices[: args.max_episodes]
    selected_set = set(selected_indices)

    task_texts = [str(item["task"]) for item in sorted(tasks, key=lambda x: int(x["task_index"]))]
    if len(task_texts) != len(set(task_texts)):
        raise ValueError("tasks.jsonl contains duplicate task text")

    partial_setup = args.max_episodes is not None
    output_episodes = (
        [episode_by_index[index] for index in selected_indices]
        if partial_setup
        else episodes
    )
    derived_info = copy.deepcopy(info)
    features = derived_info.setdefault("features", {})
    top_key = SOURCE_CAMERA_KEYS[0]
    if top_key not in features:
        reference_key = "video.overlook_camera_view"
        if reference_key not in features:
            raise KeyError(f"cannot derive missing {top_key}: {reference_key} absent")
        features[top_key] = copy.deepcopy(features[reference_key])
    derived_info["total_videos"] = len(output_episodes) * len(SOURCE_CAMERA_KEYS)
    if partial_setup:
        derived_info["total_episodes"] = len(output_episodes)
        derived_info["total_frames"] = sum(int(item["length"]) for item in output_episodes)
        derived_info["total_chunks"] = 1 + (
            max(int(item["episode_index"]) for item in output_episodes)
            // int(info["chunks_size"])
        )
        derived_info["splits"] = {"train": f"0:{len(output_episodes)}"}

    updated_episodes: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    for original in output_episodes:
        episode = dict(original)
        index = int(episode["episode_index"])
        length = int(episode["length"])
        task = str(episode["tasks"][0])
        if task not in set(task_texts):
            raise ValueError(f"episode {index} task missing from tasks.jsonl: {task!r}")
        if index in selected_set:
            episode["action_config"] = [{
                "start_frame": 0,
                "end_frame": length,
                "action_text": task,
                "skill": "",
            }]
            selected_records.append({
                "episode_index": index,
                "length": length,
                "task": task,
                "chunk": _episode_chunk(index, int(info["chunks_size"])),
                "retained_real_actions": aligned_real_action_count(length),
                "sampled_rgb_frames": usable_sampled_frame_count(length),
                "latent_frames": expected_latent_frame_count(
                    usable_sampled_frame_count(length)
                ),
            })
        else:
            raise RuntimeError("output episode unexpectedly falls outside selection")
        updated_episodes.append(episode)

    output_meta.mkdir(parents=True, exist_ok=True)
    for source_file in source_meta.iterdir():
        if not source_file.is_file() or source_file.name in {"info.json", "episodes.jsonl"}:
            continue
        if partial_setup and source_file.name == "episodes_stats.jsonl":
            stats_records = read_jsonl(source_file)
            write_jsonl(
                output_meta / source_file.name,
                [
                    item
                    for item in stats_records
                    if int(item["episode_index"]) in selected_set
                ],
            )
        else:
            shutil.copy2(source_file, output_meta / source_file.name)
    write_json(output_meta / "info.json", derived_info)
    write_jsonl(output_meta / "episodes.jsonl", updated_episodes)
    _safe_directory_symlink(source_root / "data", output_root / "data")
    _safe_directory_symlink(source_root / "videos", output_root / "videos")

    expected_real_rows = sum(item["retained_real_actions"] for item in selected_records)
    workspace = output_root / ".action_quantile_workspace.f32"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    quantile_values = np.memmap(
        workspace,
        mode="w+",
        dtype=np.float32,
        shape=(expected_real_rows, ACTION_DIM_PHYSICAL),
    )
    cursor = 0
    data_entries: list[dict[str, Any]] = []
    for ordinal, record in enumerate(selected_records, start=1):
        index = int(record["episode_index"])
        length = int(record["length"])
        source_path = source_root / _format_lerobot_path(info, "data_path", index)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_sha = sha256_file(source_path)
        sidecar_path = (
            output_root
            / "actions"
            / f"chunk-{int(record['chunk']):03d}"
            / f"episode_{index:06d}.npz"
        )
        training_action = None
        sidecar_sha = None
        if not args.overwrite_actions:
            previous_entry = previous_data_entries.get(index)
            expected_sidecar_relative = str(sidecar_path.relative_to(output_root))
            if (
                previous_entry is not None
                and previous_entry.get("source_parquet_sha256") == source_sha
                and previous_entry.get("source_parquet")
                == str(source_path.relative_to(source_root))
                and previous_entry.get("action_sidecar")
                == expected_sidecar_relative
                and sidecar_path.is_file()
                and int(previous_entry.get("action_sidecar_size", -1))
                == sidecar_path.stat().st_size
            ):
                actual_sidecar_sha = sha256_file(sidecar_path)
                if previous_entry.get("action_sidecar_sha256") == actual_sidecar_sha:
                    training_action = _load_valid_action_sidecar(
                        sidecar_path, length, source_sha
                    )
                    if training_action is not None:
                        sidecar_sha = actual_sidecar_sha
        if training_action is None:
            joints, gripper, base_delta = _read_action_columns(source_path)
            if joints.shape[0] != length:
                raise ValueError(
                    f"episode {index}: parquet rows {joints.shape[0]} != length {length}"
                )
            transformed = transform_ebench_actions(joints, gripper, base_delta)
            if not np.isfinite(transformed).all():
                raise ValueError(f"episode {index}: transformed actions are non-finite")
            stats_action, training_action = prepare_episode_actions(
                transformed, raw_length=length
            )
            _write_action_sidecar(
                sidecar_path,
                training_action,
                joints[0],
                length,
                source_sha,
            )
            sidecar_sha = sha256_file(sidecar_path)
        else:
            stats_action = training_action[ZERO_PREFIX_ROWS:]

        if sidecar_sha is None:
            raise RuntimeError(f"episode {index}: action sidecar SHA was not resolved")

        real_count = int(record["retained_real_actions"])
        if stats_action.shape != (real_count, ACTION_DIM_PHYSICAL):
            raise RuntimeError(f"episode {index}: invalid retained action shape")
        quantile_values[cursor : cursor + real_count] = stats_action
        cursor += real_count
        data_entries.append({
            "episode_index": index,
            "length": length,
            "source_parquet": str(source_path.relative_to(source_root)),
            "source_parquet_size": source_path.stat().st_size,
            "source_parquet_sha256": source_sha,
            "action_sidecar": str(sidecar_path.relative_to(output_root)),
            "action_sidecar_size": sidecar_path.stat().st_size,
            "action_sidecar_sha256": sidecar_sha,
            "retained_real_actions": real_count,
        })
        if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(selected_records):
            print(f"setup actions [{ordinal}/{len(selected_records)}] episode {index}")

    if cursor != expected_real_rows:
        raise RuntimeError(f"stats rows {cursor} != expected {expected_real_rows}")
    quantile_values.flush()
    observed_q01, observed_q99 = compute_action_quantiles(quantile_values)
    del quantile_values
    workspace.unlink()

    metadata_hashes = {
        name: sha256_file(output_meta / name)
        for name in ("info.json", "episodes.jsonl", "tasks.jsonl")
    }
    source_metadata_hashes = {
        path.name: sha256_file(path)
        for path in sorted(source_meta.iterdir())
        if path.is_file()
    }
    data_manifest = {
        "manifest_version": "ebench_lingbot_data_v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_metadata_sha256": source_metadata_hashes,
        "derived_metadata_sha256": metadata_hashes,
        "episodes": data_entries,
    }
    write_json(output_root / DATA_MANIFEST_NAME, data_manifest)
    data_manifest_sha = sha256_file(output_root / DATA_MANIFEST_NAME)
    preserved_latent_manifest_sha = None
    if previous_stats is not None:
        previous_refs = previous_stats.get("manifests", {})
        candidate_sha = previous_refs.get("latent_manifest_sha256")
        latent_manifest_path = output_root / LATENT_MANIFEST_NAME
        if (
            previous_refs.get("data_manifest_sha256") == data_manifest_sha
            and candidate_sha
            and latent_manifest_path.is_file()
            and sha256_file(latent_manifest_path) == candidate_sha
        ):
            preserved_latent_manifest_sha = str(candidate_sha)
    stats = _stats_artifact(
        observed_q01,
        observed_q99,
        expected_real_rows,
        data_manifest_sha,
        preserved_latent_manifest_sha,
    )
    write_json(output_root / STATS_NAME, stats)

    existing_text_artifact = None
    existing_text_cache = output_root / TEXT_CACHE_NAME
    existing_empty_embedding = output_root / EMPTY_EMBEDDING_NAME
    if existing_text_cache.is_file() and existing_empty_embedding.is_file():
        existing_text_artifact = {
            "path": TEXT_CACHE_NAME,
            "sha256": sha256_file(existing_text_cache),
            "empty_path": EMPTY_EMBEDDING_NAME,
            "empty_sha256": sha256_file(existing_empty_embedding),
        }
    preprocess_manifest = {
        "manifest_version": "ebench_lingbot_preprocess_v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "model_root": str(args.model_root.resolve()),
        "camera_keys": list(SOURCE_CAMERA_KEYS),
        "camera_layout": [list(row) for row in CAMERA_LAYOUT],
        "image_stride": IMAGE_STRIDE,
        "vae_temporal_stride": VAE_TEMPORAL_STRIDE,
        "height": 224,
        "width": 224,
        "fps": int(info["fps"]),
        "selected_episode_count": len(selected_records),
        "selected_episodes": selected_records,
        "task_count": len(task_texts),
        "tasks": task_texts,
        "expected": {
            "real_action_rows": expected_real_rows,
            "discarded_tail_rows": sum(
                item["length"] - item["retained_real_actions"]
                for item in selected_records
            ),
            "sampled_rgb_frames": sum(item["sampled_rgb_frames"] for item in selected_records),
            "latent_frames_per_camera": sum(item["latent_frames"] for item in selected_records),
            "latent_files": len(selected_records) * len(SOURCE_CAMERA_KEYS),
        },
        "artifacts": {
            "data_manifest": DATA_MANIFEST_NAME,
            "data_manifest_sha256": data_manifest_sha,
            "action_stats": STATS_NAME,
            "action_stats_sha256": sha256_file(output_root / STATS_NAME),
            "text_cache": existing_text_artifact,
            "latent_manifest": (
                {
                    "path": LATENT_MANIFEST_NAME,
                    "sha256": preserved_latent_manifest_sha,
                }
                if preserved_latent_manifest_sha
                else None
            ),
        },
    }
    write_json(output_root / PREPROCESS_MANIFEST_NAME, preprocess_manifest)
    if not args.skip_text:
        encode_text_cache(args)
    print(
        f"setup complete: episodes={len(selected_records)}, "
        f"retained={expected_real_rows}, output={output_root}"
    )


def encode_text_cache(args: argparse.Namespace) -> None:
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    from wan_va.modules.utils import load_text_encoder, load_tokenizer

    output_root = args.output_root.resolve()
    manifest_path = output_root / PREPROCESS_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run setup first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_root = args.model_root.resolve()
    _validate_preprocess_inputs(manifest, output_root, model_root)
    cache_path = output_root / TEXT_CACHE_NAME
    empty_path = output_root / EMPTY_EMBEDDING_NAME
    overwrite = bool(getattr(args, "overwrite_text", False))
    if cache_path.is_file() and empty_path.is_file() and not overwrite:
        text_artifact = {
            "path": TEXT_CACHE_NAME,
            "sha256": sha256_file(cache_path),
            "empty_path": EMPTY_EMBEDDING_NAME,
            "empty_sha256": sha256_file(empty_path),
        }
        if manifest["artifacts"].get("text_cache") != text_artifact:
            _invalidate_audit_success(output_root)
            raise ValueError(
                "existing text cache bytes differ from the recorded artifact; "
                "rerun with --overwrite-text and then complete a full audit"
            )
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        empty = torch.load(empty_path, map_location="cpu", weights_only=False)
        expected_texts = {"", *manifest["tasks"]}
        if set(cached) != expected_texts or not torch.equal(cached[""], empty):
            _invalidate_audit_success(output_root)
            raise ValueError(f"existing text cache is incompatible: {cache_path}")
        print(f"text cache exists and is valid, skip: {cache_path}")
        return

    # Rebuilding either shared text artifact changes the audited training
    # inputs. Revoke the gate before loading the encoder or writing output.
    _invalidate_audit_success(output_root)

    device = torch.device(args.device)
    tokenizer = load_tokenizer(str(model_root / "tokenizer"))
    text_encoder = load_text_encoder(
        str(model_root / "text_encoder"),
        torch_dtype=torch.bfloat16,
        torch_device=device,
    ).eval()
    if (
        hasattr(text_encoder, "shared")
        and hasattr(text_encoder, "encoder")
        and text_encoder.encoder.embed_tokens is not text_encoder.shared
    ):
        text_encoder.encoder.embed_tokens = text_encoder.shared

    embeddings: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for ordinal, text in enumerate([""] + list(manifest["tasks"]), start=1):
            tokenized = tokenizer(
                [prompt_clean(text)],
                padding="max_length",
                max_length=512,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            valid = int(attention_mask.gt(0).sum().item())
            hidden = text_encoder(input_ids, attention_mask).last_hidden_state
            embedding = hidden[0, :valid].to(torch.bfloat16)
            embedding = torch.cat(
                [embedding, embedding.new_zeros(512 - valid, embedding.shape[-1])],
                dim=0,
            )
            embeddings[text] = embedding.cpu().contiguous()
            if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(manifest["tasks"]) + 1:
                print(f"encode text [{ordinal}/{len(manifest['tasks']) + 1}]")

    def atomic_torch_save(value: Any, path: Path) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(value, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    atomic_torch_save(embeddings, cache_path)
    atomic_torch_save(embeddings[""], empty_path)
    del text_encoder, tokenizer, embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["text_cache"] = {
        "path": TEXT_CACHE_NAME,
        "sha256": sha256_file(cache_path),
        "empty_path": EMPTY_EMBEDDING_NAME,
        "empty_sha256": sha256_file(empty_path),
    }
    write_json(manifest_path, manifest)
    print(f"text cache complete: {cache_path}")


def decode_sampled_rgb_frames(
    path: Path,
    frame_ids: Sequence[int] | np.ndarray,
    height: int = 224,
    width: int = 224,
    expected_source_frame_count: int | None = None,
) -> np.ndarray:
    """Decode only requested raw IDs and directly resize each camera to 224²."""
    try:
        import av
    except ImportError as error:
        raise RuntimeError("video encoding requires PyAV") from error
    requested = np.asarray(frame_ids, dtype=np.int64)
    if requested.ndim != 1 or not len(requested):
        raise ValueError("frame_ids must be a non-empty 1D sequence")
    if not np.all(np.diff(requested) > 0):
        raise ValueError("frame_ids must be strictly increasing")
    decoded: list[np.ndarray] = []
    cursor = 0
    source_frame_count = 0
    with av.open(str(path)) as container:
        for raw_index, frame in enumerate(container.decode(video=0)):
            source_frame_count = raw_index + 1
            if cursor == len(requested) or raw_index < int(requested[cursor]):
                continue
            if raw_index != int(requested[cursor]):
                raise RuntimeError(
                    f"{path}: decoder skipped requested frame {requested[cursor]}"
                )
            resized = frame.reformat(width=width, height=height, format="rgb24")
            decoded.append(resized.to_ndarray())
            cursor += 1
    if cursor != len(requested):
        raise RuntimeError(
            f"{path}: decoded {cursor}/{len(requested)} requested frames"
        )
    if (
        expected_source_frame_count is not None
        and source_frame_count != int(expected_source_frame_count)
    ):
        raise RuntimeError(
            f"{path}: source frame count {source_frame_count}, "
            f"expected {expected_source_frame_count}"
        )
    return np.stack(decoded, axis=0)


def _resolve_camera_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for short, full in zip(CAMERA_SHORT_NAMES, SOURCE_CAMERA_KEYS):
        if short in values:
            resolved[short] = values[short]
        elif full in values:
            resolved[short] = values[full]
        else:
            raise KeyError(f"missing camera {short!r} ({full!r})")
    return resolved


def encode_episode_camera_batch(
    vae: Any,
    streaming_vae: Any,
    sampled_frames_by_camera: Mapping[str, Any],
    device: Any,
    temporal_chunk_size: int = 64,
) -> dict[str, Any]:
    """Encode four independent camera tensors as one batch for one episode.

    Input values are ``[sampled_rgb,H,W,3]`` uint8-like arrays/tensors.  No RGB
    mosaic is formed.  The streaming cache is cleared exactly once here.
    """
    import torch

    if temporal_chunk_size < VAE_TEMPORAL_STRIDE or (
        temporal_chunk_size % VAE_TEMPORAL_STRIDE
    ):
        raise ValueError("temporal_chunk_size must be a positive multiple of 4")
    resolved = _resolve_camera_mapping(sampled_frames_by_camera)
    shapes = {key: tuple(value.shape) for key, value in resolved.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"sampled camera shapes differ: {shapes}")
    shape = next(iter(shapes.values()))
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError(f"sampled frames must be [T,H,W,3], got {shape}")
    sampled_count = shape[0]
    expected_frames = expected_latent_frame_count(sampled_count)
    torch_device = torch.device(device)

    def input_chunk(start: int, end: int) -> Any:
        tensors = [torch.as_tensor(resolved[key][start:end]) for key in CAMERA_SHORT_NAMES]
        video = torch.stack(tensors, dim=0).permute(0, 4, 1, 2, 3)
        video = video.to(device=torch_device, dtype=torch.float32)
        return (video / 255.0 * 2.0 - 1.0).to(torch.bfloat16)

    means = torch.as_tensor(
        vae.config.latents_mean, device=torch_device, dtype=torch.float32
    ).view(1, -1, 1, 1, 1)
    inverse_std = torch.as_tensor(
        vae.config.latents_std, device=torch_device, dtype=torch.float32
    ).view(1, -1, 1, 1, 1).reciprocal()
    if means.shape[1] != 48 or inverse_std.shape[1] != 48:
        raise ValueError("EBench VAE must declare 48 latent mean/std channels")
    if not torch.isfinite(means).all() or not torch.isfinite(inverse_std).all():
        raise ValueError("EBench VAE latent normalization is non-finite")
    encoded_chunks: list[Any] = []
    streaming_vae.clear_cache()
    with torch.inference_mode():
        boundaries = [(0, 1)] + [
            (start, min(start + temporal_chunk_size, sampled_count))
            for start in range(1, sampled_count, temporal_chunk_size)
        ]
        for start, end in boundaries:
            encoded = streaming_vae.encode_chunk(input_chunk(start, end))
            posterior_mean, _ = torch.chunk(encoded, 2, dim=1)
            normalized = ((posterior_mean.float() - means) * inverse_std).to(
                posterior_mean.dtype
            )
            encoded_chunks.append(normalized.cpu())
    latent_batch = torch.cat(encoded_chunks, dim=2).contiguous()
    if latent_batch.shape[0] != len(CAMERA_SHORT_NAMES):
        raise RuntimeError(f"unexpected VAE batch shape {tuple(latent_batch.shape)}")
    expected_shape = (len(CAMERA_SHORT_NAMES), 48, expected_frames, 14, 14)
    if tuple(latent_batch.shape) != expected_shape:
        raise RuntimeError(
            f"VAE produced shape {tuple(latent_batch.shape)}, expected {expected_shape}"
        )
    if latent_batch.dtype != torch.bfloat16:
        raise RuntimeError(f"VAE produced dtype {latent_batch.dtype}, expected bfloat16")
    if not torch.isfinite(latent_batch.float()).all():
        raise RuntimeError("VAE produced non-finite latent values")
    return {
        short: latent_batch[index]
        for index, short in enumerate(CAMERA_SHORT_NAMES)
    }


def _latent_record(
    latent: Any,
    camera_key: str,
    text: str,
    raw_length: int,
    frame_ids: np.ndarray,
    fps: int,
) -> dict[str, Any]:
    import torch

    if latent.ndim != 4:
        raise ValueError(f"latent must be [C,F,H,W], got {tuple(latent.shape)}")
    channels, frames, height, width = map(int, latent.shape)
    flattened = latent.permute(1, 2, 3, 0).reshape(-1, channels).contiguous()
    return {
        "latent": flattened,
        "latent_num_frames": frames,
        "latent_height": height,
        "latent_width": width,
        "video_num_frames": int(len(frame_ids)),
        "source_video_num_frames": int(raw_length),
        "video_height": 224,
        "video_width": 224,
        "text": text,
        "camera_key": camera_key,
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "start_frame": 0,
        "end_frame": int(raw_length),
        "fps": int(fps),
        "ori_fps": int(fps),
        "image_stride": IMAGE_STRIDE,
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _latent_path(output_root: Path, record: Mapping[str, Any], camera_key: str) -> Path:
    return (
        output_root
        / "latents"
        / f"chunk-{int(record['chunk']):03d}"
        / camera_key
        / (
            f"episode_{int(record['episode_index']):06d}_0_"
            f"{int(record['length'])}.pth"
        )
    )


def _video_path(
    source_root: Path,
    source_info: Mapping[str, Any],
    episode_index: int,
    camera_key: str,
) -> Path:
    chunk = _episode_chunk(episode_index, int(source_info["chunks_size"]))
    relative = str(source_info["video_path"]).format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=camera_key,
    )
    return source_root / relative


def _valid_latent_file(
    path: Path,
    record: Mapping[str, Any],
    camera_key: str,
    frame_ids: np.ndarray,
    fps: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        latent_frames = expected_latent_frame_count(len(frame_ids))
        expected_shape = (latent_frames * 14 * 14, 48)
        return (
            tuple(value["latent"].shape) == expected_shape
            and value["latent"].dtype == torch.bfloat16
            and int(value["latent_num_frames"]) == latent_frames
            and int(value["latent_height"]) == 14
            and int(value["latent_width"]) == 14
            and int(value["video_num_frames"]) == len(frame_ids)
            and int(value["source_video_num_frames"]) == int(record["length"])
            and int(value["video_height"]) == 224
            and int(value["video_width"]) == 224
            and str(value["text"]) == str(record["task"])
            and str(value["camera_key"]) == camera_key
            and np.array_equal(value["frame_ids"], frame_ids)
            and int(value["start_frame"]) == 0
            and int(value["end_frame"]) == int(record["length"])
            and int(value["fps"]) == int(fps)
            and int(value["ori_fps"]) == int(fps)
            and int(value["image_stride"]) == IMAGE_STRIDE
            and "text_emb" not in value
            and bool(torch.isfinite(value["latent"].float()).all())
        )
    except (
        AttributeError,
        EOFError,
        IndexError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
        RuntimeError,
    ):
        return False


def encode_dataset(args: argparse.Namespace) -> None:
    import torch

    from wan_va.modules.utils import WanVAEStreamingWrapper, load_vae

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.temporal_chunk_size < 4 or args.temporal_chunk_size % 4:
        raise ValueError("temporal-chunk-size must be a positive multiple of 4")
    output_root = args.output_root.resolve()
    manifest_path = output_root / PREPROCESS_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run setup first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_root = args.model_root.resolve()
    _validate_preprocess_inputs(manifest, output_root, model_root)
    source_root = Path(manifest["source_root"])
    source_info = json.loads(
        (source_root / "meta" / "info.json").read_text(encoding="utf-8")
    )
    requested = set(args.episode_indices) if args.episode_indices is not None else None
    selected = []
    known_indices = set()
    for position, record in enumerate(manifest["selected_episodes"]):
        index = int(record["episode_index"])
        known_indices.add(index)
        if requested is not None and index not in requested:
            continue
        if position % args.num_shards == args.shard_index:
            selected.append(record)
    if requested is not None:
        unknown = sorted(requested - known_indices)
        if unknown:
            raise ValueError(f"episode indices are not in prepared train split: {unknown}")

    device = torch.device(args.device)
    vae = load_vae(
        str(model_root / "vae"),
        torch_dtype=torch.bfloat16,
        torch_device=device,
    ).eval()
    streaming_vae = WanVAEStreamingWrapper(vae)
    shard_results: list[dict[str, Any]] = []
    audit_success_invalidated = False
    for ordinal, record in enumerate(selected, start=1):
        index = int(record["episode_index"])
        length = int(record["length"])
        frame_ids = sampled_rgb_frame_ids(length)
        output_paths = {
            key: _latent_path(output_root, record, key) for key in SOURCE_CAMERA_KEYS
        }
        valid = {
            key: _valid_latent_file(
                path, record, key, frame_ids, int(manifest["fps"])
            )
            for key, path in output_paths.items()
        }
        if not args.overwrite and all(valid.values()):
            print(f"encode [{ordinal}/{len(selected)}] skip episode {index}: complete")
            shard_results.append({"episode_index": index, "status": "skipped"})
            continue

        if not audit_success_invalidated:
            # All-valid resume runs retain their gate. Any actual latent rebuild
            # revokes it before decode/encode/write begins.
            _invalidate_audit_success(output_root)
            audit_success_invalidated = True

        sampled_by_camera: dict[str, np.ndarray] = {}
        for short, camera_key in zip(CAMERA_SHORT_NAMES, SOURCE_CAMERA_KEYS):
            path = _video_path(source_root, source_info, index, camera_key)
            if not path.is_file():
                raise FileNotFoundError(path)
            sampled_by_camera[short] = decode_sampled_rgb_frames(
                path,
                frame_ids,
                expected_source_frame_count=length,
            )
        latents = encode_episode_camera_batch(
            vae,
            streaming_vae,
            sampled_by_camera,
            device,
            args.temporal_chunk_size,
        )
        for short, camera_key in zip(CAMERA_SHORT_NAMES, SOURCE_CAMERA_KEYS):
            if not args.overwrite and valid[camera_key]:
                continue
            value = _latent_record(
                latents[short],
                camera_key,
                str(record["task"]),
                length,
                frame_ids,
                int(manifest["fps"]),
            )
            _atomic_torch_save(value, output_paths[camera_key])
        del sampled_by_camera, latents
        gc.collect()
        print(
            f"encode [{ordinal}/{len(selected)}] episode {index}: "
            f"sampled_rgb={len(frame_ids)}, latent_frames={record['latent_frames']}"
        )
        shard_results.append({"episode_index": index, "status": "encoded"})

    shard_manifest = {
        "manifest_version": "ebench_lingbot_encode_shard_v1",
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "episode_filter": sorted(requested) if requested is not None else None,
        "episodes": shard_results,
    }
    shard_path = (
        output_root
        / "encode_shards"
        / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json"
    )
    write_json(shard_path, shard_manifest)
    del streaming_vae, vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"encode shard complete: {shard_path}")


def audit_dataset(args: argparse.Namespace) -> None:
    import torch

    output_root = args.output_root.resolve()
    requested = set(args.episode_indices) if args.episode_indices is not None else None
    is_full_audit = requested is None
    if is_full_audit:
        # Revoke an older gate at entry, including when this audit later fails
        # before it can read or validate the preprocessing manifests.
        _invalidate_audit_success(output_root)
    manifest_path = output_root / PREPROCESS_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run setup first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = Path(manifest["source_root"])
    source_info = json.loads(
        (source_root / "meta" / "info.json").read_text(encoding="utf-8")
    )
    derived_info = json.loads(
        (output_root / "meta" / "info.json").read_text(encoding="utf-8")
    )
    selected = [
        record
        for record in manifest["selected_episodes"]
        if requested is None or int(record["episode_index"]) in requested
    ]
    known = {int(record["episode_index"]) for record in manifest["selected_episodes"]}
    if requested is not None and requested - known:
        raise ValueError(f"unknown episode indices: {sorted(requested - known)}")
    errors: list[str] = []

    if Path(str(manifest.get("output_root", ""))).resolve() != output_root:
        errors.append("preprocess manifest output_root mismatch")
    if Path(str(manifest.get("source_root", ""))).resolve() != source_root.resolve():
        errors.append("preprocess manifest source_root mismatch")
    if len(known) != len(manifest["selected_episodes"]):
        errors.append("preprocess manifest contains duplicate episode indices")
    if int(manifest.get("selected_episode_count", -1)) != len(known):
        errors.append("preprocess manifest selected_episode_count mismatch")

    if SOURCE_CAMERA_KEYS[0] not in derived_info.get("features", {}):
        errors.append("derived info.json still lacks top camera")
    if int(derived_info["total_episodes"]) != len(known):
        errors.append(
            f"total_episodes={derived_info['total_episodes']}, expected {len(known)}"
        )
    expected_total_videos = int(derived_info["total_episodes"]) * len(SOURCE_CAMERA_KEYS)
    if int(derived_info["total_videos"]) != expected_total_videos:
        errors.append(
            f"total_videos={derived_info['total_videos']}, expected {expected_total_videos}"
        )

    episode_metadata = {
        int(value["episode_index"]): value
        for value in read_jsonl(output_root / "meta" / "episodes.jsonl")
    }
    text_cache_path = output_root / TEXT_CACHE_NAME
    empty_path = output_root / EMPTY_EMBEDDING_NAME
    text_cache: dict[str, torch.Tensor] | None = None
    if not text_cache_path.is_file() or not empty_path.is_file():
        errors.append("shared text cache or empty embedding is missing")
    else:
        text_artifact = manifest.get("artifacts", {}).get("text_cache")
        if not isinstance(text_artifact, Mapping):
            errors.append("preprocess manifest text_cache artifact is missing")
        else:
            if text_artifact.get("path") != TEXT_CACHE_NAME:
                errors.append("preprocess manifest text cache path mismatch")
            if text_artifact.get("empty_path") != EMPTY_EMBEDDING_NAME:
                errors.append("preprocess manifest empty embedding path mismatch")
            if text_artifact.get("sha256") != sha256_file(text_cache_path):
                errors.append("preprocess manifest text cache SHA mismatch")
            if text_artifact.get("empty_sha256") != sha256_file(empty_path):
                errors.append("preprocess manifest empty embedding SHA mismatch")
        try:
            text_cache = torch.load(
                text_cache_path, map_location="cpu", weights_only=False
            )
            empty = torch.load(empty_path, map_location="cpu", weights_only=False)
            if not isinstance(text_cache, dict):
                raise TypeError("text cache is not a dictionary")
            if set(text_cache) != {"", *manifest["tasks"]}:
                errors.append("shared text cache keys differ from manifest tasks")
            elif not torch.equal(text_cache[""], empty):
                errors.append("empty_emb.pt differs from text_embeddings.pt['']")
            for text, embedding in text_cache.items():
                if not torch.is_tensor(embedding) or tuple(embedding.shape) != (
                    512,
                    4096,
                ):
                    errors.append(
                        f"text cache entry {text!r} does not have shape [512,4096]"
                    )
                    continue
                if embedding.dtype != torch.bfloat16:
                    errors.append(f"text cache entry {text!r} has dtype {embedding.dtype}")
                if not torch.isfinite(embedding.float()).all():
                    errors.append(
                        f"text cache entry {text!r} contains non-finite values"
                    )
        except (AttributeError, EOFError, OSError, TypeError, ValueError, RuntimeError) as error:
            errors.append(f"shared text cache audit failed: {error}")

    data_manifest_path = output_root / DATA_MANIFEST_NAME
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    data_entry_records = list(data_manifest["episodes"])
    data_entries = {
        int(value["episode_index"]): value for value in data_entry_records
    }
    stats_path = output_root / STATS_NAME
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    actual_data_manifest_sha = sha256_file(data_manifest_path)
    actual_stats_sha = sha256_file(stats_path)
    preprocess_artifacts = manifest.get("artifacts", {})
    if preprocess_artifacts.get("data_manifest") != DATA_MANIFEST_NAME:
        errors.append("preprocess manifest data_manifest path mismatch")
    if (
        preprocess_artifacts.get("data_manifest_sha256")
        != actual_data_manifest_sha
    ):
        errors.append("preprocess manifest data_manifest_sha256 mismatch")
    if preprocess_artifacts.get("action_stats") != STATS_NAME:
        errors.append("preprocess manifest action_stats path mismatch")
    if preprocess_artifacts.get("action_stats_sha256") != actual_stats_sha:
        errors.append("preprocess manifest action_stats_sha256 mismatch")
    if data_manifest.get("manifest_version") != "ebench_lingbot_data_v1":
        errors.append("data manifest version mismatch")
    if Path(str(data_manifest.get("source_root", ""))).resolve() != source_root.resolve():
        errors.append("data manifest source_root mismatch")
    if Path(str(data_manifest.get("output_root", ""))).resolve() != output_root:
        errors.append("data manifest output_root mismatch")
    if len(data_entries) != len(data_entry_records):
        errors.append("data manifest contains duplicate episode indices")
    if is_full_audit and set(data_entries) != known:
        errors.append("data manifest episode set differs from preprocess manifest")
    for name, expected_sha in data_manifest.get(
        "source_metadata_sha256", {}
    ).items():
        metadata_path = source_root / "meta" / name
        if not metadata_path.is_file() or sha256_file(metadata_path) != expected_sha:
            errors.append(f"source metadata SHA mismatch: {name}")
    for name, expected_sha in data_manifest.get(
        "derived_metadata_sha256", {}
    ).items():
        metadata_path = output_root / "meta" / name
        if not metadata_path.is_file() or sha256_file(metadata_path) != expected_sha:
            errors.append(f"derived metadata SHA mismatch: {name}")
    if stats["manifests"]["data_manifest_sha256"] != actual_data_manifest_sha:
        errors.append("action stats data_manifest_sha256 mismatch")
    try:
        recorded_q01 = np.asarray(stats["quantiles"]["observed_q01_19d"])
        recorded_q99 = np.asarray(stats["quantiles"]["observed_q99_19d"])
        recorded_frame_count = int(stats["quantiles"]["frame_count"])
        expected_stats = _stats_artifact(
            recorded_q01,
            recorded_q99,
            recorded_frame_count,
            actual_data_manifest_sha,
            stats["manifests"].get("latent_manifest_sha256"),
        )
        if stats != expected_stats:
            errors.append("action stats representation or normalization is inconsistent")
        if recorded_frame_count != int(manifest["expected"]["real_action_rows"]):
            errors.append("action stats frame_count differs from preprocess manifest")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"action stats artifact is malformed: {error}")

    stats_rows = sum(int(record["retained_real_actions"]) for record in selected)
    workspace = output_root / ".audit_action_workspace.f32"
    quantile_values = np.memmap(
        workspace,
        mode="w+",
        dtype=np.float32,
        shape=(stats_rows, ACTION_DIM_PHYSICAL),
    )
    cursor = 0
    latent_entries: list[dict[str, Any]] = []
    latent_value_count = 0
    latent_sum = 0.0
    latent_square_sum = 0.0
    latent_min = float("inf")
    latent_max = float("-inf")

    for ordinal, record in enumerate(selected, start=1):
        index = int(record["episode_index"])
        length = int(record["length"])
        real_count = aligned_real_action_count(length)
        latent_frames = 1 + real_count // ACTION_PER_LATENT_FRAME
        metadata = episode_metadata.get(index)
        expected_action_config = [{
            "start_frame": 0,
            "end_frame": length,
            "action_text": str(record["task"]),
            "skill": "",
        }]
        if metadata is None or metadata.get("action_config") != expected_action_config:
            errors.append(f"episode {index}: action_config is not one full episode")

        sidecar_path = (
            output_root
            / "actions"
            / f"chunk-{int(record['chunk']):03d}"
            / f"episode_{index:06d}.npz"
        )
        source_path = source_root / _format_lerobot_path(source_info, "data_path", index)
        try:
            source_sha = sha256_file(source_path)
            entry = data_entries[index]
            if entry.get("source_parquet") != str(source_path.relative_to(source_root)):
                errors.append(f"episode {index}: source parquet path mismatch")
            if int(entry.get("source_parquet_size", -1)) != source_path.stat().st_size:
                errors.append(f"episode {index}: source parquet size mismatch")
            if source_sha != entry["source_parquet_sha256"]:
                errors.append(f"episode {index}: source parquet SHA changed")
            if entry.get("action_sidecar") != str(sidecar_path.relative_to(output_root)):
                errors.append(f"episode {index}: action sidecar path mismatch")
            if int(entry.get("action_sidecar_size", -1)) != sidecar_path.stat().st_size:
                errors.append(f"episode {index}: action sidecar size mismatch")
            if sha256_file(sidecar_path) != entry["action_sidecar_sha256"]:
                errors.append(f"episode {index}: action sidecar SHA mismatch")
            with np.load(sidecar_path, allow_pickle=False) as sidecar:
                training_action = np.asarray(sidecar["action_19d"], np.float32)
                training_mask = np.asarray(sidecar["loss_mask_19d"], bool)
                source_frame_ids = np.asarray(sidecar["source_frame_ids"], np.int64)
                expected_rows = latent_frames * ACTION_PER_LATENT_FRAME
                if training_action.shape != (expected_rows, ACTION_DIM_PHYSICAL):
                    errors.append(
                        f"episode {index}: sidecar shape {training_action.shape}, "
                        f"expected {(expected_rows, ACTION_DIM_PHYSICAL)}"
                    )
                if training_mask.shape != training_action.shape or not training_mask.all():
                    errors.append(f"episode {index}: physical loss mask mismatch")
                if not np.isfinite(training_action).all():
                    errors.append(f"episode {index}: sidecar actions are non-finite")
                if not np.all(training_action[:ZERO_PREFIX_ROWS] == 0):
                    errors.append(f"episode {index}: zero16 prefix mismatch")
                expected_source_ids = np.concatenate([
                    np.full(ZERO_PREFIX_ROWS, -1, dtype=np.int64),
                    np.arange(real_count, dtype=np.int64),
                ])
                if not np.array_equal(source_frame_ids, expected_source_ids):
                    errors.append(f"episode {index}: action source_frame_ids mismatch")
                for key, expected in (
                    ("source_length", length),
                    ("real_action_count", real_count),
                    ("zero_prefix_count", ZERO_PREFIX_ROWS),
                    ("tail_discarded_count", length - real_count),
                ):
                    if int(sidecar[key]) != expected:
                        errors.append(
                            f"episode {index}: sidecar {key}={int(sidecar[key])}, "
                            f"expected {expected}"
                        )
                if str(sidecar["source_parquet_sha256"]) != source_sha:
                    errors.append(
                        f"episode {index}: sidecar source parquet SHA mismatch"
                    )

                joints, gripper, base_delta = _read_action_columns(source_path)
                transformed = transform_ebench_actions(joints, gripper, base_delta)
                stats_action, expected_training = prepare_episode_actions(
                    transformed, length
                )
                if not np.array_equal(training_action, expected_training):
                    errors.append(f"episode {index}: sidecar action values mismatch")
                if not np.array_equal(sidecar["episode_anchor_12d"], joints[0]):
                    errors.append(f"episode {index}: episode anchor mismatch")
                quantile_values[cursor : cursor + real_count] = stats_action
                cursor += real_count
        except (
            AttributeError,
            EOFError,
            IndexError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
            RuntimeError,
            zipfile.BadZipFile,
        ) as error:
            errors.append(f"episode {index}: sidecar audit failed: {error}")
            # Keep the memmap cursor aligned so later episodes remain auditable.
            quantile_values[cursor : cursor + real_count] = np.nan
            cursor += real_count

        frame_ids = sampled_rgb_frame_ids(length)
        camera_frame_ids: list[np.ndarray] = []
        for camera_key in SOURCE_CAMERA_KEYS:
            path = _latent_path(output_root, record, camera_key)
            if not path.is_file():
                errors.append(f"episode {index}: missing latent {path}")
                continue
            try:
                value = torch.load(
                    path, map_location="cpu", weights_only=False, mmap=True
                )
                if "text_emb" in value:
                    errors.append(f"{path}: embeds text instead of shared lookup")
                expected_shape = (latent_frames * 14 * 14, 48)
                if tuple(value["latent"].shape) != expected_shape:
                    errors.append(
                        f"{path}: latent shape {tuple(value['latent'].shape)} "
                        f"!= {expected_shape}"
                    )
                if value["latent"].dtype != torch.bfloat16:
                    errors.append(f"{path}: latent dtype {value['latent'].dtype}")
                if int(value["latent_num_frames"]) != latent_frames:
                    errors.append(f"{path}: latent frame count mismatch")
                current_ids = np.asarray(value["frame_ids"], dtype=np.int64)
                camera_frame_ids.append(current_ids)
                if not np.array_equal(current_ids, frame_ids):
                    errors.append(f"{path}: sampled frame IDs mismatch")
                for key, expected in (
                    ("latent_height", 14),
                    ("latent_width", 14),
                    ("video_num_frames", len(frame_ids)),
                    ("source_video_num_frames", length),
                    ("video_height", 224),
                    ("video_width", 224),
                    ("text", str(record["task"])),
                    ("camera_key", camera_key),
                    ("start_frame", 0),
                    ("end_frame", length),
                    ("fps", int(manifest["fps"])),
                    ("ori_fps", int(manifest["fps"])),
                    ("image_stride", IMAGE_STRIDE),
                ):
                    if value.get(key) != expected:
                        errors.append(f"{path}: {key}={value.get(key)!r}, expected {expected!r}")
                latent_float = value["latent"].float()
                if not torch.isfinite(latent_float).all():
                    errors.append(f"{path}: non-finite latent")
                else:
                    latent_value_count += latent_float.numel()
                    latent_sum += float(latent_float.sum())
                    latent_square_sum += float(latent_float.square().sum())
                    latent_min = min(latent_min, float(latent_float.min()))
                    latent_max = max(latent_max, float(latent_float.max()))
                latent_entries.append({
                    "episode_index": index,
                    "camera_key": camera_key,
                    "path": str(path.relative_to(output_root)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "latent_num_frames": latent_frames,
                })
            except (
                AttributeError,
                EOFError,
                IndexError,
                OSError,
                TypeError,
                ValueError,
                KeyError,
                RuntimeError,
            ) as error:
                errors.append(f"{path}: latent audit failed: {error}")
        if camera_frame_ids and not all(
            np.array_equal(camera_frame_ids[0], ids) for ids in camera_frame_ids[1:]
        ):
            errors.append(f"episode {index}: four camera frame IDs differ")
        if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(selected):
            print(f"audit [{ordinal}/{len(selected)}] episode {index}")

    quantile_values.flush()
    if cursor != stats_rows:
        errors.append(f"audit stats cursor {cursor} != {stats_rows}")
    if np.isfinite(quantile_values).all():
        observed_q01, observed_q99 = compute_action_quantiles(quantile_values)
        if is_full_audit:
            expected_q01 = np.asarray(stats["quantiles"]["observed_q01_19d"])
            expected_q99 = np.asarray(stats["quantiles"]["observed_q99_19d"])
            if not np.allclose(observed_q01, expected_q01, rtol=0, atol=1e-7):
                errors.append("observed q01 no longer matches action stats")
            if not np.allclose(observed_q99, expected_q99, rtol=0, atol=1e-7):
                errors.append("observed q99 no longer matches action stats")
    else:
        errors.append("retained real action population contains non-finite values")
    del quantile_values
    workspace.unlink()

    if is_full_audit:
        actual_latent_files = len(list((output_root / "latents").rglob("*.pth")))
        expected_latent_files = int(manifest["expected"]["latent_files"])
        if actual_latent_files != expected_latent_files:
            errors.append(
                f"latent file count {actual_latent_files}, expected {expected_latent_files}"
            )
    else:
        actual_latent_files = len(latent_entries)
        expected_latent_files = len(selected) * len(SOURCE_CAMERA_KEYS)

    latent_mean = latent_sum / latent_value_count if latent_value_count else None
    if latent_value_count:
        variance = max(
            latent_square_sum / latent_value_count - float(latent_mean) ** 2,
            0.0,
        )
        latent_std = variance ** 0.5
    else:
        latent_std = None
        latent_min = None  # type: ignore[assignment]
        latent_max = None  # type: ignore[assignment]

    latent_manifest = {
        "manifest_version": "ebench_lingbot_latents_v1",
        "complete_dataset": is_full_audit,
        "camera_keys": list(SOURCE_CAMERA_KEYS),
        "camera_layout": [list(row) for row in CAMERA_LAYOUT],
        "episodes_audited": len(selected),
        "files": latent_entries,
    }
    if is_full_audit and not errors:
        write_json(output_root / LATENT_MANIFEST_NAME, latent_manifest)
        latent_manifest_sha = sha256_file(output_root / LATENT_MANIFEST_NAME)
        stats["manifests"]["latent_manifest_sha256"] = latent_manifest_sha
        write_json(stats_path, stats)
        manifest["artifacts"]["latent_manifest"] = {
            "path": LATENT_MANIFEST_NAME,
            "sha256": latent_manifest_sha,
        }
        manifest["artifacts"]["action_stats_sha256"] = sha256_file(stats_path)
        write_json(manifest_path, manifest)
    else:
        latent_manifest_sha = None

    report = {
        "passed": not errors,
        "complete_dataset": is_full_audit,
        "errors": errors,
        "episodes_audited": len(selected),
        "latent_files": {"actual": actual_latent_files, "expected": expected_latent_files},
        "tensor_schema": {
            "per_camera": [48, "F", 14, 14],
            "mosaic": [48, "F", 28, 28],
            "action_19d": ["16*F", 19],
            "model_action": [30, "F", 16, 1],
            "latent_dtype": "torch.bfloat16",
        },
        "latent_statistics": {
            "count": latent_value_count,
            "mean": latent_mean,
            "std": latent_std,
            "min": latent_min,
            "max": latent_max,
        },
        "data_manifest_sha256": actual_data_manifest_sha,
        "latent_manifest_sha256": latent_manifest_sha,
    }
    report_path = args.report_path or (output_root / AUDIT_REPORT_NAME)
    write_json(report_path.resolve(), report)
    if is_full_audit and not errors:
        # This atomic rename is the final commit point after stats, manifests,
        # preprocess provenance, and the audit report have all been written.
        _write_audit_success_marker(output_root)
    print(
        f"audit {'passed' if report['passed'] else 'failed'}: "
        f"episodes={len(selected)}, latent_files={actual_latent_files}, "
        f"errors={len(errors)}, report={report_path.resolve()}"
    )
    if errors:
        raise RuntimeError(f"EBench audit found {len(errors)} errors")


if __name__ == "__main__":
    main()
