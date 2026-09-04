#!/usr/bin/env python3
"""Build audited virtual EBench training segments without copying latents.

The source EBench derivative remains immutable.  This command verifies its
full-audit provenance, every latent file, and every action sidecar before it
publishes three small derived artifacts:

* a manifest describing virtual latent/action slices;
* action quantiles recomputed after segment-local re-anchoring; and
* a final success marker binding the source and derived artifacts together.

Segment actions are deterministic and therefore are not written as duplicate
sidecars.  The manifest records a content hash that the dataset loader can
verify after reconstructing a segment from its audited parent sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.prepare_ebench_lingbot import (  # noqa: E402
    _stats_artifact,
    compute_action_quantiles,
    sha256_file,
    write_json,
)
from wan_va.dataset.ebench_latent_dataset import (  # noqa: E402
    EBENCH_ACTION_STATS_VERSION,
    EBENCH_AUDIT_SUCCESS_MARKER_NAME,
    EBENCH_AUDIT_SUCCESS_MARKER_VERSION,
    EBENCH_CAMERA_KEYS,
    EBENCH_DATA_MANIFEST_VERSION,
    EBENCH_FIXED_ZERO_PHYSICAL_CHANNEL,
    EBENCH_LATENT_MANIFEST_VERSION,
    EBENCH_PREPROCESS_MANIFEST_VERSION,
    EBENCH_SEGMENT_AUDIT_VERSION,
    EBENCH_SEGMENT_MANIFEST_VERSION,
    EBENCH_SEGMENTED_ACTION_STATS_VERSION,
    EBENCH_SELF_TOKENS_PER_FRAME,
    EBENCH_ZERO_PREFIX_ROWS,
    ebench_segment_action_sha256,
    ebench_self_token_count,
    plan_ebench_segments,
    reanchor_ebench_segment_actions,
)


DEFAULT_MAX_SELF_TOKENS = 81_920
DEFAULT_SEGMENT_MANIFEST_NAME = "segment_manifest_81920_v1.json"
DEFAULT_ACTION_STATS_NAME = "action_stats_segment_81920_v1.json"
DEFAULT_SEGMENT_AUDIT_NAME = "segment_audit_success_81920.json"

SOURCE_ACTION_STATS_NAME = "action_stats_v1.json"
SOURCE_DATA_MANIFEST_NAME = "data_manifest.json"
SOURCE_LATENT_MANIFEST_NAME = "latent_manifest.json"
SOURCE_PREPROCESS_MANIFEST_NAME = "preprocess_manifest.json"
SOURCE_TEXT_CACHE_NAME = "text_embeddings.pt"
SOURCE_EMPTY_EMBEDDING_NAME = "empty_emb.pt"

PHYSICAL_ACTION_DIM = 19
ACTION_PER_FRAME = EBENCH_ZERO_PREFIX_ROWS

SOURCE_AUDIT_ARTIFACTS = (
    SOURCE_ACTION_STATS_NAME,
    SOURCE_DATA_MANIFEST_NAME,
    SOURCE_LATENT_MANIFEST_NAME,
    SOURCE_PREPROCESS_MANIFEST_NAME,
    SOURCE_TEXT_CACHE_NAME,
    SOURCE_EMPTY_EMBEDDING_NAME,
)
SEGMENT_SOURCE_ARTIFACTS = (
    SOURCE_ACTION_STATS_NAME,
    SOURCE_DATA_MANIFEST_NAME,
    SOURCE_LATENT_MANIFEST_NAME,
    SOURCE_PREPROCESS_MANIFEST_NAME,
    EBENCH_AUDIT_SUCCESS_MARKER_NAME,
)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _validated_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error
    return value.lower()


def _resolve_relative_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise ValueError(f"Unsafe {label}: {value!r}")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} escapes dataset root: {value!r}")
    return path


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Malformed {label} line {line_number}: {path}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{label} line {line_number} must contain an object"
                )
            records.append(value)
    return records


def _check_file_sha(
    path: Path,
    expected_sha256: Any,
    label: str,
    *,
    expected_size: Any | None = None,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(
            f"{label} size mismatch: expected={int(expected_size)}, "
            f"actual={path.stat().st_size}, path={path}"
        )
    expected = _validated_sha256(expected_sha256, f"{label} SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}, "
            f"path={path}"
        )
    return actual


def _maximum_segment_frames(max_self_tokens: int) -> int:
    max_self_tokens = int(max_self_tokens)
    if max_self_tokens <= 0:
        raise ValueError("--max-self-tokens must be positive")
    budget_frame_count = max_self_tokens // EBENCH_SELF_TOKENS_PER_FRAME
    frame_count = budget_frame_count
    while frame_count > 0 and ebench_self_token_count(frame_count) > max_self_tokens:
        frame_count -= 1
    if frame_count < 2:
        raise ValueError(
            f"max_self_tokens={max_self_tokens} cannot fit an anchor plus actions"
        )
    if frame_count != budget_frame_count:
        raise ValueError(
            "--max-self-tokens must admit its integer frame budget after "
            "128-token padding; use a 128-token-aligned cap such as 81920"
        )
    return frame_count


def _resolve_output_path(
    dataset_root: Path,
    configured: Path | None,
    default_name: str,
) -> Path:
    if configured is None:
        return dataset_root / default_name
    configured = configured.expanduser()
    if not configured.is_absolute():
        configured = dataset_root / configured
    return configured.resolve()


def _artifact_path_value(dataset_root: Path, path: Path) -> str:
    if path.parent != dataset_root:
        raise ValueError(f"Segment overlay artifact must live at {dataset_root}: {path}")
    return path.name


def _validate_output_paths(
    dataset_root: Path,
    segment_manifest_path: Path,
    action_stats_path: Path,
    audit_path: Path,
) -> None:
    outputs = (segment_manifest_path, action_stats_path, audit_path)
    if len(set(outputs)) != len(outputs):
        raise ValueError("Segment manifest, action stats, and audit paths must differ")
    if any(path.parent != dataset_root for path in outputs):
        raise ValueError(
            "Segment manifest, action stats, and audit marker must be direct "
            f"children of the dataset root: {dataset_root}"
        )
    protected = {
        (dataset_root / filename).resolve()
        for filename in (*SOURCE_AUDIT_ARTIFACTS, EBENCH_AUDIT_SUCCESS_MARKER_NAME)
    }
    overlap = protected.intersection(outputs)
    if overlap:
        raise ValueError(
            "Refusing to overwrite audited source artifacts: "
            + ", ".join(str(path) for path in sorted(overlap))
        )


def _validate_source_audit(dataset_root: Path) -> dict[str, str]:
    marker_path = dataset_root / EBENCH_AUDIT_SUCCESS_MARKER_NAME
    marker = _load_json_object(marker_path, "EBench full-audit marker")
    if marker.get("marker_version") != EBENCH_AUDIT_SUCCESS_MARKER_VERSION:
        raise ValueError("Unsupported EBench full-audit marker version")
    try:
        marker_root = Path(str(marker["dataset_root"])).resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Malformed EBench full-audit dataset_root") from error
    if marker_root != dataset_root:
        raise ValueError(
            f"EBench audit belongs to {marker_root}, not {dataset_root}"
        )
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        SOURCE_AUDIT_ARTIFACTS
    ):
        raise ValueError("EBench full-audit artifact set is incomplete or unexpected")
    actual: dict[str, str] = {}
    for filename in SOURCE_AUDIT_ARTIFACTS:
        actual[filename] = _check_file_sha(
            dataset_root / filename,
            artifacts[filename],
            f"audited source artifact {filename}",
        )
    actual[EBENCH_AUDIT_SUCCESS_MARKER_NAME] = sha256_file(marker_path)
    return actual


def _validate_metadata_hashes(
    dataset_root: Path, data_manifest: Mapping[str, Any]
) -> None:
    for mapping_key, metadata_root in (
        ("derived_metadata_sha256", dataset_root / "meta"),
        (
            "source_metadata_sha256",
            Path(str(data_manifest.get("source_root", ""))).resolve() / "meta",
        ),
    ):
        hashes = data_manifest.get(mapping_key)
        if not isinstance(hashes, Mapping) or not hashes:
            raise ValueError(f"Data manifest lacks {mapping_key}")
        for filename, expected_sha in hashes.items():
            if Path(str(filename)).name != str(filename):
                raise ValueError(f"Unsafe metadata filename in {mapping_key}")
            _check_file_sha(
                metadata_root / str(filename),
                expected_sha,
                f"{mapping_key} {filename}",
            )


def _validate_episode_metadata(
    dataset_root: Path,
) -> tuple[dict[str, Any], dict[int, str]]:
    info = _load_json_object(dataset_root / "meta" / "info.json", "EBench info")
    records = _read_jsonl(
        dataset_root / "meta" / "episodes.jsonl", "EBench episode metadata"
    )
    tasks: dict[int, str] = {}
    for record in records:
        try:
            episode_index = int(record["episode_index"])
            length = int(record["length"])
            episode_tasks = record["tasks"]
            action_configs = record["action_config"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed EBench episode metadata") from error
        if episode_index in tasks or length <= 1:
            raise ValueError(
                f"Duplicate episode or invalid length for episode {episode_index}"
            )
        if not isinstance(episode_tasks, list) or not episode_tasks:
            raise ValueError(f"Episode {episode_index} has no task")
        if not isinstance(action_configs, list) or len(action_configs) != 1:
            raise ValueError(
                f"Episode {episode_index} must have one full action_config"
            )
        config = action_configs[0]
        if (
            int(config.get("start_frame", -1)) != 0
            or int(config.get("end_frame", -1)) != length
        ):
            raise ValueError(
                f"Episode {episode_index} action_config does not cover [0,{length})"
            )
        if str(config.get("action_text", "")) != str(episode_tasks[0]):
            raise ValueError(f"Episode {episode_index} task/action text mismatch")
        tasks[episode_index] = str(episode_tasks[0])
    if int(info.get("total_episodes", -1)) != len(tasks):
        raise ValueError("EBench info total_episodes does not match metadata")
    return info, tasks


def _validate_data_manifest(
    dataset_root: Path,
    expected_sha256: str,
    info: Mapping[str, Any],
    episode_tasks: Mapping[int, str],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    path = dataset_root / SOURCE_DATA_MANIFEST_NAME
    _check_file_sha(path, expected_sha256, "EBench data manifest")
    manifest = _load_json_object(path, "EBench data manifest")
    if manifest.get("manifest_version") != EBENCH_DATA_MANIFEST_VERSION:
        raise ValueError("Unsupported EBench data manifest version")
    if Path(str(manifest.get("output_root", ""))).resolve() != dataset_root:
        raise ValueError("EBench data manifest output_root mismatch")
    _validate_metadata_hashes(dataset_root, manifest)

    raw_entries = manifest.get("episodes")
    if not isinstance(raw_entries, list):
        raise ValueError("EBench data manifest episodes must be a list")
    entries: dict[int, dict[str, Any]] = {}
    action_paths: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("EBench data manifest episode entry is not an object")
        try:
            episode_index = int(raw_entry["episode_index"])
            length = int(raw_entry["length"])
            retained_real_actions = int(raw_entry["retained_real_actions"])
            action_relative = str(raw_entry["action_sidecar"])
            action_size = int(raw_entry["action_sidecar_size"])
            action_sha = _validated_sha256(
                raw_entry["action_sidecar_sha256"],
                f"episode {episode_index} action sidecar SHA256",
            )
            _validated_sha256(
                raw_entry["source_parquet_sha256"],
                f"episode {episode_index} source parquet SHA256",
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed EBench data manifest episode entry") from error
        if episode_index in entries or episode_index not in episode_tasks:
            raise ValueError(f"Unexpected or duplicate data episode {episode_index}")
        if length <= 1 or retained_real_actions != ACTION_PER_FRAME * (
            (length - 1) // ACTION_PER_FRAME
        ):
            raise ValueError(f"Episode {episode_index} retained-action mismatch")
        action_path = _resolve_relative_file(
            dataset_root, action_relative, f"episode {episode_index} action sidecar"
        )
        if action_relative in action_paths or action_size <= 0:
            raise ValueError(f"Duplicate or invalid action sidecar for {episode_index}")
        if not action_path.is_file() or action_path.stat().st_size != action_size:
            raise ValueError(f"Episode {episode_index} action sidecar size mismatch")
        action_paths.add(action_relative)
        entry = dict(raw_entry)
        entry["episode_index"] = episode_index
        entry["length"] = length
        entry["retained_real_actions"] = retained_real_actions
        entry["action_sidecar"] = action_relative
        entry["action_sidecar_size"] = action_size
        entry["action_sidecar_sha256"] = action_sha
        entries[episode_index] = entry

    if set(entries) != set(episode_tasks):
        raise ValueError("Data manifest does not cover episode metadata exactly")
    if len(entries) != int(info.get("total_episodes", -1)):
        raise ValueError("Data manifest episode count mismatch")
    actual_action_paths = {
        path.relative_to(dataset_root).as_posix()
        for path in (dataset_root / "actions").rglob("*.npz")
    }
    if actual_action_paths != action_paths:
        raise ValueError(
            "Action sidecar tree differs from data manifest: "
            f"missing={len(action_paths - actual_action_paths)}, "
            f"extra={len(actual_action_paths - action_paths)}"
        )
    return manifest, entries


def _validate_latent_manifest(
    dataset_root: Path,
    expected_sha256: str,
    info: Mapping[str, Any],
    data_entries: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[int, int]]:
    path = dataset_root / SOURCE_LATENT_MANIFEST_NAME
    _check_file_sha(path, expected_sha256, "EBench latent manifest")
    manifest = _load_json_object(path, "EBench latent manifest")
    if (
        manifest.get("manifest_version") != EBENCH_LATENT_MANIFEST_VERSION
        or manifest.get("complete_dataset") is not True
        or tuple(manifest.get("camera_keys", ())) != EBENCH_CAMERA_KEYS
    ):
        raise ValueError("EBench latent manifest contract mismatch")
    if int(manifest.get("episodes_audited", -1)) != len(data_entries):
        raise ValueError("EBench latent manifest episode count mismatch")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("EBench latent manifest files must be a list")
    expected_keys = {
        (episode_index, camera_key)
        for episode_index in data_entries
        for camera_key in EBENCH_CAMERA_KEYS
    }
    seen_keys: set[tuple[int, str]] = set()
    seen_paths: set[str] = set()
    frames_by_episode: dict[int, set[int]] = {
        episode_index: set() for episode_index in data_entries
    }
    for ordinal, raw_entry in enumerate(raw_files, start=1):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("EBench latent manifest file entry is not an object")
        try:
            episode_index = int(raw_entry["episode_index"])
            camera_key = str(raw_entry["camera_key"])
            relative = str(raw_entry["path"])
            size = int(raw_entry["size"])
            latent_frames = int(raw_entry["latent_num_frames"])
            expected_sha = _validated_sha256(
                raw_entry["sha256"],
                f"episode {episode_index} {camera_key} latent SHA256",
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed EBench latent manifest file entry") from error
        key = (episode_index, camera_key)
        if key not in expected_keys or key in seen_keys or relative in seen_paths:
            raise ValueError(f"Unexpected or duplicate latent entry: {key}")
        latent_path = _resolve_relative_file(
            dataset_root, relative, f"episode {episode_index} {camera_key} latent"
        )
        _check_file_sha(
            latent_path,
            expected_sha,
            f"episode {episode_index} {camera_key} latent",
            expected_size=size,
        )
        if latent_frames <= 1:
            raise ValueError(f"Episode {episode_index} latent is too short")
        seen_keys.add(key)
        seen_paths.add(relative)
        frames_by_episode[episode_index].add(latent_frames)
        if ordinal == 1 or ordinal % 1000 == 0 or ordinal == len(raw_files):
            print(f"verify latents [{ordinal}/{len(raw_files)}]")

    if seen_keys != expected_keys:
        raise ValueError(
            "Latent manifest does not cover every episode/camera exactly once"
        )
    if len(raw_files) != int(info.get("total_videos", -1)):
        raise ValueError("Latent manifest file count differs from EBench info")
    actual_paths = {
        path.relative_to(dataset_root).as_posix()
        for path in (dataset_root / "latents").rglob("*.pth")
    }
    if actual_paths != seen_paths:
        raise ValueError(
            "Latent file tree differs from latent manifest: "
            f"missing={len(seen_paths - actual_paths)}, "
            f"extra={len(actual_paths - seen_paths)}"
        )

    frame_counts: dict[int, int] = {}
    for episode_index, values in frames_by_episode.items():
        if len(values) != 1:
            raise ValueError(
                f"Episode {episode_index} camera latent frame counts differ: {values}"
            )
        frame_count = next(iter(values))
        expected_frame_count = 1 + int(
            data_entries[episode_index]["retained_real_actions"]
        ) // ACTION_PER_FRAME
        if frame_count != expected_frame_count:
            raise ValueError(
                f"Episode {episode_index} latent/action frame mismatch: "
                f"latent={frame_count}, expected={expected_frame_count}"
            )
        frame_counts[episode_index] = frame_count
    return manifest, frame_counts


def _validate_preprocess_manifest(
    dataset_root: Path,
    expected_sha256: str,
    source_shas: Mapping[str, str],
    episode_tasks: Mapping[int, str],
    data_entries: Mapping[int, Mapping[str, Any]],
    latent_frame_counts: Mapping[int, int],
) -> dict[str, Any]:
    path = dataset_root / SOURCE_PREPROCESS_MANIFEST_NAME
    _check_file_sha(path, expected_sha256, "EBench preprocess manifest")
    manifest = _load_json_object(path, "EBench preprocess manifest")
    if (
        manifest.get("manifest_version") != EBENCH_PREPROCESS_MANIFEST_VERSION
        or Path(str(manifest.get("output_root", ""))).resolve() != dataset_root
        or tuple(manifest.get("camera_keys", ())) != EBENCH_CAMERA_KEYS
    ):
        raise ValueError("EBench preprocess manifest contract mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("EBench preprocess artifact map is missing")
    if (
        artifacts.get("data_manifest") != SOURCE_DATA_MANIFEST_NAME
        or artifacts.get("data_manifest_sha256")
        != source_shas[SOURCE_DATA_MANIFEST_NAME]
        or artifacts.get("action_stats") != SOURCE_ACTION_STATS_NAME
        or artifacts.get("action_stats_sha256")
        != source_shas[SOURCE_ACTION_STATS_NAME]
    ):
        raise ValueError("EBench preprocess data/stats references mismatch")
    latent_artifact = artifacts.get("latent_manifest")
    if (
        not isinstance(latent_artifact, Mapping)
        or latent_artifact.get("path") != SOURCE_LATENT_MANIFEST_NAME
        or latent_artifact.get("sha256")
        != source_shas[SOURCE_LATENT_MANIFEST_NAME]
    ):
        raise ValueError("EBench preprocess latent reference mismatch")
    text_artifact = artifacts.get("text_cache")
    if (
        not isinstance(text_artifact, Mapping)
        or text_artifact.get("path") != SOURCE_TEXT_CACHE_NAME
        or text_artifact.get("sha256") != source_shas[SOURCE_TEXT_CACHE_NAME]
        or text_artifact.get("empty_path") != SOURCE_EMPTY_EMBEDDING_NAME
        or text_artifact.get("empty_sha256")
        != source_shas[SOURCE_EMPTY_EMBEDDING_NAME]
    ):
        raise ValueError("EBench preprocess text artifact references mismatch")

    selected = manifest.get("selected_episodes")
    if not isinstance(selected, list) or int(
        manifest.get("selected_episode_count", -1)
    ) != len(selected):
        raise ValueError("EBench preprocess selected episode list is inconsistent")
    selected_indices: set[int] = set()
    for record in selected:
        if not isinstance(record, Mapping):
            raise ValueError("Malformed preprocess selected episode")
        episode_index = int(record.get("episode_index", -1))
        if episode_index in selected_indices or episode_index not in episode_tasks:
            raise ValueError(f"Unexpected preprocess episode {episode_index}")
        if (
            int(record.get("length", -1)) != int(data_entries[episode_index]["length"])
            or int(record.get("retained_real_actions", -1))
            != int(data_entries[episode_index]["retained_real_actions"])
            or int(record.get("latent_frames", -1))
            != latent_frame_counts[episode_index]
            or str(record.get("task", "")) != episode_tasks[episode_index]
        ):
            raise ValueError(
                f"Preprocess episode {episode_index} differs from audited artifacts"
            )
        selected_indices.add(episode_index)
    if selected_indices != set(episode_tasks):
        raise ValueError("Preprocess manifest does not cover all episodes")

    expected = manifest.get("expected")
    source_real_rows = sum(
        int(entry["retained_real_actions"]) for entry in data_entries.values()
    )
    source_latent_frames = sum(latent_frame_counts.values())
    if (
        not isinstance(expected, Mapping)
        or int(expected.get("real_action_rows", -1)) != source_real_rows
        or int(expected.get("latent_frames_per_camera", -1))
        != source_latent_frames
        or int(expected.get("latent_files", -1))
        != len(latent_frame_counts) * len(EBENCH_CAMERA_KEYS)
    ):
        raise ValueError("EBench preprocess expected counts mismatch")
    return manifest


def _validate_source_stats(
    dataset_root: Path,
    expected_sha256: str,
    data_manifest_sha256: str,
    latent_manifest_sha256: str,
    expected_real_rows: int,
) -> dict[str, Any]:
    path = dataset_root / SOURCE_ACTION_STATS_NAME
    _check_file_sha(path, expected_sha256, "EBench source action stats")
    artifact = _load_json_object(path, "EBench source action stats")
    if artifact.get("artifact_version") != EBENCH_ACTION_STATS_VERSION:
        raise ValueError("Unsupported EBench source action-stats version")
    try:
        quantiles = artifact["quantiles"]
        observed_q01 = np.asarray(quantiles["observed_q01_19d"], dtype=np.float64)
        observed_q99 = np.asarray(quantiles["observed_q99_19d"], dtype=np.float64)
        frame_count = int(quantiles["frame_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Malformed EBench source action stats") from error
    expected = _stats_artifact(
        observed_q01,
        observed_q99,
        frame_count,
        data_manifest_sha256,
        latent_manifest_sha256,
    )
    if artifact != expected:
        raise ValueError("EBench source action-stats contract is inconsistent")
    if frame_count != expected_real_rows:
        raise ValueError("EBench source action-stats population size mismatch")
    return artifact


def _load_verified_parent_actions(
    dataset_root: Path,
    entry: Mapping[str, Any],
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_index = int(entry["episode_index"])
    action_path = _resolve_relative_file(
        dataset_root,
        entry["action_sidecar"],
        f"episode {episode_index} action sidecar",
    )
    _check_file_sha(
        action_path,
        entry["action_sidecar_sha256"],
        f"episode {episode_index} action sidecar",
        expected_size=entry["action_sidecar_size"],
    )
    try:
        with np.load(action_path, allow_pickle=False) as sidecar:
            raw_action = sidecar["action_19d"]
            raw_mask = sidecar["loss_mask_19d"]
            if raw_action.dtype != np.dtype(np.float32):
                raise ValueError("action_19d must use float32")
            if raw_mask.dtype != np.dtype(bool):
                raise ValueError("loss_mask_19d must use bool")
            action = np.ascontiguousarray(raw_action)
            mask = np.ascontiguousarray(raw_mask)
            source_frame_ids = np.asarray(sidecar["source_frame_ids"], dtype=np.int64)
            episode_anchor = np.asarray(
                sidecar["episode_anchor_12d"], dtype=np.float32
            )
            source_length = int(sidecar["source_length"])
            real_action_count = int(sidecar["real_action_count"])
            zero_prefix_count = int(sidecar["zero_prefix_count"])
            tail_discarded_count = int(sidecar["tail_discarded_count"])
            source_parquet_sha256 = str(sidecar["source_parquet_sha256"])
    except (EOFError, KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"Malformed action sidecar for episode {episode_index}: {action_path}"
        ) from error

    expected_real_rows = (frame_count - 1) * ACTION_PER_FRAME
    expected_rows = frame_count * ACTION_PER_FRAME
    expected_source_ids = np.concatenate(
        (
            np.full(EBENCH_ZERO_PREFIX_ROWS, -1, dtype=np.int64),
            np.arange(expected_real_rows, dtype=np.int64),
        )
    )
    if action.shape != (expected_rows, PHYSICAL_ACTION_DIM) or mask.shape != action.shape:
        raise ValueError(f"Episode {episode_index} action sidecar shape mismatch")
    if not np.isfinite(action).all() or not np.isfinite(episode_anchor).all():
        raise ValueError(f"Episode {episode_index} action sidecar is non-finite")
    if episode_anchor.shape != (12,) or not mask.all():
        raise ValueError(f"Episode {episode_index} anchor/mask contract mismatch")
    if not np.all(action[:EBENCH_ZERO_PREFIX_ROWS] == 0):
        raise ValueError(f"Episode {episode_index} raw-zero16 prefix mismatch")
    if not np.all(action[:, EBENCH_FIXED_ZERO_PHYSICAL_CHANNEL] == 0):
        raise ValueError(f"Episode {episode_index} yaw is not fixed zero")
    if not np.array_equal(source_frame_ids, expected_source_ids):
        raise ValueError(f"Episode {episode_index} source frame IDs mismatch")
    if (
        source_length != int(entry["length"])
        or real_action_count != expected_real_rows
        or real_action_count != int(entry["retained_real_actions"])
        or zero_prefix_count != EBENCH_ZERO_PREFIX_ROWS
        or tail_discarded_count != source_length - real_action_count
        or source_parquet_sha256 != str(entry["source_parquet_sha256"])
    ):
        raise ValueError(f"Episode {episode_index} action provenance mismatch")
    return action, mask, episode_anchor


def _segment_sample_key(
    dataset_root: Path,
    episode_index: int,
    segment_ordinal: int,
    real_block_start: int,
    real_block_end: int,
) -> str:
    return "|".join(
        (
            str(dataset_root),
            str(episode_index),
            "segment",
            str(segment_ordinal),
            str(real_block_start),
            str(real_block_end),
            EBENCH_SEGMENT_MANIFEST_VERSION,
        )
    )


def _build_segments_and_quantiles(
    dataset_root: Path,
    max_self_tokens: int,
    data_entries: Mapping[int, Mapping[str, Any]],
    latent_frame_counts: Mapping[int, int],
    episode_tasks: Mapping[int, str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, int]]:
    source_real_rows = sum(
        int(entry["retained_real_actions"]) for entry in data_entries.values()
    )
    workspace_handle = tempfile.NamedTemporaryFile(
        prefix=".ebench_segment_quantiles_",
        suffix=".f32",
        dir=dataset_root,
        delete=False,
    )
    workspace_path = Path(workspace_handle.name)
    workspace_handle.close()
    values: np.memmap | None = None
    try:
        values = np.memmap(
            workspace_path,
            mode="w+",
            dtype=np.float32,
            shape=(source_real_rows, PHYSICAL_ACTION_DIM),
        )
        entries: list[dict[str, Any]] = []
        cursor = 0
        split_episodes = 0
        segment_latent_frames = 0
        sample_keys: set[str] = set()
        for episode_ordinal, episode_index in enumerate(sorted(data_entries), start=1):
            parent_entry = data_entries[episode_index]
            parent_frame_count = latent_frame_counts[episode_index]
            parent_action, parent_mask, parent_joint_anchor = (
                _load_verified_parent_actions(
                dataset_root, parent_entry, parent_frame_count
            )
            )
            specs = plan_ebench_segments(parent_frame_count, max_self_tokens)
            if (
                specs[0].real_block_start != 0
                or specs[-1].real_block_end != parent_frame_count - 1
                or any(
                    left.real_block_end != right.real_block_start
                    for left, right in zip(specs, specs[1:])
                )
            ):
                raise RuntimeError(
                    f"Segment planner did not partition episode {episode_index}"
                )
            if len(specs) > 1:
                split_episodes += 1
            episode_real_rows = 0
            for spec in specs:
                segment_action, segment_mask, joint_anchor, base_anchor = (
                    reanchor_ebench_segment_actions(
                        parent_action,
                        parent_mask,
                        spec,
                        action_per_frame=ACTION_PER_FRAME,
                    )
                )
                expected_shape = (
                    spec.frame_count * ACTION_PER_FRAME,
                    PHYSICAL_ACTION_DIM,
                )
                if (
                    segment_action.shape != expected_shape
                    or segment_mask.shape != expected_shape
                    or not segment_mask.all()
                    or not np.isfinite(segment_action).all()
                    or not np.all(
                        segment_action[:EBENCH_ZERO_PREFIX_ROWS] == 0
                    )
                    or not np.all(
                        segment_action[:, EBENCH_FIXED_ZERO_PHYSICAL_CHANNEL] == 0
                    )
                ):
                    raise RuntimeError(
                        f"Invalid reconstructed segment for episode {episode_index} "
                        f"ordinal {spec.ordinal}"
                    )
                token_count = ebench_self_token_count(spec.frame_count)
                if token_count > max_self_tokens:
                    raise RuntimeError("Segment planner emitted an over-budget segment")
                segment_joint_anchor = parent_joint_anchor.copy()
                segment_joint_anchor[:6] += joint_anchor[:6]
                segment_joint_anchor[6:12] += joint_anchor[6:12]
                real_rows = segment_action[EBENCH_ZERO_PREFIX_ROWS:]
                next_cursor = cursor + len(real_rows)
                values[cursor:next_cursor] = real_rows
                cursor = next_cursor
                episode_real_rows += len(real_rows)
                segment_latent_frames += spec.frame_count
                sample_key = _segment_sample_key(
                    dataset_root,
                    episode_index,
                    spec.ordinal,
                    spec.real_block_start,
                    spec.real_block_end,
                )
                if sample_key in sample_keys:
                    raise RuntimeError(f"Duplicate segment sample_key: {sample_key}")
                sample_keys.add(sample_key)
                entries.append(
                    {
                        "segment_index": len(entries),
                        "episode_index": episode_index,
                        "segment_ordinal": spec.ordinal,
                        "real_block_start": spec.real_block_start,
                        "real_block_end": spec.real_block_end,
                        "real_action_row_start": (
                            spec.real_block_start * ACTION_PER_FRAME
                        ),
                        "real_action_row_end": (
                            spec.real_block_end * ACTION_PER_FRAME
                        ),
                        "latent_start": spec.latent_start,
                        "latent_stop": spec.latent_stop,
                        "frame_count": spec.frame_count,
                        "token_count": token_count,
                        "task": episode_tasks[episode_index],
                        "segment_joint_anchor_12d": segment_joint_anchor.tolist(),
                        "segment_base_anchor_3d": base_anchor.tolist(),
                        "segment_action_sha256": ebench_segment_action_sha256(
                            segment_action, segment_mask
                        ),
                        "sample_key": sample_key,
                    }
                )
            if episode_real_rows != int(parent_entry["retained_real_actions"]):
                raise RuntimeError(
                    f"Segments changed the real-row population for episode {episode_index}"
                )
            if (
                episode_ordinal == 1
                or episode_ordinal % 100 == 0
                or episode_ordinal == len(data_entries)
            ):
                print(
                    f"build segments [{episode_ordinal}/{len(data_entries)}] "
                    f"episode={episode_index} total_segments={len(entries)}"
                )

        if cursor != source_real_rows:
            raise RuntimeError(
                f"Segment quantile cursor {cursor} != expected {source_real_rows}"
            )
        values.flush()
        observed_q01, observed_q99 = compute_action_quantiles(values)
        zero_prefix_rows = len(entries) * EBENCH_ZERO_PREFIX_ROWS
        counts = {
            "parent_episodes": len(data_entries),
            "split_episodes": split_episodes,
            "segments": len(entries),
            "source_latent_frames": sum(latent_frame_counts.values()),
            "virtual_segment_frames": segment_latent_frames,
            "real_action_rows": cursor,
            "zero_prefix_rows": zero_prefix_rows,
            "training_action_rows": cursor + zero_prefix_rows,
            "total_padded_self_tokens": sum(
                int(entry["token_count"]) for entry in entries
            ),
        }
        return entries, observed_q01, observed_q99, counts
    finally:
        if values is not None:
            del values
        workspace_path.unlink(missing_ok=True)


def prepare_segment_artifacts(
    dataset_root: Path,
    *,
    max_self_tokens: int = DEFAULT_MAX_SELF_TOKENS,
    segment_manifest_path: Path | None = None,
    action_stats_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"EBench dataset root does not exist: {dataset_root}")
    max_self_tokens = int(max_self_tokens)
    max_segment_frames = _maximum_segment_frames(max_self_tokens)
    max_real_blocks = max_segment_frames - 1
    segment_manifest_path = _resolve_output_path(
        dataset_root,
        segment_manifest_path,
        DEFAULT_SEGMENT_MANIFEST_NAME,
    )
    action_stats_path = _resolve_output_path(
        dataset_root,
        action_stats_path,
        DEFAULT_ACTION_STATS_NAME,
    )
    audit_path = _resolve_output_path(
        dataset_root,
        audit_path,
        DEFAULT_SEGMENT_AUDIT_NAME,
    )
    _validate_output_paths(
        dataset_root, segment_manifest_path, action_stats_path, audit_path
    )

    source_shas = _validate_source_audit(dataset_root)
    info, episode_tasks = _validate_episode_metadata(dataset_root)
    _, data_entries = _validate_data_manifest(
        dataset_root,
        source_shas[SOURCE_DATA_MANIFEST_NAME],
        info,
        episode_tasks,
    )
    _, latent_frame_counts = _validate_latent_manifest(
        dataset_root,
        source_shas[SOURCE_LATENT_MANIFEST_NAME],
        info,
        data_entries,
    )
    _validate_preprocess_manifest(
        dataset_root,
        source_shas[SOURCE_PREPROCESS_MANIFEST_NAME],
        source_shas,
        episode_tasks,
        data_entries,
        latent_frame_counts,
    )
    source_real_rows = sum(
        int(entry["retained_real_actions"]) for entry in data_entries.values()
    )
    _validate_source_stats(
        dataset_root,
        source_shas[SOURCE_ACTION_STATS_NAME],
        source_shas[SOURCE_DATA_MANIFEST_NAME],
        source_shas[SOURCE_LATENT_MANIFEST_NAME],
        source_real_rows,
    )

    segments, observed_q01, observed_q99, counts = _build_segments_and_quantiles(
        dataset_root,
        max_self_tokens,
        data_entries,
        latent_frame_counts,
        episode_tasks,
    )
    source_artifacts = {
        filename: source_shas[filename]
        for filename in SEGMENT_SOURCE_ARTIFACTS
    }
    segment_manifest = {
        "manifest_version": EBENCH_SEGMENT_MANIFEST_VERSION,
        "dataset_root": str(dataset_root),
        "max_self_tokens": max_self_tokens,
        "max_segment_frames": max_segment_frames,
        "max_real_blocks_per_segment": max_real_blocks,
        "source_artifacts": source_artifacts,
        "counts": counts,
        "segments": segments,
    }

    # Removing the success marker is the commit barrier.  A failure after this
    # point can leave atomically written intermediate artifacts, but never a
    # marker that authorizes a mismatched pair.
    audit_path.unlink(missing_ok=True)
    write_json(segment_manifest_path, segment_manifest)
    segment_manifest_sha256 = sha256_file(segment_manifest_path)

    stats = _stats_artifact(
        observed_q01,
        observed_q99,
        counts["real_action_rows"],
        source_shas[SOURCE_DATA_MANIFEST_NAME],
        source_shas[SOURCE_LATENT_MANIFEST_NAME],
    )
    stats["artifact_version"] = EBENCH_SEGMENTED_ACTION_STATS_VERSION
    stats["representation"]["joint_anchor"] = (
        "segment first retained action.joints"
    )
    stats["representation"]["base_representation"] = (
        "segment-local inclusive cumsum(action.base_delta); yaw=0"
    )
    stats["quantiles"]["population"] = (
        "train segment-relative retained real actions; excludes every segment "
        "zero16 prefix and source tails"
    )
    stats["manifests"].update(
        {
            "segment_manifest_path": _artifact_path_value(
                dataset_root, segment_manifest_path
            ),
            "segment_manifest_sha256": segment_manifest_sha256,
            "source_action_stats_sha256": source_shas[SOURCE_ACTION_STATS_NAME],
        }
    )
    stats["segmentation"] = {
        "strategy": "contiguous real-action blocks with one overlapping latent anchor",
        "max_self_tokens": max_self_tokens,
        "max_segment_frames": max_segment_frames,
        "max_real_blocks_per_segment": max_real_blocks,
        "action_rows_per_frame": ACTION_PER_FRAME,
        "zero_prefix_rows_per_segment": EBENCH_ZERO_PREFIX_ROWS,
        "latent_boundary_overlap_frames": 1,
        "segment_action_storage": "reconstructed from audited parent sidecar",
        "segment_action_hash": "sha256(float32 action bytes || uint8 mask bytes)",
        "counts": counts,
    }
    write_json(action_stats_path, stats)
    action_stats_sha256 = sha256_file(action_stats_path)

    marker = {
        "marker_version": EBENCH_SEGMENT_AUDIT_VERSION,
        "dataset_root": str(dataset_root),
        "max_self_tokens": max_self_tokens,
        "artifacts": {
            "segment_manifest": {
                "path": _artifact_path_value(dataset_root, segment_manifest_path),
                "sha256": segment_manifest_sha256,
            },
            "action_stats": {
                "path": _artifact_path_value(dataset_root, action_stats_path),
                "sha256": action_stats_sha256,
            },
            "source": source_artifacts,
        },
        "counts": counts,
    }
    write_json(audit_path, marker)

    result = {
        "dataset_root": str(dataset_root),
        "max_self_tokens": max_self_tokens,
        "segment_manifest": str(segment_manifest_path),
        "segment_manifest_sha256": segment_manifest_sha256,
        "action_stats": str(action_stats_path),
        "action_stats_sha256": action_stats_sha256,
        "audit_success": str(audit_path),
        "audit_success_sha256": sha256_file(audit_path),
        "counts": counts,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build audited, virtual EBench segments and segment-relative action stats "
            "without copying latent tensors."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Existing fully audited EBench LingBot-VA derivative root.",
    )
    parser.add_argument(
        "--max-self-tokens",
        type=int,
        default=DEFAULT_MAX_SELF_TOKENS,
        help=f"Per-segment self-token cap (default: {DEFAULT_MAX_SELF_TOKENS}).",
    )
    parser.add_argument(
        "--segment-manifest-path",
        type=Path,
        default=None,
        help=(
            f"Output manifest path (default: DATASET_ROOT/{DEFAULT_SEGMENT_MANIFEST_NAME}); "
            "relative paths are resolved under DATASET_ROOT."
        ),
    )
    parser.add_argument(
        "--action-stats-path",
        type=Path,
        default=None,
        help=(
            f"Output stats path (default: DATASET_ROOT/{DEFAULT_ACTION_STATS_NAME}); "
            "relative paths are resolved under DATASET_ROOT."
        ),
    )
    parser.add_argument(
        "--audit-success-path",
        type=Path,
        default=None,
        help=(
            f"Output success-marker path (default: DATASET_ROOT/{DEFAULT_SEGMENT_AUDIT_NAME}); "
            "relative paths are resolved under DATASET_ROOT."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare_segment_artifacts(
        args.dataset_root,
        max_self_tokens=args.max_self_tokens,
        segment_manifest_path=args.segment_manifest_path,
        action_stats_path=args.action_stats_path,
        audit_path=args.audit_success_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
