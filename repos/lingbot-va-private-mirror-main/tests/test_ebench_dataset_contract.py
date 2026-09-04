import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wan_va.dataset.ebench_latent_dataset import (
    EBENCH_AUDIT_SUCCESS_MARKER_NAME,
    EBENCH_AUDIT_SUCCESS_MARKER_VERSION,
    EBENCH_SEGMENT_AUDIT_VERSION,
    EBENCH_SEGMENT_MANIFEST_VERSION,
    EBENCH_SEGMENTED_ACTION_STATS_VERSION,
    EBENCH_SELF_TOKENS_PER_FRAME,
    EBenchSegmentSpec,
    EBenchLatentDataset,
    assemble_ebench_camera_grid,
    ebench_segment_action_sha256,
    ebench_self_token_count,
    map_and_normalize_ebench_actions,
    reanchor_ebench_segment_actions,
)
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from wan_va.dataset.packing import PackedSampleRef


CAMERA_KEYS = (
    "video.top_camera_view",
    "video.overlook_camera_view",
    "video.left_camera_view",
    "video.right_camera_view",
)
MODEL_CHANNELS = np.array(
    [14, 15, 16, 17, 18, 19, 28, 20, 21, 22, 23, 24, 25, 26, 29, 27, 0, 1, 2]
)


def _latent_record(value, frame_ids=None):
    return {
        "latent": torch.full(
            (2 * 14 * 14, 48), float(value), dtype=torch.bfloat16
        ),
        "latent_num_frames": 2,
        "latent_height": 14,
        "latent_width": 14,
        "frame_ids": np.arange(0, 17, 4) if frame_ids is None else frame_ids,
        "video_num_frames": 5,
        "text": "Put the cup on the tray.",
    }


def _write_tiny_ebench_dataset(root):
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {"chunks_size": 1000, "total_episodes": 1, "total_videos": 4}
        ),
        encoding="utf-8",
    )
    episode = {
        "episode_index": 0,
        "length": 17,
        "tasks": ["Put the cup on the tray."],
        "action_config": [{"start_frame": 0, "end_frame": 17}],
    }
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps(episode) + "\n", encoding="utf-8"
    )
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps(
            {"task_index": 0, "task": "Put the cup on the tray."}
        )
        + "\n",
        encoding="utf-8",
    )

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    data_manifest = {
        "manifest_version": "ebench_lingbot_data_v1",
        "derived_metadata_sha256": {
            filename: sha256(root / "meta" / filename)
            for filename in ("info.json", "episodes.jsonl", "tasks.jsonl")
        },
        "episodes": [{"episode_index": 0, "length": 17}],
    }
    (root / "data_manifest.json").write_text(
        json.dumps(data_manifest), encoding="utf-8"
    )
    latent_entries = []
    for value, camera_key in enumerate(CAMERA_KEYS, start=1):
        latent_dir = root / "latents" / "chunk-000" / camera_key
        latent_dir.mkdir(parents=True)
        latent_path = latent_dir / "episode_000000_0_17.pth"
        torch.save(_latent_record(value), latent_path)
        latent_entries.append(
            {
                "episode_index": 0,
                "camera_key": camera_key,
                "path": latent_path.relative_to(root).as_posix(),
                "size": latent_path.stat().st_size,
                "sha256": sha256(latent_path),
                "latent_num_frames": 2,
            }
        )
    latent_manifest = {
        "manifest_version": "ebench_lingbot_latents_v1",
        "complete_dataset": True,
        "camera_keys": list(CAMERA_KEYS),
        "camera_layout": [list(CAMERA_KEYS[:2]), list(CAMERA_KEYS[2:])],
        "episodes_audited": 1,
        "files": latent_entries,
    }
    (root / "latent_manifest.json").write_text(
        json.dumps(latent_manifest), encoding="utf-8"
    )
    stats_path = root / "action_stats_v1.json"
    stats_path.write_text(
        json.dumps(
            {
                "artifact_version": "ebench_lingbot_action_stats_v1",
                "representation": {
                    "physical_action_dim": 19,
                    "model_action_dim": 30,
                    "joint_anchor": "episode action.joints[0]",
                    "gripper_representation": "absolute action.gripper",
                    "base_representation": (
                        "inclusive cumsum(action.base_delta); yaw=0"
                    ),
                    "zero_prefix_rows": 16,
                    "tail_alignment": "16 * floor((length - 1) / 16)",
                },
                "channel_mapping": {
                    "physical_19d_to_model_30d": MODEL_CHANNELS.tolist(),
                    "used_model_channels": sorted(MODEL_CHANNELS.tolist()),
                    "unused_model_channels": sorted(
                        set(range(30)) - set(MODEL_CHANNELS.tolist())
                    ),
                },
                "normalization": {
                    "clip": [-1.5, 1.5],
                    "fixed_zero_neutral_range": [-1.0, 1.0],
                },
                "norm_stat": {
                    "q01": [-1.0] * 30,
                    "q99": [1.0] * 30,
                },
                "manifests": {
                    "data_manifest_sha256": sha256(root / "data_manifest.json"),
                    "latent_manifest_sha256": sha256(
                        root / "latent_manifest.json"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    action_dir = root / "actions" / "chunk-000"
    action_dir.mkdir(parents=True)
    actions = np.zeros((32, 19), dtype=np.float32)
    actions[16:] = np.linspace(-0.5, 0.5, 16 * 19).reshape(16, 19)
    actions[:, 18] = 0.0
    action_path = action_dir / "episode_000000.npz"
    np.savez(
        action_path,
        action_19d=actions,
        loss_mask_19d=np.ones_like(actions, dtype=bool),
        real_action_count=np.int64(16),
        episode_anchor_12d=np.arange(12, dtype=np.float32),
        source_frame_ids=np.concatenate(
            (np.full(16, -1, dtype=np.int64), np.arange(16, dtype=np.int64))
        ),
    )
    data_manifest["episodes"][0].update(
        {
            "action_sidecar": "actions/chunk-000/episode_000000.npz",
            "action_sidecar_size": action_path.stat().st_size,
            "action_sidecar_sha256": sha256(action_path),
        }
    )
    (root / "data_manifest.json").write_text(
        json.dumps(data_manifest), encoding="utf-8"
    )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["manifests"]["data_manifest_sha256"] = sha256(
        root / "data_manifest.json"
    )
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    text_cache_path = root / "text_embeddings.pt"
    empty_embedding_path = root / "empty_emb.pt"
    text_cache_path.write_bytes(b"tiny audited text cache")
    empty_embedding_path.write_bytes(b"tiny audited empty embedding")
    preprocess_path = root / "preprocess_manifest.json"
    preprocess_path.write_text(
        json.dumps(
            {
                "manifest_version": "ebench_lingbot_preprocess_v1",
                "output_root": str(root.resolve()),
                "artifacts": {
                    "data_manifest": "data_manifest.json",
                    "data_manifest_sha256": sha256(root / "data_manifest.json"),
                    "action_stats": "action_stats_v1.json",
                    "action_stats_sha256": sha256(stats_path),
                    "latent_manifest": {
                        "path": "latent_manifest.json",
                        "sha256": sha256(root / "latent_manifest.json"),
                    },
                    "text_cache": {
                        "path": "text_embeddings.pt",
                        "sha256": sha256(text_cache_path),
                        "empty_path": "empty_emb.pt",
                        "empty_sha256": sha256(empty_embedding_path),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (root / EBENCH_AUDIT_SUCCESS_MARKER_NAME).write_text(
        json.dumps(
            {
                "marker_version": EBENCH_AUDIT_SUCCESS_MARKER_VERSION,
                "dataset_root": str(root.resolve()),
                "artifacts": {
                    "action_stats_v1.json": sha256(stats_path),
                    "data_manifest.json": sha256(root / "data_manifest.json"),
                    "latent_manifest.json": sha256(root / "latent_manifest.json"),
                    "preprocess_manifest.json": sha256(preprocess_path),
                    "text_embeddings.pt": sha256(text_cache_path),
                    "empty_emb.pt": sha256(empty_embedding_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        obs_cam_keys=CAMERA_KEYS,
        used_action_channel_ids=tuple(MODEL_CHANNELS),
        action_dim=30,
        action_per_frame=16,
        patch_size=(1, 2, 2),
    )


def _rewrite_stats_manifest_sha(root, manifest_name):
    stats_path = root / "action_stats_v1.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["manifests"][f"{manifest_name.removesuffix('.json')}_sha256"] = (
        hashlib.sha256((root / manifest_name).read_bytes()).hexdigest()
    )
    stats_path.write_text(json.dumps(stats), encoding="utf-8")


def _write_tiny_segment_overlay(root, config):
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    max_self_tokens = 81_920
    max_segment_frames = max_self_tokens // EBENCH_SELF_TOKENS_PER_FRAME
    spec = EBenchSegmentSpec(ordinal=0, real_block_start=0, real_block_end=1)
    token_count = ebench_self_token_count(spec.frame_count)

    action_path = root / "actions" / "chunk-000" / "episode_000000.npz"
    with np.load(action_path, allow_pickle=False) as sidecar:
        parent_action = np.asarray(sidecar["action_19d"]).copy()
        parent_mask = np.asarray(sidecar["loss_mask_19d"]).copy()
        parent_joint_anchor = np.asarray(
            sidecar["episode_anchor_12d"], dtype=np.float32
        ).copy()
    segment_action, segment_mask, joint_delta, base_anchor = (
        reanchor_ebench_segment_actions(parent_action, parent_mask, spec)
    )
    segment_joint_anchor = parent_joint_anchor.copy()
    segment_joint_anchor[:6] += joint_delta[:6]
    segment_joint_anchor[6:12] += joint_delta[6:12]
    segment_action_sha256 = ebench_segment_action_sha256(
        segment_action, segment_mask
    )

    counts = {
        "parent_episodes": 1,
        "split_episodes": 0,
        "segments": 1,
        "source_latent_frames": 2,
        "virtual_segment_frames": 2,
        "real_action_rows": 16,
        "zero_prefix_rows": 16,
        "training_action_rows": 32,
        "total_padded_self_tokens": token_count,
    }
    source_artifacts = {
        filename: sha256(root / filename)
        for filename in (
            "action_stats_v1.json",
            "data_manifest.json",
            "latent_manifest.json",
            "preprocess_manifest.json",
            EBENCH_AUDIT_SUCCESS_MARKER_NAME,
        )
    }
    manifest_path = root / "segment_manifest_81920_v1.json"
    manifest = {
        "manifest_version": EBENCH_SEGMENT_MANIFEST_VERSION,
        "dataset_root": str(root.resolve()),
        "max_self_tokens": max_self_tokens,
        "max_segment_frames": max_segment_frames,
        "max_real_blocks_per_segment": max_segment_frames - 1,
        "source_artifacts": source_artifacts,
        "counts": counts,
        "segments": [
            {
                "segment_index": 0,
                "episode_index": 0,
                "segment_ordinal": 0,
                "real_block_start": 0,
                "real_block_end": 1,
                "real_action_row_start": 0,
                "real_action_row_end": 16,
                "latent_start": 0,
                "latent_stop": 2,
                "frame_count": 2,
                "token_count": token_count,
                "task": "Put the cup on the tray.",
                "segment_joint_anchor_12d": segment_joint_anchor.tolist(),
                "segment_base_anchor_3d": base_anchor.tolist(),
                "segment_action_sha256": segment_action_sha256,
                "sample_key": "|".join(
                    (
                        str(root.resolve()),
                        "0",
                        "segment",
                        "0",
                        "0",
                        "1",
                        EBENCH_SEGMENT_MANIFEST_VERSION,
                    )
                ),
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    source_stats_path = root / "action_stats_v1.json"
    segmented_stats_path = root / "action_stats_segment_81920_v1.json"
    stats = json.loads(source_stats_path.read_text(encoding="utf-8"))
    stats["artifact_version"] = EBENCH_SEGMENTED_ACTION_STATS_VERSION
    stats["representation"]["joint_anchor"] = (
        "segment first retained action.joints"
    )
    stats["representation"]["base_representation"] = (
        "segment-local inclusive cumsum(action.base_delta); yaw=0"
    )
    stats["manifests"].update(
        {
            "segment_manifest_path": manifest_path.name,
            "segment_manifest_sha256": sha256(manifest_path),
            "source_action_stats_sha256": sha256(source_stats_path),
        }
    )
    stats["quantiles"] = {"frame_count": 16}
    stats["segmentation"] = {
        "max_self_tokens": max_self_tokens,
        "max_segment_frames": max_segment_frames,
        "counts": counts,
    }
    segmented_stats_path.write_text(json.dumps(stats), encoding="utf-8")

    audit_path = root / "segment_audit_success_81920.json"
    audit = {
        "marker_version": EBENCH_SEGMENT_AUDIT_VERSION,
        "dataset_root": str(root.resolve()),
        "max_self_tokens": max_self_tokens,
        "artifacts": {
            "segment_manifest": {
                "path": manifest_path.name,
                "sha256": sha256(manifest_path),
            },
            "action_stats": {
                "path": segmented_stats_path.name,
                "sha256": sha256(segmented_stats_path),
            },
            "source": source_artifacts,
        },
        "counts": counts,
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    config.action_stats_path = segmented_stats_path.name
    config.segment_manifest_path = manifest_path.name
    config.segment_audit_path = audit_path.name
    config.max_self_tokens = max_self_tokens
    return token_count, segment_action_sha256


def test_loader_assembles_grid_and_rejects_unsynchronized_camera_frame_ids():
    records = {
        key: _latent_record(value)
        for value, key in enumerate(CAMERA_KEYS, start=1)
    }

    grid, frame_ids = assemble_ebench_camera_grid(records, CAMERA_KEYS)

    assert grid.shape == (48, 2, 28, 28)
    expected_shape = (48, 2, 14, 14)
    torch.testing.assert_close(
        grid[:, :, :14, :14], torch.ones(expected_shape, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        grid[:, :, :14, 14:],
        torch.full(expected_shape, 2.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        grid[:, :, 14:, :14],
        torch.full(expected_shape, 3.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        grid[:, :, 14:, 14:],
        torch.full(expected_shape, 4.0, dtype=torch.bfloat16),
    )
    np.testing.assert_array_equal(frame_ids, np.arange(0, 17, 4))

    records[CAMERA_KEYS[-1]] = _latent_record(
        4, frame_ids=np.array([0, 4, 8, 12, 20])
    )
    with pytest.raises(ValueError, match="frame"):
        assemble_ebench_camera_grid(records, CAMERA_KEYS)


def test_loader_rejects_wrong_dtype_and_non_finite_camera_latents():
    records = {
        key: _latent_record(value)
        for value, key in enumerate(CAMERA_KEYS, start=1)
    }
    records[CAMERA_KEYS[0]]["latent"] = records[CAMERA_KEYS[0]][
        "latent"
    ].float()
    with pytest.raises(ValueError, match="dtype.*bfloat16"):
        assemble_ebench_camera_grid(records, CAMERA_KEYS)

    records[CAMERA_KEYS[0]] = _latent_record(1)
    records[CAMERA_KEYS[0]]["latent"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        assemble_ebench_camera_grid(records, CAMERA_KEYS)


def test_loader_maps_normalizes_and_reshapes_actions_without_unmasking_padding():
    frame_count = 2
    action_19d = np.linspace(-0.75, 0.75, frame_count * 16 * 19).reshape(
        frame_count * 16, 19
    )
    action_19d[:, 18] = 0.0
    loss_mask_19d = np.ones_like(action_19d, dtype=bool)
    loss_mask_19d[0, 6] = False
    q01 = np.full(30, -1.0)
    q99 = np.full(30, 1.0)

    actions, action_mask = map_and_normalize_ebench_actions(
        action_19d,
        loss_mask_19d,
        q01,
        q99,
        MODEL_CHANNELS,
        frame_count=frame_count,
    )

    assert actions.shape == (30, 2, 16, 1)
    assert action_mask.shape == (30, 2, 16, 1)
    flat_actions = actions.squeeze(-1).permute(1, 2, 0).reshape(32, 30).numpy()
    flat_mask = action_mask.squeeze(-1).permute(1, 2, 0).reshape(32, 30).numpy()
    unused = np.setdiff1d(np.arange(30), MODEL_CHANNELS)
    expected_used_actions = (action_19d + 1.0) / (2.0 + 1e-6) * 2.0 - 1.0
    expected_used_actions[:, 18] = 0.0
    expected_used_actions[~loss_mask_19d] = 0
    np.testing.assert_allclose(
        flat_actions[:, MODEL_CHANNELS], expected_used_actions, atol=1e-6
    )
    np.testing.assert_array_equal(flat_actions[:, unused], 0)
    np.testing.assert_array_equal(flat_mask[:, MODEL_CHANNELS], loss_mask_19d)
    np.testing.assert_array_equal(flat_mask[:, unused], False)
    assert flat_actions[0, MODEL_CHANNELS[6]] == 0
    # Fixed physical yaw maps to model channel 2 and must be bit-exact zero,
    # despite the shared normalization formula's denominator epsilon.
    assert torch.count_nonzero(actions[2]).item() == 0


def test_loader_rejects_nonzero_fixed_base_yaw():
    action_19d = np.zeros((16, 19), dtype=np.float32)
    action_19d[7, 18] = 1e-4

    with pytest.raises(ValueError, match="yaw.*fixed raw zero"):
        map_and_normalize_ebench_actions(
            action_19d,
            np.ones_like(action_19d, dtype=bool),
            np.full(30, -1.0),
            np.full(30, 1.0),
            MODEL_CHANNELS,
            frame_count=1,
        )


def test_tiny_on_disk_episode_loads_full_shapes_and_packing_metadata(tmp_path):
    config = _write_tiny_ebench_dataset(tmp_path)
    dataset = EBenchLatentDataset("tiny-ebench", config, root=tmp_path)

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["latents"].shape == (48, 2, 28, 28)
    assert sample["actions"].shape == (30, 2, 16, 1)
    assert sample["actions_mask"].shape == (30, 2, 16, 1)
    assert sample["text"] == "Put the cup on the tray."
    # The raw-zero prefix occupies the first action chunk and remains trainable.
    torch.testing.assert_close(sample["actions"][:, 0], torch.zeros(30, 16, 1))
    assert sample["actions_mask"][:, 0].sum().item() == 19 * 16

    packing = dataset.get_packing_metadata(0)
    assert packing["frame_count"] == 2
    assert packing["shape_signature"] == (48, 28, 28, 30, 16, 1)
    assert packing["token_count"] == 896


def test_tiny_segment_overlay_contract_and_stats_version_gate(tmp_path):
    config = _write_tiny_ebench_dataset(tmp_path)
    config.segment_manifest_path = "segment_manifest_81920_v1.json"
    config.segment_audit_path = "segment_audit_success_81920.json"
    config.max_self_tokens = 81_920

    # Merely selecting segment paths must not reinterpret the old episode-level
    # quantiles as segment-relative statistics.
    with pytest.raises(ValueError, match="action stats artifact_version"):
        EBenchLatentDataset("tiny-ebench", config, root=tmp_path)

    expected_tokens, expected_segment_sha = _write_tiny_segment_overlay(
        tmp_path, config
    )
    dataset = EBenchLatentDataset("tiny-ebench", config, root=tmp_path)

    assert len(dataset) == 1
    assert expected_tokens == ebench_self_token_count(2) == 896
    packing = dataset.get_packing_metadata(0)
    assert packing["frame_count"] == 2
    assert packing["token_count"] == expected_tokens
    assert packing["shape_signature"] == (48, 28, 28, 30, 16, 1)
    assert dataset.segment_entries[0]["segment_action_sha256"] == (
        expected_segment_sha
    )

    sample = dataset[0]
    assert sample["latents"].shape == (48, 2, 28, 28)
    assert sample["actions"].shape == (30, 2, 16, 1)
    assert sample["actions_mask"].shape == (30, 2, 16, 1)
    assert sample["text"] == "Put the cup on the tray."


def test_loader_rejects_stats_mapping_mismatch(tmp_path):
    config = _write_tiny_ebench_dataset(tmp_path)
    stats_path = tmp_path / "action_stats_v1.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["channel_mapping"]["physical_19d_to_model_30d"][0] = 13
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match="channel mapping mismatch"):
        EBenchLatentDataset("tiny-ebench", config, root=tmp_path)


def test_loader_requires_completed_sha_matching_manifests(tmp_path):
    config = _write_tiny_ebench_dataset(tmp_path)
    latent_path = tmp_path / "latent_manifest.json"
    latent = json.loads(latent_path.read_text(encoding="utf-8"))
    latent["complete_dataset"] = False
    latent_path.write_text(json.dumps(latent), encoding="utf-8")
    _rewrite_stats_manifest_sha(tmp_path, "latent_manifest.json")

    with pytest.raises(ValueError, match="not a complete full-data audit"):
        EBenchLatentDataset("tiny-ebench", config, root=tmp_path)

    # A post-audit manifest mutation is rejected even if its JSON remains
    # structurally plausible, because stats pins the exact audited bytes.
    latent["complete_dataset"] = True
    latent_path.write_text(json.dumps(latent), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        EBenchLatentDataset("tiny-ebench", config, root=tmp_path)


def test_loader_requires_current_full_audit_success_marker(tmp_path):
    config = _write_tiny_ebench_dataset(tmp_path)
    marker_path = tmp_path / EBENCH_AUDIT_SUCCESS_MARKER_NAME
    marker_path.unlink()

    with pytest.raises(FileNotFoundError, match="full-audit success marker"):
        EBenchLatentDataset("tiny-ebench", config, root=tmp_path)


def test_loader_rejects_text_cache_changed_after_audit(tmp_path):
    config = _write_tiny_ebench_dataset(tmp_path)
    text_cache_path = tmp_path / "text_embeddings.pt"
    text_cache_path.write_bytes(text_cache_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="text cache SHA256 mismatch"):
        EBenchLatentDataset("tiny-ebench", config, root=tmp_path)


def test_loader_rejects_non_finite_action_sidecar(tmp_path):
    config = _write_tiny_ebench_dataset(tmp_path)
    action_path = tmp_path / "actions" / "chunk-000" / "episode_000000.npz"
    with np.load(action_path, allow_pickle=False) as sidecar:
        values = {key: np.asarray(sidecar[key]).copy() for key in sidecar.files}
    values["action_19d"][16, 0] = np.nan
    np.savez(action_path, **values)

    dataset = EBenchLatentDataset("tiny-ebench", config, root=tmp_path)
    with pytest.raises(ValueError, match="non-finite"):
        dataset[0]


def test_loader_rejects_finite_action_sidecar_changed_after_audit(tmp_path):
    config = _write_tiny_ebench_dataset(tmp_path)
    action_path = tmp_path / "actions" / "chunk-000" / "episode_000000.npz"
    with np.load(action_path, allow_pickle=False) as sidecar:
        values = {key: np.asarray(sidecar[key]).copy() for key in sidecar.files}
    values["action_19d"][16, 0] += 0.125
    np.savez(action_path, **values)

    dataset = EBenchLatentDataset("tiny-ebench", config, root=tmp_path)
    with pytest.raises(ValueError, match="audited data manifest"):
        dataset[0]


def test_loader_rejects_latent_file_changed_after_audit(tmp_path):
    config = _write_tiny_ebench_dataset(tmp_path)
    latent_path = (
        tmp_path
        / "latents"
        / "chunk-000"
        / CAMERA_KEYS[0]
        / "episode_000000_0_17.pth"
    )
    record = torch.load(latent_path, map_location="cpu", weights_only=False)
    record["latent"][0, 0] += 1
    torch.save(record, latent_path)

    dataset = EBenchLatentDataset("tiny-ebench", config, root=tmp_path)
    with pytest.raises(ValueError, match="audited latent manifest"):
        dataset[0]


def test_loader_episode_index_filter_rejects_duplicate_and_unknown_values(tmp_path):
    config = _write_tiny_ebench_dataset(tmp_path)
    config.ebench_episode_indices = [0]
    dataset = EBenchLatentDataset("tiny-ebench", config, root=tmp_path)
    assert dataset.episode_indices == (0,)
    assert len(dataset) == 1

    config.ebench_episode_indices = [0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        EBenchLatentDataset("tiny-ebench", config, root=tmp_path)

    config.ebench_episode_indices = [1750]
    with pytest.raises(ValueError, match="unknown.*1750"):
        EBenchLatentDataset("tiny-ebench", config, root=tmp_path)


class _TinyEBenchLeafDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return {
            "latents": torch.zeros(48, 2, 28, 28),
            "actions": torch.zeros(30, 2, 16, 1),
            "actions_mask": torch.ones(30, 2, 16, 1, dtype=torch.bool),
            "text": "Put the cup on the tray.",
        }

    def get_packing_metadata(self, index):
        assert index == 0
        return {
            "sample_key": "ebench|episode-000000",
            "frame_count": 2,
            "token_count": 8192,
            "shape_signature": (48, 28, 28, 30, 16, 1),
        }


def _multi_dataset_with_tiny_ebench_leaf():
    dataset = object.__new__(MultiLatentLeRobotDataset)
    dataset._datasets = [_TinyEBenchLeafDataset()]
    dataset.item_id_to_dataset_id = {0: 0}
    dataset.acc_dset_num = {0: 0}
    dataset._text_to_id = None
    dataset._packing_manifest = None
    return dataset


def test_ebench_sample_shape_shared_text_lookup_and_packed_reference_metadata():
    dataset = _multi_dataset_with_tiny_ebench_leaf()
    dataset.set_text_lookup(["", "Put the cup on the tray."])
    packed_ref = PackedSampleRef(
        index=0,
        sample_uid=41,
        update_id=7,
        micro_idx=1,
        micro_count=3,
        planned_tokens=8192,
    )

    sample = dataset[packed_ref]

    assert sample["latents"].shape == (48, 2, 28, 28)
    assert sample["actions"].shape == (30, 2, 16, 1)
    assert sample["actions_mask"].shape == (30, 2, 16, 1)
    assert sample["text_id"] == 1
    assert "text" not in sample
    for name in (
        "sample_uid",
        "update_id",
        "micro_idx",
        "micro_count",
        "planned_tokens",
    ):
        assert sample[name] == getattr(packed_ref, name)

    packing = dataset.get_packing_manifest()
    assert len(packing) == 1
    assert packing[0].frame_count == 2
    assert packing[0].token_count == 8192
    assert packing[0].shape_signature == (48, 28, 28, 30, 16, 1)
