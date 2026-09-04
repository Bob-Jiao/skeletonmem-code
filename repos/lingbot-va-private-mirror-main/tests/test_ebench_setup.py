import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import script.prepare_ebench_lingbot as ebench_preprocess
from script.prepare_ebench_lingbot import (
    AUDIT_REPORT_NAME,
    AUDIT_SUCCESS_MARKER_NAME,
    DATA_MANIFEST_NAME,
    LATENT_MANIFEST_NAME,
    MODEL_CHANNELS,
    PREPROCESS_MANIFEST_NAME,
    SOURCE_CAMERA_KEYS,
    STATS_NAME,
    TEXT_CACHE_NAME,
    EMPTY_EMBEDDING_NAME,
    ZERO_PREFIX_ROWS,
    audit_dataset,
    encode_dataset,
    encode_text_cache,
    setup_dataset,
    transform_ebench_actions,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_tiny_source(root):
    video_feature = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "info": {"video.height": 480, "video.width": 640, "video.fps": 15},
    }
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "total_frames": 33,
        "total_tasks": 1,
        "total_videos": 3,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 15,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "video.left_camera_view": video_feature,
            "video.right_camera_view": video_feature,
            "video.overlook_camera_view": video_feature,
        },
    }
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {
                "episode_index": 0,
                "tasks": ["Put the cup on the tray."],
                "length": 33,
            }
        ],
    )
    _write_jsonl(
        meta / "tasks.jsonl",
        [{"task_index": 0, "task": "Put the cup on the tray."}],
    )
    _write_jsonl(meta / "episodes_stats.jsonl", [{"episode_index": 0}])

    time = np.arange(33, dtype=np.float32)[:, None]
    joints = 100.0 + time * np.arange(1, 13, dtype=np.float32)[None]
    gripper = 10.0 + time * np.arange(1, 5, dtype=np.float32)[None]
    base_delta = np.tile(np.array([[1.0, -2.0, 50.0]], np.float32), (33, 1))
    # An extreme incomplete tail catches accidental inclusion in quantiles.
    joints[-1] = 1_000_000
    gripper[-1] = 1_000_000
    base_delta[-1] = 1_000_000
    table = pa.table(
        {
            "action.joints": pa.array(
                joints.tolist(), type=pa.list_(pa.float32(), 12)
            ),
            "action.gripper": pa.array(
                gripper.tolist(), type=pa.list_(pa.float32(), 4)
            ),
            "action.base_delta": pa.array(
                base_delta.tolist(), type=pa.list_(pa.float32(), 3)
            ),
        }
    )
    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(table, parquet)
    for camera_key in SOURCE_CAMERA_KEYS:
        video_path = (
            root
            / "videos"
            / "chunk-000"
            / camera_key
            / "episode_000000.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()
    return joints, gripper, base_delta, parquet


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_setup_derives_metadata_sidecar_and_stats_without_touching_source(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    joints, gripper, base_delta, parquet = _write_tiny_source(source)
    source_sha = _sha256(parquet)
    args = SimpleNamespace(
        source_root=source,
        output_root=output,
        model_root=tmp_path / "model",
        max_episodes=None,
        overwrite_actions=False,
        skip_text=True,
        overwrite_text=False,
        device="cpu",
    )

    setup_dataset(args)

    assert _sha256(parquet) == source_sha
    assert (output / "data").is_symlink()
    assert (output / "videos").is_symlink()
    derived_info = json.loads((output / "meta" / "info.json").read_text())
    assert derived_info["total_videos"] == 4
    assert (
        derived_info["features"]["video.top_camera_view"]
        == derived_info["features"]["video.overlook_camera_view"]
    )
    episode = json.loads(
        (output / "meta" / "episodes.jsonl").read_text().strip()
    )
    assert episode["action_config"] == [
        {
            "start_frame": 0,
            "end_frame": 33,
            "action_text": "Put the cup on the tray.",
            "skill": "",
        }
    ]

    transformed = transform_ebench_actions(joints, gripper, base_delta)
    expected_real = transformed[:32]
    sidecar_path = output / "actions" / "chunk-000" / "episode_000000.npz"
    with np.load(sidecar_path, allow_pickle=False) as sidecar:
        assert sidecar["action_19d"].shape == (48, 19)
        np.testing.assert_array_equal(sidecar["action_19d"][:16], 0)
        np.testing.assert_array_equal(sidecar["action_19d"][16:], expected_real)
        np.testing.assert_array_equal(sidecar["loss_mask_19d"], True)
        np.testing.assert_array_equal(sidecar["episode_anchor_12d"], joints[0])
        assert int(sidecar["real_action_count"]) == 32
        assert int(sidecar["tail_discarded_count"]) == 1
        assert np.count_nonzero(np.all(sidecar["action_19d"] == 0, axis=1)) == 16

    stats = json.loads((output / STATS_NAME).read_text())
    expected_q01, expected_q99 = np.quantile(expected_real, [0.01, 0.99], axis=0)
    np.testing.assert_allclose(stats["quantiles"]["observed_q01_19d"], expected_q01)
    np.testing.assert_allclose(stats["quantiles"]["observed_q99_19d"], expected_q99)
    assert stats["quantiles"]["frame_count"] == 32
    assert stats["quantiles"]["effective_q01_19d"][18] == -1
    assert stats["quantiles"]["effective_q99_19d"][18] == 1
    assert stats["norm_stat"]["q01"][2] == -1
    assert stats["norm_stat"]["q99"][2] == 1
    unused = np.setdiff1d(np.arange(30), MODEL_CHANNELS)
    np.testing.assert_array_equal(np.asarray(stats["norm_stat"]["q01"])[unused], -1)
    np.testing.assert_array_equal(np.asarray(stats["norm_stat"]["q99"])[unused], 1)

    data_manifest_sha = _sha256(output / DATA_MANIFEST_NAME)
    assert stats["manifests"]["data_manifest_sha256"] == data_manifest_sha
    manifest = json.loads((output / PREPROCESS_MANIFEST_NAME).read_text())
    assert manifest["expected"] == {
        "real_action_rows": 32,
        "discarded_tail_rows": 1,
        "sampled_rgb_frames": 9,
        "latent_frames_per_camera": 3,
        "latent_files": 4,
    }

    # Setup is resumable: a valid action sidecar is reused byte-for-byte.
    sidecar_sha = _sha256(sidecar_path)

    def unexpected_source_read(*_args, **_kwargs):
        raise AssertionError("resume unexpectedly reread source action columns")

    monkeypatch.setattr(
        ebench_preprocess,
        "_read_action_columns",
        unexpected_source_read,
    )
    setup_dataset(args)
    assert _sha256(sidecar_path) == sidecar_sha

    # Any setup pass rewrites derived metadata/manifests and therefore revokes
    # a prior audit gate before those writes begin.
    success_marker = output / AUDIT_SUCCESS_MARKER_NAME
    success_marker.write_text("stale", encoding="utf-8")
    setup_dataset(args)
    assert not success_marker.exists()


def test_setup_rebuilds_schema_valid_sidecar_when_previous_sha_changes(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    joints, _, _, parquet = _write_tiny_source(source)
    args = SimpleNamespace(
        source_root=source,
        output_root=output,
        model_root=tmp_path / "model",
        max_episodes=None,
        overwrite_actions=False,
        skip_text=True,
        overwrite_text=False,
        device="cpu",
    )
    setup_dataset(args)

    sidecar_path = output / "actions" / "chunk-000" / "episode_000000.npz"
    with np.load(sidecar_path, allow_pickle=False) as sidecar:
        corrupted = np.asarray(sidecar["action_19d"], dtype=np.float32).copy()
    # Keep every schema/provenance field self-consistent while changing a
    # middle retained action.  Only the previous manifest SHA catches this.
    corrupted[ZERO_PREFIX_ROWS + 7, 0] += 123.0
    ebench_preprocess._write_action_sidecar(
        sidecar_path,
        corrupted,
        joints[0],
        raw_length=33,
        source_sha256=_sha256(parquet),
    )

    source_reads = 0
    original_read = ebench_preprocess._read_action_columns

    def counted_source_read(*call_args, **call_kwargs):
        nonlocal source_reads
        source_reads += 1
        return original_read(*call_args, **call_kwargs)

    monkeypatch.setattr(
        ebench_preprocess, "_read_action_columns", counted_source_read
    )
    setup_dataset(args)

    assert source_reads == 1
    expected = transform_ebench_actions(*original_read(parquet))[:32]
    with np.load(sidecar_path, allow_pickle=False) as repaired:
        np.testing.assert_array_equal(
            repaired["action_19d"][ZERO_PREFIX_ROWS:], expected
        )


def test_text_cache_resume_rejects_changed_bytes_and_revokes_audit_gate(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    _write_tiny_source(source)
    model_root = tmp_path / "model"
    setup_args = SimpleNamespace(
        source_root=source,
        output_root=output,
        model_root=model_root,
        max_episodes=None,
        overwrite_actions=False,
        skip_text=True,
        overwrite_text=False,
        device="cpu",
    )
    setup_dataset(setup_args)

    embeddings = {
        "": torch.zeros(2, 3, dtype=torch.bfloat16),
        "Put the cup on the tray.": torch.ones(2, 3, dtype=torch.bfloat16),
    }
    text_path = output / TEXT_CACHE_NAME
    empty_path = output / EMPTY_EMBEDDING_NAME
    torch.save(embeddings, text_path)
    torch.save(embeddings[""], empty_path)
    manifest_path = output / PREPROCESS_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["text_cache"] = {
        "path": TEXT_CACHE_NAME,
        "sha256": _sha256(text_path),
        "empty_path": EMPTY_EMBEDDING_NAME,
        "empty_sha256": _sha256(empty_path),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker_path = output / AUDIT_SUCCESS_MARKER_NAME
    marker_path.write_text("previous audit gate", encoding="utf-8")

    replacement = {key: value + 1 for key, value in embeddings.items()}
    torch.save(replacement, text_path)
    torch.save(replacement[""], empty_path)
    with pytest.raises(ValueError, match="bytes differ"):
        encode_text_cache(
            SimpleNamespace(
                output_root=output,
                model_root=model_root,
                overwrite_text=False,
                device="cpu",
            )
        )
    assert not marker_path.exists()


def test_setup_rejects_output_nested_inside_source(tmp_path):
    source = tmp_path / "source"
    _write_tiny_source(source)
    nested_output = source / "derived"
    args = SimpleNamespace(
        source_root=source,
        output_root=nested_output,
        model_root=tmp_path / "model",
        max_episodes=None,
        overwrite_actions=False,
        skip_text=True,
        overwrite_text=False,
        device="cpu",
    )

    with pytest.raises(ValueError, match="non-nested"):
        setup_dataset(args)
    assert not nested_output.exists()


class _FakeVAE:
    def __init__(self):
        self.config = SimpleNamespace(
            latents_mean=[0.0] * 48,
            latents_std=[1.0] * 48,
        )

    def eval(self):
        return self


class _FakeStreamingVAE:
    def __init__(self, _vae):
        self.clear_count = 0
        self.input_shapes = []

    def clear_cache(self):
        self.clear_count += 1

    def encode_chunk(self, video):
        self.input_shapes.append(tuple(video.shape))
        batch, _, sampled_frames, _, _ = video.shape
        latent_frames = 1 if sampled_frames == 1 else sampled_frames // 4
        mean = torch.stack(
            [
                torch.full(
                    (48, latent_frames, 14, 14),
                    float(camera_index + 1),
                    dtype=video.dtype,
                    device=video.device,
                )
                for camera_index in range(batch)
            ]
        )
        return torch.cat((mean, torch.zeros_like(mean)), dim=1)


def test_fake_encode_resume_and_full_audit(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    _write_tiny_source(source)
    setup_args = SimpleNamespace(
        source_root=source,
        output_root=output,
        model_root=tmp_path / "model",
        max_episodes=None,
        overwrite_actions=False,
        skip_text=True,
        overwrite_text=False,
        device="cpu",
    )
    setup_dataset(setup_args)

    import wan_va.modules.utils as module_utils

    streaming_instances = []

    def make_streaming(vae):
        streaming = _FakeStreamingVAE(vae)
        streaming_instances.append(streaming)
        return streaming

    decode_calls = []

    def fake_decode(path, frame_ids, **kwargs):
        decode_calls.append((path, np.asarray(frame_ids).copy(), kwargs))
        assert kwargs["expected_source_frame_count"] == 33
        return np.zeros((len(frame_ids), 2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(module_utils, "load_vae", lambda *args, **kwargs: _FakeVAE())
    monkeypatch.setattr(module_utils, "WanVAEStreamingWrapper", make_streaming)
    monkeypatch.setattr(
        ebench_preprocess, "decode_sampled_rgb_frames", fake_decode
    )
    encode_args = SimpleNamespace(
        output_root=output,
        model_root=tmp_path / "model",
        device="cpu",
        num_shards=1,
        shard_index=0,
        episode_indices=None,
        temporal_chunk_size=4,
        overwrite=False,
    )

    encode_dataset(encode_args)

    assert len(decode_calls) == 4
    assert streaming_instances[0].clear_count == 1
    assert streaming_instances[0].input_shapes == [
        (4, 3, 1, 2, 2),
        (4, 3, 4, 2, 2),
        (4, 3, 4, 2, 2),
    ]
    expected_ids = np.arange(0, 33, 4)
    for camera_key in SOURCE_CAMERA_KEYS:
        latent_path = (
            output
            / "latents"
            / "chunk-000"
            / camera_key
            / "episode_000000_0_33.pth"
        )
        record = torch.load(latent_path, weights_only=False)
        assert record["latent"].shape == (3 * 14 * 14, 48)
        assert record["latent"].dtype == torch.bfloat16
        np.testing.assert_array_equal(record["frame_ids"], expected_ids)
        assert "text_emb" not in record

    # A second sharded encode validates all four files and performs no decode
    # or VAE cache operation.
    encode_dataset(encode_args)
    assert len(decode_calls) == 4
    assert streaming_instances[1].clear_count == 0

    embeddings = {
        "": torch.zeros(512, 4096, dtype=torch.bfloat16),
        "Put the cup on the tray.": torch.ones(512, 4096, dtype=torch.bfloat16),
    }
    torch.save(embeddings, output / TEXT_CACHE_NAME)
    torch.save(embeddings[""], output / EMPTY_EMBEDDING_NAME)
    preprocess_manifest_path = output / PREPROCESS_MANIFEST_NAME
    preprocess_manifest = json.loads(preprocess_manifest_path.read_text())
    preprocess_manifest["artifacts"]["text_cache"] = {
        "path": TEXT_CACHE_NAME,
        "sha256": _sha256(output / TEXT_CACHE_NAME),
        "empty_path": EMPTY_EMBEDDING_NAME,
        "empty_sha256": _sha256(output / EMPTY_EMBEDDING_NAME),
    }
    preprocess_manifest_path.write_text(
        json.dumps(preprocess_manifest, indent=2) + "\n", encoding="utf-8"
    )
    audit_dataset(
        SimpleNamespace(
            output_root=output,
            report_path=None,
            episode_indices=None,
        )
    )

    report = json.loads((output / AUDIT_REPORT_NAME).read_text())
    assert report["passed"] is True
    assert report["complete_dataset"] is True
    assert report["episodes_audited"] == 1
    assert report["latent_files"] == {"actual": 4, "expected": 4}
    latent_manifest_sha = _sha256(output / LATENT_MANIFEST_NAME)
    stats = json.loads((output / STATS_NAME).read_text())
    assert stats["manifests"]["latent_manifest_sha256"] == latent_manifest_sha
    success_marker = output / AUDIT_SUCCESS_MARKER_NAME
    assert success_marker.is_file()

    from wan_va.dataset.ebench_latent_dataset import EBenchLatentDataset

    loader_config = SimpleNamespace(
        obs_cam_keys=SOURCE_CAMERA_KEYS,
        used_action_channel_ids=tuple(MODEL_CHANNELS),
        action_dim=30,
        action_per_frame=16,
        patch_size=(1, 2, 2),
        action_stats_path=output / STATS_NAME,
    )
    assert len(EBenchLatentDataset("tiny-ebench", loader_config, root=output)) == 1

    # A no-op resume preserves the marker, but an actual overwrite revokes it
    # before writing and the loader must refuse the now unaudited dataset.
    marker_bytes = success_marker.read_bytes()
    encode_dataset(encode_args)
    assert success_marker.read_bytes() == marker_bytes
    encode_args.overwrite = True
    encode_dataset(encode_args)
    assert not success_marker.exists()
    with pytest.raises(FileNotFoundError, match="full-audit success marker"):
        EBenchLatentDataset("tiny-ebench", loader_config, root=output)

    # A new successful full audit atomically restores the gate. A partial audit
    # is diagnostic only and leaves the current success marker unchanged.
    encode_args.overwrite = False
    audit_dataset(
        SimpleNamespace(output_root=output, report_path=None, episode_indices=None)
    )
    assert success_marker.is_file()
    marker_bytes = success_marker.read_bytes()
    audit_dataset(
        SimpleNamespace(
            output_root=output,
            report_path=output / "partial_audit.json",
            episode_indices=[0],
        )
    )
    assert success_marker.read_bytes() == marker_bytes

    # Same keys/schema/content relationship is insufficient: replacing both
    # shared files must still fail their setup-recorded SHA provenance check.
    replacement = {
        key: value + torch.ones_like(value) for key, value in embeddings.items()
    }
    torch.save(replacement, output / TEXT_CACHE_NAME)
    torch.save(replacement[""], output / EMPTY_EMBEDDING_NAME)
    with pytest.raises(RuntimeError, match="audit found"):
        audit_dataset(
            SimpleNamespace(
                output_root=output,
                report_path=None,
                episode_indices=None,
            )
        )
    assert not success_marker.exists()
    failed_report = json.loads((output / AUDIT_REPORT_NAME).read_text())
    assert any("text cache SHA mismatch" in error for error in failed_report["errors"])
    assert any(
        "empty embedding SHA mismatch" in error
        for error in failed_report["errors"]
    )
