import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from wan_va.dataset.packing import stable_hash
from wan_va.train import (
    Trainer,
    _dataset_fingerprint,
    _sample_nonpacked_attention_layout,
)


def test_non_ebench_dataset_fingerprint_keeps_the_legacy_payload(tmp_path):
    config = SimpleNamespace(
        dataset_path=tmp_path,
        obs_cam_keys=["video.image"],
        env_type="robotwin_tshape",
        used_action_channel_ids=[0, 1],
        norm_stat={"q01": [0.0], "q99": [1.0]},
    )

    class LegacyDataset:
        packing_manifest_hash = None

        def __len__(self):
            return 7

    expected = stable_hash(
        {
            "dataset_path": str(tmp_path.resolve()),
            "length": 7,
            "obs_cam_keys": ["video.image"],
            "env_type": "robotwin_tshape",
            "used_action_channel_ids": [0, 1],
            "norm_stat": {"q01": [0.0], "q99": [1.0]},
            "packing_manifest_hash": None,
        }
    )

    assert _dataset_fingerprint(config, LegacyDataset()) == expected


def test_checkpoint_copies_the_exact_versioned_action_stats(tmp_path):
    stats_path = tmp_path / "action_stats_v1.json"
    stats_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "representation": "ebench_episode_relative_19d",
                "norm_stat": {"q01": [-1.0] * 30, "q99": [1.0] * 30},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    expected_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    trainer = object.__new__(Trainer)
    trainer.action_stats_path = stats_path
    trainer.action_stats_sha256 = expected_sha
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    trainer._save_action_stats_artifact(checkpoint)

    copied = checkpoint / stats_path.name
    assert copied.read_bytes() == stats_path.read_bytes()
    manifest = json.loads((checkpoint / "dataset_artifacts.json").read_text())
    assert manifest["filename"] == "action_stats_v1.json"
    assert manifest["sha256"] == expected_sha

    stats_path.write_text("changed after dataset initialization", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        trainer._save_action_stats_artifact(checkpoint)


def test_ebench_attention_layout_is_broadcast_from_rank_zero(monkeypatch):
    shared = {}
    current_rank = {"value": 0}
    randint_values = iter((torch.tensor([3]), torch.tensor([27])))

    monkeypatch.setattr("wan_va.train.dist.is_initialized", lambda: True)
    monkeypatch.setattr("wan_va.train.dist.get_world_size", lambda: 2)
    monkeypatch.setattr(
        "wan_va.train.dist.get_rank", lambda: current_rank["value"]
    )

    def fake_broadcast(values, src):
        assert src == 0
        if current_rank["value"] == src:
            shared["values"] = values.clone()
        else:
            values.copy_(shared["values"])

    monkeypatch.setattr("wan_va.train.dist.broadcast", fake_broadcast)
    monkeypatch.setattr(
        "wan_va.train.torch.randint", lambda *args, **kwargs: next(randint_values)
    )

    rank_zero = _sample_nonpacked_attention_layout(
        "cpu", sync_across_ranks=True
    )
    current_rank["value"] = 1
    rank_one = _sample_nonpacked_attention_layout(
        "cpu", sync_across_ranks=True
    )

    assert rank_zero == (3, 27)
    assert rank_one == rank_zero


def test_legacy_attention_layout_does_not_touch_distributed(monkeypatch):
    randint_values = iter((torch.tensor([2]), torch.tensor([41])))
    monkeypatch.setattr(
        "wan_va.train.dist.is_initialized",
        lambda: (_ for _ in ()).throw(AssertionError("dist should not be queried")),
    )
    monkeypatch.setattr(
        "wan_va.train.torch.randint", lambda *args, **kwargs: next(randint_values)
    )

    assert _sample_nonpacked_attention_layout(
        "cpu", sync_across_ranks=False
    ) == (2, 41)
