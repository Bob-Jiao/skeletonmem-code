from wan_va.configs import VA_CONFIGS


def test_ebench_protocol_configuration_is_registered():
    config = VA_CONFIGS["ebench"]

    assert config.dataset_type == "ebench"
    assert config.frame_chunk_size == 2
    assert config.action_per_frame == 16
    assert config.action_dim == 30
    assert (config.height, config.width) == (224, 224)
    assert config.attn_window == 72
    assert config.obs_cam_keys == [
        "video.top_camera_view",
        "video.overlook_camera_view",
        "video.left_camera_view",
        "video.right_camera_view",
    ]
    assert config.used_action_channel_ids == [
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
    ]


def test_ebench_full_training_keeps_robotwin_optimizer_recipe():
    config = VA_CONFIGS["ebench_train"]
    robotwin = VA_CONFIGS["robotwin_train"]

    assert config.training_mode == "full"
    assert config.shared_text_embeddings is True
    assert config.pin_memory is True
    assert config.persistent_workers is True
    assert config.prefetch_factor == 2
    assert config.packing_enabled is True
    assert config.global_episodes_per_update == 128
    assert config.max_self_tokens == 81_920
    assert config.max_episodes_per_pack == 4
    assert config.segment_manifest_path.endswith("segment_manifest_81920_v1.json")
    assert config.action_stats_path.endswith("action_stats_segment_81920_v1.json")
    assert config.segment_audit_path.endswith("segment_audit_success_81920.json")
    assert config.packing_seed == 42
    assert config.sync_nonpacked_attention_layout_across_ranks is True
    assert config.num_steps == 50_000
    for name in (
        "learning_rate",
        "beta1",
        "beta2",
        "weight_decay",
        "warmup_steps",
        "batch_size",
        "gradient_accumulation_steps",
    ):
        assert config[name] == robotwin[name]
