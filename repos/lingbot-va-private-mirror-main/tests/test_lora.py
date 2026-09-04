from __future__ import annotations

import torch

from wan_va.modules.lora import (
    LORA_ACTION_MODULES_TO_SAVE,
    LORA_TARGETS_PER_BLOCK,
    add_lora_adapter,
    get_adapter_state_dict,
    load_lora_adapter,
    lora_optimizer_param_groups,
    save_lora_adapter,
    validate_lora_model,
)
from wan_va.modules.model import WanTransformer3DModel


def tiny_model() -> WanTransformer3DModel:
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=4,
        out_channels=4,
        action_dim=6,
        text_dim=32,
        freq_dim=8,
        ffn_dim=48,
        num_layers=2,
        rope_max_seq_len=32,
        attn_mode="torch",
    )


def test_lora_targets_and_freezing():
    model = tiny_model()
    stats = add_lora_adapter(model, rank=4, alpha=4, validate_percentage=False)

    assert stats.target_modules == len(model.blocks) * LORA_TARGETS_PER_BLOCK
    assert model.peft_config["default"].r == 4
    assert model.peft_config["default"].lora_alpha == 4
    assert model.peft_config["default"].lora_dropout == 0.0
    assert model.peft_config["default"].bias == "none"

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names
    assert all("lora_A." in name or "lora_B." in name for name in trainable_names)
    assert all(
        torch.count_nonzero(parameter) == 0
        for name, parameter in model.named_parameters()
        if "lora_B." in name
    )

    validated = validate_lora_model(
        model,
        min_trainable_percent=0.0,
        max_trainable_percent=100.0,
    )
    assert validated.trainable_params == stats.trainable_params


def test_lora_is_initially_a_noop():
    torch.manual_seed(1)
    model = tiny_model()
    add_lora_adapter(model, rank=4, alpha=4, validate_percentage=False)
    layer = model.blocks[0].attn1.to_q
    inputs = torch.randn(2, 3, layer.get_base_layer().in_features)

    with_adapter = layer(inputs)
    model.disable_lora()
    without_adapter = layer(inputs)
    model.enable_lora()
    torch.testing.assert_close(with_adapter, without_adapter)


def test_adapter_round_trip(tmp_path):
    torch.manual_seed(7)
    source = tiny_model()
    legacy_state = {
        name: tensor.detach().clone() for name, tensor in source.state_dict().items()
    }
    add_lora_adapter(source, rank=4, alpha=4, validate_percentage=False)
    for name, parameter in source.named_parameters():
        if "lora_" in name:
            torch.nn.init.normal_(parameter)

    adapter_state = get_adapter_state_dict(source, source.state_dict())
    adapter_dir = tmp_path / "adapter"
    save_lora_adapter(source, adapter_state, adapter_dir)

    restored = tiny_model()
    restored.load_state_dict(legacy_state, strict=True)
    load_lora_adapter(restored, adapter_dir)

    source_layer = source.blocks[0].attn1.to_q
    restored_layer = restored.blocks[0].attn1.to_q
    inputs = torch.randn(2, 3, source_layer.get_base_layer().in_features)
    torch.testing.assert_close(source_layer(inputs), restored_layer(inputs))
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "pytorch_lora_weights.safetensors").is_file()


def test_full_action_modules_are_trainable_grouped_and_saved(tmp_path):
    torch.manual_seed(11)
    source = tiny_model()
    legacy_state = {
        name: tensor.detach().clone() for name, tensor in source.state_dict().items()
    }
    add_lora_adapter(
        source,
        rank=4,
        alpha=4,
        modules_to_save=LORA_ACTION_MODULES_TO_SAVE,
        validate_percentage=False,
    )

    trainable_names = {
        name for name, parameter in source.named_parameters() if parameter.requires_grad
    }
    assert any("lora_A." in name for name in trainable_names)
    assert any(
        name.startswith("action_embedder.modules_to_save.default.")
        for name in trainable_names
    )
    assert any(
        name.startswith("condition_embedder_action.modules_to_save.default.")
        for name in trainable_names
    )
    assert any(
        name.startswith("action_proj_out.modules_to_save.default.")
        for name in trainable_names
    )
    assert all(
        "lora_" in name or ".modules_to_save." in name
        for name in trainable_names
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in source.named_parameters()
        if ".original_module." in name
    )

    groups = lora_optimizer_param_groups(
        source,
        lora_learning_rate=1e-4,
        lora_weight_decay=1e-2,
        action_learning_rate=1e-5,
        action_weight_decay=1e-1,
    )
    assert [group["group_name"] for group in groups] == ["lora", "action"]
    assert groups[0]["lr"] == 1e-4
    assert groups[0]["weight_decay"] == 1e-2
    assert groups[1]["lr"] == 1e-5
    assert groups[1]["weight_decay"] == 1e-1
    grouped_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {
        id(parameter)
        for parameter in source.parameters()
        if parameter.requires_grad
    }

    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in source.named_parameters()
        if parameter.requires_grad
    }
    frozen_action_before = (
        source.action_embedder.original_module.weight.detach().clone()
    )
    optimizer = torch.optim.AdamW(groups)
    action_inputs = torch.randn(2, 3, 6)
    attention_inputs = torch.randn(
        2,
        3,
        source.blocks[0].attn1.to_q.get_base_layer().in_features,
    )
    loss = (
        source.action_proj_out(source.action_embedder(action_inputs)).square().mean()
        + source.blocks[0].attn1.to_q(attention_inputs).square().mean()
    )
    loss.backward()
    optimizer.step()
    trainable_after = dict(source.named_parameters())
    assert any(
        not torch.equal(before, trainable_after[name])
        for name, before in trainable_before.items()
        if "lora_" in name
    )
    assert any(
        not torch.equal(before, trainable_after[name])
        for name, before in trainable_before.items()
        if ".modules_to_save." in name
    )
    torch.testing.assert_close(
        source.action_embedder.original_module.weight,
        frozen_action_before,
    )

    for name, parameter in source.named_parameters():
        if parameter.requires_grad:
            torch.nn.init.normal_(parameter)
    adapter_state = get_adapter_state_dict(source, source.state_dict())
    adapter_dir = tmp_path / "adapter_with_action"
    save_lora_adapter(source, adapter_state, adapter_dir)

    restored = tiny_model()
    restored.load_state_dict(legacy_state, strict=True)
    load_lora_adapter(restored, adapter_dir)

    action_inputs = torch.randn(2, 3, 6)
    text_inputs = torch.randn(2, 3, 32)
    torch.testing.assert_close(
        source.action_embedder(action_inputs),
        restored.action_embedder(action_inputs),
    )
    torch.testing.assert_close(
        source.condition_embedder_action.text_embedder(text_inputs),
        restored.condition_embedder_action.text_embedder(text_inputs),
    )
    torch.testing.assert_close(
        source.action_proj_out(source.action_embedder(action_inputs)),
        restored.action_proj_out(restored.action_embedder(action_inputs)),
    )


def test_legacy_model_has_no_peft_state_keys():
    model = tiny_model()
    assert all("lora_" not in key and ".base_layer." not in key for key in model.state_dict())
