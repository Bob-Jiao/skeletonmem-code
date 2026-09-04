"""LoRA helpers for the LingBot-VA transformer.

The adapter is injected in place so the custom transformer's public shape stays
unchanged for activation checkpointing and FSDP2.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors.torch import save_file


LORA_ADAPTER_NAME = "default"
LORA_TARGET_PATTERN = (
    r"^blocks\.\d+\."
    r"(?:attn[12]\.(?:to_q|to_k|to_v|to_out\.0)|"
    r"ffn\.net\.(?:0\.proj|2))$"
)
LORA_TARGETS_PER_BLOCK = 10
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
LORA_ACTION_MODULES_TO_SAVE = (
    "action_embedder",
    "condition_embedder_action",
    "action_proj_out",
)


@dataclass(frozen=True)
class LoraStats:
    target_modules: int
    trainable_params: int
    base_params: int
    total_params: int
    trainable_percent: float

    def to_dict(self) -> dict:
        return asdict(self)


def _is_lora_parameter(name: str) -> bool:
    return "lora_A." in name or "lora_B." in name


def _is_module_to_save_parameter(name: str) -> bool:
    return ".modules_to_save." in name


def _is_adapter_trainable_parameter(name: str) -> bool:
    return _is_lora_parameter(name) or _is_module_to_save_parameter(name)


def _set_only_adapter_trainable(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_adapter_trainable_parameter(name))


def lora_optimizer_param_groups(
    model: torch.nn.Module,
    *,
    lora_learning_rate: float,
    lora_weight_decay: float,
    action_learning_rate: float,
    action_weight_decay: float,
) -> list[dict]:
    """Split trainable adapter and full action-module parameters."""
    lora_parameters = []
    action_parameters = []
    unexpected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_lora_parameter(name):
            lora_parameters.append(parameter)
        elif _is_module_to_save_parameter(name):
            action_parameters.append(parameter)
        else:
            unexpected.append(name)

    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise ValueError(f"Unexpected trainable parameters in LoRA mode: {preview}")
    if not lora_parameters:
        raise ValueError("LoRA mode has no trainable LoRA parameters")

    groups = [
        {
            "params": lora_parameters,
            "lr": lora_learning_rate,
            "weight_decay": lora_weight_decay,
            "group_name": "lora",
        }
    ]
    if action_parameters:
        groups.append(
            {
                "params": action_parameters,
                "lr": action_learning_rate,
                "weight_decay": action_weight_decay,
                "group_name": "action",
            }
        )
    return groups


def _target_module_names(model: torch.nn.Module) -> list[str]:
    import re

    pattern = re.compile(LORA_TARGET_PATTERN)
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and pattern.fullmatch(name)
    ]


def _adapter_target_count(model: torch.nn.Module) -> int:
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError as exc:
        raise ImportError(
            "LoRA training requires PEFT. Install the project training dependencies."
        ) from exc

    return sum(1 for module in model.modules() if isinstance(module, BaseTunerLayer))


def lora_stats(model: torch.nn.Module, target_modules: int | None = None) -> LoraStats:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    base = total - trainable
    percent = 100.0 * trainable / total if total else 0.0
    return LoraStats(
        target_modules=_adapter_target_count(model) if target_modules is None else target_modules,
        trainable_params=trainable,
        base_params=base,
        total_params=total,
        trainable_percent=percent,
    )


def validate_lora_model(
    model: torch.nn.Module,
    *,
    min_trainable_percent: float = 2.0,
    max_trainable_percent: float = 6.0,
) -> LoraStats:
    expected_targets = len(model.blocks) * LORA_TARGETS_PER_BLOCK
    stats = lora_stats(model)
    if stats.target_modules != expected_targets:
        raise ValueError(
            "LoRA target mismatch: "
            f"expected {expected_targets}, found {stats.target_modules}."
        )

    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not _is_adapter_trainable_parameter(name)
    ]
    if unexpected_trainable:
        preview = ", ".join(unexpected_trainable[:5])
        raise ValueError(f"Non-adapter parameters are trainable: {preview}")

    if not min_trainable_percent <= stats.trainable_percent <= max_trainable_percent:
        raise ValueError(
            "LoRA trainable parameter percentage is outside the configured range: "
            f"{stats.trainable_percent:.4f}% not in "
            f"[{min_trainable_percent:.4f}%, {max_trainable_percent:.4f}%]."
        )
    return stats


def add_lora_adapter(
    model: torch.nn.Module,
    *,
    rank: int = 64,
    alpha: int | None = None,
    dropout: float = 0.0,
    modules_to_save: Sequence[str] | None = None,
    validate_percentage: bool = True,
) -> LoraStats:
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError(
            "LoRA training requires PEFT. Install the project training dependencies."
        ) from exc

    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}.")
    alpha = rank if alpha is None else alpha
    if alpha != rank:
        raise ValueError(f"LoRA alpha must equal rank, got alpha={alpha}, rank={rank}.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}.")

    target_names = _target_module_names(model)
    expected_targets = len(model.blocks) * LORA_TARGETS_PER_BLOCK
    if len(target_names) != expected_targets:
        raise ValueError(
            "LoRA target discovery failed before injection: "
            f"expected {expected_targets}, found {len(target_names)}."
        )

    model.requires_grad_(False)
    model.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            init_lora_weights="gaussian",
            target_modules=LORA_TARGET_PATTERN,
            modules_to_save=list(modules_to_save) if modules_to_save else None,
        ),
        adapter_name=LORA_ADAPTER_NAME,
    )
    _set_only_adapter_trainable(model)
    if validate_percentage:
        return validate_lora_model(model)
    return lora_stats(model)


def load_lora_adapter(
    model: torch.nn.Module,
    adapter_path: str | Path,
    *,
    is_trainable: bool = False,
    validate_percentage: bool = False,
) -> LoraStats:
    adapter_path = Path(adapter_path)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter_path}")
    model.requires_grad_(False)
    model.load_lora_adapter(
        adapter_path,
        prefix=None,
        adapter_name=LORA_ADAPTER_NAME,
        weight_name=LORA_WEIGHT_NAME,
        use_safetensors=True,
    )
    if is_trainable:
        _set_only_adapter_trainable(model)
    else:
        model.requires_grad_(False)
    if validate_percentage:
        return validate_lora_model(model)
    return lora_stats(model)


def get_adapter_state_dict(
    model: torch.nn.Module,
    full_model_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    try:
        from peft.utils import get_peft_model_state_dict
    except ImportError as exc:
        raise ImportError(
            "Saving a LoRA adapter requires PEFT."
        ) from exc

    return get_peft_model_state_dict(
        model,
        state_dict=dict(full_model_state_dict),
        adapter_name=LORA_ADAPTER_NAME,
        save_embedding_layers=False,
    )


def save_lora_adapter(
    model: torch.nn.Module,
    adapter_state_dict: Mapping[str, torch.Tensor],
    save_directory: str | Path,
) -> None:
    try:
        from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY
    except ImportError:
        LORA_ADAPTER_METADATA_KEY = "lora_adapter_metadata"

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    config = model.peft_config[LORA_ADAPTER_NAME]
    config.save_pretrained(save_directory)

    config_dict = config.to_dict()
    for key, value in tuple(config_dict.items()):
        if isinstance(value, set):
            config_dict[key] = sorted(value)
    metadata = {
        "format": "pt",
        LORA_ADAPTER_METADATA_KEY: json.dumps(config_dict, indent=2, sort_keys=True),
    }
    weights = {name: tensor.contiguous().cpu() for name, tensor in adapter_state_dict.items()}
    save_file(weights, save_directory / LORA_WEIGHT_NAME, metadata=metadata)
