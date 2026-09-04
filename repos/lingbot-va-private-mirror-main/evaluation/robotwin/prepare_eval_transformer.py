#!/usr/bin/env python3
"""Create a read-only-friendly transformer view with an evaluation attention mode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def find_transformer_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    transformer = path / "transformer"
    if (transformer / "config.json").is_file():
        return transformer.resolve()
    raise FileNotFoundError(
        f"Expected config.json in {path} or {transformer}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an eval transformer directory whose weights are symlinked "
            "to a training checkpoint and whose config uses torch/flashattn."
        )
    )
    parser.add_argument("source", type=Path, help="Checkpoint or transformer directory")
    parser.add_argument("output", type=Path, help="Output transformer directory")
    parser.add_argument(
        "--attn-mode",
        choices=("torch", "flashattn"),
        default="torch",
    )
    args = parser.parse_args()

    source = find_transformer_dir(args.source)
    output = args.output.expanduser().resolve()
    if output == source:
        raise ValueError("Output must differ from the training checkpoint")
    output.mkdir(parents=True, exist_ok=True)

    with (source / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    training_attn_mode = config.get("attn_mode")
    config["attn_mode"] = args.attn_mode

    config_tmp = output / ".config.json.tmp"
    with config_tmp.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(config_tmp, output / "config.json")

    weight_files = 0
    for source_file in sorted(source.iterdir()):
        if source_file.name == "config.json" or not source_file.is_file():
            continue
        destination = output / source_file.name
        if destination.is_symlink() and destination.resolve() == source_file.resolve():
            pass
        elif destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to replace existing non-matching file: {destination}"
            )
        else:
            destination.symlink_to(source_file.resolve())
        if source_file.suffix == ".safetensors":
            weight_files += 1

    if not weight_files:
        raise FileNotFoundError(f"No .safetensors weights found in {source}")

    print(f"source={source}")
    print(f"output={output}")
    print(f"attn_mode={training_attn_mode!r} -> {args.attn_mode!r}")
    print(f"safetensors_files={weight_files}")


if __name__ == "__main__":
    main()
