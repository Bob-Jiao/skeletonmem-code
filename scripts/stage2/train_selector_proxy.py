#!/usr/bin/env python3
"""Train a lightweight future-action proxy on selector manifests.

Default LIBERO protocol uses the full official task data. The optional
holdout mode is retained only for historical diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DatasetArrays:
    features: dict[int, np.ndarray]
    targets: dict[int, np.ndarray]
    lengths: dict[int, int]


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_manifest_timesteps(manifest_path: Path, train_episodes: set[int], target_offset: int) -> list[tuple[int, int]]:
    timesteps: set[tuple[int, int]] = set()
    for item in read_jsonl(manifest_path):
        episode_index = int(item["episode_index"])
        if episode_index not in train_episodes:
            continue
        first = int(item["action_loss_first"])
        last = int(item["action_loss_last"])
        for timestep in range(first, last + 1):
            if timestep + target_offset <= last + target_offset:
                timesteps.add((episode_index, timestep))
    return sorted(timesteps)


def load_episode_arrays(dataset_root: Path, episode_indices: list[int], target_offset: int) -> DatasetArrays:
    features: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    lengths: dict[int, int] = {}
    for episode_index in episode_indices:
        path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        frame = pd.read_parquet(path, columns=["observation.state", "action", "frame_index"])
        state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(frame["action"].to_numpy()).astype(np.float32)
        length = len(frame)
        progress = (frame["frame_index"].to_numpy(dtype=np.float32) / max(1, length - 1))[:, None]
        feature = np.concatenate([state, progress], axis=1)
        valid = max(0, length - target_offset)
        features[episode_index] = feature[:valid]
        targets[episode_index] = action[target_offset:target_offset + valid]
        lengths[episode_index] = valid
    return DatasetArrays(features=features, targets=targets, lengths=lengths)


def gather_samples(arrays: DatasetArrays, timesteps: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for episode_index, timestep in timesteps:
        if timestep < arrays.lengths[episode_index]:
            xs.append(arrays.features[episode_index][timestep])
            ys.append(arrays.targets[episode_index][timestep])
    if not xs:
        raise ValueError("manifest produced zero valid samples")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def all_timesteps(arrays: DatasetArrays, episode_indices: list[int]) -> list[tuple[int, int]]:
    return [
        (episode_index, timestep)
        for episode_index in episode_indices
        for timestep in range(arrays.lengths[episode_index])
    ]


def normalize(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = np.maximum(train_x.std(axis=0, keepdims=True), 1e-4)
    y_mean = train_y.mean(axis=0, keepdims=True)
    y_std = np.maximum(train_y.std(axis=0, keepdims=True), 1e-4)
    return (
        (train_x - x_mean) / x_std,
        (train_y - y_mean) / y_std,
        (val_x - x_mean) / x_std,
        (val_y - y_mean) / y_std,
    )


def train_one(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_x, train_y, val_x, val_y = normalize(train_x, train_y, val_x, val_y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(train_x.shape[1], train_y.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_inputs = torch.from_numpy(val_x).to(device)
    val_targets = torch.from_numpy(val_y).to(device)

    history = []
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.inference_mode():
            val_predictions = model(val_inputs)
            val_loss = float(criterion(val_predictions, val_targets).detach().cpu())
        best_val = min(best_val, val_loss)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})

    with torch.inference_mode():
        val_predictions = model(val_inputs)
        per_sample = torch.mean((val_predictions - val_targets) ** 2, dim=1).detach().cpu().numpy()
    top_indices = np.argsort(per_sample)[-10:][::-1]

    return {
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "best_val_loss": best_val,
        "history": history,
        "device": str(device),
        "top_val_errors": [
            {"index": int(index), "mse": float(per_sample[index])}
            for index in top_indices
        ],
    }


def parse_selector_budget(path: Path) -> tuple[str, float]:
    stem = path.stem
    selector, budget_token = stem.rsplit("_b", 1)
    return selector, int(budget_token) / 100.0


def plot_results(rows: list[dict], output_path: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["selector"], []).append(row)
    plt.figure(figsize=(7, 4.5))
    for selector, items in sorted(grouped.items()):
        items = sorted(items, key=lambda row: float(row["unique_action_ratio"]))
        xs = [float(row["unique_action_ratio"]) for row in items]
        ys = [float(row["best_val_loss"]) for row in items]
        plt.plot(xs, ys, marker="o", label=selector)
    plt.xlabel("Unique action-label budget ratio")
    plt.ylabel("Best validation MSE, normalized future action")
    plt.title("Stage2 selector proxy loss")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--budget-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-offset", type=int, default=16)
    parser.add_argument("--train-episodes", type=int, default=50)
    parser.add_argument(
        "--eval-episodes",
        type=str,
        default="same",
        choices=["same", "heldout"],
        help="Use the same full episode pool for proxy evaluation by default; heldout is legacy diagnostic mode.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.budget_table.open() as handle:
        budget_rows = {(row["selector"], float(row["budget"])): row for row in csv.DictReader(handle)}

    episode_indices = list(range(50))
    train_episodes = set(range(args.train_episodes))
    if args.eval_episodes == "heldout":
        val_episodes = list(range(args.train_episodes, 50))
        if not val_episodes:
            raise ValueError("--eval-episodes heldout requires --train-episodes < 50")
    else:
        val_episodes = list(range(50))
    arrays = load_episode_arrays(args.dataset_root, episode_indices, args.target_offset)
    val_timesteps = all_timesteps(arrays, val_episodes)
    val_x, val_y = gather_samples(arrays, val_timesteps)

    result_rows = []
    for manifest_path in sorted(args.manifest_dir.glob("*_b*.jsonl")):
        selector, budget = parse_selector_budget(manifest_path)
        train_timesteps = load_manifest_timesteps(manifest_path, train_episodes, args.target_offset)
        train_x, train_y = gather_samples(arrays, train_timesteps)
        metrics = train_one(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
        )
        budget_info = budget_rows.get((selector, budget), {})
        row = {
            "selector": selector,
            "budget": budget,
            "train_samples": len(train_x),
            "val_samples": len(val_x),
            "target_offset": args.target_offset,
            "epochs": args.epochs,
            "seed": args.seed,
            "final_train_loss": metrics["final_train_loss"],
            "final_val_loss": metrics["final_val_loss"],
            "best_val_loss": metrics["best_val_loss"],
            "device": metrics["device"],
            "unique_action_labels": budget_info.get("unique_action_labels", ""),
            "unique_action_ratio": budget_info.get("unique_action_ratio", ""),
            "action_overlap_ratio": budget_info.get("action_overlap_ratio", ""),
            "oracle_selector": budget_info.get("oracle_selector", ""),
            "deployable_selector": budget_info.get("deployable_selector", ""),
        }
        result_rows.append(row)
        history_path = args.output_dir / f"{selector}_b{int(round(budget * 100)):03d}_history.json"
        history_path.write_text(json.dumps(metrics["history"], indent=2) + "\n")
        failure_examples = []
        for item in metrics["top_val_errors"]:
            episode_index, timestep = val_timesteps[item["index"]]
            failure_examples.append({
                "episode_index": episode_index,
                "timestep": timestep,
                "future_timestep": timestep + args.target_offset,
                "normalized_mse": item["mse"],
            })
        failure_path = args.output_dir / f"{selector}_b{int(round(budget * 100)):03d}_failure_examples.json"
        failure_path.write_text(json.dumps(failure_examples, indent=2) + "\n")
        print(json.dumps(row))

    result_csv = args.output_dir / "proxy_loss_results.csv"
    with result_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)
    plot_results(result_rows, args.output_dir / "proxy_loss_curve.png")

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {result_csv}")


if __name__ == "__main__":
    main()
