from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.models.baseline_heads import ClipCandidateBaseline


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/clip_baseline_1pct")
    parser.add_argument("--log-file", type=str, default="outputs/milestone2/train_clip_baseline_1pct_log.csv")
    parser.add_argument("--summary-file", type=str, default="outputs/milestone2/train_clip_baseline_1pct_summary.txt")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-cardinality", type=float, default=1.0)

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--count-class-weights",
        type=float,
        nargs=4,
        default=None,
        metavar=("W0", "W1", "W2", "W3"),
        help="Optional non-negative weights for count classes 0, 1, 2, 3+.",
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_count_loss(
    count_class_weights: List[float] | None,
    device: torch.device,
) -> nn.CrossEntropyLoss:
    """Build the cardinality loss, with optional validated class weights."""
    if count_class_weights is None:
        return nn.CrossEntropyLoss()

    if len(count_class_weights) != 4:
        raise ValueError(
            "count_class_weights must contain exactly four values for "
            "classes 0, 1, 2, and 3+."
        )

    count_weights = torch.as_tensor(
        count_class_weights,
        dtype=torch.float32,
        device=device,
    )

    if not torch.isfinite(count_weights).all():
        raise ValueError("count_class_weights must all be finite.")
    if torch.any(count_weights < 0):
        raise ValueError("count_class_weights must be non-negative.")
    if not torch.any(count_weights > 0):
        raise ValueError("at least one count class weight must be positive.")

    return nn.CrossEntropyLoss(weight=count_weights)


def compute_losses(
    outputs: Dict[str, object],
    batch: Dict[str, object],
    bce_loss: nn.Module,
    ce_loss: nn.Module,
    device: torch.device,
    lambda_cardinality: float,
):
    membership_logits: List[torch.Tensor] = outputs["membership_logits"]
    count_logits: torch.Tensor = outputs["count_logits"]

    membership_loss = torch.zeros((), device=device)

    for logits, labels in zip(membership_logits, batch["candidate_labels"]):
        labels = labels.to(device).float()
        membership_loss = membership_loss + bce_loss(logits, labels)

    membership_loss = membership_loss / len(membership_logits)

    count_class = batch["count_class"].to(device).long()
    count_loss = ce_loss(count_logits, count_class)

    total_loss = membership_loss + lambda_cardinality * count_loss

    return total_loss, membership_loss, count_loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_loss: nn.Module,
    ce_loss: nn.Module,
    device: torch.device,
    lambda_cardinality: float,
):
    model.train()

    total_loss_sum = 0.0
    membership_loss_sum = 0.0
    count_loss_sum = 0.0
    correct_count = 0
    total_samples = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch)

        total_loss, membership_loss, count_loss = compute_losses(
            outputs=outputs,
            batch=batch,
            bce_loss=bce_loss,
            ce_loss=ce_loss,
            device=device,
            lambda_cardinality=lambda_cardinality,
        )

        total_loss.backward()
        optimizer.step()

        batch_size = len(batch["expressions"])

        total_loss_sum += float(total_loss.item()) * batch_size
        membership_loss_sum += float(membership_loss.item()) * batch_size
        count_loss_sum += float(count_loss.item()) * batch_size

        count_logits = outputs["count_logits"]
        pred_count = count_logits.argmax(dim=1).detach().cpu()
        true_count = batch["count_class"].detach().cpu()
        correct_count += int((pred_count == true_count).sum().item())
        total_samples += batch_size

    return {
        "total_loss": total_loss_sum / total_samples,
        "membership_loss": membership_loss_sum / total_samples,
        "count_loss": count_loss_sum / total_samples,
        "count_accuracy": correct_count / total_samples,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args,
    feature_dim: int,
):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "feature_dim": feature_dim,
    }
    torch.save(ckpt, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    dataset = ClipFeatureDataset(
        feature_file=args.feature_file,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=clip_feature_collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = ClipCandidateBaseline(
        feature_dim=dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = build_count_loss(args.count_class_weights, device)

    best_loss = float("inf")

    with open(args.log_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "total_loss",
                "membership_loss",
                "count_loss",
                "count_accuracy",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            metrics = train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                bce_loss=bce_loss,
                ce_loss=ce_loss,
                device=device,
                lambda_cardinality=args.lambda_cardinality,
            )

            row = {"epoch": epoch, **metrics}
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"total_loss={metrics['total_loss']:.4f} | "
                f"membership_loss={metrics['membership_loss']:.4f} | "
                f"count_loss={metrics['count_loss']:.4f} | "
                f"count_acc={metrics['count_accuracy']:.4f}"
            )

            save_checkpoint(
                path=os.path.join(args.output_dir, "last.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                args=args,
                feature_dim=dataset.feature_dim,
            )

            if metrics["total_loss"] < best_loss:
                best_loss = metrics["total_loss"]
                save_checkpoint(
                    path=os.path.join(args.output_dir, "best.pt"),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    args=args,
                    feature_dim=dataset.feature_dim,
                )

    with open(args.summary_file, "w") as f:
        f.write("CLIP baseline training summary\n")
        f.write("==============================\n")
        f.write(f"Feature file: {args.feature_file}\n")
        f.write(f"Dataset size: {len(dataset)}\n")
        f.write(f"CLIP model: {dataset.clip_model}\n")
        f.write(f"Feature dimension: {dataset.feature_dim}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Hidden dimension: {args.hidden_dim}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Lambda cardinality: {args.lambda_cardinality}\n")
        f.write(f"Best training loss: {best_loss:.6f}\n")
        f.write(f"Best checkpoint: {os.path.join(args.output_dir, 'best.pt')}\n")
        f.write(f"Last checkpoint: {os.path.join(args.output_dir, 'last.pt')}\n")
        f.write(f"Count class weights: {args.count_class_weights}\n")

    print("Training complete.")
    print("Log file:", args.log_file)
    print("Summary file:", args.summary_file)
    print("Best checkpoint:", os.path.join(args.output_dir, "best.pt"))
    print("Last checkpoint:", os.path.join(args.output_dir, "last.pt"))


if __name__ == "__main__":
    main()
