from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.models.baseline_heads import ClipCandidateBaseline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = ClipFeatureDataset(
        feature_file=args.feature_file,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clip_feature_collate_fn,
    )

    model = ClipCandidateBaseline(
        candidate_feature_dim=dataset.candidate_feature_dim,
        text_feature_dim=dataset.text_feature_dim,
        hidden_dim=args.hidden_dim,
    )

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    batch = next(iter(loader))
    outputs = model(batch)

    membership_logits = outputs["membership_logits"]
    count_logits = outputs["count_logits"]

    membership_loss = 0.0

    for logits, labels in zip(membership_logits, batch["candidate_labels"]):
        membership_loss = membership_loss + bce_loss(logits.cpu(), labels.float())

    membership_loss = membership_loss / len(membership_logits)
    count_loss = ce_loss(count_logits.cpu(), batch["count_class"])
    total_loss = membership_loss + count_loss

    print("Feature file:", args.feature_file)
    print("Dataset size:", len(dataset))
    print("Representation:", dataset.representation)
    print("Candidate feature dim:", dataset.candidate_feature_dim)
    print("Text feature dim:", dataset.text_feature_dim)
    print("Batch size:", len(batch["expressions"]))

    print("Number of membership logit tensors:", len(membership_logits))
    print("First membership logits shape:", membership_logits[0].shape)
    print("First candidate labels shape:", batch["candidate_labels"][0].shape)
    print("Count logits shape:", count_logits.shape)
    print("Count class shape:", batch["count_class"].shape)

    print("Membership loss:", float(membership_loss.item()))
    print("Count loss:", float(count_loss.item()))
    print("Total loss:", float(total_loss.item()))

    print("First expression:", batch["expressions"][0])
    print("First target type:", batch["metadata"][0]["target_type"])
    print("Forward smoke test passed.")


if __name__ == "__main__":
    main()
