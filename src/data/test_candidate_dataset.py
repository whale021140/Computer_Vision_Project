import argparse
from collections import Counter
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.candidate_dataset import CandidateBoxDataset, candidate_collate_fn


def check_sample(sample: Dict, idx: int, counters: Counter) -> None:
    candidate_boxes = sample["candidate_boxes_xyxy"]
    candidate_boxes_norm = sample["candidate_boxes_norm"]
    candidate_labels = sample["candidate_labels"]
    target_boxes = sample["target_boxes_xyxy"]
    target_best_proposal_ious = sample["target_best_proposal_ious"]
    count_class = int(sample["count_class"].item())
    metadata = sample["metadata"]

    target_type = metadata["target_type"]
    num_targets = int(metadata["num_targets"])

    if candidate_boxes.ndim != 2 or candidate_boxes.shape[1] != 4:
        raise ValueError(f"Bad candidate_boxes_xyxy shape at idx={idx}: {candidate_boxes.shape}")

    if candidate_boxes_norm.ndim != 2 or candidate_boxes_norm.shape[1] != 4:
        raise ValueError(f"Bad candidate_boxes_norm shape at idx={idx}: {candidate_boxes_norm.shape}")

    if target_boxes.ndim != 2 or target_boxes.shape[1] != 4:
        raise ValueError(f"Bad target_boxes_xyxy shape at idx={idx}: {target_boxes.shape}")

    if candidate_boxes.shape[0] != candidate_labels.shape[0]:
        raise ValueError(
            f"Candidate boxes and labels mismatch at idx={idx}: "
            f"{candidate_boxes.shape[0]} vs {candidate_labels.shape[0]}"
        )

    if candidate_boxes.shape[0] == 0:
        raise ValueError(f"Zero candidate boxes at idx={idx}")

    positive_count = int(candidate_labels.sum().item())

    if target_best_proposal_ious.numel() > 0:
        if target_best_proposal_ious.numel() != num_targets:
            raise ValueError(
                f"Target best-IoU count mismatch at idx={idx}: "
                f"{target_best_proposal_ious.numel()} vs {num_targets}"
            )
        matched_targets = int((target_best_proposal_ious >= 0.5).sum().item())
        if (positive_count > 0) != (matched_targets > 0):
            raise ValueError(
                f"Candidate labels and target coverage disagree at idx={idx}: "
                f"positives={positive_count}, matched_targets={matched_targets}"
            )
        counters["matched_targets"] += matched_targets

    counters["samples"] += 1
    counters[f"type:{target_type}"] += 1
    counters["total_candidates"] += candidate_boxes.shape[0]
    counters["total_positive_labels"] += positive_count

    if target_type == "no-target":
        if num_targets != 0:
            raise ValueError(f"No-target sample has num_targets={num_targets} at idx={idx}")
        if positive_count != 0:
            raise ValueError(f"No-target sample has positive candidate labels at idx={idx}")
        if count_class != 0:
            raise ValueError(f"No-target sample has count_class={count_class} at idx={idx}")

    elif target_type == "single-target":
        if num_targets != 1:
            raise ValueError(f"Single-target sample has num_targets={num_targets} at idx={idx}")
        if count_class != 1:
            raise ValueError(f"Single-target sample has count_class={count_class} at idx={idx}")

    elif target_type == "multi-target":
        if num_targets < 2:
            raise ValueError(f"Multi-target sample has num_targets={num_targets} at idx={idx}")
        expected_count_class = 2 if num_targets == 2 else 3
        if count_class != expected_count_class:
            raise ValueError(
                f"Multi-target sample has count_class={count_class}; "
                f"expected {expected_count_class} at idx={idx}"
            )

    else:
        raise ValueError(f"Unknown target_type={target_type} at idx={idx}")

    if torch.any(candidate_boxes_norm < -1e-6) or torch.any(candidate_boxes_norm > 1.0 + 1e-6):
        raise ValueError(f"Normalized candidate boxes out of range at idx={idx}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-file",
        type=str,
        default="cache/candidates/train_1pct_coco_candidates.jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="data/coco/train2014",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-image", action="store_true")
    args = parser.parse_args()

    dataset = CandidateBoxDataset(
        candidate_file=args.candidate_file,
        image_root=args.image_root,
        load_image=not args.no_image,
        max_samples=args.max_samples,
    )

    print(f"Dataset size: {len(dataset)}")

    counters = Counter()
    image_failures = 0

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
        except Exception as exc:
            image_failures += 1
            raise RuntimeError(f"Failed to load sample idx={idx}") from exc

        check_sample(sample, idx, counters)

    print("Sample-level checks passed.")
    print(f"Image loading failures: {image_failures}")
    print(f"No-target samples: {counters['type:no-target']}")
    print(f"Single-target samples: {counters['type:single-target']}")
    print(f"Multi-target samples: {counters['type:multi-target']}")
    print(f"Total candidates: {counters['total_candidates']}")
    print(f"Total positive candidate labels: {counters['total_positive_labels']}")
    print(f"Matched targets with recorded proposal IoUs: {counters['matched_targets']}")

    avg_candidates = counters["total_candidates"] / max(counters["samples"], 1)
    print(f"Average candidates per sample: {avg_candidates:.4f}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=candidate_collate_fn,
    )

    first_batch = next(iter(dataloader))

    print("DataLoader batch test passed.")
    print(f"Batch size: {len(first_batch['expressions'])}")
    print(f"Batch count_class shape: {first_batch['count_class'].shape}")
    print(f"First expression: {first_batch['expressions'][0]}")
    print(f"First sample candidate shape: {first_batch['candidate_boxes_xyxy'][0].shape}")
    print(f"First sample target type: {first_batch['metadata'][0]['target_type']}")


if __name__ == "__main__":
    main()
