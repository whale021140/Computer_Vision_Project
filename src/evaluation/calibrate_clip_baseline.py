from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.evaluation.evaluate_clip_baseline import (
    get_device,
    load_model,
    normalized_boxes_to_xyxy,
)
from src.evaluation.grec_metrics import PredictionRecord, evaluate_records
from src.evaluation.metrics import select_cardinality_gated_indices


DEFAULT_THRESHOLDS = [round(value / 20, 2) for value in range(2, 19)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the 3+ membership threshold on validation data."
    )
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-metric", choices=["iou", "giou"], default="giou")
    parser.add_argument("--image-f1-threshold", type=float, default=1.0)
    return parser.parse_args()


@torch.no_grad()
def collect_model_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
) -> List[Dict[str, Any]]:
    model.eval()
    collected: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc="Collecting validation outputs"):
        outputs = model(batch)
        pred_count_classes = outputs["count_logits"].argmax(dim=1).detach().cpu()
        for index, logits in enumerate(outputs["membership_logits"]):
            metadata = batch["metadata"][index]
            collected.append(
                {
                    "sample_id": str(batch["sample_ids"][index]),
                    "target_type": metadata.get("target_type", "unknown"),
                    "membership_logits": logits.detach().cpu().float(),
                    "predicted_count_class": int(pred_count_classes[index].item()),
                    "candidate_boxes": normalized_boxes_to_xyxy(
                        batch["candidate_boxes_norm"][index],
                        width=int(metadata["width"]),
                        height=int(metadata["height"]),
                    ),
                    "target_boxes": batch["target_boxes_xyxy"][index]
                    .detach()
                    .cpu()
                    .float(),
                }
            )
    if not collected:
        raise ValueError("Validation feature cache contains no samples.")
    return collected


def evaluate_threshold(
    collected: Sequence[Dict[str, Any]],
    membership_threshold: float,
    match_threshold: float = 0.5,
    overlap_metric: str = "giou",
    image_f1_threshold: float = 1.0,
) -> Dict[str, Any]:
    records = []
    for sample in collected:
        logits = sample["membership_logits"]
        indices = sorted(
            select_cardinality_gated_indices(
                logits,
                sample["predicted_count_class"],
                membership_threshold=membership_threshold,
            )
        )
        records.append(
            PredictionRecord(
                sample_id=sample["sample_id"],
                predicted_boxes=(
                    sample["candidate_boxes"][indices]
                    if indices
                    else torch.empty((0, 4), dtype=torch.float32)
                ),
                predicted_scores=(
                    torch.sigmoid(logits)[indices]
                    if indices
                    else torch.empty(0, dtype=torch.float32)
                ),
                target_boxes=sample["target_boxes"],
                target_type=sample["target_type"],
                predicted_count_class=sample["predicted_count_class"],
            )
        )
    return evaluate_records(
        records,
        match_threshold=match_threshold,
        overlap_metric=overlap_metric,
        image_f1_threshold=image_f1_threshold,
    )


def choose_best_threshold(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("At least one calibration row is required.")
    return max(
        rows,
        key=lambda row: (
            float(row["official"]["F1_score"]),
            float(row["diagnostics"]["mean_f1"]),
            float(row["official"]["N_acc"]),
            -abs(float(row["membership_threshold"]) - 0.5),
            -float(row["membership_threshold"]),
        ),
    )


def validate_thresholds(thresholds: Sequence[float]) -> List[float]:
    values = sorted(set(float(value) for value in thresholds))
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Calibration thresholds must be non-empty and in [0, 1].")
    return values


def format_summary(result: Dict[str, Any]) -> str:
    best = result["best"]
    lines = [
        "Frozen Representation Validation Calibration",
        "============================================",
        f"Feature file: {result['feature_file']}",
        f"Checkpoint: {result['checkpoint']}",
        f"Overlap metric: {result['config']['overlap_metric']}",
        f"Best membership threshold: {best['membership_threshold']:.4f}",
        f"Best F1_score: {best['official']['F1_score']:.6f}",
        f"Best T_acc: {best['official']['T_acc']:.6f}",
        f"Best N_acc: {best['official']['N_acc']:.6f}",
        f"Best mean F1: {best['diagnostics']['mean_f1']:.6f}",
        f"Predicted count distribution: {result['predicted_count_distribution']}",
        f"Threshold-sensitive samples: {result['threshold_sensitive_samples']}",
        "",
        "[threshold sweep]",
    ]
    for row in result["rows"]:
        lines.append(
            f"threshold={row['membership_threshold']:.4f} "
            f"F1_score={row['official']['F1_score']:.6f} "
            f"mean_f1={row['diagnostics']['mean_f1']:.6f} "
            f"T_acc={row['official']['T_acc']:.6f} "
            f"N_acc={row['official']['N_acc']:.6f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    thresholds = validate_thresholds(args.thresholds)
    device = get_device(args.device)
    dataset = ClipFeatureDataset(args.feature_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=clip_feature_collate_fn,
    )
    model = load_model(
        checkpoint_path=args.checkpoint,
        candidate_feature_dim=dataset.candidate_feature_dim,
        text_feature_dim=dataset.text_feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=device,
    )
    collected = collect_model_outputs(model, loader)
    rows = []
    for threshold in tqdm(thresholds, desc="Sweeping thresholds"):
        metrics = evaluate_threshold(
            collected,
            membership_threshold=threshold,
            match_threshold=args.match_threshold,
            overlap_metric=args.overlap_metric,
            image_f1_threshold=args.image_f1_threshold,
        )
        rows.append({"membership_threshold": threshold, **metrics})
    best = choose_best_threshold(rows)
    result = {
        "feature_file": args.feature_file,
        "checkpoint": args.checkpoint,
        "clip_model": dataset.clip_model,
        "feature_dim": dataset.candidate_feature_dim,
        "candidate_feature_dim": dataset.candidate_feature_dim,
        "text_feature_dim": dataset.text_feature_dim,
        "representation": dataset.representation,
        "device": str(device),
        "selection_criterion": [
            "official.F1_score",
            "diagnostics.mean_f1",
            "official.N_acc",
            "membership_threshold closest to 0.5",
            "lower membership_threshold",
        ],
        "predicted_count_distribution": dict(
            sorted(
                Counter(
                    str(sample["predicted_count_class"]) for sample in collected
                ).items()
            )
        ),
        "threshold_sensitive_samples": sum(
            sample["predicted_count_class"] == 3 for sample in collected
        ),
        "config": {
            "match_threshold": args.match_threshold,
            "overlap_metric": args.overlap_metric,
            "image_f1_threshold": args.image_f1_threshold,
        },
        "best": best,
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = format_summary(result)
    output_txt.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
