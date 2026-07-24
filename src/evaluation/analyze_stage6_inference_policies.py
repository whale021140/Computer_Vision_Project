"""Compare Stage 6 inference ablations on the locked development set only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.evaluation.evaluate_clip_baseline import get_device, load_model
from src.evaluation.grec_metrics import PredictionRecord, evaluate_records
from src.evaluation.recalibrate_stage5_6 import DEFAULT_MEMBERSHIP_THRESHOLDS
from src.evaluation.select_stage5_5_checkpoint import (
    calibration_key,
    collect_outputs,
    count_macro_mean_f1,
    evaluate_setting,
)


def membership_only_indices(
    membership_logits: torch.Tensor, threshold: float
) -> list[int]:
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("membership threshold must be in [0, 1]")
    probabilities = torch.sigmoid(torch.as_tensor(membership_logits).float())
    return (
        torch.nonzero(probabilities >= float(threshold), as_tuple=False)
        .flatten()
        .tolist()
    )


def evaluate_membership_only(
    rows: Sequence[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    records = []
    for row in rows:
        selected = membership_only_indices(row["membership_logits"], threshold)
        predicted_count_class = min(len(selected), 3)
        records.append(
            PredictionRecord(
                sample_id=str(row["sample_id"]),
                predicted_boxes=(
                    row["candidate_boxes"][selected]
                    if selected
                    else torch.empty((0, 4), dtype=torch.float32)
                ),
                predicted_scores=(
                    torch.sigmoid(row["membership_logits"])[selected]
                    if selected
                    else torch.empty(0, dtype=torch.float32)
                ),
                target_boxes=row["target_boxes"],
                target_type=str(row["target_type"]),
                predicted_count_class=predicted_count_class,
            )
        )
    metrics = evaluate_records(
        records,
        match_threshold=0.5,
        overlap_metric="giou",
        image_f1_threshold=1.0,
    )
    return {
        "policy": "membership_only",
        "membership_threshold": float(threshold),
        "count_macro_mean_f1": count_macro_mean_f1(metrics),
        "official": metrics["official"],
        "diagnostics": metrics["diagnostics"],
        "by_target_type": metrics["by_target_type"],
        "by_count_group": metrics["by_count_group"],
    }


def membership_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["count_macro_mean_f1"]),
        float(row["official"]["F1_score"]),
        -abs(float(row["membership_threshold"]) - 0.5),
        -float(row["membership_threshold"]),
    )


def compact_policy(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "policy",
            "class0_logit_bias",
            "class3_logit_bias",
            "count_logit_bias",
            "membership_threshold",
            "count_macro_mean_f1",
            "official",
            "diagnostics",
            "by_target_type",
            "by_count_group",
        )
        if key in row
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--dev-split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-calibration", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--membership-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_MEMBERSHIP_THRESHOLDS,
    )
    args = parser.parse_args()

    thresholds = [float(value) for value in args.membership_thresholds]
    if thresholds != sorted(set(thresholds)):
        raise ValueError("membership thresholds must be unique and sorted")

    device = get_device(args.device)
    dataset = ClipFeatureDataset(args.feature_file, split_file=args.dev_split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=clip_feature_collate_fn,
    )
    model = load_model(
        args.checkpoint,
        candidate_feature_dim=dataset.candidate_feature_dim,
        text_feature_dim=dataset.text_feature_dim,
        hidden_dim=256,
        dropout=0.1,
        device=device,
    )
    collected = collect_outputs(model, loader)

    calibration = json.loads(
        Path(args.baseline_calibration).read_text(encoding="utf-8")
    )
    baseline_best = calibration["best"]
    baseline = evaluate_setting(
        collected,
        float(baseline_best["class0_logit_bias"]),
        float(baseline_best["class3_logit_bias"]),
        float(baseline_best["membership_threshold"]),
    )
    baseline["policy"] = "cardinality_gated_v2_wide"

    neutral = evaluate_setting(collected, 0.0, 0.0, 0.5)
    neutral["policy"] = "cardinality_gated_neutral"

    no_bias_rows = []
    for threshold in thresholds:
        row = evaluate_setting(collected, 0.0, 0.0, threshold)
        row["policy"] = "cardinality_gated_no_bias_threshold_sweep"
        no_bias_rows.append(row)
    no_bias_best = max(no_bias_rows, key=calibration_key)

    membership_rows = [
        evaluate_membership_only(collected, threshold)
        for threshold in thresholds
    ]
    membership_best = max(membership_rows, key=membership_key)
    result = {
        "stage": "6.1",
        "scope": "development-only inference mechanism audit",
        "feature_file": args.feature_file,
        "development_split": args.dev_split,
        "checkpoint": args.checkpoint,
        "baseline_calibration": args.baseline_calibration,
        "num_samples": len(collected),
        "thresholds": thresholds,
        "policies": {
            "C2_baseline": compact_policy(baseline),
            "C3_membership_only": compact_policy(membership_best),
            "C4a_no_bias_best_threshold": compact_policy(no_bias_best),
            "C4b_fully_neutral": compact_policy(neutral),
        },
        "membership_only_sweep": [
            compact_policy(row) for row in membership_rows
        ],
        "no_bias_threshold_sweep": [
            compact_policy(row) for row in no_bias_rows
        ],
    }
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Stage 6.1 Development-Only Inference Audit",
        "==========================================",
        "policy | threshold | count-macro mean F1 | F1_score | T_acc | N_acc",
        "--- | ---: | ---: | ---: | ---: | ---:",
    ]
    for name, row in result["policies"].items():
        lines.append(
            f"{name} | {row['membership_threshold']:.3f} | "
            f"{row['count_macro_mean_f1']:.6f} | "
            f"{row['official']['F1_score']:.6f} | "
            f"{row['official']['T_acc']:.6f} | "
            f"{row['official']['N_acc']:.6f}"
        )
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_txt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
