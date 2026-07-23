"""Select and calibrate a Stage 5.5 checkpoint on locked shadow-dev only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

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


COUNT_GROUPS = ("0", "1", "2", "3+")


def count_macro_mean_f1(metrics: dict[str, Any]) -> float:
    return sum(
        float(metrics["by_count_group"][group]["mean_f1"])
        for group in COUNT_GROUPS
    ) / len(COUNT_GROUPS)


@torch.no_grad()
def collect_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="Collecting shadow-dev logits", leave=False):
        outputs = model(batch)
        count_logits = outputs["count_logits"].detach().cpu().float()
        for index, membership_logits in enumerate(outputs["membership_logits"]):
            metadata = batch["metadata"][index]
            rows.append(
                {
                    "sample_id": str(batch["sample_ids"][index]),
                    "target_type": metadata["target_type"],
                    "count_logits": count_logits[index],
                    "membership_logits": membership_logits.detach().cpu().float(),
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
    if not rows:
        raise ValueError("Shadow-dev contains no samples.")
    return rows


def evaluate_setting(
    rows: Sequence[dict[str, Any]],
    class0_bias: float,
    class3_bias: float,
    membership_threshold: float,
) -> dict[str, Any]:
    bias = torch.tensor([class0_bias, 0.0, 0.0, class3_bias])
    records = []
    for row in rows:
        predicted_count = int(torch.argmax(row["count_logits"] + bias).item())
        selected = sorted(
            select_cardinality_gated_indices(
                row["membership_logits"],
                predicted_count,
                membership_threshold=membership_threshold,
            )
        )
        records.append(
            PredictionRecord(
                sample_id=row["sample_id"],
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
                target_type=row["target_type"],
                predicted_count_class=predicted_count,
            )
        )
    metrics = evaluate_records(
        records,
        match_threshold=0.5,
        overlap_metric="giou",
        image_f1_threshold=1.0,
    )
    return {
        "class0_logit_bias": float(class0_bias),
        "class3_logit_bias": float(class3_bias),
        "count_logit_bias": [float(class0_bias), 0.0, 0.0, float(class3_bias)],
        "membership_threshold": float(membership_threshold),
        "count_macro_mean_f1": count_macro_mean_f1(metrics),
        "official": metrics["official"],
        "diagnostics": metrics["diagnostics"],
        "by_target_type": metrics["by_target_type"],
        "by_count_group": metrics["by_count_group"],
    }


def epoch_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row["count_macro_mean_f1"]),
        float(row["official"]["F1_score"]),
        -int(row["epoch"]),
    )


def calibration_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["count_macro_mean_f1"]),
        float(row["official"]["F1_score"]),
        -(abs(float(row["class0_logit_bias"])) + abs(float(row["class3_logit_bias"]))),
        -abs(float(row["membership_threshold"]) - 0.5),
        -float(row["class0_logit_bias"]),
        -float(row["class3_logit_bias"]),
        -float(row["membership_threshold"]),
    )


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "epoch",
            "checkpoint",
            "count_macro_mean_f1",
            "official",
            "diagnostics",
            "by_count_group",
        )
        if key in row
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--shadow-split", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--class0-biases", type=float, nargs="+", default=[-1, -0.5, 0, 0.5, 1]
    )
    parser.add_argument(
        "--class3-biases", type=float, nargs="+", default=[0, 0.5, 1, 1.5, 2]
    )
    parser.add_argument(
        "--membership-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    dataset = ClipFeatureDataset(args.feature_file, split_file=args.shadow_split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=clip_feature_collate_fn,
    )
    checkpoint_paths = sorted(Path(args.checkpoint_dir).glob("epoch_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No epoch checkpoints under {args.checkpoint_dir}")

    epoch_rows = []
    for path in tqdm(checkpoint_paths, desc="Selecting shadow-dev epoch"):
        model = load_model(
            str(path),
            candidate_feature_dim=dataset.candidate_feature_dim,
            text_feature_dim=dataset.text_feature_dim,
            hidden_dim=256,
            dropout=0.1,
            device=device,
        )
        collected = collect_outputs(model, loader)
        row = evaluate_setting(collected, 0.0, 0.0, 0.5)
        row.update({"epoch": int(path.stem.split("_")[-1]), "checkpoint": str(path)})
        epoch_rows.append(row)
        del model, collected
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected_epoch = max(epoch_rows, key=epoch_key)
    shutil.copy2(selected_epoch["checkpoint"], args.selected_checkpoint)

    model = load_model(
        args.selected_checkpoint,
        candidate_feature_dim=dataset.candidate_feature_dim,
        text_feature_dim=dataset.text_feature_dim,
        hidden_dim=256,
        dropout=0.1,
        device=device,
    )
    collected = collect_outputs(model, loader)
    calibration_rows = []
    settings = [
        (class0_bias, class3_bias, threshold)
        for class0_bias in args.class0_biases
        for class3_bias in args.class3_biases
        for threshold in args.membership_thresholds
    ]
    for class0_bias, class3_bias, threshold in tqdm(
        settings, desc="Calibrating shadow-dev"
    ):
        calibration_rows.append(
            evaluate_setting(collected, class0_bias, class3_bias, threshold)
        )
    best_calibration = max(calibration_rows, key=calibration_key)

    selection = {
        "feature_file": args.feature_file,
        "shadow_split": args.shadow_split,
        "device": str(device),
        "criterion": [
            "count_macro_mean_f1",
            "official.F1_score",
            "earlier_epoch",
        ],
        "selected": compact(selected_epoch),
        "epochs": [compact(row) for row in epoch_rows],
    }
    calibration = {
        "feature_file": args.feature_file,
        "shadow_split": args.shadow_split,
        "checkpoint": args.selected_checkpoint,
        "device": str(device),
        "criterion": [
            "count_macro_mean_f1",
            "official.F1_score",
            "smaller_absolute_bias",
            "threshold_closest_to_0.5",
            "lexicographic_setting",
        ],
        "best": best_calibration,
        "rows": calibration_rows,
    }
    for path, value in (
        (args.selection_json, selection),
        (args.calibration_json, calibration),
    ):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    summary = "\n".join(
        [
            "Stage 5.5 Shadow-Dev Selection",
            "================================",
            f"Selected epoch: {selected_epoch['epoch']}",
            f"Uncalibrated count-macro mean F1: {selected_epoch['count_macro_mean_f1']:.6f}",
            f"Calibration: bias={best_calibration['count_logit_bias']} threshold={best_calibration['membership_threshold']}",
            f"Calibrated count-macro mean F1: {best_calibration['count_macro_mean_f1']:.6f}",
            f"Calibrated F1_score: {best_calibration['official']['F1_score']:.6f}",
            f"Calibrated T_acc: {best_calibration['official']['T_acc']:.6f}",
            f"Calibrated N_acc: {best_calibration['official']['N_acc']:.6f}",
        ]
    ) + "\n"
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
