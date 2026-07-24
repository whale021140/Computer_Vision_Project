"""Select an epoch and calibrate inference on the locked Stage 5.6 dev set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.evaluation.evaluate_clip_baseline import get_device, load_model
from src.evaluation.select_stage5_5_checkpoint import (
    calibration_key,
    collect_outputs,
    compact,
    epoch_key,
    evaluate_setting,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--dev-split", required=True)
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
    dataset = ClipFeatureDataset(args.feature_file, split_file=args.dev_split)
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
    for path in tqdm(checkpoint_paths, desc="Selecting Stage 5.6 dev epoch"):
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
    selected_path = Path(args.selected_checkpoint)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_epoch["checkpoint"], selected_path)

    model = load_model(
        str(selected_path),
        candidate_feature_dim=dataset.candidate_feature_dim,
        text_feature_dim=dataset.text_feature_dim,
        hidden_dim=256,
        dropout=0.1,
        device=device,
    )
    collected = collect_outputs(model, loader)
    calibration_rows = [
        evaluate_setting(collected, class0_bias, class3_bias, threshold)
        for class0_bias in args.class0_biases
        for class3_bias in args.class3_biases
        for threshold in args.membership_thresholds
    ]
    best_calibration = max(calibration_rows, key=calibration_key)

    selection = {
        "stage": "5.6",
        "feature_file": args.feature_file,
        "development_split": args.dev_split,
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
        "stage": "5.6",
        "feature_file": args.feature_file,
        "development_split": args.dev_split,
        "checkpoint": str(selected_path),
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
            "Stage 5.6 Development Selection",
            "================================",
            f"Selected epoch: {selected_epoch['epoch']}",
            (
                "Uncalibrated count-macro mean F1: "
                f"{selected_epoch['count_macro_mean_f1']:.6f}"
            ),
            (
                f"Calibration: bias={best_calibration['count_logit_bias']} "
                f"threshold={best_calibration['membership_threshold']}"
            ),
            (
                "Calibrated count-macro mean F1: "
                f"{best_calibration['count_macro_mean_f1']:.6f}"
            ),
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
