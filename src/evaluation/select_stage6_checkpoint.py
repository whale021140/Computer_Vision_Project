"""Select and calibrate one Stage 6 pilot checkpoint on development only."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.evaluation.analyze_stage6_inference_policies import (
    compact_policy,
    evaluate_membership_only,
    membership_key,
)
from src.evaluation.evaluate_clip_baseline import get_device, load_model
from src.evaluation.recalibrate_stage5_6 import (
    DEFAULT_CLASS0_BIASES,
    DEFAULT_CLASS3_BIASES,
    DEFAULT_MEMBERSHIP_THRESHOLDS,
    fast_grid_rows,
    precompute_option_metrics,
)
from src.evaluation.select_stage5_5_checkpoint import (
    calibration_key,
    collect_outputs,
    compact,
    epoch_key,
    evaluate_setting,
)


def active_boundaries(
    best: dict[str, Any],
    class0_biases: list[float] | None,
    class3_biases: list[float] | None,
    thresholds: list[float],
) -> dict[str, bool]:
    result = {
        "threshold_at_min": (
            best["membership_threshold"] == min(thresholds)
            and min(thresholds) > 0.0
        ),
        "threshold_at_max": (
            best["membership_threshold"] == max(thresholds)
            and max(thresholds) < 1.0
        ),
    }
    if class0_biases is not None and class3_biases is not None:
        result.update(
            {
                "class0_at_min": best["class0_logit_bias"]
                == min(class0_biases),
                "class0_at_max": best["class0_logit_bias"]
                == max(class0_biases),
                "class3_at_min": best["class3_logit_bias"]
                == min(class3_biases),
                "class3_at_max": best["class3_logit_bias"]
                == max(class3_biases),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--dev-split", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument(
        "--policy",
        choices=["cardinality-gated", "membership-only"],
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def membership_epoch_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row["count_macro_mean_f1"]),
        float(row["official"]["F1_score"]),
        -int(row["epoch"]),
    )


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
    for path in tqdm(checkpoint_paths, desc=f"Stage 6 epoch selection ({args.policy})"):
        model = load_model(
            str(path),
            candidate_feature_dim=dataset.candidate_feature_dim,
            text_feature_dim=dataset.text_feature_dim,
            hidden_dim=256,
            dropout=0.1,
            device=device,
        )
        collected = collect_outputs(model, loader)
        if args.policy == "membership-only":
            row = evaluate_membership_only(collected, 0.5)
        else:
            row = evaluate_setting(collected, 0.0, 0.0, 0.5)
        row.update({"epoch": int(path.stem.split("_")[-1]), "checkpoint": str(path)})
        epoch_rows.append(row)
        del model, collected
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_epoch = max(
        epoch_rows,
        key=membership_epoch_key if args.policy == "membership-only" else epoch_key,
    )
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
    thresholds = [float(value) for value in DEFAULT_MEMBERSHIP_THRESHOLDS]
    if args.policy == "membership-only":
        rows = [evaluate_membership_only(collected, value) for value in thresholds]
        best = max(rows, key=membership_key)
        grid = {
            "membership_thresholds": thresholds,
            "num_settings": len(rows),
        }
        boundary = active_boundaries(best, None, None, thresholds)
        stored_rows = sorted(rows, key=membership_key, reverse=True)
    else:
        class0_biases = [float(value) for value in DEFAULT_CLASS0_BIASES]
        class3_biases = [float(value) for value in DEFAULT_CLASS3_BIASES]
        option_metrics = precompute_option_metrics(collected, thresholds)
        compact_rows = fast_grid_rows(
            option_metrics, class0_biases, class3_biases, thresholds
        )
        best_compact = max(compact_rows, key=calibration_key)
        best = evaluate_setting(
            collected,
            float(best_compact["class0_logit_bias"]),
            float(best_compact["class3_logit_bias"]),
            float(best_compact["membership_threshold"]),
        )
        best["policy"] = "cardinality-gated"
        grid = {
            "class0_biases": class0_biases,
            "class3_biases": class3_biases,
            "membership_thresholds": thresholds,
            "num_settings": len(compact_rows),
        }
        boundary = active_boundaries(
            best, class0_biases, class3_biases, thresholds
        )
        stored_rows = sorted(
            compact_rows, key=calibration_key, reverse=True
        )[:100]

    selection = {
        "stage": "6.1",
        "selection_source": "Stage 5.6 image-disjoint development split only",
        "policy": args.policy,
        "feature_file": args.feature_file,
        "development_split": args.dev_split,
        "criterion": [
            "count_macro_mean_f1",
            "official.F1_score",
            "earlier_epoch",
        ],
        "selected": (
            {
                **compact_policy(selected_epoch),
                "epoch": selected_epoch["epoch"],
                "checkpoint": selected_epoch["checkpoint"],
            }
            if args.policy == "membership-only"
            else compact(selected_epoch)
        ),
        "epochs": [
            (
                {
                    **compact_policy(row),
                    "epoch": row["epoch"],
                    "checkpoint": row["checkpoint"],
                }
                if args.policy == "membership-only"
                else compact(row)
            )
            for row in epoch_rows
        ],
    }
    calibration = {
        "stage": "6.1",
        "calibration_revision": "stage6_dev_only_v1",
        "selection_source": "Stage 5.6 image-disjoint development split only",
        "policy": args.policy,
        "feature_file": args.feature_file,
        "development_split": args.dev_split,
        "checkpoint": str(selected_path),
        "criterion": [
            "count_macro_mean_f1",
            "official.F1_score",
            "neutral-setting tie-break",
        ],
        "grid": grid,
        "best": best,
        "best_on_boundary": boundary,
        "num_rows_evaluated": grid["num_settings"],
        "top_rows": stored_rows,
    }
    for path, value in (
        (Path(args.selection_json), selection),
        (Path(args.calibration_json), calibration),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    active = [key for key, value in boundary.items() if value]
    lines = [
        "Stage 6.1 Development Selection",
        "================================",
        f"Policy: {args.policy}",
        f"Selected epoch: {selected_epoch['epoch']}",
        f"Calibrated count-macro mean F1: {best['count_macro_mean_f1']:.6f}",
        f"Calibrated F1_score: {best['official']['F1_score']:.6f}",
        f"T_acc: {best['official']['T_acc']:.6f}",
        f"N_acc: {best['official']['N_acc']:.6f}",
        f"Calibration boundary flags: {active or 'none'}",
    ]
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if active:
        raise SystemExit(
            "Stage 6 calibration selected a truncatable boundary; stop before test."
        )


if __name__ == "__main__":
    main()
