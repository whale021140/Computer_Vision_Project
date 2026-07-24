"""Recalibrate one selected Stage 5.6 checkpoint on the locked dev split.

This entry point deliberately does not repeat epoch selection or training. It
loads the already selected checkpoint once, caches its dev-set logits, evaluates
the declared inference grid, and writes a versioned calibration artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.evaluation.evaluate_clip_baseline import (
    get_device,
    load_model,
    select_cardinality_gated_indices,
)
from src.evaluation.grec_metrics import (
    greedy_one_to_one_matches,
    pairwise_generalized_iou,
)
from src.evaluation.select_stage5_5_checkpoint import (
    calibration_key,
    collect_outputs,
    evaluate_setting,
)


DEFAULT_CLASS0_BIASES = [-1.0, -0.5] + [
    value / 2 for value in range(0, 33)
]
DEFAULT_CLASS3_BIASES = [value / 2 for value in range(0, 65)]
DEFAULT_MEMBERSHIP_THRESHOLDS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.925,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    0.995,
    0.999,
    1.0,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--dev-split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--revision", default="v2_wide")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--class0-biases",
        type=float,
        nargs="+",
        default=DEFAULT_CLASS0_BIASES,
    )
    parser.add_argument(
        "--class3-biases",
        type=float,
        nargs="+",
        default=DEFAULT_CLASS3_BIASES,
    )
    parser.add_argument(
        "--membership-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_MEMBERSHIP_THRESHOLDS,
    )
    return parser.parse_args()


def validate_grid(args: argparse.Namespace) -> None:
    for name in (
        "class0_biases",
        "class3_biases",
        "membership_thresholds",
    ):
        values = [float(value) for value in getattr(args, name)]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be non-empty and contain no duplicates.")
        if values != sorted(values):
            raise ValueError(f"{name} must be sorted.")
    if any(
        not 0.0 <= float(value) <= 1.0
        for value in args.membership_thresholds
    ):
        raise ValueError("membership thresholds must be in [0, 1].")


def _selection_f1(
    overlaps: torch.Tensor,
    selected: list[int],
    num_targets: int,
) -> tuple[float, float, float]:
    num_predictions = len(selected)
    if num_predictions == 0 and num_targets == 0:
        return 1.0, 1.0, 0.0
    true_positives = len(
        greedy_one_to_one_matches(overlaps[selected], threshold=0.5)
    )
    denominator = num_predictions + num_targets
    f1 = 2.0 * true_positives / denominator if denominator else 0.0
    return f1, float(f1 + 1e-12 >= 1.0), float(num_predictions > 0)


def precompute_option_metrics(
    collected: list[dict[str, object]],
    thresholds: list[float],
) -> dict[str, torch.Tensor]:
    """Precompute GIoU matching once per possible inference decision.

    Count classes 0/1/2 do not depend on the 3+ threshold. Count class 3 has
    one option per threshold, so every sample has ``3 + len(thresholds)``
    possible output decisions.
    """
    num_options = 3 + len(thresholds)
    num_samples = len(collected)
    f1 = torch.empty((num_samples, num_options), dtype=torch.float64)
    exact = torch.empty_like(f1)
    has_prediction = torch.empty_like(f1)
    count_groups = torch.empty(num_samples, dtype=torch.long)
    count_logits = torch.empty((num_samples, 4), dtype=torch.float64)
    for index, row in enumerate(
        tqdm(collected, desc="Precomputing dev decision metrics")
    ):
        membership_logits = torch.as_tensor(row["membership_logits"]).float()
        candidate_boxes = torch.as_tensor(row["candidate_boxes"]).float()
        target_boxes = torch.as_tensor(row["target_boxes"]).float()
        num_targets = int(target_boxes.shape[0])
        count_groups[index] = min(num_targets, 3)
        count_logits[index] = torch.as_tensor(row["count_logits"]).double()
        overlaps = pairwise_generalized_iou(candidate_boxes, target_boxes)
        selections = [
            sorted(
                select_cardinality_gated_indices(
                    membership_logits,
                    predicted_count_class,
                    membership_threshold=0.5,
                )
            )
            for predicted_count_class in (0, 1, 2)
        ]
        selections.extend(
            sorted(
                select_cardinality_gated_indices(
                    membership_logits,
                    3,
                    membership_threshold=threshold,
                )
            )
            for threshold in thresholds
        )
        for option, selected in enumerate(selections):
            values = _selection_f1(overlaps, selected, num_targets)
            f1[index, option], exact[index, option], has_prediction[index, option] = (
                values
            )
    return {
        "f1": f1,
        "exact": exact,
        "has_prediction": has_prediction,
        "count_groups": count_groups,
        "count_logits": count_logits,
    }


def fast_grid_rows(
    metrics: dict[str, torch.Tensor],
    class0_biases: list[float],
    class3_biases: list[float],
    thresholds: list[float],
) -> list[dict[str, object]]:
    sample_indices = torch.arange(metrics["f1"].shape[0])
    targeted = metrics["count_groups"] != 0
    no_target = ~targeted
    group_masks = [metrics["count_groups"] == group for group in range(4)]
    rows = []
    for class0_bias in class0_biases:
        for class3_bias in class3_biases:
            bias = torch.tensor(
                [class0_bias, 0.0, 0.0, class3_bias],
                dtype=metrics["count_logits"].dtype,
            )
            predicted_count = torch.argmax(metrics["count_logits"] + bias, dim=1)
            for threshold_index, threshold in enumerate(thresholds):
                options = torch.where(
                    predicted_count == 3,
                    3 + threshold_index,
                    predicted_count,
                )
                selected_f1 = metrics["f1"][sample_indices, options]
                selected_exact = metrics["exact"][sample_indices, options]
                selected_has_prediction = metrics["has_prediction"][
                    sample_indices, options
                ]
                rows.append(
                    {
                        "class0_logit_bias": float(class0_bias),
                        "class3_logit_bias": float(class3_bias),
                        "count_logit_bias": [
                            float(class0_bias),
                            0.0,
                            0.0,
                            float(class3_bias),
                        ],
                        "membership_threshold": float(threshold),
                        "count_macro_mean_f1": float(
                            torch.stack(
                                [
                                    selected_f1[mask].mean()
                                    for mask in group_masks
                                ]
                            ).mean()
                        ),
                        "official": {
                            "F1_score": float(selected_exact.mean()),
                            "T_acc": float(
                                selected_has_prediction[targeted].mean()
                            ),
                            "N_acc": float(
                                1.0
                                - selected_has_prediction[no_target].mean()
                            ),
                        },
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    validate_grid(args)
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
    option_metrics = precompute_option_metrics(
        collected,
        [float(value) for value in args.membership_thresholds],
    )
    rows = fast_grid_rows(
        option_metrics,
        [float(value) for value in args.class0_biases],
        [float(value) for value in args.class3_biases],
        [float(value) for value in args.membership_thresholds],
    )
    best_compact = max(rows, key=calibration_key)
    best = evaluate_setting(
        collected,
        float(best_compact["class0_logit_bias"]),
        float(best_compact["class3_logit_bias"]),
        float(best_compact["membership_threshold"]),
    )
    grid = {
        "class0_biases": [float(value) for value in args.class0_biases],
        "class3_biases": [float(value) for value in args.class3_biases],
        "membership_thresholds": [
            float(value) for value in args.membership_thresholds
        ],
        "num_settings": len(rows),
    }
    boundary = {
        "class0_at_min": best["class0_logit_bias"] == min(args.class0_biases),
        "class0_at_max": best["class0_logit_bias"] == max(args.class0_biases),
        "class3_at_min": best["class3_logit_bias"] == min(args.class3_biases),
        "class3_at_max": best["class3_logit_bias"] == max(args.class3_biases),
        "threshold_at_min": (
            best["membership_threshold"] == min(args.membership_thresholds)
            and min(args.membership_thresholds) > 0.0
        ),
        "threshold_at_max": (
            best["membership_threshold"] == max(args.membership_thresholds)
            and max(args.membership_thresholds) < 1.0
        ),
    }
    result = {
        "stage": "5.6",
        "calibration_revision": args.revision,
        "selection_source": "locked image-disjoint development set only",
        "feature_file": args.feature_file,
        "development_split": args.dev_split,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "criterion": [
            "count_macro_mean_f1",
            "official.F1_score",
            "smaller_absolute_bias",
            "threshold_closest_to_0.5",
            "lexicographic_setting",
        ],
        "grid": grid,
        "best": best,
        "best_on_boundary": boundary,
        "best_at_natural_threshold_endpoint": {
            "threshold_at_zero": best["membership_threshold"] == 0.0,
            "threshold_at_one": best["membership_threshold"] == 1.0,
        },
        "row_storage": (
            "Only the best 100 settings are stored; all settings are "
            "deterministically reproducible from grid and criterion."
        ),
        "num_rows_evaluated": len(rows),
        "top_rows": sorted(rows, key=calibration_key, reverse=True)[:100],
    }
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = "\n".join(
        [
            "Stage 5.6 Wide Development Calibration",
            "======================================",
            f"Revision: {args.revision}",
            f"Settings: {len(rows)}",
            (
                "Best: "
                f"class0_bias={best['class0_logit_bias']}, "
                f"class3_bias={best['class3_logit_bias']}, "
                f"threshold={best['membership_threshold']}"
            ),
            f"Count-macro mean F1: {best['count_macro_mean_f1']:.6f}",
            f"Official F1_score: {best['official']['F1_score']:.6f}",
            f"T_acc: {best['official']['T_acc']:.6f}",
            f"N_acc: {best['official']['N_acc']:.6f}",
            f"Boundary flags: {boundary}",
        ]
    ) + "\n"
    output_txt.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
