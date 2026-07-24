"""Diagnose exact 3/4/5/6+ failures and inference-time duplicate suppression."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.evaluation.evaluate_clip_baseline import (
    get_device,
    load_model,
    normalized_boxes_to_xyxy,
)
from src.evaluation.grec_metrics import (
    PredictionRecord,
    evaluate_records,
    greedy_one_to_one_matches,
    pairwise_iou,
)
from src.evaluation.metrics import select_cardinality_gated_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="")
    parser.add_argument("--nms-thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument(
        "--pre-nms-membership-thresholds",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Optional additional 3+ membership thresholds evaluated after "
            "pre-selection NMS; the calibration threshold is always included."
        ),
    )
    return parser.parse_args()


def exact_group(count: int) -> str:
    return str(count) if count <= 5 else "6+"


def greedy_nms(
    boxes: torch.Tensor, scores: torch.Tensor, threshold: float
) -> list[int]:
    if boxes.shape[0] == 0:
        return []
    return [
        int(index)
        for index in nms(boxes.float(), scores.float(), float(threshold)).tolist()
    ]


def classify_failure(
    candidate_boxes: torch.Tensor,
    selected_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
) -> tuple[str, dict[str, Any]]:
    candidate_overlaps = pairwise_iou(candidate_boxes, target_boxes)
    target_covered = (
        candidate_overlaps.max(dim=0).values >= 0.5
        if target_boxes.shape[0]
        else torch.empty(0, dtype=torch.bool)
    )
    full_coverage = bool(target_covered.all()) if target_covered.numel() else True
    selected_overlaps = pairwise_iou(selected_boxes, target_boxes)
    matches = greedy_one_to_one_matches(selected_overlaps, threshold=0.5)
    success = (
        len(matches) == target_boxes.shape[0]
        and selected_boxes.shape[0] == target_boxes.shape[0]
    )
    duplicate = False
    if selected_boxes.shape[0] >= 2 and target_boxes.shape[0]:
        best_values, best_targets = selected_overlaps.max(dim=1)
        assigned = [
            int(target)
            for target, value in zip(best_targets.tolist(), best_values.tolist())
            if value >= 0.5
        ]
        duplicate = len(assigned) != len(set(assigned))

    if success:
        category = "success"
    elif not full_coverage:
        category = "proposal_miss"
    elif selected_boxes.shape[0] < target_boxes.shape[0]:
        category = "count_under"
    elif selected_boxes.shape[0] > target_boxes.shape[0]:
        category = "count_over"
    elif duplicate:
        category = "duplicate_selection"
    else:
        category = "ranking_selection_error"
    return category, {
        "num_candidates": int(candidate_boxes.shape[0]),
        "num_predictions": int(selected_boxes.shape[0]),
        "num_targets": int(target_boxes.shape[0]),
        "matched_targets": len(matches),
        "full_proposal_coverage": full_coverage,
        "duplicate_selected_target": duplicate,
    }


def main() -> None:
    args = parse_args()
    calibration_payload = json.loads(Path(args.calibration_json).read_text())
    calibration = calibration_payload["best"]
    if any(calibration_payload["best_on_boundary"].values()):
        raise ValueError("calibration has an active artificial boundary")
    threshold = float(calibration["membership_threshold"])
    bias = torch.tensor(calibration["count_logit_bias"], dtype=torch.float32)

    device = get_device(args.device)
    dataset = ClipFeatureDataset(args.feature_file, split_file=args.split_file)
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

    nms_thresholds = sorted(set(float(value) for value in args.nms_thresholds))
    pre_nms_membership_thresholds = sorted(
        set(
            [threshold]
            + [float(value) for value in args.pre_nms_membership_thresholds]
        )
    )
    if any(
        value < 0.0 or value > 1.0
        for value in pre_nms_membership_thresholds
    ):
        raise ValueError("pre-NMS membership thresholds must be in [0, 1]")
    records: dict[str, list[PredictionRecord]] = {
        "none": [],
        **{f"nms_{value:g}": [] for value in nms_thresholds},
        **{f"pre_nms_{value:g}": [] for value in nms_thresholds},
        **{
            f"pre_nms_{value:g}_thr_{selection_threshold:g}": []
            for value in nms_thresholds
            for selection_threshold in pre_nms_membership_thresholds
            if selection_threshold != threshold
        },
    }
    failures: dict[str, Counter[str]] = defaultdict(Counter)
    failures_by_policy: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qualitative_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_counts: Counter[str] = Counter()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Stage 6 multi-target diagnosis"):
            outputs = model(batch)
            count_logits = outputs["count_logits"].detach().cpu().float()
            for index, logits_device in enumerate(outputs["membership_logits"]):
                logits = logits_device.detach().cpu().float()
                metadata = batch["metadata"][index]
                candidate_boxes = normalized_boxes_to_xyxy(
                    batch["candidate_boxes_norm"][index],
                    int(metadata["width"]),
                    int(metadata["height"]),
                )
                target_boxes = batch["target_boxes_xyxy"][index].float()
                pred_count = int(torch.argmax(count_logits[index] + bias).item())
                selected = sorted(
                    select_cardinality_gated_indices(
                        logits, pred_count, membership_threshold=threshold
                    )
                )
                probabilities = torch.sigmoid(logits)
                selected_boxes = (
                    candidate_boxes[selected]
                    if selected
                    else torch.empty((0, 4), dtype=torch.float32)
                )
                selected_scores = (
                    probabilities[selected]
                    if selected
                    else torch.empty(0, dtype=torch.float32)
                )
                selected_boxes_by_policy = {"none": selected_boxes}
                sample_id = str(batch["sample_ids"][index])
                target_type = metadata["target_type"]
                variants = {"none": list(range(len(selected)))}
                variants.update(
                    {
                        f"nms_{value:g}": greedy_nms(
                            selected_boxes, selected_scores, value
                        )
                        for value in nms_thresholds
                    }
                )
                for name, keep in variants.items():
                    records[name].append(
                        PredictionRecord(
                            sample_id=sample_id,
                            predicted_boxes=selected_boxes[keep],
                            predicted_scores=selected_scores[keep],
                            target_boxes=target_boxes,
                            target_type=target_type,
                            predicted_count_class=pred_count,
                        )
                    )
                # Suppressing the full ranked candidate pool before cardinality
                # selection can replace a duplicate top box with the next distinct
                # region. Post-selection NMS cannot do that replacement and may
                # therefore reduce an already small predicted set.
                for value in nms_thresholds:
                    available = greedy_nms(candidate_boxes, probabilities, value)
                    filtered_logits = logits[available]
                    for selection_threshold in pre_nms_membership_thresholds:
                        policy_name = (
                            f"pre_nms_{value:g}"
                            if selection_threshold == threshold
                            else (
                                f"pre_nms_{value:g}_thr_"
                                f"{selection_threshold:g}"
                            )
                        )
                        filtered_selected = sorted(
                            select_cardinality_gated_indices(
                                filtered_logits,
                                pred_count,
                                membership_threshold=selection_threshold,
                            )
                        )
                        global_selected = [
                            available[i] for i in filtered_selected
                        ]
                        pre_nms_boxes = (
                            candidate_boxes[global_selected]
                            if global_selected
                            else torch.empty((0, 4), dtype=torch.float32)
                        )
                        selected_boxes_by_policy[policy_name] = pre_nms_boxes
                        records[policy_name].append(
                            PredictionRecord(
                                sample_id=sample_id,
                                predicted_boxes=pre_nms_boxes,
                                predicted_scores=(
                                    probabilities[global_selected]
                                    if global_selected
                                    else torch.empty(0, dtype=torch.float32)
                                ),
                                target_boxes=target_boxes,
                                target_type=target_type,
                                predicted_count_class=pred_count,
                            )
                        )

                base_overlaps = pairwise_iou(selected_boxes, target_boxes)
                base_matches = greedy_one_to_one_matches(
                    base_overlaps, threshold=0.5
                )
                is_success = (
                    len(base_matches) == target_boxes.shape[0]
                    and selected_boxes.shape[0] == target_boxes.shape[0]
                )
                num_targets = int(target_boxes.shape[0])
                broad_group = (
                    "no_target"
                    if num_targets == 0
                    else "single_target"
                    if num_targets == 1
                    else "two_target"
                    if num_targets == 2
                    else "multi_target"
                )
                qualitative_category = (
                    f"{broad_group}_{'success' if is_success else 'failure'}"
                )
                if len(qualitative_examples[qualitative_category]) < 2:
                    qualitative_examples[qualitative_category].append(
                        {
                            "sample_id": sample_id,
                            "image_id": metadata.get("image_id"),
                            "file_name": metadata.get("file_name"),
                            "expression": batch["expressions"][index],
                            "category": qualitative_category,
                            "num_targets": num_targets,
                            "num_predictions": int(selected_boxes.shape[0]),
                            "matched_targets": len(base_matches),
                            "target_boxes_xyxy": target_boxes.tolist(),
                            "predicted_boxes_xyxy": selected_boxes.tolist(),
                            "predicted_scores": selected_scores.tolist(),
                        }
                    )
                enhanced_policy = "pre_nms_0.3_thr_0.5"
                if enhanced_policy in selected_boxes_by_policy:
                    enhanced_boxes = selected_boxes_by_policy[enhanced_policy]
                    enhanced_overlaps = pairwise_iou(
                        enhanced_boxes, target_boxes
                    )
                    enhanced_matches = greedy_one_to_one_matches(
                        enhanced_overlaps, threshold=0.5
                    )
                    enhanced_success = (
                        len(enhanced_matches) == target_boxes.shape[0]
                        and enhanced_boxes.shape[0] == target_boxes.shape[0]
                    )
                    enhanced_category = (
                        f"enhanced_{broad_group}_"
                        f"{'success' if enhanced_success else 'failure'}"
                    )
                    if len(qualitative_examples[enhanced_category]) < 2:
                        enhanced_scores = torch.tensor(
                            [
                                float(probabilities[
                                    torch.argmin(
                                        torch.sum(
                                            torch.abs(
                                                candidate_boxes - box
                                            ),
                                            dim=1,
                                        )
                                    )
                                ])
                                for box in enhanced_boxes
                            ],
                            dtype=torch.float32,
                        )
                        qualitative_examples[enhanced_category].append(
                            {
                                "sample_id": sample_id,
                                "image_id": metadata.get("image_id"),
                                "file_name": metadata.get("file_name"),
                                "expression": batch["expressions"][index],
                                "category": enhanced_category,
                                "policy": enhanced_policy,
                                "num_targets": num_targets,
                                "num_predictions": int(
                                    enhanced_boxes.shape[0]
                                ),
                                "matched_targets": len(enhanced_matches),
                                "target_boxes_xyxy": target_boxes.tolist(),
                                "predicted_boxes_xyxy": enhanced_boxes.tolist(),
                                "predicted_scores": enhanced_scores.tolist(),
                            }
                        )

                if target_boxes.shape[0] >= 3:
                    group = exact_group(int(target_boxes.shape[0]))
                    exact_counts[group] += 1
                    category, details = classify_failure(
                        candidate_boxes, selected_boxes, target_boxes
                    )
                    failures[group][category] += 1
                    for policy_name, policy_boxes in selected_boxes_by_policy.items():
                        policy_category, _ = classify_failure(
                            candidate_boxes, policy_boxes, target_boxes
                        )
                        failures_by_policy[policy_name][group][
                            policy_category
                        ] += 1
                    if len(examples[category]) < 10:
                        examples[category].append(
                            {
                                "sample_id": sample_id,
                                "image_id": metadata.get("image_id"),
                                "file_name": metadata.get("file_name"),
                                "expression": batch["expressions"][index],
                                "group": group,
                                "category": category,
                                "target_boxes_xyxy": target_boxes.tolist(),
                                "predicted_boxes_xyxy": selected_boxes.tolist(),
                                "predicted_scores": selected_scores.tolist(),
                                **details,
                            }
                        )

    metric_rows = {
        name: evaluate_records(
            value,
            match_threshold=0.5,
            overlap_metric="giou",
            image_f1_threshold=1.0,
        )
        for name, value in records.items()
    }
    failure_rows = {}
    categories = (
        "success",
        "proposal_miss",
        "count_under",
        "count_over",
        "duplicate_selection",
        "ranking_selection_error",
    )
    for group in ("3", "4", "5", "6+"):
        total = exact_counts[group]
        failure_rows[group] = {
            "num_samples": total,
            "counts": {name: failures[group][name] for name in categories},
            "fractions": {
                name: failures[group][name] / total if total else None
                for name in categories
            },
        }

    payload = {
        "stage": "6.3 multi-target failure and duplicate-suppression diagnosis",
        "scope": "locked development split",
        "checkpoint": args.checkpoint,
        "calibration_json": args.calibration_json,
        "membership_threshold": threshold,
        "count_logit_bias": bias.tolist(),
        "exact_count_failures": failure_rows,
        "exact_count_failures_by_policy": {
            policy_name: {
                group: {
                    "num_samples": exact_counts[group],
                    "counts": {
                        name: policy_groups[group][name]
                        for name in categories
                    },
                    "fractions": {
                        name: (
                            policy_groups[group][name] / exact_counts[group]
                            if exact_counts[group]
                            else None
                        )
                        for name in categories
                    },
                }
                for group in ("3", "4", "5", "6+")
            }
            for policy_name, policy_groups in failures_by_policy.items()
        },
        "inference_suppression": {
            name: {
                "count_macro_mean_f1": sum(
                    row["by_count_group"][group]["mean_f1"]
                    for group in ("0", "1", "2", "3+")
                )
                / 4,
                "official": row["official"],
                "by_count_group": row["by_count_group"],
            }
            for name, row in metric_rows.items()
        },
        "examples": examples,
        "qualitative_examples": qualitative_examples,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "Stage 6.3 Multi-Target Failure Diagnosis",
        "========================================",
        "group | n | success | proposal miss | count under | count over | duplicate | ranking",
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for group, row in failure_rows.items():
        count = row["counts"]
        lines.append(
            f"{group} | {row['num_samples']} | {count['success']} | "
            f"{count['proposal_miss']} | {count['count_under']} | "
            f"{count['count_over']} | {count['duplicate_selection']} | "
            f"{count['ranking_selection_error']}"
        )
    lines.extend(
        [
            "",
            "policy | macro F1 | official F1",
            "--- | ---: | ---:",
        ]
    )
    for name, row in payload["inference_suppression"].items():
        lines.append(
            f"{name} | {row['count_macro_mean_f1']:.6f} | "
            f"{row['official']['F1_score']:.6f}"
        )
    Path(args.output_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(Path(args.output_txt).read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
