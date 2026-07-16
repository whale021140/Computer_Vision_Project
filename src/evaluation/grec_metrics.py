from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Literal, Sequence

import torch


OverlapMetric = Literal["iou", "giou"]


def _as_boxes(value: Any, name: str) -> torch.Tensor:
    boxes = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    boxes = boxes.reshape(-1, 4)
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} must contain only finite values.")
    if torch.any(boxes[:, 2:] < boxes[:, :2]):
        raise ValueError(f"{name} must use valid [x1, y1, x2, y2] boxes.")
    return boxes


def _as_scores(value: Any, num_boxes: int) -> torch.Tensor:
    if value is None:
        return torch.ones(num_boxes, dtype=torch.float32)
    scores = torch.as_tensor(value, dtype=torch.float32).detach().cpu().reshape(-1)
    if scores.numel() != num_boxes:
        raise ValueError(
            "predicted_scores must contain one score per predicted box: "
            f"got {scores.numel()} scores for {num_boxes} boxes."
        )
    if not torch.isfinite(scores).all():
        raise ValueError("predicted_scores must contain only finite values.")
    return scores


def target_type_from_count(num_targets: int) -> str:
    if num_targets == 0:
        return "no-target"
    if num_targets == 1:
        return "single-target"
    return "multi-target"


def count_to_class(num_targets: int) -> int:
    return min(max(int(num_targets), 0), 3)


@dataclass
class PredictionRecord:
    sample_id: str
    predicted_boxes: torch.Tensor
    target_boxes: torch.Tensor
    predicted_scores: torch.Tensor | None = None
    target_type: str | None = None
    predicted_count_class: int | None = None

    def __post_init__(self) -> None:
        self.predicted_boxes = _as_boxes(self.predicted_boxes, "predicted_boxes")
        self.target_boxes = _as_boxes(self.target_boxes, "target_boxes")
        self.predicted_scores = _as_scores(
            self.predicted_scores,
            self.predicted_boxes.shape[0],
        )

        inferred_type = target_type_from_count(self.target_boxes.shape[0])
        if self.target_type is None:
            self.target_type = inferred_type
        elif self.target_type != inferred_type:
            raise ValueError(
                f"target_type={self.target_type!r} conflicts with "
                f"{self.target_boxes.shape[0]} target boxes ({inferred_type!r})."
            )

        if self.predicted_count_class is not None:
            predicted_class = int(self.predicted_count_class)
            if predicted_class not in (0, 1, 2, 3):
                raise ValueError("predicted_count_class must be one of 0, 1, 2, or 3.")
            self.predicted_count_class = predicted_class

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PredictionRecord":
        return cls(
            sample_id=str(value["sample_id"]),
            predicted_boxes=value.get(
                "predicted_boxes",
                value.get("predicted_boxes_xyxy", []),
            ),
            predicted_scores=value.get("predicted_scores"),
            target_boxes=value.get("target_boxes", value.get("target_boxes_xyxy", [])),
            target_type=value.get("target_type"),
            predicted_count_class=value.get("predicted_count_class"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "predicted_boxes_xyxy": self.predicted_boxes.tolist(),
            "predicted_scores": self.predicted_scores.tolist(),
            "target_boxes_xyxy": self.target_boxes.tolist(),
            "target_type": self.target_type,
            "predicted_count_class": self.predicted_count_class,
        }


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    boxes1 = _as_boxes(boxes1, "boxes1")
    boxes2 = _as_boxes(boxes2, "boxes2")
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.empty((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def pairwise_generalized_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    boxes1 = _as_boxes(boxes1, "boxes1")
    boxes2 = _as_boxes(boxes2, "boxes2")
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.empty((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    iou = pairwise_iou(boxes1, boxes2)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    top_left_intersection = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right_intersection = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_size = (bottom_right_intersection - top_left_intersection).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    union = area1[:, None] + area2[None, :] - intersection

    top_left_enclosing = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right_enclosing = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_size = (bottom_right_enclosing - top_left_enclosing).clamp(min=0)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    penalty = torch.where(
        enclosing_area > 0,
        (enclosing_area - union) / enclosing_area,
        torch.zeros_like(enclosing_area),
    )
    return iou - penalty


def pairwise_overlap(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    metric: OverlapMetric,
) -> torch.Tensor:
    if metric == "iou":
        return pairwise_iou(boxes1, boxes2)
    if metric == "giou":
        return pairwise_generalized_iou(boxes1, boxes2)
    raise ValueError(f"Unknown overlap metric: {metric!r}")


def greedy_one_to_one_matches(
    overlaps: torch.Tensor,
    threshold: float = 0.5,
) -> List[tuple[int, int, float]]:
    """Match the highest-overlap remaining pair, as in the released evaluator."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("match threshold must be between 0 and 1.")
    if overlaps.ndim != 2:
        raise ValueError("overlaps must be a two-dimensional matrix.")
    if overlaps.shape[0] == 0 or overlaps.shape[1] == 0:
        return []

    work = overlaps.detach().cpu().clone()
    matches: List[tuple[int, int, float]] = []
    for _ in range(min(work.shape)):
        flat_index = int(torch.argmax(work).item())
        row = flat_index // work.shape[1]
        col = flat_index % work.shape[1]
        value = float(work[row, col].item())
        if value < threshold:
            break
        matches.append((row, col, value))
        work[row, :] = float("-inf")
        work[:, col] = float("-inf")
    return matches


@dataclass
class SampleMetrics:
    sample_id: str
    target_type: str
    num_predictions: int
    num_targets: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    exact_set: float
    image_correct: float
    has_prediction: bool
    has_localized_target: bool
    cardinality_correct: bool
    count_class_correct: bool | None


def evaluate_sample(
    record: PredictionRecord,
    match_threshold: float = 0.5,
    overlap_metric: OverlapMetric = "iou",
    prediction_score_threshold: float | None = None,
    image_f1_threshold: float = 1.0,
) -> SampleMetrics:
    if not 0.0 <= image_f1_threshold <= 1.0:
        raise ValueError("image_f1_threshold must be between 0 and 1.")
    if prediction_score_threshold is None:
        keep = torch.ones(record.predicted_boxes.shape[0], dtype=torch.bool)
    else:
        keep = record.predicted_scores >= float(prediction_score_threshold)

    predicted_boxes = record.predicted_boxes[keep]
    num_predictions = int(predicted_boxes.shape[0])
    num_targets = int(record.target_boxes.shape[0])

    overlaps = pairwise_overlap(predicted_boxes, record.target_boxes, overlap_metric)
    matches = greedy_one_to_one_matches(overlaps, threshold=match_threshold)
    true_positives = len(matches)
    false_positives = num_predictions - true_positives
    false_negatives = num_targets - true_positives

    if num_predictions == 0 and num_targets == 0:
        precision = recall = f1 = 1.0
    else:
        precision = true_positives / num_predictions if num_predictions else 0.0
        recall = true_positives / num_targets if num_targets else 0.0
        denominator = 2 * true_positives + false_positives + false_negatives
        f1 = 2 * true_positives / denominator if denominator else 0.0

    true_count_class = count_to_class(num_targets)
    count_class_correct = (
        None
        if record.predicted_count_class is None
        else record.predicted_count_class == true_count_class
    )

    return SampleMetrics(
        sample_id=record.sample_id,
        target_type=str(record.target_type),
        num_predictions=num_predictions,
        num_targets=num_targets,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_set=float(f1 + 1e-12 >= 1.0),
        image_correct=float(f1 + 1e-12 >= image_f1_threshold),
        has_prediction=num_predictions > 0,
        has_localized_target=true_positives > 0,
        cardinality_correct=num_predictions == num_targets,
        count_class_correct=count_class_correct,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _summarize_samples(samples: Sequence[SampleMetrics]) -> Dict[str, Any]:
    tp = sum(sample.true_positives for sample in samples)
    fp = sum(sample.false_positives for sample in samples)
    fn = sum(sample.false_negatives for sample in samples)
    count_class_values = [
        float(sample.count_class_correct)
        for sample in samples
        if sample.count_class_correct is not None
    ]

    return {
        "num_samples": len(samples),
        "mean_precision": _mean(sample.precision for sample in samples),
        "mean_recall": _mean(sample.recall for sample in samples),
        "mean_f1": _mean(sample.f1 for sample in samples),
        "exact_set_accuracy": _mean(sample.exact_set for sample in samples),
        "cardinality_accuracy": _mean(float(sample.cardinality_correct) for sample in samples),
        "count_class_accuracy": _mean(count_class_values),
        "count_class_evaluated": len(count_class_values),
        "micro_precision": _safe_ratio(tp, tp + fp),
        "micro_recall": _safe_ratio(tp, tp + fn),
        "micro_f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "total_tp": tp,
        "total_fp": fp,
        "total_fn": fn,
    }


def evaluate_records(
    records: Sequence[PredictionRecord],
    match_threshold: float = 0.5,
    overlap_metric: OverlapMetric = "iou",
    prediction_score_threshold: float | None = None,
    image_f1_threshold: float = 1.0,
    include_sample_metrics: bool = False,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("At least one prediction record is required.")

    samples = [
        evaluate_sample(
            record,
            match_threshold=match_threshold,
            overlap_metric=overlap_metric,
            prediction_score_threshold=prediction_score_threshold,
            image_f1_threshold=image_f1_threshold,
        )
        for record in records
    ]
    no_target = [sample for sample in samples if sample.target_type == "no-target"]
    targeted = [sample for sample in samples if sample.target_type != "no-target"]
    single = [sample for sample in samples if sample.target_type == "single-target"]
    multi = [sample for sample in samples if sample.target_type == "multi-target"]

    official = {
        "F1_score": _mean(sample.image_correct for sample in samples),
        "T_acc": _mean(float(sample.has_prediction) for sample in targeted),
        "N_acc": _mean(float(not sample.has_prediction) for sample in no_target),
    }

    diagnostics = _summarize_samples(samples)
    diagnostics.update(
        {
            "no_target_total": len(no_target),
            "no_target_accuracy": _mean(float(not sample.has_prediction) for sample in no_target),
            "false_grounding_rate": _mean(float(sample.has_prediction) for sample in no_target),
            "single_target_total": len(single),
            "single_target_localization_accuracy": _mean(
                float(sample.has_localized_target) for sample in single
            ),
            "single_target_exact_accuracy": _mean(sample.exact_set for sample in single),
            "multi_target_total": len(multi),
            "multi_target_mean_f1": _mean(sample.f1 for sample in multi),
            "multi_target_exact_accuracy": _mean(sample.exact_set for sample in multi),
        }
    )

    result: Dict[str, Any] = {
        "config": {
            "match_threshold": match_threshold,
            "overlap_metric": overlap_metric,
            "prediction_score_threshold": prediction_score_threshold,
            "image_f1_threshold": image_f1_threshold,
        },
        "official": official,
        "diagnostics": diagnostics,
        "by_target_type": {
            target_type: _summarize_samples(
                [sample for sample in samples if sample.target_type == target_type]
            )
            for target_type in ("no-target", "single-target", "multi-target")
        },
    }
    if include_sample_metrics:
        result["samples"] = [asdict(sample) for sample in samples]
    return result
