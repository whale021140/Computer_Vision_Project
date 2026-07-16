from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


def count_class_to_k(count_class: int, num_candidates: int) -> int:
    """Map count class {0,1,2,3} to number of selected boxes.

    Class 3 means 3-or-more. For this first baseline evaluator, we select top-3.
    """
    if count_class <= 0:
        return 0
    if count_class == 1:
        return min(1, num_candidates)
    if count_class == 2:
        return min(2, num_candidates)
    return min(3, num_candidates)


def select_topk_indices(membership_logits: torch.Tensor, count_class: int) -> set[int]:
    """Legacy exact top-k selection retained for old Milestone 2 artifacts."""
    num_candidates = int(membership_logits.numel())
    k = count_class_to_k(int(count_class), num_candidates)
    if k <= 0 or num_candidates == 0:
        return set()

    topk = torch.topk(membership_logits.detach().cpu(), k=k).indices.tolist()
    return set(int(i) for i in topk)


def select_cardinality_gated_indices(
    membership_logits: torch.Tensor,
    count_class: int,
    membership_threshold: float = 0.5,
) -> set[int]:
    """Select a variable-size set while retaining the cardinality head's gate.

    Count classes 0, 1, and 2 select exactly 0, 1, and 2 candidates. Class 3
    means 3-or-more: every candidate above the probability threshold is kept,
    with the top three added when fewer than three pass the threshold.
    """
    if not 0.0 <= membership_threshold <= 1.0:
        raise ValueError("membership_threshold must be between 0 and 1.")

    logits = membership_logits.detach().cpu().reshape(-1)
    num_candidates = int(logits.numel())
    count_class = int(count_class)
    if count_class not in (0, 1, 2, 3):
        raise ValueError("count_class must be one of 0, 1, 2, or 3.")
    if count_class == 0 or num_candidates == 0:
        return set()
    if count_class in (1, 2):
        k = min(count_class, num_candidates)
        return set(int(i) for i in torch.topk(logits, k=k).indices.tolist())

    probabilities = torch.sigmoid(logits)
    selected = set(
        int(i)
        for i in torch.nonzero(
            probabilities >= membership_threshold,
            as_tuple=False,
        ).flatten().tolist()
    )
    minimum_k = min(3, num_candidates)
    if len(selected) < minimum_k:
        selected.update(
            int(i) for i in torch.topk(logits, k=minimum_k).indices.tolist()
        )
    return selected


def positive_indices(candidate_labels: torch.Tensor) -> set[int]:
    labels = candidate_labels.detach().cpu()
    return set(int(i) for i in torch.nonzero(labels > 0.5, as_tuple=False).flatten().tolist())


@dataclass
class SetMetrics:
    precision: float
    recall: float
    f1: float
    exact: float
    tp: int
    fp: int
    fn: int


def compute_set_metrics(pred: set[int], gt: set[int]) -> SetMetrics:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    precision = 1.0 if len(pred) == 0 and len(gt) == 0 else (tp / len(pred) if len(pred) > 0 else 0.0)
    recall = 1.0 if len(pred) == 0 and len(gt) == 0 else (tp / len(gt) if len(gt) > 0 else 0.0)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    exact = 1.0 if pred == gt else 0.0
    return SetMetrics(precision=precision, recall=recall, f1=f1, exact=exact, tp=tp, fp=fp, fn=fn)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
