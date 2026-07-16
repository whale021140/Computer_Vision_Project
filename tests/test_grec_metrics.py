from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

from src.evaluation.calibrate_clip_baseline import (
    choose_best_threshold,
    evaluate_threshold,
)

from src.evaluation.grec_metrics import (
    PredictionRecord,
    evaluate_records,
    evaluate_sample,
    greedy_one_to_one_matches,
    pairwise_generalized_iou,
    pairwise_iou,
)
from src.evaluation.metrics import select_cardinality_gated_indices


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_released_generalized_box_iou():
    source = PROJECT_ROOT / "gRefCOCO" / "mdetr" / "util" / "box_ops.py"
    spec = importlib.util.spec_from_file_location("released_grefcoco_box_ops", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load released box operations from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generalized_box_iou


def released_reference_summary(
    records: list[PredictionRecord],
    score_threshold: float,
) -> dict[str, float]:
    """Small direct transcription of gRefCOCO's released summarize loop."""
    generalized_box_iou = load_released_generalized_box_iou()
    correct_images = 0
    target_present = 0
    target_present_predicted = 0
    no_target = 0
    no_target_empty = 0

    for record in records:
        predicted_boxes = record.predicted_boxes[
            record.predicted_scores >= score_threshold
        ]
        if record.target_boxes.shape[0] == 0:
            no_target += 1
            no_target_empty += int(predicted_boxes.shape[0] == 0)
            sample_f1 = float(predicted_boxes.shape[0] == 0)
        else:
            target_present += 1
            target_present_predicted += int(predicted_boxes.shape[0] > 0)
            overlaps = generalized_box_iou(predicted_boxes, record.target_boxes)
            true_positives = 0
            for _ in range(min(overlaps.shape)):
                best_value, best_index = torch.topk(overlaps.flatten(), 1)
                if float(best_value.item()) < 0.5:
                    break
                row = int(best_index.item()) // overlaps.shape[1]
                col = int(best_index.item()) % overlaps.shape[1]
                true_positives += 1
                overlaps[row, :] = 0.0
                overlaps[:, col] = 0.0
            false_positives = predicted_boxes.shape[0] - true_positives
            false_negatives = record.target_boxes.shape[0] - true_positives
            sample_f1 = 2 * true_positives / (
                2 * true_positives + false_positives + false_negatives
            )
        correct_images += int(sample_f1 >= 1.0)

    return {
        "F1_score": correct_images / len(records),
        "T_acc": target_present_predicted / target_present,
        "N_acc": no_target_empty / no_target,
    }


class BoxMatchingTests(unittest.TestCase):
    def test_pairwise_iou_and_greedy_matching(self) -> None:
        predictions = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
        )
        targets = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [25.0, 25.0, 35.0, 35.0]]
        )
        overlaps = pairwise_iou(predictions, targets)
        self.assertAlmostEqual(float(overlaps[0, 0]), 1.0)
        self.assertAlmostEqual(float(overlaps[1, 1]), 25.0 / 175.0)
        matches = greedy_one_to_one_matches(overlaps, threshold=0.5)
        self.assertEqual([(row, col) for row, col, _ in matches], [(0, 0)])

    def test_generalized_iou_agrees_with_released_grefcoco_code(self) -> None:
        predictions = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 14.0, 16.0]]
        )
        targets = torch.tensor(
            [[1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]]
        )
        released_generalized_box_iou = load_released_generalized_box_iou()
        expected = released_generalized_box_iou(predictions, targets)
        actual = pairwise_generalized_iou(predictions, targets)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_duplicate_predictions_match_one_target_only_once(self) -> None:
        record = PredictionRecord(
            sample_id="duplicate",
            predicted_boxes=[
                [0, 0, 10, 10],
                [0, 0, 10, 10],
            ],
            target_boxes=[[0, 0, 10, 10]],
        )
        result = evaluate_sample(record)
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 0)
        self.assertAlmostEqual(result.f1, 2 / 3)
        self.assertEqual(result.exact_set, 0)


class CardinalitySelectionTests(unittest.TestCase):
    def test_three_plus_can_return_more_than_three_candidates(self) -> None:
        probabilities = torch.tensor([0.95, 0.9, 0.8, 0.7, 0.1])
        logits = torch.logit(probabilities)
        selected = select_cardinality_gated_indices(
            logits,
            count_class=3,
            membership_threshold=0.6,
        )
        self.assertEqual(selected, {0, 1, 2, 3})

    def test_three_plus_always_returns_at_least_three_when_available(self) -> None:
        probabilities = torch.tensor([0.4, 0.3, 0.2, 0.1])
        logits = torch.logit(probabilities)
        selected = select_cardinality_gated_indices(
            logits,
            count_class=3,
            membership_threshold=0.9,
        )
        self.assertEqual(selected, {0, 1, 2})

    def test_exact_count_classes_remain_exact(self) -> None:
        logits = torch.tensor([0.1, 3.0, 2.0, -1.0])
        self.assertEqual(select_cardinality_gated_indices(logits, 0), set())
        self.assertEqual(select_cardinality_gated_indices(logits, 1), {1})
        self.assertEqual(select_cardinality_gated_indices(logits, 2), {1, 2})


class CalibrationTests(unittest.TestCase):
    def test_threshold_sweep_prefers_complete_three_plus_set(self) -> None:
        boxes = torch.tensor(
            [[i * 20.0, 0.0, i * 20.0 + 10.0, 10.0] for i in range(4)]
        )
        collected = [
            {
                "sample_id": "four-targets",
                "target_type": "multi-target",
                "membership_logits": torch.logit(
                    torch.tensor([0.9, 0.8, 0.7, 0.6])
                ),
                "predicted_count_class": 3,
                "candidate_boxes": boxes,
                "target_boxes": boxes.clone(),
            },
            {
                "sample_id": "no-target",
                "target_type": "no-target",
                "membership_logits": torch.tensor([2.0]),
                "predicted_count_class": 0,
                "candidate_boxes": boxes[:1],
                "target_boxes": torch.empty((0, 4)),
            },
        ]
        low = {
            "membership_threshold": 0.5,
            **evaluate_threshold(collected, 0.5),
        }
        high = {
            "membership_threshold": 0.8,
            **evaluate_threshold(collected, 0.8),
        }
        self.assertEqual(low["official"]["F1_score"], 1.0)
        self.assertEqual(high["official"]["F1_score"], 0.5)
        self.assertEqual(choose_best_threshold([high, low]), low)

    def test_calibration_tie_prefers_neutral_threshold(self) -> None:
        metrics = {
            "official": {"F1_score": 0.5, "N_acc": 0.5},
            "diagnostics": {"mean_f1": 0.5},
        }
        rows = [
            {"membership_threshold": threshold, **metrics}
            for threshold in (0.1, 0.5, 0.9)
        ]
        self.assertEqual(
            choose_best_threshold(rows)["membership_threshold"],
            0.5,
        )


class GRECMetricTests(unittest.TestCase):
    @staticmethod
    def records() -> list[PredictionRecord]:
        return [
            PredictionRecord("no-target-correct", [], [], []),
            PredictionRecord(
                "no-target-false",
                [[0, 0, 10, 10]],
                [],
                [0.9],
            ),
            PredictionRecord(
                "single-correct",
                [[0, 0, 10, 10]],
                [[0, 0, 10, 10]],
                [0.9],
                predicted_count_class=1,
            ),
            PredictionRecord(
                "single-empty",
                [],
                [[0, 0, 10, 10]],
                [],
                predicted_count_class=0,
            ),
            PredictionRecord(
                "multi-correct",
                [[0, 0, 10, 10], [20, 20, 30, 30]],
                [[0, 0, 10, 10], [20, 20, 30, 30]],
                [0.95, 0.85],
                predicted_count_class=2,
            ),
        ]

    def test_released_grec_summary_values(self) -> None:
        records = self.records()
        result = evaluate_records(
            records,
            overlap_metric="giou",
            match_threshold=0.5,
            prediction_score_threshold=0.7,
        )
        reference = released_reference_summary(records, score_threshold=0.7)
        self.assertEqual(result["official"], reference)
        self.assertAlmostEqual(result["official"]["F1_score"], 3 / 5)
        self.assertAlmostEqual(result["official"]["T_acc"], 2 / 3)
        self.assertAlmostEqual(result["official"]["N_acc"], 1 / 2)
        self.assertAlmostEqual(result["diagnostics"]["false_grounding_rate"], 1 / 2)
        self.assertAlmostEqual(
            result["diagnostics"]["single_target_localization_accuracy"],
            1 / 2,
        )

    def test_score_threshold_is_applied_before_matching(self) -> None:
        record = PredictionRecord(
            "low-score",
            [[0, 0, 10, 10]],
            [[0, 0, 10, 10]],
            [0.69],
        )
        below = evaluate_sample(record, prediction_score_threshold=0.7)
        included = evaluate_sample(record, prediction_score_threshold=0.6)
        self.assertEqual(below.num_predictions, 0)
        self.assertEqual(below.f1, 0)
        self.assertEqual(included.f1, 1)

    def test_partial_multi_target_and_more_than_three_targets(self) -> None:
        target_boxes = [[i * 20, 0, i * 20 + 10, 10] for i in range(5)]
        record = PredictionRecord(
            "five-targets",
            target_boxes[:4],
            target_boxes,
            [0.9] * 4,
            predicted_count_class=3,
        )
        result = evaluate_sample(record)
        self.assertEqual((result.true_positives, result.false_negatives), (4, 1))
        self.assertAlmostEqual(result.f1, 8 / 9)
        self.assertEqual(result.exact_set, 0)

    def test_record_rejects_inconsistent_target_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            PredictionRecord(
                "bad-target-type",
                [],
                [[0, 0, 10, 10]],
                [],
                target_type="no-target",
            )


if __name__ == "__main__":
    unittest.main()
