from __future__ import annotations

import unittest

import torch

from src.data.create_stage5_6_splits import (
    apportion_count_targets,
    build_image_disjoint_dev,
    build_nested_count_stratified_subsets,
)
from src.data.create_fewshot_splits import sample_key
from src.evaluation.recalibrate_stage5_6 import (
    fast_grid_rows,
    precompute_option_metrics,
)
from src.evaluation.select_stage5_5_checkpoint import evaluate_setting


def sample(ref_id: int, image_id: int, count: int) -> dict:
    return {
        "ref_id": ref_id,
        "sent_id": ref_id,
        "image_id": image_id,
        "target_type": (
            "no-target" if count == 0 else
            "single-target" if count == 1 else "multi-target"
        ),
        "num_targets": count,
    }


class Stage56SplitTests(unittest.TestCase):
    def test_largest_remainder_preserves_total_budget(self) -> None:
        counts = {"0": 19140, "1": 120624, "2": 67848, "3+": 1732}
        self.assertEqual(
            apportion_count_targets(counts, 1),
            {"0": 191, "1": 1206, "2": 679, "3+": 17},
        )
        self.assertEqual(sum(apportion_count_targets(counts, 10).values()), 20934)

    def test_dev_is_image_disjoint_and_covers_all_groups(self) -> None:
        rows = []
        ref_id = 0
        for count in (0, 1, 2, 4):
            for _ in range(6):
                ref_id += 1
                rows.append(sample(ref_id, ref_id, count))
        dev, pool = build_image_disjoint_dev(
            rows,
            {"0": 2, "1": 2, "2": 2, "3+": 2},
            seed=56,
        )
        self.assertFalse(
            {row["image_id"] for row in dev}
            & {row["image_id"] for row in pool}
        )
        groups = {
            "0": sum(row["num_targets"] == 0 for row in dev),
            "1": sum(row["num_targets"] == 1 for row in dev),
            "2": sum(row["num_targets"] == 2 for row in dev),
            "3+": sum(row["num_targets"] >= 3 for row in dev),
        }
        self.assertTrue(all(value >= 2 for value in groups.values()))

    def test_count_stratified_subsets_are_nested(self) -> None:
        rows = []
        ref_id = 0
        source_counts = {"0": 100, "1": 100, "2": 100, "3+": 100}
        for count in (0, 1, 2, 4):
            for _ in range(100):
                ref_id += 1
                rows.append(sample(ref_id, ref_id, count))
        subsets, targets = build_nested_count_stratified_subsets(
            rows, source_counts, [1, 5, 10], seed=0
        )
        keys = {
            percentage: {sample_key(row) for row in subset}
            for percentage, subset in subsets.items()
        }
        self.assertTrue(keys[1] <= keys[5] <= keys[10])
        for percentage, subset in subsets.items():
            self.assertEqual(len(subset), sum(targets[percentage].values()))


class Stage56WideCalibrationTests(unittest.TestCase):
    def test_fast_grid_matches_reference_evaluator(self) -> None:
        boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
                [60.0, 60.0, 70.0, 70.0],
            ]
        )
        rows = []
        target_counts = (0, 1, 2, 4)
        for index, count in enumerate(target_counts):
            rows.append(
                {
                    "sample_id": str(index),
                    "membership_logits": torch.tensor([2.0, 1.0, 0.0, -1.0]),
                    "count_logits": torch.tensor(
                        [0.1, 0.2, 0.3, 0.4]
                    ),
                    "candidate_boxes": boxes,
                    "target_boxes": boxes[:count],
                    "target_type": (
                        "no-target"
                        if count == 0
                        else "single-target"
                        if count == 1
                        else "multi-target"
                    ),
                }
            )
        thresholds = [0.3, 0.7]
        fast_rows = fast_grid_rows(
            precompute_option_metrics(rows, thresholds),
            class0_biases=[0.0, 1.0],
            class3_biases=[0.0, 1.0],
            thresholds=thresholds,
        )
        for fast in fast_rows:
            reference = evaluate_setting(
                rows,
                fast["class0_logit_bias"],
                fast["class3_logit_bias"],
                fast["membership_threshold"],
            )
            self.assertAlmostEqual(
                fast["count_macro_mean_f1"],
                reference["count_macro_mean_f1"],
            )
            for metric in ("F1_score", "T_acc", "N_acc"):
                self.assertAlmostEqual(
                    fast["official"][metric],
                    reference["official"][metric],
                )


if __name__ == "__main__":
    unittest.main()
