from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.data.candidate_dataset import CandidateBoxDataset, candidate_collate_fn
from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.evaluation.metrics import compute_set_metrics
from src.models.baseline_heads import ClipCandidateBaseline
from src.training.train_clip_baseline import build_count_loss, compute_losses
from src.utils.boxes import count_to_class, normalize_xyxy, xywh_to_xyxy


class BoxUtilityTests(unittest.TestCase):
    def test_box_conversion_and_normalization(self) -> None:
        box = xywh_to_xyxy([10, 20, 30, 40])
        self.assertEqual(box, [10, 20, 40, 60])
        self.assertEqual(normalize_xyxy(box, 100, 200), [0.1, 0.1, 0.4, 0.3])

    def test_count_classes(self) -> None:
        self.assertEqual([count_to_class(i) for i in range(6)], [0, 1, 2, 3, 3, 3])


class MetricTests(unittest.TestCase):
    def test_set_metrics_for_empty_and_partial_predictions(self) -> None:
        empty = compute_set_metrics(set(), set())
        self.assertEqual((empty.precision, empty.recall, empty.f1, empty.exact), (1, 1, 1, 1))

        partial = compute_set_metrics({0, 2}, {0, 1})
        self.assertEqual((partial.tp, partial.fp, partial.fn), (1, 1, 1))
        self.assertAlmostEqual(partial.f1, 0.5)
        self.assertEqual(partial.exact, 0)


class CandidateDatasetTests(unittest.TestCase):
    def test_candidate_dataset_and_collate_without_images(self) -> None:
        record = {
            "sample_id": "sample-0",
            "expression": "the object",
            "target_type": "single-target",
            "target_ann_ids": [11],
            "target_boxes_xyxy": [[1, 2, 11, 12]],
            "target_boxes_norm": [[0.01, 0.02, 0.11, 0.12]],
            "candidate_ann_ids": [11, 12],
            "candidate_boxes_xyxy": [[1, 2, 11, 12], [20, 20, 30, 30]],
            "candidate_boxes_norm": [[0.01, 0.02, 0.11, 0.12], [0.2, 0.2, 0.3, 0.3]],
            "candidate_labels": [1, 0],
            "count_class": 1,
            "num_targets": 1,
            "num_candidates": 2,
            "ref_id": 1,
            "sent_id": 2,
            "image_id": 3,
            "file_name": "unused.jpg",
            "width": 100,
            "height": 100,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = Path(tmpdir) / "candidates.jsonl"
            candidate_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = CandidateBoxDataset(
                candidate_file=str(candidate_path),
                image_root=tmpdir,
                load_image=False,
            )

            sample = dataset[0]
            self.assertEqual(tuple(sample["candidate_boxes_xyxy"].shape), (2, 4))
            self.assertEqual(sample["candidate_labels"].tolist(), [1, 0])

            batch = candidate_collate_fn([sample, sample])
            self.assertEqual(tuple(batch["count_class"].shape), (2,))
            self.assertEqual(len(batch["candidate_boxes_xyxy"]), 2)


class BaselineTrainingTests(unittest.TestCase):
    @staticmethod
    def _record(sample_id: str, count_class: int) -> dict:
        feature_dim = 4
        return {
            "sample_id": sample_id,
            "metadata": {"target_type": "single-target"},
            "expression": "the object",
            "text_feature": torch.ones(feature_dim),
            "candidate_features": torch.eye(3, feature_dim),
            "candidate_text_similarity": torch.tensor([0.5, 0.2, -0.1]),
            "candidate_boxes_norm": torch.tensor(
                [[0.0, 0.0, 0.2, 0.2], [0.2, 0.2, 0.4, 0.4], [0.4, 0.4, 0.6, 0.6]]
            ),
            "candidate_labels": torch.tensor([1.0, 0.0, 0.0]),
            "count_class": torch.tensor(count_class),
            "target_boxes_xyxy": torch.tensor([[0.0, 0.0, 20.0, 20.0]]),
            "target_boxes_norm": torch.tensor([[0.0, 0.0, 0.2, 0.2]]),
        }

    def test_weighted_and_unweighted_count_losses(self) -> None:
        device = torch.device("cpu")
        unweighted = build_count_loss(None, device)
        self.assertIsNone(unweighted.weight)

        weighted = build_count_loss([15.0, 1.0, 1.5, 2.0], device)
        self.assertTrue(torch.equal(weighted.weight, torch.tensor([15.0, 1.0, 1.5, 2.0])))

        for invalid in ([1.0, 2.0], [-1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_count_loss(invalid, device)

    def test_feature_dataset_forward_and_losses(self) -> None:
        records = [self._record("sample-0", 1), self._record("sample-1", 1)]

        with tempfile.TemporaryDirectory() as tmpdir:
            feature_path = Path(tmpdir) / "features.pt"
            torch.save(
                {
                    "feature_dim": 4,
                    "clip_model": "synthetic",
                    "records": records,
                },
                feature_path,
            )
            dataset = ClipFeatureDataset(str(feature_path))
            batch = clip_feature_collate_fn([dataset[0], dataset[1]])

            model = ClipCandidateBaseline(feature_dim=4, hidden_dim=8, dropout=0.0)
            outputs = model(batch)
            self.assertEqual(len(outputs["membership_logits"]), 2)
            self.assertEqual(tuple(outputs["membership_logits"][0].shape), (3,))
            self.assertEqual(tuple(outputs["count_logits"].shape), (2, 4))

            total, membership, cardinality = compute_losses(
                outputs=outputs,
                batch=batch,
                bce_loss=nn.BCEWithLogitsLoss(),
                ce_loss=build_count_loss(None, torch.device("cpu")),
                device=torch.device("cpu"),
                lambda_cardinality=1.0,
            )
            self.assertTrue(torch.isfinite(total))
            self.assertTrue(torch.isfinite(membership))
            self.assertTrue(torch.isfinite(cardinality))

    def test_model_rejects_empty_candidate_sets(self) -> None:
        model = ClipCandidateBaseline(feature_dim=4, hidden_dim=8)
        batch = {
            "text_features": [torch.ones(4)],
            "candidate_features": [torch.empty((0, 4))],
            "candidate_text_similarity": [torch.empty(0)],
            "candidate_boxes_norm": [torch.empty((0, 4))],
        }
        with self.assertRaisesRegex(ValueError, "no candidate features"):
            model(batch)


if __name__ == "__main__":
    unittest.main()
