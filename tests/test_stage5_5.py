from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.data.create_stage5_5_splits import (
    build_shadow_dev,
    sample_exact_count_groups,
)
from src.data.feature_dataset import ClipFeatureDataset
from src.models.baseline_heads import ClipCandidateBaseline
from src.training.train_clip_baseline import effective_number_weights


class Stage55ModelTests(unittest.TestCase):
    def test_hierarchical_mean_max_stats_forward(self) -> None:
        model = ClipCandidateBaseline(
            candidate_feature_dim=4,
            text_feature_dim=4,
            hidden_dim=8,
            pooling="mean_max_stats",
            hierarchical_cardinality=True,
        )
        batch = {
            "text_features": [torch.randn(4), torch.randn(4)],
            "candidate_features": [torch.randn(3, 4), torch.randn(2, 4)],
            "candidate_text_similarity": [torch.randn(3), torch.randn(2)],
            "candidate_boxes_norm": [torch.rand(3, 4), torch.rand(2, 4)],
        }
        outputs = model(batch)
        self.assertEqual(tuple(outputs["count_logits"].shape), (2, 4))
        self.assertEqual(tuple(outputs["presence_logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["positive_count_logits"].shape), (2, 3))
        probabilities = torch.softmax(outputs["count_logits"], dim=1)
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(2)))

    def test_effective_number_upweights_rare_class(self) -> None:
        weights = effective_number_weights([100, 1000, 500, 10], beta=0.9999)
        self.assertAlmostEqual(sum(weights), 4.0)
        self.assertGreater(weights[3], weights[0])
        self.assertGreater(weights[0], weights[1])


class Stage55DataTests(unittest.TestCase):
    def test_one_to_one_policy_keeps_one_candidate_per_target(self) -> None:
        cache = {
            "feature_dim": 2,
            "candidate_feature_dim": 2,
            "text_feature_dim": 2,
            "images": {
                "1": {
                    "candidate_features": torch.ones(2, 2),
                    "candidate_boxes_norm": torch.tensor(
                        [[0.0, 0.0, 0.5, 0.5], [0.02, 0.02, 0.52, 0.52]]
                    ),
                }
            },
            "records": [
                {
                    "sample_id": "x",
                    "image_id": 1,
                    "metadata": {"ref_id": 1, "sent_id": 1, "image_id": 1},
                    "expression": "object",
                    "text_feature": torch.ones(2),
                    "candidate_labels": torch.ones(2),
                    "count_class": torch.tensor(1),
                    "target_boxes_xyxy": torch.tensor([[0.0, 0.0, 50.0, 50.0]]),
                    "target_boxes_norm": torch.tensor([[0.0, 0.0, 0.5, 0.5]]),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pt"
            torch.save(cache, path)
            dataset = ClipFeatureDataset(str(path), label_policy="one-to-one")
            labels = dataset[0]["candidate_labels"]
        self.assertEqual(labels.tolist(), [1.0, 0.0])

    def test_shadow_images_are_excluded_before_exact_sampling(self) -> None:
        samples = []
        ref_id = 1
        for group, targets in (("0", 0), ("1", 1), ("2", 2), ("3+", 4)):
            for image_offset in range(4):
                samples.append(
                    {
                        "ref_id": ref_id,
                        "sent_id": ref_id,
                        "image_id": ref_id,
                        "target_type": (
                            "no-target" if group == "0" else
                            "single-target" if group == "1" else "multi-target"
                        ),
                        "num_targets": targets,
                    }
                )
                ref_id += 1
        shadow, pool = build_shadow_dev(
            samples, {group: 1 for group in ("0", "1", "2", "3+")}, seed=3
        )
        train = sample_exact_count_groups(
            pool, {group: 1 for group in ("0", "1", "2", "3+")}, seed=4
        )
        self.assertFalse(
            {row["image_id"] for row in shadow}
            & {row["image_id"] for row in train}
        )


if __name__ == "__main__":
    unittest.main()
