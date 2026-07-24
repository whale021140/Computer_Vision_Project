import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluation.audit_stage6_baseline import (
    active_calibration_boundaries,
    evaluation_matches_calibration,
    sha256_file,
)
from src.evaluation.analyze_stage6_inference_policies import (
    membership_only_indices,
)
from src.evaluation.analyze_stage6_multitarget_failures import (
    classify_failure,
    greedy_nms,
)
from src.evaluation.evaluate_clip_baseline import select_prediction_indices
from src.models.baseline_heads import ClipCandidateBaseline


class Stage60AuditTests(unittest.TestCase):
    def test_sha256_file_is_chunk_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(b"stage6-baseline" * 1024)
            self.assertEqual(
                sha256_file(path, chunk_size=7),
                sha256_file(path, chunk_size=4096),
            )

    def test_boundary_dictionary_requires_an_active_axis(self):
        self.assertEqual(
            active_calibration_boundaries(
                {
                    "best_on_boundary": {
                        "class0_at_min": False,
                        "class0_at_max": False,
                    }
                }
            ),
            [],
        )
        self.assertEqual(
            active_calibration_boundaries(
                {
                    "best_on_boundary": {
                        "class0_at_min": False,
                        "class0_at_max": True,
                    }
                }
            ),
            ["class0_at_max"],
        )

    def test_evaluation_must_reference_and_match_calibration(self):
        path = Path("outputs/example/calibration_dev.json")
        calibration = {
            "best": {
                "count_logit_bias": [2.0, 0.0, 0.0, 5.0],
                "membership_threshold": 0.85,
            }
        }
        evaluation = {
            "calibration_json": str(path),
            "count_logit_bias": [2.0, 0.0, 0.0, 5.0],
            "membership_threshold": 0.85,
        }
        self.assertTrue(
            evaluation_matches_calibration(evaluation, path, calibration)
        )
        evaluation["membership_threshold"] = 0.9
        self.assertFalse(
            evaluation_matches_calibration(evaluation, path, calibration)
        )


class Stage61InferenceTests(unittest.TestCase):
    def test_membership_only_allows_empty_and_variable_cardinality(self):
        logits = torch.tensor([-2.0, 0.0, 2.0])
        self.assertEqual(membership_only_indices(logits, 0.9), [])
        self.assertEqual(membership_only_indices(logits, 0.5), [1, 2])
        self.assertEqual(membership_only_indices(logits, 0.1), [0, 1, 2])
        self.assertEqual(
            select_prediction_indices(logits, 0, "membership-only", 0.9),
            set(),
        )
        self.assertEqual(
            select_prediction_indices(logits, 0, "membership-only", 0.5),
            {1, 2},
        )

    def test_membership_only_model_has_no_cardinality_parameters(self):
        model = ClipCandidateBaseline(
            candidate_feature_dim=8,
            text_feature_dim=8,
            hidden_dim=4,
            pooling="mean_max_stats",
            membership_only=True,
        )
        parameter_names = [name for name, _ in model.named_parameters()]
        self.assertFalse(any("count_head" in name for name in parameter_names))
        self.assertFalse(any("presence_head" in name for name in parameter_names))

    def test_input_ablation_dimensions_are_exact(self):
        full = ClipCandidateBaseline(
            candidate_feature_dim=8,
            text_feature_dim=6,
        )
        no_coordinates = ClipCandidateBaseline(
            candidate_feature_dim=8,
            text_feature_dim=6,
            use_box_coordinates=False,
        )
        no_similarity = ClipCandidateBaseline(
            candidate_feature_dim=8,
            text_feature_dim=6,
            use_explicit_similarity=False,
        )
        self.assertEqual(full.input_dim, 19)
        self.assertEqual(no_coordinates.input_dim, 15)
        self.assertEqual(no_similarity.input_dim, 18)

    def test_disabled_inputs_do_not_affect_candidate_input(self):
        model = ClipCandidateBaseline(
            candidate_feature_dim=2,
            text_feature_dim=2,
            use_box_coordinates=False,
            use_explicit_similarity=False,
        )
        candidate_features = torch.tensor([[1.0, 2.0]])
        text_feature = torch.tensor([3.0, 4.0])
        first = model._build_candidate_input(
            text_feature,
            candidate_features,
            torch.tensor([0.1]),
            torch.tensor([[0.0, 0.0, 0.5, 0.5]]),
        )
        second = model._build_candidate_input(
            text_feature,
            candidate_features,
            torch.tensor([0.9]),
            torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
        )
        torch.testing.assert_close(first, second)

    def test_failure_precedence_distinguishes_proposal_and_count(self):
        targets = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 3.0, 1.0], [4.0, 0.0, 5.0, 1.0]]
        )
        complete_candidates = targets.clone()
        category, _ = classify_failure(
            complete_candidates, complete_candidates[:2], targets
        )
        self.assertEqual(category, "count_under")
        category, _ = classify_failure(
            complete_candidates[:2], complete_candidates[:2], targets
        )
        self.assertEqual(category, "proposal_miss")

    def test_greedy_nms_suppresses_overlapping_lower_score_box(self):
        boxes = torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [0.1, 0.1, 2.1, 2.1], [3.0, 3.0, 4.0, 4.0]]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        self.assertEqual(greedy_nms(boxes, scores, 0.5), [0, 2])


if __name__ == "__main__":
    unittest.main()
