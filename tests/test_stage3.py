import unittest

from src.evaluation.compare_candidate_sources import build_comparison


class CandidateSourceComparisonTests(unittest.TestCase):
    @staticmethod
    def evaluation(f1: float, samples: int = 10) -> dict:
        return {
            "feature_file": "features.pt",
            "official": {"F1_score": f1, "T_acc": f1 + 0.1, "N_acc": 0.9},
            "diagnostics": {
                "num_samples": samples,
                "mean_f1": f1 + 0.05,
                "exact_set_accuracy": f1,
                "cardinality_accuracy": 0.8,
                "false_grounding_rate": 0.1,
                "multi_target_mean_f1": 0.4,
                "multi_target_exact_accuracy": 0.2,
            },
        }

    @staticmethod
    def recall() -> dict:
        return {
            "iou_threshold": 0.5,
            "unique_target_recall": 0.99,
            "overall": {
                "target_recall": 0.995,
                "full_target_coverage": 0.98,
            },
        }

    def test_detector_minus_oracle_delta(self) -> None:
        result = build_comparison(
            self.evaluation(0.7), self.evaluation(0.64), self.recall()
        )
        self.assertAlmostEqual(
            result["metrics"]["F1_score"]["detector_minus_oracle"], -0.06
        )
        self.assertAlmostEqual(
            result["interpretation"]["unique_target_miss_rate"], 0.01
        )

    def test_rejects_mismatched_evaluation_sets(self) -> None:
        with self.assertRaises(ValueError):
            build_comparison(
                self.evaluation(0.7, samples=10),
                self.evaluation(0.64, samples=9),
                self.recall(),
            )


if __name__ == "__main__":
    unittest.main()
