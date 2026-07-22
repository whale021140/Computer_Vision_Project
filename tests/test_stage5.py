from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.data.audit_split_integrity import audit_splits
from src.data.create_fewshot_splits import (
    build_nested_subsets,
    build_union,
    expand_train_expressions,
    sample_key,
    summarize,
)
from src.data.feature_dataset import ClipFeatureDataset
from src.evaluation.summarize_stage5_grid import summarize_grid
from src.evaluation.summarize_stage5_test_grid import summarize_test_grid


def synthetic_refs() -> list[dict]:
    refs = []
    ref_id = 1
    sent_id = 100
    for target_count, no_target, count in ((0, True, 20), (1, False, 30), (2, False, 20), (4, False, 10)):
        for _ in range(count):
            refs.append(
                {
                    "ref_id": ref_id,
                    "image_id": ref_id + 1000,
                    "split": "train",
                    "no_target": no_target,
                    "ann_id": [-1] if no_target else list(range(target_count)),
                    "sentences": [{"sent_id": sent_id, "sent": "object"}],
                }
            )
            ref_id += 1
            sent_id += 1
    return refs


class FewShotSplitTests(unittest.TestCase):
    def test_stratified_splits_are_deterministic_nested_and_counted(self) -> None:
        samples = expand_train_expressions(synthetic_refs())
        first = build_nested_subsets(samples, seed=7, percentages=(10, 50, 100))
        second = build_nested_subsets(samples, seed=7, percentages=(10, 50, 100))
        self.assertEqual(first, second)
        key_sets = {pct: {sample_key(row) for row in rows} for pct, rows in first.items()}
        self.assertLessEqual(key_sets[10], key_sets[50])
        self.assertLessEqual(key_sets[50], key_sets[100])
        self.assertEqual(
            summarize(first[10])["by_target_type"],
            {"no-target": 2, "single-target": 3, "multi-target": 3},
        )
        count_summary = summarize(first[10])["by_count_group"]
        self.assertEqual(count_summary["0"], 2)
        self.assertEqual(count_summary["1"], 3)
        self.assertEqual(count_summary["2"] + count_summary["3+"], 3)

    def test_different_data_seeds_and_union(self) -> None:
        samples = expand_train_expressions(synthetic_refs())
        split0 = build_nested_subsets(samples, seed=0, percentages=(10,))
        split1 = build_nested_subsets(samples, seed=1, percentages=(10,))
        self.assertNotEqual(
            {sample_key(row) for row in split0[10]},
            {sample_key(row) for row in split1[10]},
        )
        union = build_union({0: split0, 1: split1})
        expected = {
            sample_key(row) for row in split0[10]
        } | {sample_key(row) for row in split1[10]}
        self.assertEqual({sample_key(row) for row in union}, expected)


class SharedFeatureBankTests(unittest.TestCase):
    def _cache(self) -> dict:
        records = []
        images = {}
        for index, (ref_id, sent_id, image_id) in enumerate(((1, 11, 101), (2, 22, 102), (3, 33, 103))):
            images[str(image_id)] = {
                "candidate_features": torch.ones((1, 2)),
                "candidate_boxes_norm": torch.zeros((1, 4)),
            }
            records.append(
                {
                    "sample_id": str(index),
                    "image_id": image_id,
                    "metadata": {
                        "ref_id": ref_id,
                        "sent_id": sent_id,
                        "image_id": image_id,
                    },
                    "expression": "object",
                    "text_feature": torch.ones(2),
                    "candidate_labels": torch.ones(1),
                    "count_class": torch.tensor(1),
                    "target_boxes_xyxy": torch.zeros((1, 4)),
                    "target_boxes_norm": torch.zeros((1, 4)),
                }
            )
        return {
            "cache_format": "frozen_representation_v1",
            "feature_dim": 2,
            "candidate_feature_dim": 2,
            "text_feature_dim": 2,
            "images": images,
            "records": records,
        }

    def test_split_selects_exact_records_in_split_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "bank.pt"
            split_path = Path(tmpdir) / "split.json"
            torch.save(self._cache(), cache_path)
            split_path.write_text(
                json.dumps(
                    [
                        {"ref_id": 3, "sent_id": 33},
                        {"ref_id": 1, "sent_id": 11},
                    ]
                ),
                encoding="utf-8",
            )
            dataset = ClipFeatureDataset(
                str(cache_path), split_file=str(split_path)
            )
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["metadata"]["ref_id"], 3)
        self.assertEqual(dataset[1]["metadata"]["ref_id"], 1)

    def test_missing_split_expression_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "bank.pt"
            split_path = Path(tmpdir) / "split.json"
            torch.save(self._cache(), cache_path)
            split_path.write_text(
                json.dumps([{"ref_id": 99, "sent_id": 999}]),
                encoding="utf-8",
            )
            with self.assertRaises(KeyError):
                ClipFeatureDataset(str(cache_path), split_file=str(split_path))


class SplitIntegrityAuditTests(unittest.TestCase):
    def test_audit_accepts_nested_disjoint_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train1 = root / "train_1pct_seed0.json"
            train5 = root / "train_5pct_seed0.json"
            val = root / "val.json"
            train1.write_text(
                json.dumps([{"ref_id": 1, "sent_id": 10, "image_id": 100}]),
                encoding="utf-8",
            )
            train5.write_text(
                json.dumps(
                    [
                        {"ref_id": 1, "sent_id": 10, "image_id": 100},
                        {"ref_id": 2, "sent_id": 20, "image_id": 200},
                    ]
                ),
                encoding="utf-8",
            )
            val.write_text(
                json.dumps([{"ref_id": 3, "sent_id": 30, "image_id": 300}]),
                encoding="utf-8",
            )
            result = audit_splits([train1, train5], [val])
        self.assertEqual(result["status"], "passed")

    def test_audit_rejects_image_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train_1pct_seed0.json"
            val = root / "val.json"
            train.write_text(
                json.dumps([{"ref_id": 1, "sent_id": 10, "image_id": 100}]),
                encoding="utf-8",
            )
            val.write_text(
                json.dumps([{"ref_id": 2, "sent_id": 20, "image_id": 100}]),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                audit_splits([train], [val])


def fake_manifest(representation: str, percentage: int, seed: int) -> dict:
    value = percentage / 100 + seed / 1000
    group = {
        "mean_f1": value,
        "exact_set_accuracy": value,
        "cardinality_accuracy": value,
    }
    return {
        "cell": {
            "representation": representation,
            "percentage": percentage,
            "seed": seed,
        },
        "git_commit": "commit",
        "split": {"sha256": f"split-{percentage}-{seed}"},
        "features": {"candidate_file_sha256": "candidates"},
        "training": {"checkpoint": f"checkpoint-{representation}-{percentage}-{seed}"},
        "calibration": {"path": f"calibration-{representation}-{percentage}-{seed}"},
        "validation_evaluation": {
            "official": {
                "F1_score": value,
                "T_acc": value,
                "N_acc": value,
            },
            "diagnostics": {
                "mean_f1": value,
                "cardinality_accuracy": value,
                "false_grounding_rate": value,
                "multi_target_mean_f1": value,
                "multi_target_exact_accuracy": value,
            },
            "by_count_group": {key: group for key in ("0", "1", "2", "3+")},
        },
    }


class Stage5AggregationTests(unittest.TestCase):
    def test_complete_paired_grid_is_aggregated(self) -> None:
        representations = ["a", "b"]
        percentages = [1, 5]
        seeds = [0, 1, 2]
        manifests = [
            fake_manifest(representation, percentage, seed)
            for representation in representations
            for percentage in percentages
            for seed in seeds
        ]
        result = summarize_grid(manifests, representations, percentages, seeds)
        self.assertEqual(result["num_cells"], 12)
        self.assertAlmostEqual(
            result["summary"][0]["metrics"]["F1_score"]["mean"],
            0.011,
        )
        self.assertAlmostEqual(
            result["summary"][0]["metrics"]["F1_score"]["std"],
            0.001,
        )

    def test_missing_grid_cell_is_rejected(self) -> None:
        manifests = [fake_manifest("a", 1, 0)]
        with self.assertRaises(ValueError):
            summarize_grid(manifests, ["a"], [1], [0, 1])

    def test_locked_test_grid_is_aggregated_and_validated(self) -> None:
        representations = ["a", "b"]
        percentages = [1]
        seeds = [0, 1]
        splits = ["testA", "testB"]
        checkpoint_policies = ["last", "best"]
        manifests = [
            fake_manifest(representation, 1, seed)
            for representation in representations
            for seed in seeds
        ]
        by_cell = {
            (
                manifest["cell"]["representation"],
                manifest["cell"]["percentage"],
                manifest["cell"]["seed"],
            ): manifest
            for manifest in manifests
        }
        evaluations = {}
        for cell, manifest in by_cell.items():
            representation, percentage, seed = cell
            for split in splits:
                for checkpoint_policy in checkpoint_policies:
                    evaluation = dict(manifest["validation_evaluation"])
                    evaluation.update(
                        {
                            "checkpoint": (
                                f"checkpoints/stage5/{representation}_"
                                f"{percentage}pct_seed{seed}/"
                                f"{checkpoint_policy}.pt"
                            ),
                            "membership_threshold": 0.5,
                            "representation": {"name": representation},
                        }
                    )
                    evaluation["diagnostics"] = dict(evaluation["diagnostics"])
                    evaluation["diagnostics"]["num_samples"] = 10
                    evaluation["by_target_type"] = {
                        key: {
                            "mean_f1": 0.5,
                            "exact_set_accuracy": 0.5,
                            "cardinality_accuracy": 0.5,
                        }
                        for key in ("no-target", "single-target", "multi-target")
                    }
                    evaluations[(*cell, split, checkpoint_policy)] = evaluation
        feature_stats = {
            (representation, split): {
                "candidate_file_sha256": f"candidate-{split}",
                "num_samples": 10,
            }
            for representation in representations
            for split in splits
        }
        result = summarize_test_grid(
            manifests,
            evaluations,
            feature_stats,
            representations,
            percentages,
            seeds,
            splits,
            checkpoint_policies,
        )
        self.assertEqual(result["num_evaluations"], 16)
        self.assertEqual(result["expected_samples"], {"testA": 10, "testB": 10})
        self.assertEqual(len(result["paired_differences"]), 4)
        paired = result["paired_differences"][0]
        self.assertEqual(paired["difference"], "left_minus_right")
        self.assertEqual(paired["left_representation"], "a")
        self.assertEqual(paired["right_representation"], "b")
        self.assertEqual(paired["metrics"]["F1_score"]["values"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
