from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.data.build_proposal_candidate_samples import (
    associate_proposals,
    build_candidate_sample,
    summarize_candidate_samples,
)
from src.data.candidate_dataset import CandidateBoxDataset
from src.proposals.fasterrcnn import ProposalConfig, filter_detector_output
from src.proposals.generate_fasterrcnn_proposals import (
    collect_image_ids,
    load_existing_records,
)


class DetectorFilteringTests(unittest.TestCase):
    def test_proposal_config_rejects_unknown_precision(self) -> None:
        with self.assertRaisesRegex(ValueError, "inference_precision"):
            ProposalConfig(inference_precision="bfloat16").validate()

    def setUp(self) -> None:
        self.config = ProposalConfig(
            score_threshold=0.5,
            nms_threshold=0.5,
            max_proposals=3,
            detector_output_limit=5,
        )

    def test_class_agnostic_nms_and_score_order(self) -> None:
        output = {
            "boxes": torch.tensor(
                [
                    [0, 0, 10, 10],
                    [1, 1, 11, 11],
                    [20, 20, 30, 30],
                    [40, 40, 50, 50],
                ],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.9, 0.8, 0.7, 0.4]),
            "labels": torch.tensor([1, 2, 3, 4]),
        }
        result = filter_detector_output(output, 100, 100, self.config)
        self.assertEqual(result["num_raw_detections"], 4)
        self.assertEqual(result["num_proposals"], 2)
        self.assertEqual(result["detector_labels"], [1, 3])
        self.assertEqual(result["fallback"], "none")

    def test_below_threshold_fallback_keeps_highest_score(self) -> None:
        output = {
            "boxes": torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]]),
            "scores": torch.tensor([0.2, 0.3]),
            "labels": torch.tensor([1, 2]),
        }
        result = filter_detector_output(output, 100, 100, self.config)
        self.assertEqual(result["num_proposals"], 1)
        self.assertEqual(result["detector_labels"], [2])
        self.assertEqual(result["fallback"], "highest-score-below-threshold")

    def test_empty_detector_output_uses_full_image_fallback(self) -> None:
        output = {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty(0),
            "labels": torch.empty(0, dtype=torch.long),
        }
        result = filter_detector_output(output, 80, 120, self.config)
        self.assertEqual(result["proposal_boxes_xyxy"], [[0.0, 0.0, 120.0, 80.0]])
        self.assertEqual(result["fallback"], "full-image")

    def test_collect_image_ids_is_sorted_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.json"
            second = Path(tmpdir) / "second.json"
            first.write_text(json.dumps([{"image_id": 5}, {"image_id": 2}]))
            second.write_text(json.dumps([{"image_id": 5}, {"image_id": 3}]))
            self.assertEqual(collect_image_ids([first, second]), [2, 3, 5])

    def test_resume_loads_records_and_rejects_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "proposals.jsonl"
            record = {
                "image_id": 7,
                "proposal_config": self.config.to_dict(),
            }
            cache_file.write_text(json.dumps(record) + "\n")

            records, seen = load_existing_records(
                cache_file,
                self.config,
                resume=True,
            )
            self.assertEqual(records, [record])
            self.assertEqual(seen, {7})

            with self.assertRaisesRegex(ValueError, "different configuration"):
                load_existing_records(
                    cache_file,
                    ProposalConfig(),
                    resume=True,
                )

    def test_resume_discards_only_a_truncated_final_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "proposals.jsonl"
            record = {
                "image_id": 7,
                "proposal_config": self.config.to_dict(),
            }
            valid_line = json.dumps(record) + "\n"
            cache_file.write_bytes((valid_line + '{"image_id": 8').encode())

            records, seen = load_existing_records(
                cache_file,
                self.config,
                resume=True,
            )
            self.assertEqual(records, [record])
            self.assertEqual(seen, {7})
            self.assertEqual(cache_file.read_text(), valid_line)


class ProposalAssociationTests(unittest.TestCase):
    def test_iou_association_labels_candidates_and_target_coverage(self) -> None:
        proposals = torch.tensor(
            [
                [0, 0, 10, 10],
                [1, 1, 9, 9],
                [20, 20, 30, 30],
                [50, 50, 60, 60],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [[0, 0, 10, 10], [20, 20, 32, 32]],
            dtype=torch.float32,
        )
        result = associate_proposals(proposals, targets, iou_threshold=0.5)
        self.assertEqual(result["candidate_labels"].tolist(), [1, 1, 1, 0])
        self.assertEqual(result["candidate_target_indices"].tolist(), [0, 0, 1, -1])
        self.assertTrue(torch.all(result["target_best_proposal_ious"] >= 0.5))

    def test_no_target_association_produces_only_negative_labels(self) -> None:
        result = associate_proposals(
            torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
            torch.empty((0, 4)),
            iou_threshold=0.5,
        )
        self.assertEqual(result["candidate_labels"].tolist(), [0])
        self.assertEqual(result["candidate_target_indices"].tolist(), [-1])
        self.assertEqual(result["target_best_proposal_ious"].numel(), 0)

    @staticmethod
    def _fixtures():
        ref_by_id = {
            1: {
                "ref_id": 1,
                "ann_id": [10, 11],
                "sentences": [{"sent_id": 100, "sent": "the two objects"}],
            },
            2: {
                "ref_id": 2,
                "ann_id": [-1],
                "sentences": [{"sent_id": 200, "sent": "missing object"}],
            },
        }
        ann_by_id = {
            10: {"id": 10, "bbox": [0, 0, 10, 10]},
            11: {"id": 11, "bbox": [20, 20, 10, 10]},
        }
        proposal_by_image = {
            7: {
                "image_id": 7,
                "file_name": "image.jpg",
                "width": 100,
                "height": 100,
                "proposal_config": {"detector_id": "synthetic"},
                "proposal_boxes_xyxy": [
                    [0, 0, 10, 10],
                    [20, 20, 30, 30],
                    [50, 50, 60, 60],
                ],
                "proposal_scores": [0.9, 0.8, 0.7],
                "detector_labels": [1, 2, 3],
            }
        }
        return ref_by_id, ann_by_id, proposal_by_image

    def test_candidate_sample_is_dataset_compatible(self) -> None:
        ref_by_id, ann_by_id, proposal_by_image = self._fixtures()
        sample = build_candidate_sample(
            {
                "ref_id": 1,
                "sent_id": 100,
                "image_id": 7,
                "target_type": "multi-target",
            },
            ref_by_id,
            ann_by_id,
            proposal_by_image,
            sample_index=0,
            iou_threshold=0.5,
        )
        self.assertEqual(sample["candidate_labels"], [1.0, 1.0, 0.0])
        self.assertEqual(sample["target_best_proposal_ious"], [1.0, 1.0])
        self.assertEqual(sample["count_class"], 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_file = Path(tmpdir) / "candidates.jsonl"
            candidate_file.write_text(json.dumps(sample) + "\n")
            dataset = CandidateBoxDataset(
                str(candidate_file),
                image_root=tmpdir,
                load_image=False,
            )
            self.assertEqual(dataset[0]["candidate_labels"].tolist(), [1, 1, 0])
            self.assertEqual(
                dataset[0]["target_best_proposal_ious"].tolist(),
                [1.0, 1.0],
            )

    def test_recall_summary_reports_unique_and_full_coverage(self) -> None:
        ref_by_id, ann_by_id, proposal_by_image = self._fixtures()
        multi = build_candidate_sample(
            {"ref_id": 1, "sent_id": 100, "image_id": 7},
            ref_by_id,
            ann_by_id,
            proposal_by_image,
            sample_index=0,
            iou_threshold=0.5,
        )
        no_target = build_candidate_sample(
            {"ref_id": 2, "sent_id": 200, "image_id": 7},
            ref_by_id,
            ann_by_id,
            proposal_by_image,
            sample_index=1,
            iou_threshold=0.5,
        )
        stats = summarize_candidate_samples([multi, no_target], iou_threshold=0.5)
        self.assertEqual(stats["unique_images"], 1)
        self.assertEqual(stats["unique_target_recall"], 1.0)
        self.assertEqual(stats["overall"]["target_recall"], 1.0)
        self.assertEqual(stats["overall"]["full_target_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
