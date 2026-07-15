from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

import torch

from src.evaluation.grec_metrics import pairwise_iou
from src.utils.boxes import count_to_class, normalize_xyxy, xywh_to_xyxy


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_proposal_jsonl(path: str | Path) -> Dict[int, Dict[str, Any]]:
    proposals: Dict[int, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = int(record["image_id"])
            if image_id in proposals:
                raise ValueError(f"Duplicate image_id={image_id} in proposal cache.")
            proposals[image_id] = record
    if not proposals:
        raise ValueError(f"No proposals loaded from {path}")
    return proposals


def build_annotation_lookups(
    grefs: Sequence[Dict[str, Any]],
    instances: Dict[str, Any],
) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    ref_by_id = {int(ref["ref_id"]): ref for ref in grefs}
    ann_by_id = {int(ann["id"]): ann for ann in instances["annotations"]}
    return ref_by_id, ann_by_id


def find_sentence(ref: Dict[str, Any], sent_id: int) -> str:
    for sentence in ref["sentences"]:
        if int(sentence["sent_id"]) == sent_id:
            return str(sentence["sent"])
    raise KeyError(f"sent_id={sent_id} not found in ref_id={ref['ref_id']}")


def valid_target_ann_ids(ref: Dict[str, Any]) -> List[int]:
    return [int(ann_id) for ann_id in ref.get("ann_id", []) if int(ann_id) != -1]


def target_type_from_count(num_targets: int) -> str:
    if num_targets == 0:
        return "no-target"
    if num_targets == 1:
        return "single-target"
    return "multi-target"


def associate_proposals(
    proposal_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    iou_threshold: float,
) -> Dict[str, Any]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1.")
    proposal_boxes = proposal_boxes.float().reshape(-1, 4)
    target_boxes = target_boxes.float().reshape(-1, 4)
    if proposal_boxes.shape[0] == 0:
        raise ValueError("Every image must contain at least one proposal.")

    if target_boxes.shape[0] == 0:
        return {
            "candidate_best_ious": torch.zeros(proposal_boxes.shape[0]),
            "candidate_target_indices": torch.full(
                (proposal_boxes.shape[0],),
                -1,
                dtype=torch.long,
            ),
            "candidate_labels": torch.zeros(proposal_boxes.shape[0]),
            "target_best_proposal_ious": torch.empty(0),
        }

    overlaps = pairwise_iou(proposal_boxes, target_boxes)
    candidate_best_ious, candidate_target_indices = overlaps.max(dim=1)
    target_best_proposal_ious = overlaps.max(dim=0).values
    candidate_labels = (candidate_best_ious >= iou_threshold).float()
    candidate_target_indices = candidate_target_indices.masked_fill(
        candidate_labels == 0,
        -1,
    )
    return {
        "candidate_best_ious": candidate_best_ious,
        "candidate_target_indices": candidate_target_indices,
        "candidate_labels": candidate_labels,
        "target_best_proposal_ious": target_best_proposal_ious,
    }


def build_candidate_sample(
    split_sample: Dict[str, Any],
    ref_by_id: Dict[int, Dict[str, Any]],
    ann_by_id: Dict[int, Dict[str, Any]],
    proposal_by_image: Dict[int, Dict[str, Any]],
    sample_index: int,
    iou_threshold: float,
) -> Dict[str, Any]:
    ref_id = int(split_sample["ref_id"])
    sent_id = int(split_sample["sent_id"])
    image_id = int(split_sample["image_id"])
    ref = ref_by_id[ref_id]
    proposal_record = proposal_by_image.get(image_id)
    if proposal_record is None:
        raise KeyError(f"No proposal cache entry for image_id={image_id}")

    width = int(proposal_record["width"])
    height = int(proposal_record["height"])
    proposal_boxes_list = proposal_record["proposal_boxes_xyxy"]
    proposal_boxes = torch.tensor(proposal_boxes_list, dtype=torch.float32).reshape(-1, 4)
    target_ann_ids = valid_target_ann_ids(ref)
    target_boxes_list = [
        xywh_to_xyxy(ann_by_id[ann_id]["bbox"])
        for ann_id in target_ann_ids
    ]
    target_boxes = torch.tensor(target_boxes_list, dtype=torch.float32).reshape(-1, 4)
    association = associate_proposals(
        proposal_boxes,
        target_boxes,
        iou_threshold=iou_threshold,
    )
    num_targets = len(target_ann_ids)
    target_type = target_type_from_count(num_targets)
    declared_type = split_sample.get("target_type")
    if declared_type is not None and declared_type != target_type:
        raise ValueError(
            f"Split target_type={declared_type!r} conflicts with "
            f"{num_targets} targets for ref_id={ref_id}."
        )

    return {
        "sample_id": f"{sample_index:07d}",
        "ref_id": ref_id,
        "sent_id": sent_id,
        "image_id": image_id,
        "file_name": proposal_record["file_name"],
        "width": width,
        "height": height,
        "expression": find_sentence(ref, sent_id),
        "target_type": target_type,
        "target_ann_ids": target_ann_ids,
        "target_boxes_xyxy": target_boxes_list,
        "target_boxes_norm": [
            normalize_xyxy(box, width, height) for box in target_boxes_list
        ],
        "candidate_source": proposal_record["proposal_config"]["detector_id"],
        "proposal_config": proposal_record["proposal_config"],
        "candidate_boxes_xyxy": proposal_boxes_list,
        "candidate_boxes_norm": [
            normalize_xyxy(box, width, height) for box in proposal_boxes_list
        ],
        "candidate_scores": proposal_record["proposal_scores"],
        "candidate_detector_labels": proposal_record["detector_labels"],
        "candidate_best_ious": association["candidate_best_ious"].tolist(),
        "candidate_target_indices": association["candidate_target_indices"].tolist(),
        "candidate_labels": association["candidate_labels"].tolist(),
        "target_best_proposal_ious": association[
            "target_best_proposal_ious"
        ].tolist(),
        "num_targets": num_targets,
        "count_class": count_to_class(num_targets),
        "num_candidates": len(proposal_boxes_list),
    }


def _new_group() -> Dict[str, Any]:
    return {
        "samples": 0,
        "targeted_samples": 0,
        "fully_covered_samples": 0,
        "samples_with_positive_candidate": 0,
        "targets": 0,
        "matched_targets": 0,
        "candidate_counts": [],
        "positive_candidate_counts": [],
    }


def _update_group(group: Dict[str, Any], sample: Dict[str, Any], threshold: float) -> None:
    num_targets = int(sample["num_targets"])
    best_ious = sample["target_best_proposal_ious"]
    matched_targets = sum(float(iou) >= threshold for iou in best_ious)
    positive_count = sum(float(label) > 0.5 for label in sample["candidate_labels"])
    group["samples"] += 1
    group["targets"] += num_targets
    group["matched_targets"] += matched_targets
    group["candidate_counts"].append(int(sample["num_candidates"]))
    group["positive_candidate_counts"].append(positive_count)
    group["samples_with_positive_candidate"] += int(positive_count > 0)
    if num_targets > 0:
        group["targeted_samples"] += 1
        group["fully_covered_samples"] += int(matched_targets == num_targets)


def _finalize_group(group: Dict[str, Any]) -> Dict[str, Any]:
    targets = group["targets"]
    targeted_samples = group["targeted_samples"]
    samples = group["samples"]
    return {
        "samples": samples,
        "targeted_samples": targeted_samples,
        "targets": targets,
        "matched_targets": group["matched_targets"],
        "target_recall": group["matched_targets"] / targets if targets else 0.0,
        "fully_covered_samples": group["fully_covered_samples"],
        "full_target_coverage": (
            group["fully_covered_samples"] / targeted_samples
            if targeted_samples
            else 0.0
        ),
        "samples_with_positive_candidate": group["samples_with_positive_candidate"],
        "positive_candidate_sample_rate": (
            group["samples_with_positive_candidate"] / samples if samples else 0.0
        ),
        "average_candidates": mean(group["candidate_counts"])
        if group["candidate_counts"]
        else 0.0,
        "average_positive_candidates": mean(group["positive_candidate_counts"])
        if group["positive_candidate_counts"]
        else 0.0,
    }


def summarize_candidate_samples(
    samples: Sequence[Dict[str, Any]],
    iou_threshold: float,
) -> Dict[str, Any]:
    overall = _new_group()
    by_target_type: Dict[str, Dict[str, Any]] = defaultdict(_new_group)
    by_count_group: Dict[str, Dict[str, Any]] = defaultdict(_new_group)
    unique_targets: Dict[tuple[int, int], float] = {}
    unique_images = set()

    for sample in samples:
        _update_group(overall, sample, iou_threshold)
        _update_group(by_target_type[sample["target_type"]], sample, iou_threshold)
        num_targets = int(sample["num_targets"])
        count_group = str(num_targets) if num_targets <= 2 else "3+"
        _update_group(by_count_group[count_group], sample, iou_threshold)
        unique_images.add(int(sample["image_id"]))
        for ann_id, best_iou in zip(
            sample["target_ann_ids"],
            sample["target_best_proposal_ious"],
        ):
            key = (int(sample["image_id"]), int(ann_id))
            unique_targets[key] = max(unique_targets.get(key, 0.0), float(best_iou))

    unique_matched = sum(iou >= iou_threshold for iou in unique_targets.values())
    return {
        "iou_threshold": iou_threshold,
        "unique_images": len(unique_images),
        "unique_targets": len(unique_targets),
        "unique_matched_targets": unique_matched,
        "unique_target_recall": (
            unique_matched / len(unique_targets) if unique_targets else 0.0
        ),
        "overall": _finalize_group(overall),
        "by_target_type": {
            key: _finalize_group(value)
            for key, value in sorted(by_target_type.items())
        },
        "by_count_group": {
            key: _finalize_group(value)
            for key, value in sorted(by_count_group.items())
        },
    }


def format_stats(stats: Dict[str, Any]) -> str:
    overall = stats["overall"]
    lines = [
        "Detector Proposal Candidate Statistics",
        "======================================",
        f"IoU threshold: {stats['iou_threshold']}",
        f"Unique images: {stats['unique_images']}",
        f"Unique targets: {stats['unique_targets']}",
        f"Unique matched targets: {stats['unique_matched_targets']}",
        f"Unique target recall: {stats['unique_target_recall']:.6f}",
        f"Expression samples: {overall['samples']}",
        f"Expression-weighted target recall: {overall['target_recall']:.6f}",
        f"Full-target sample coverage: {overall['full_target_coverage']:.6f}",
        f"Average candidates: {overall['average_candidates']:.4f}",
        f"Average positive candidates: {overall['average_positive_candidates']:.4f}",
    ]
    for section in ("by_target_type", "by_count_group"):
        lines.extend(["", f"[{section}]"])
        for name, group in stats[section].items():
            lines.append(
                f"{name}: samples={group['samples']}, targets={group['targets']}, "
                f"target_recall={group['target_recall']:.6f}, "
                f"full_coverage={group['full_target_coverage']:.6f}"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build expression-level detector-proposal candidate samples."
    )
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--proposal-file", required=True)
    parser.add_argument(
        "--grefs",
        default="data/grefcoco/annotations/grefs(unc).json",
    )
    parser.add_argument(
        "--instances",
        default="data/grefcoco/annotations/instances.json",
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--stats-txt", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_samples = load_json(args.split_file)
    if args.max_samples is not None:
        split_samples = split_samples[: args.max_samples]
    grefs = load_json(args.grefs)
    instances = load_json(args.instances)
    proposal_by_image = load_proposal_jsonl(args.proposal_file)
    ref_by_id, ann_by_id = build_annotation_lookups(grefs, instances)

    samples = [
        build_candidate_sample(
            split_sample,
            ref_by_id=ref_by_id,
            ann_by_id=ann_by_id,
            proposal_by_image=proposal_by_image,
            sample_index=index,
            iou_threshold=args.iou_threshold,
        )
        for index, split_sample in enumerate(split_samples)
    ]
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats = summarize_candidate_samples(samples, args.iou_threshold)
    stats.update(
        {
            "split_file": args.split_file,
            "split_sha256": sha256_file(args.split_file),
            "proposal_file": args.proposal_file,
            "proposal_sha256": sha256_file(args.proposal_file),
            "output_file": args.output_file,
            "command": shlex.join(sys.argv),
        }
    )
    stats_json = Path(args.stats_json)
    stats_txt = Path(args.stats_txt)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_txt.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    summary = format_stats(stats)
    stats_txt.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
