"""Audit 50-vs-100 cached detector proposals on the locked development split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.evaluation.grec_metrics import pairwise_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["ref_id"]), int(row["sent_id"])


def count_group(count: int) -> str:
    if count <= 5:
        return str(count)
    return "6+"


def new_stats() -> dict[str, Any]:
    return {
        "samples": 0,
        "targets": 0,
        "matched_targets": 0,
        "targeted_samples": 0,
        "full_coverage_samples": 0,
        "candidate_sum": 0,
        "at_cap_samples": 0,
    }


def update(
    stats: dict[str, Any],
    candidate_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    cap: int,
    threshold: float,
) -> list[bool]:
    candidates = candidate_boxes[:cap]
    overlaps = pairwise_iou(candidates, target_boxes)
    matched = (
        (overlaps.max(dim=0).values >= threshold).tolist()
        if target_boxes.shape[0]
        else []
    )
    stats["samples"] += 1
    stats["targets"] += len(matched)
    stats["matched_targets"] += sum(bool(value) for value in matched)
    stats["targeted_samples"] += int(bool(matched))
    stats["full_coverage_samples"] += int(bool(matched) and all(matched))
    stats["candidate_sum"] += int(candidates.shape[0])
    stats["at_cap_samples"] += int(candidate_boxes.shape[0] >= cap)
    return matched


def finalize(stats: dict[str, Any]) -> dict[str, Any]:
    samples = int(stats["samples"])
    targets = int(stats["targets"])
    targeted_samples = int(stats["targeted_samples"])
    return {
        **stats,
        "target_recall": (
            float(stats["matched_targets"]) / targets if targets else None
        ),
        "full_target_coverage": (
            float(stats["full_coverage_samples"]) / targeted_samples
            if targeted_samples
            else None
        ),
        "average_candidates": (
            float(stats["candidate_sum"]) / samples if samples else None
        ),
        "at_cap_fraction": (
            float(stats["at_cap_samples"]) / samples if samples else None
        ),
    }


def main() -> None:
    args = parse_args()
    caps = sorted(set(args.caps))
    if not caps or any(cap <= 0 for cap in caps):
        raise ValueError("caps must contain positive integers")
    requested = {key(row) for row in json.loads(Path(args.split_file).read_text())}
    seen: set[tuple[int, int]] = set()
    overall = {cap: new_stats() for cap in caps}
    by_exact_count = {
        cap: defaultdict(new_stats)
        for cap in caps
    }
    unique_targets: dict[int, dict[tuple[int, int], bool]] = {
        cap: {} for cap in caps
    }

    with Path(args.candidate_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_key = key(row)
            if row_key not in requested:
                continue
            if row_key in seen:
                raise ValueError(f"duplicate candidate record: {row_key}")
            seen.add(row_key)
            candidate_boxes = torch.tensor(
                row["candidate_boxes_xyxy"], dtype=torch.float32
            ).reshape(-1, 4)
            target_boxes = torch.tensor(
                row["target_boxes_xyxy"], dtype=torch.float32
            ).reshape(-1, 4)
            group = count_group(int(target_boxes.shape[0]))
            target_ids = [
                (int(row["image_id"]), int(annotation_id))
                for annotation_id in row.get("target_ann_ids", [])
            ]
            if len(target_ids) != target_boxes.shape[0]:
                raise ValueError(f"target ID/box mismatch for {row_key}")
            for cap in caps:
                matched = update(
                    overall[cap],
                    candidate_boxes,
                    target_boxes,
                    cap,
                    args.iou_threshold,
                )
                update(
                    by_exact_count[cap][group],
                    candidate_boxes,
                    target_boxes,
                    cap,
                    args.iou_threshold,
                )
                for target_id, is_matched in zip(target_ids, matched):
                    unique_targets[cap][target_id] = (
                        unique_targets[cap].get(target_id, False)
                        or bool(is_matched)
                    )

    missing = requested - seen
    if missing:
        raise KeyError(
            f"candidate file is missing {len(missing)} dev records; "
            f"first={next(iter(missing))}"
        )

    results = {}
    for cap in caps:
        unique = unique_targets[cap]
        results[str(cap)] = {
            "overall": finalize(overall[cap]),
            "unique_target_recall": (
                sum(unique.values()) / len(unique) if unique else None
            ),
            "unique_targets": len(unique),
            "by_exact_count": {
                group: finalize(stats)
                for group, stats in sorted(by_exact_count[cap].items())
            },
        }
    payload = {
        "stage": "6.3 cached proposal-cap audit",
        "scope": "locked development split; no detector rerun",
        "candidate_file": args.candidate_file,
        "split_file": args.split_file,
        "iou_threshold": args.iou_threshold,
        "caps": caps,
        "results": results,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "Stage 6.3 Cached Proposal-Cap Audit",
        "===================================",
        "cap | target recall | unique recall | full coverage | avg candidates | at cap",
        "---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for cap in caps:
        row = results[str(cap)]
        overall_row = row["overall"]
        lines.append(
            f"{cap} | {overall_row['target_recall']:.6f} | "
            f"{row['unique_target_recall']:.6f} | "
            f"{overall_row['full_target_coverage']:.6f} | "
            f"{overall_row['average_candidates']:.3f} | "
            f"{overall_row['at_cap_fraction']:.6f}"
        )
    Path(args.output_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(Path(args.output_txt).read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
