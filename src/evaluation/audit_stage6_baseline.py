"""Freeze and audit the accepted Stage 5.6 baseline before Stage 6."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


REPRESENTATIONS = ("clip", "clip_dinov2", "siglip2")
PERCENTAGES = (1, 5, 10)
SEEDS = (0, 1, 2)
SPLITS = ("testA", "testB")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> Any:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def file_record(path: Path, hash_contents: bool = True) -> dict[str, Any]:
    path = require_file(path)
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
    }
    if hash_contents:
        result["sha256"] = sha256_file(path)
    return result


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def split_image_ids(path: Path) -> set[int]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list split in {path}")
    return {int(row["image_id"]) for row in rows}


def active_calibration_boundaries(calibration: dict[str, Any]) -> list[str]:
    boundary = calibration.get("best_on_boundary", {})
    if not isinstance(boundary, dict):
        raise ValueError("best_on_boundary must be a per-axis dictionary")
    return sorted(key for key, value in boundary.items() if bool(value))


def evaluation_matches_calibration(
    evaluation: dict[str, Any],
    calibration_path: Path,
    calibration: dict[str, Any],
) -> bool:
    best = calibration["best"]
    return (
        Path(evaluation.get("calibration_json", "")) == calibration_path
        and evaluation.get("count_logit_bias") == best["count_logit_bias"]
        and evaluation.get("membership_threshold")
        == best["membership_threshold"]
    )


def validate_split_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    dev_path = Path(manifest["dev"]["path"])
    dev_ids = split_image_ids(dev_path)
    test_paths = {"testA": Path("splits/testA.json"), "testB": Path("splits/testB.json")}
    test_ids = {name: split_image_ids(path) for name, path in test_paths.items()}
    if dev_ids & test_ids["testA"] or dev_ids & test_ids["testB"]:
        raise ValueError("Stage 5.6 development images overlap official test images")
    if test_ids["testA"] & test_ids["testB"]:
        raise ValueError("testA and testB image sets overlap")

    split_records = []
    for item in manifest["splits"]:
        path = Path(item["path"])
        actual_sha = sha256_file(require_file(path))
        if actual_sha != item["sha256"]:
            raise ValueError(f"Split hash mismatch: {path}")
        train_ids = split_image_ids(path)
        if train_ids & dev_ids:
            raise ValueError(f"Train/dev image overlap: {path}")
        if train_ids & test_ids["testA"] or train_ids & test_ids["testB"]:
            raise ValueError(f"Train/test image overlap: {path}")
        if item["by_count_group"] != item["requested_by_count_group"]:
            raise ValueError(f"Count budget mismatch: {path}")
        split_records.append(
            {
                "path": str(path),
                "sha256": actual_sha,
                "num_expressions": item["expressions"],
                "num_unique_images": len(train_ids),
            }
        )
    return {
        "development": {
            **file_record(dev_path),
            "num_expressions": manifest["dev"]["expressions"],
            "num_unique_images": len(dev_ids),
            "by_count_group": manifest["dev"]["by_count_group"],
        },
        "tests": {
            name: {
                **file_record(path),
                "num_unique_images": len(test_ids[name]),
            }
            for name, path in test_paths.items()
        },
        "training_splits": split_records,
        "image_disjointness_passed": True,
    }


def validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if summary["calibration_revision"] != "v2_wide":
        raise ValueError("Stage 6 requires the v2_wide Stage 5.6 baseline")
    if summary["selected_variant"] != "hierarchical":
        raise ValueError("Unexpected Stage 5.6 selected variant")
    if summary["num_evaluations"] != 54:
        raise ValueError("Stage 5.6 benchmark is incomplete")
    if not summary["calibration_audit"]["test_gate_passed"]:
        raise ValueError("Stage 5.6 calibration audit did not pass")

    rows = summary["summary"]
    if len(rows) != 18:
        raise ValueError(f"Expected 18 aggregate rows, found {len(rows)}")
    for row in rows:
        for metric in row["metrics"].values():
            if not math.isfinite(metric["mean"]) or not math.isfinite(metric["std"]):
                raise ValueError("Non-finite Stage 5.6 aggregate")
    for split in SPLITS:
        for representation in REPRESENTATIONS:
            values = [
                row["metrics"]["F1_score"]["mean"]
                for row in rows
                if row["split"] == split
                and row["representation"] == representation
            ]
            if len(values) != 3 or values != sorted(values):
                raise ValueError(
                    f"Non-monotonic or incomplete F1 curve: {split}/{representation}"
                )
    ten_percent = {
        split: {
            representation: next(
                row["metrics"]["F1_score"]["mean"]
                for row in rows
                if row["split"] == split
                and row["representation"] == representation
                and row["percentage"] == 10
            )
            for representation in REPRESENTATIONS
        }
        for split in SPLITS
    }
    return {
        "num_evaluations": summary["num_evaluations"],
        "aggregate_rows": len(rows),
        "all_finite": True,
        "all_f1_supervision_curves_monotonic": True,
        "ten_percent_f1": ten_percent,
    }


def validate_cells(hash_contents: bool) -> dict[str, Any]:
    records = []
    for representation in REPRESENTATIONS:
        for percentage in PERCENTAGES:
            for seed in SEEDS:
                tag = f"{representation}_{percentage}pct_seed{seed}_hierarchical"
                checkpoint = Path("checkpoints/stage5_6") / tag / "selected.pt"
                root = Path("outputs/stage5_6/cells") / tag
                selection_path = root / "selection_dev.json"
                calibration_path = root / "calibration_dev.json"
                selection = load_json(selection_path)
                calibration = load_json(calibration_path)
                if calibration["calibration_revision"] != "v2_wide":
                    raise ValueError(f"Non-v2 calibration: {tag}")
                active_boundaries = active_calibration_boundaries(calibration)
                if active_boundaries:
                    raise ValueError(
                        f"Truncated calibration boundary {active_boundaries}: {tag}"
                    )
                if calibration["num_rows_evaluated"] != 45_500:
                    raise ValueError(f"Incomplete calibration grid: {tag}")
                evaluations = {}
                for split in SPLITS:
                    path = root / f"evaluation_{split}.json"
                    result = load_json(path)
                    if not evaluation_matches_calibration(
                        result, calibration_path, calibration
                    ):
                        raise ValueError(
                            f"Evaluation does not use canonical v2 calibration: "
                            f"{tag}/{split}"
                        )
                    evaluations[split] = file_record(path)
                records.append(
                    {
                        "tag": tag,
                        "representation": representation,
                        "percentage": percentage,
                        "seed": seed,
                        "selected_epoch": selection["selected"]["epoch"],
                        "checkpoint": file_record(
                            checkpoint, hash_contents=hash_contents
                        ),
                        "selection": file_record(selection_path),
                        "calibration": file_record(calibration_path),
                        "calibration_best": {
                            "class0_logit_bias": calibration["best"][
                                "class0_logit_bias"
                            ],
                            "class3_logit_bias": calibration["best"][
                                "class3_logit_bias"
                            ],
                            "membership_threshold": calibration["best"][
                                "membership_threshold"
                            ],
                            "count_macro_mean_f1": calibration["best"][
                                "count_macro_mean_f1"
                            ],
                        },
                        "evaluations": evaluations,
                    }
                )
    return {
        "expected_cells": 27,
        "validated_cells": len(records),
        "checkpoint_contents_hashed": hash_contents,
        "cells": records,
    }


def validate_features(hash_large_files: bool) -> dict[str, Any]:
    records = {}
    expected_candidate_hash = None
    for representation in REPRESENTATIONS:
        stats_path = Path(
            f"outputs/stage5_6/{representation}_feature_union_stats.json"
        )
        stats = load_json(stats_path)
        if expected_candidate_hash is None:
            expected_candidate_hash = stats["candidate_file_sha256"]
        elif stats["candidate_file_sha256"] != expected_candidate_hash:
            raise ValueError("Representations do not share one candidate file")
        feature_path = Path(stats["output_file"])
        records[representation] = {
            "statistics": file_record(stats_path),
            "feature_file": file_record(
                feature_path, hash_contents=hash_large_files
            ),
            "representation": stats["representation"],
            "num_samples": stats["num_samples"],
            "num_unique_images": stats["num_unique_images"],
        }
    candidate_path = Path(
        load_json(Path("outputs/stage5_6/clip_feature_union_stats.json"))[
            "candidate_file"
        ]
    )
    candidate_record = file_record(
        candidate_path, hash_contents=hash_large_files
    )
    if hash_large_files and candidate_record["sha256"] != expected_candidate_hash:
        raise ValueError("Candidate file hash does not match feature statistics")
    return {
        "shared_candidate_sha256": expected_candidate_hash,
        "candidate_file": candidate_record,
        "large_file_contents_hashed": hash_large_files,
        "representations": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json", default="outputs/stage6/baseline_lock.json"
    )
    parser.add_argument(
        "--output-txt", default="outputs/stage6/baseline_lock.txt"
    )
    parser.add_argument(
        "--hash-large-files",
        action="store_true",
        help="Hash multi-gigabyte feature banks and the candidate JSONL.",
    )
    args = parser.parse_args()

    critical_paths = [
        Path("outputs/stage5_6/protocol_lock.json"),
        Path("outputs/stage5_6/split_manifest.json"),
        Path("outputs/stage5_6/calibration_revision_v2_protocol.json"),
        Path("outputs/stage5_6/calibration_v2_audit.json"),
        Path("outputs/stage5_6/test_summary.json"),
        Path("outputs/stage5_6/pilot_selection.json"),
        Path("outputs/stage5_6/proposal_recall_feature_union.json"),
    ]
    manifest = load_json(critical_paths[1])
    summary = load_json(critical_paths[4])
    audit = load_json(critical_paths[3])
    if audit["test_gate_passed"] is not True or audit["num_validated"] != 31:
        raise ValueError("Stage 5.6 wide calibration audit is incomplete")

    result = {
        "stage": "6.0",
        "purpose": "immutable Stage 5.6 v2 baseline for Stage 6 comparisons",
        "git_commit_at_stage6_start": git_head(),
        "critical_artifacts": {
            path.name: file_record(path) for path in critical_paths
        },
        "split_contract": validate_split_contract(manifest),
        "feature_contract": validate_features(args.hash_large_files),
        "cell_contract": validate_cells(args.hash_large_files),
        "benchmark_contract": validate_summary(summary),
        "calibration_audit": {
            "num_unique_calibrations": audit["num_unique_calibrations"],
            "num_validated": audit["num_validated"],
            "test_gate_passed": audit["test_gate_passed"],
        },
        "passed": True,
    }
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    feature_hash_note = (
        "full SHA-256"
        if args.hash_large_files
        else "size/path only (use --hash-large-files for full SHA-256)"
    )
    lines = [
        "Stage 6.0 Baseline Lock",
        "=======================",
        f"Stage 5.6 source commit: {result['git_commit_at_stage6_start']}",
        "Validated formal cells: 27/27",
        "Validated test evaluations: 54/54",
        "Validated wide calibrations: 31/31",
        "Calibration boundary gate: passed",
        "Train/dev/test image disjointness: passed",
        "All F1 supervision curves monotonic: passed",
        f"Large feature/candidate identity: {feature_hash_note}",
        "Stage 6 baseline lock: passed",
    ]
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
