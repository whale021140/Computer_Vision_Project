"""Write a compact reproducibility manifest for one Stage 5 validation cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import torch


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--percentage", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--feature-stats", required=True)
    parser.add_argument("--val-feature-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--evaluation-json", required=True)
    parser.add_argument("--runner-command", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = load_json(args.split_file)
    feature_stats = load_json(args.feature_stats)
    calibration = load_json(args.calibration_json)
    evaluation = load_json(args.evaluation_json)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    representation = evaluation["representation"]["name"]
    if representation != args.representation:
        raise ValueError(
            f"Evaluation representation {representation!r} does not match "
            f"{args.representation!r}."
        )
    checkpoint_args = checkpoint.get("args", {})
    if int(checkpoint_args.get("seed", -1)) != args.seed:
        raise ValueError("Checkpoint seed does not match the Stage 5 cell.")
    if checkpoint_args.get("train_split_file") != args.split_file:
        raise ValueError("Checkpoint split file does not match the Stage 5 cell.")

    manifest = {
        "stage": 5,
        "cell": {
            "representation": args.representation,
            "percentage": args.percentage,
            "seed": args.seed,
        },
        "git_commit": git_commit(),
        "environment": "ece485",
        "runner_command": args.runner_command,
        "manifest_command": shlex.join(sys.argv),
        "split": {
            "path": args.split_file,
            "sha256": sha256_file(args.split_file),
            "expressions": len(split),
        },
        "features": {
            "train_union_stats": args.feature_stats,
            "candidate_file_sha256": feature_stats["candidate_file_sha256"],
            "representation": feature_stats["representation"],
            "num_samples_in_bank": feature_stats["num_samples"],
            "num_unique_images": feature_stats["num_unique_images"],
            "unique_candidate_regions": feature_stats["unique_candidate_regions"],
            "validation_file": args.val_feature_file,
        },
        "training": {
            "checkpoint": args.checkpoint,
            "best_epoch": int(checkpoint["epoch"]),
            "checkpoint_metrics": checkpoint["metrics"],
            "trainable_parameter_count": checkpoint["trainable_parameter_count"],
            "args": checkpoint_args,
        },
        "calibration": {
            "path": args.calibration_json,
            "membership_threshold": calibration["best"]["membership_threshold"],
            "threshold_sensitive_samples": calibration[
                "threshold_sensitive_samples"
            ],
        },
        "validation_evaluation": {
            "path": args.evaluation_json,
            "official": evaluation["official"],
            "diagnostics": evaluation["diagnostics"],
            "by_target_type": evaluation["by_target_type"],
            "by_count_group": evaluation["by_count_group"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
