"""Create the locked Stage 5.6 development and few-shot training splits.

The development split is selected by whole image from the complete official
gRefCOCO training set and explicitly covers every target-count group. All
few-shot subsets are sampled only after removing those development images.
Within each seed, 1% is nested in 5%, which is nested in 10%.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import shlex
import subprocess
import sys
from typing import Any, Sequence

from src.data.create_fewshot_splits import (
    count_group,
    expand_train_expressions,
    load_json,
    sample_key,
    save_json,
    sha256_file,
    summarize,
)


COUNT_GROUPS = ("0", "1", "2", "3+")
DEFAULT_DEV_TARGETS = (1500, 1500, 1500, 300)
DEFAULT_PERCENTAGES = (1, 5, 10)
DEFAULT_SEEDS = (0, 1, 2)


def group_of(sample: dict[str, Any]) -> str:
    return count_group(int(sample["num_targets"]))


def validate_dev_targets(values: Sequence[int]) -> dict[str, int]:
    if len(values) != len(COUNT_GROUPS):
        raise ValueError("dev-count-targets must contain counts for 0, 1, 2, 3+.")
    targets = {group: int(value) for group, value in zip(COUNT_GROUPS, values)}
    if any(value <= 0 for value in targets.values()):
        raise ValueError("All development count targets must be positive.")
    return targets


def apportion_count_targets(
    source_counts: dict[str, int],
    percentage: int,
) -> dict[str, int]:
    """Use largest remainders while preserving the rounded total label budget."""
    if not 0 < percentage <= 100:
        raise ValueError("percentage must be in [1, 100].")
    quotas = {
        group: source_counts[group] * percentage / 100.0
        for group in COUNT_GROUPS
    }
    targets = {group: math.floor(quotas[group]) for group in COUNT_GROUPS}
    desired_total = round(sum(source_counts.values()) * percentage / 100.0)
    remaining = desired_total - sum(targets.values())
    order = sorted(
        COUNT_GROUPS,
        key=lambda group: (quotas[group] - targets[group], -COUNT_GROUPS.index(group)),
        reverse=True,
    )
    for group in order[:remaining]:
        targets[group] += 1
    return targets


def build_image_disjoint_dev(
    samples: Sequence[dict[str, Any]],
    targets: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_image[int(sample["image_id"])].append(dict(sample))

    selected_images: set[int] = set()
    counts = {group: 0 for group in COUNT_GROUPS}
    # The rare 3+ group is protected first. Adding an image always adds every
    # expression on that image, preventing visual leakage into training.
    for group in ("3+", "0", "2", "1"):
        candidates = [
            image_id
            for image_id, rows in by_image.items()
            if any(group_of(row) == group for row in rows)
        ]
        random.Random(f"stage5.6-dev-{seed}-{group}").shuffle(candidates)
        for image_id in candidates:
            if counts[group] >= targets[group]:
                break
            if image_id in selected_images:
                continue
            selected_images.add(image_id)
            for row in by_image[image_id]:
                counts[group_of(row)] += 1

    missing = {
        group: targets[group] - counts[group]
        for group in COUNT_GROUPS
        if counts[group] < targets[group]
    }
    if missing:
        raise ValueError(f"Could not satisfy development targets: {missing}")
    dev = [
        dict(sample)
        for sample in samples
        if int(sample["image_id"]) in selected_images
    ]
    pool = [
        dict(sample)
        for sample in samples
        if int(sample["image_id"]) not in selected_images
    ]
    return dev, pool


def build_nested_count_stratified_subsets(
    pool: Sequence[dict[str, Any]],
    source_counts: dict[str, int],
    percentages: Sequence[int],
    seed: int,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, dict[str, int]],
]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in pool:
        grouped[group_of(sample)].append(dict(sample))
    for group in COUNT_GROUPS:
        random.Random(f"stage5.6-train-{seed}-{group}").shuffle(grouped[group])

    subsets: dict[int, list[dict[str, Any]]] = {}
    targets_by_percentage: dict[int, dict[str, int]] = {}
    previous_keys: set[tuple[int, int]] = set()
    for percentage in sorted(set(int(value) for value in percentages)):
        targets = apportion_count_targets(source_counts, percentage)
        selected = []
        for group in COUNT_GROUPS:
            if len(grouped[group]) < targets[group]:
                raise ValueError(
                    f"Pool has {len(grouped[group])} rows for {group}, "
                    f"but {targets[group]} are required at {percentage}%."
                )
            selected.extend(grouped[group][: targets[group]])
        random.Random(f"stage5.6-order-{seed}-{percentage}").shuffle(selected)
        keys = {sample_key(row) for row in selected}
        if not previous_keys <= keys:
            raise AssertionError("Few-shot subsets are not nested.")
        previous_keys = keys
        subsets[percentage] = selected
        targets_by_percentage[percentage] = targets
    return subsets, targets_by_percentage


def build_union(*sample_sets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for samples in sample_sets:
        for sample in samples:
            by_key[sample_key(sample)] = dict(sample)
    return [by_key[key] for key in sorted(by_key)]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grefs", default="data/grefcoco/annotations/grefs(unc).json"
    )
    parser.add_argument("--output-dir", default="splits/stage5_6")
    parser.add_argument(
        "--manifest", default="outputs/stage5_6/split_manifest.json"
    )
    parser.add_argument(
        "--protocol-lock", default="outputs/stage5_6/protocol_lock.json"
    )
    parser.add_argument("--union-output", default="splits/stage5_6/feature_union.json")
    parser.add_argument("--dev-seed", type=int, default=56)
    parser.add_argument(
        "--dev-count-targets", type=int, nargs=4, default=DEFAULT_DEV_TARGETS
    )
    parser.add_argument(
        "--percentages", type=int, nargs="+", default=DEFAULT_PERCENTAGES
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--official-eval-splits",
        nargs="+",
        default=["splits/testA.json", "splits/testB.json"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    percentages = tuple(sorted(set(int(value) for value in args.percentages)))
    seeds = tuple(dict.fromkeys(int(value) for value in args.seeds))
    if percentages != DEFAULT_PERCENTAGES:
        raise ValueError("Locked Stage 5.6 percentages are 1, 5, and 10.")
    if seeds != DEFAULT_SEEDS:
        raise ValueError("Locked Stage 5.6 seeds are 0, 1, and 2.")

    grefs_path = Path(args.grefs)
    full_train = expand_train_expressions(load_json(grefs_path))
    full_keys = [sample_key(row) for row in full_train]
    if len(full_keys) != len(set(full_keys)):
        raise ValueError("Official training expressions contain duplicate keys.")
    source_summary = summarize(full_train)
    source_counts = source_summary["by_count_group"]

    dev_targets = validate_dev_targets(args.dev_count_targets)
    dev, pool = build_image_disjoint_dev(full_train, dev_targets, args.dev_seed)
    dev_images = {int(row["image_id"]) for row in dev}
    output_dir = Path(args.output_dir)
    dev_path = output_dir / "dev.json"
    save_json(dev, dev_path)

    split_records = []
    largest_splits = []
    integrity = {}
    for seed in seeds:
        subsets, targets = build_nested_count_stratified_subsets(
            pool, source_counts, percentages, seed
        )
        seed_key_sets = {}
        for percentage in percentages:
            samples = subsets[percentage]
            path = output_dir / f"train_{percentage}pct_seed{seed}.json"
            save_json(samples, path)
            images = {int(row["image_id"]) for row in samples}
            if images & dev_images:
                raise AssertionError("Development image leaked into training.")
            seed_key_sets[percentage] = {sample_key(row) for row in samples}
            split_records.append(
                {
                    "seed": seed,
                    "percentage": percentage,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "requested_by_count_group": targets[percentage],
                    **summarize(samples),
                }
            )
        largest_splits.append(subsets[max(percentages)])
        integrity[f"seed{seed}_nested"] = all(
            seed_key_sets[left] <= seed_key_sets[right]
            for left, right in zip(percentages, percentages[1:])
        )

    feature_union = build_union(dev, *largest_splits)
    union_path = Path(args.union_output)
    save_json(feature_union, union_path)

    eval_records = []
    train_images = {int(row["image_id"]) for row in full_train}
    for split_name in args.official_eval_splits:
        path = Path(split_name)
        rows = load_json(path)
        overlap = train_images & {int(row["image_id"]) for row in rows}
        if overlap:
            raise ValueError(f"Official train image overlap with {path}: {len(overlap)}")
        eval_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "expressions": len(rows),
                "train_image_overlap": 0,
            }
        )

    manifest = {
        "stage": "5.6",
        "protocol": (
            "whole-image balanced development split from complete official train; "
            "development images removed before exact count-stratified nested "
            "few-shot sampling"
        ),
        "grefs": str(grefs_path),
        "grefs_sha256": sha256_file(grefs_path),
        "source": source_summary,
        "dev_seed": args.dev_seed,
        "dev": {
            "path": str(dev_path),
            "sha256": sha256_file(dev_path),
            "requested_minimum_by_count_group": dev_targets,
            **summarize(dev),
        },
        "remaining_pool": summarize(pool),
        "splits": split_records,
        "feature_union": {
            "path": str(union_path),
            "sha256": sha256_file(union_path),
            **summarize(feature_union),
        },
        "official_evaluation_splits": eval_records,
        "integrity": {
            "dev_train_expression_overlap": 0,
            "dev_train_image_overlap": 0,
            **integrity,
        },
        "command": shlex.join(sys.argv),
    }
    manifest_path = Path(args.manifest)
    save_json(manifest, manifest_path)

    protocol_lock = {
        "stage": "5.6",
        "name": "final unified protocol retraining",
        "locked_before_new_training": True,
        "git_commit_at_lock": git_commit(),
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "development_policy": (
            "All recipe, epoch, and inference-calibration choices use only "
            "splits/stage5_6/dev.json with count-macro mean F1 as the primary "
            "criterion and official F1_score as the first tie-break."
        ),
        "pilot_policy": {
            "representation": "siglip2",
            "percentage": 10,
            "seed": 0,
            "variants_in_declared_order": [
                "selection_only",
                "balanced",
                "hierarchical",
                "one_to_one",
                "combined",
            ],
        },
        "final_grid": {
            "representations": ["clip", "clip_dinov2", "siglip2"],
            "percentages": list(percentages),
            "seeds": list(seeds),
            "epochs": 40,
            "recipe": "winner of the locked development-only pilot",
        },
        "test_policy": (
            "No testA/testB metric is used for recipe choice, checkpoint choice, "
            "calibration, or any retraining decision. After all 27 models are "
            "selected and calibrated, testA and testB are evaluated as the sole "
            "final benchmark splits for Stage 5.6."
        ),
        "historical_result_policy": (
            "Stages 5 and 5.5 remain archived development history and are not "
            "mixed with Stage 5.6 model selection or final aggregate tables."
        ),
    }
    save_json(protocol_lock, args.protocol_lock)
    print(json.dumps(manifest, indent=2))
    print(f"Protocol locked in {args.protocol_lock}")


if __name__ == "__main__":
    main()
