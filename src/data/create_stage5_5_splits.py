"""Create the locked Stage 5.5 image-disjoint shadow-dev and 10% splits.

The source is the pre-test Stage 5 multi-seed train union, so every selected
expression and image already has frozen proposals and representation features.
Shadow-dev images are removed before sampling any enhanced training split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence

from src.data.create_fewshot_splits import count_group, sample_key, summarize


COUNT_GROUPS = ("0", "1", "2", "3+")
DEFAULT_DEV_TARGETS = (1000, 1000, 1000, 120)
DEFAULT_TRAIN_TARGETS = (1914, 12062, 6785, 173)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(value: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_of(sample: dict[str, Any]) -> str:
    return count_group(int(sample["num_targets"]))


def validate_targets(values: Sequence[int], name: str) -> dict[str, int]:
    if len(values) != len(COUNT_GROUPS):
        raise ValueError(f"{name} must contain counts for 0, 1, 2, and 3+.")
    counts = {group: int(value) for group, value in zip(COUNT_GROUPS, values)}
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"{name} counts must all be positive.")
    return counts


def build_shadow_dev(
    samples: Sequence[dict[str, Any]],
    targets: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_image[int(sample["image_id"])].append(dict(sample))

    selected_images: set[int] = set()
    counts = {group: 0 for group in COUNT_GROUPS}
    # Protect the rare 3+ group first, then fill the other deficits. Selecting
    # an image always selects all union expressions for that image.
    for group in ("3+", "0", "2", "1"):
        candidates = [
            image_id
            for image_id, rows in by_image.items()
            if any(group_of(row) == group for row in rows)
        ]
        random.Random(f"stage5.5-shadow-{seed}-{group}").shuffle(candidates)
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
        raise ValueError(f"Could not satisfy shadow-dev targets: {missing}")

    shadow = [dict(row) for row in samples if int(row["image_id"]) in selected_images]
    pool = [dict(row) for row in samples if int(row["image_id"]) not in selected_images]
    return shadow, pool


def sample_exact_count_groups(
    pool: Sequence[dict[str, Any]],
    targets: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in pool:
        grouped[group_of(sample)].append(dict(sample))
    chosen = []
    for group in COUNT_GROUPS:
        rows = grouped[group]
        random.Random(f"stage5.5-train-{seed}-{group}").shuffle(rows)
        if len(rows) < targets[group]:
            raise ValueError(
                f"Training pool has {len(rows)} rows for {group}, "
                f"but {targets[group]} are required."
            )
        chosen.extend(rows[: targets[group]])
    random.Random(f"stage5.5-train-order-{seed}").shuffle(chosen)
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-union", default="splits/train_stage5_union_seed0-2.json"
    )
    parser.add_argument("--output-dir", default="splits/stage5_5")
    parser.add_argument("--manifest", default="outputs/stage5_5/split_manifest.json")
    parser.add_argument("--shadow-seed", type=int, default=55)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--dev-count-targets", type=int, nargs=4, default=DEFAULT_DEV_TARGETS
    )
    parser.add_argument(
        "--train-count-targets", type=int, nargs=4, default=DEFAULT_TRAIN_TARGETS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_union)
    source = load_json(source_path)
    source_keys = [sample_key(row) for row in source]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("Source train union contains duplicate expression keys.")
    dev_targets = validate_targets(args.dev_count_targets, "dev-count-targets")
    train_targets = validate_targets(args.train_count_targets, "train-count-targets")
    shadow, pool = build_shadow_dev(source, dev_targets, args.shadow_seed)

    output_dir = Path(args.output_dir)
    shadow_path = output_dir / "shadow_dev.json"
    save_json(shadow, shadow_path)
    shadow_images = {int(row["image_id"]) for row in shadow}
    records = []
    train_key_sets = []
    for seed in dict.fromkeys(args.seeds):
        train = sample_exact_count_groups(pool, train_targets, int(seed))
        if shadow_images & {int(row["image_id"]) for row in train}:
            raise AssertionError("Shadow-dev image leakage into enhanced training.")
        path = output_dir / f"train_10pct_seed{seed}.json"
        save_json(train, path)
        train_key_sets.append({sample_key(row) for row in train})
        records.append(
            {
                "seed": int(seed),
                "path": str(path),
                "sha256": sha256_file(path),
                **summarize(train),
            }
        )

    manifest = {
        "stage": "5.5",
        "protocol": (
            "pre-test-train-union image-disjoint shadow-dev; exact count-group "
            "10% training targets; no test-based sampling"
        ),
        "source_union": str(source_path),
        "source_union_sha256": sha256_file(source_path),
        "source_summary": summarize(source),
        "shadow_seed": args.shadow_seed,
        "shadow_dev": {
            "path": str(shadow_path),
            "sha256": sha256_file(shadow_path),
            "requested_minimum_by_count_group": dev_targets,
            **summarize(shadow),
        },
        "remaining_pool": summarize(pool),
        "train_requested_by_count_group": train_targets,
        "train_splits": records,
        "integrity": {
            "shadow_train_expression_overlap": 0,
            "shadow_train_image_overlap": 0,
            "pairwise_train_expression_overlap": {
                f"seed{left}-seed{right}": len(train_key_sets[left] & train_key_sets[right])
                for left in range(len(train_key_sets))
                for right in range(left + 1, len(train_key_sets))
            },
        },
    }
    save_json(manifest, args.manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
