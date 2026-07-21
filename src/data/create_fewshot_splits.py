"""Create deterministic, nested, expression-level few-shot train splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shlex
import sys
from typing import Any, Iterable, Sequence


DEFAULT_GREFS = Path("data/grefcoco/annotations/grefs(unc).json")
DEFAULT_SPLIT_DIR = Path("splits")
TARGET_TYPES = ("no-target", "single-target", "multi-target")
DEFAULT_PERCENTAGES = (1, 5, 10)
DEFAULT_SEEDS = (0,)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_target_type(ref: dict[str, Any]) -> str:
    if ref["no_target"]:
        return "no-target"
    if len(ref["ann_id"]) == 1:
        return "single-target"
    return "multi-target"


def expand_train_expressions(refs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    for ref in refs:
        if ref["split"] != "train":
            continue
        target_type = get_target_type(ref)
        num_targets = 0 if ref["no_target"] else len(ref["ann_id"])
        for sentence in ref["sentences"]:
            samples.append(
                {
                    "ref_id": ref["ref_id"],
                    "sent_id": sentence["sent_id"],
                    "image_id": ref["image_id"],
                    "target_type": target_type,
                    "num_targets": num_targets,
                }
            )
    return samples


def sample_key(sample: dict[str, Any]) -> tuple[int, int]:
    return int(sample["ref_id"]), int(sample["sent_id"])


def count_group(num_targets: int) -> str:
    return str(num_targets) if num_targets <= 2 else "3+"


def summarize(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    target_types = Counter(sample["target_type"] for sample in samples)
    count_groups = Counter(count_group(int(sample["num_targets"])) for sample in samples)
    return {
        "expressions": len(samples),
        "unique_refs": len({int(sample["ref_id"]) for sample in samples}),
        "unique_images": len({int(sample["image_id"]) for sample in samples}),
        "by_target_type": {
            key: int(target_types.get(key, 0)) for key in TARGET_TYPES
        },
        "by_count_group": {
            key: int(count_groups.get(key, 0)) for key in ("0", "1", "2", "3+")
        },
    }


def validate_percentages(percentages: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(value) for value in percentages)))
    if not normalized or any(value <= 0 or value > 100 for value in normalized):
        raise ValueError("Percentages must be unique integers in [1, 100].")
    return normalized


def build_nested_subsets(
    train_samples: Sequence[dict[str, Any]],
    seed: int,
    percentages: Sequence[int] = DEFAULT_PERCENTAGES,
) -> dict[int, list[dict[str, Any]]]:
    """Stratify by target type and take nested prefixes for one data seed."""
    percentages = validate_percentages(percentages)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in train_samples:
        groups[str(sample["target_type"])].append(dict(sample))
    missing = [target_type for target_type in TARGET_TYPES if not groups[target_type]]
    if missing:
        raise ValueError(f"Training data has empty target-type groups: {missing}")

    for target_type in TARGET_TYPES:
        random.Random(f"{seed}-{target_type}").shuffle(groups[target_type])

    subsets: dict[int, list[dict[str, Any]]] = {}
    for percentage in percentages:
        subset = []
        for target_type in TARGET_TYPES:
            group = groups[target_type]
            sample_count = round(len(group) * percentage / 100)
            subset.extend(group[:sample_count])
        random.Random(seed).shuffle(subset)
        subsets[percentage] = subset
    validate_nested_subsets(train_samples, subsets)
    return subsets


def validate_nested_subsets(
    train_samples: Sequence[dict[str, Any]],
    subsets: dict[int, Sequence[dict[str, Any]]],
) -> None:
    source_keys = [sample_key(sample) for sample in train_samples]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("Full training expressions contain duplicate (ref_id, sent_id) keys.")
    source_key_set = set(source_keys)
    previous: set[tuple[int, int]] = set()
    for percentage in sorted(subsets):
        keys = [sample_key(sample) for sample in subsets[percentage]]
        key_set = set(keys)
        if len(keys) != len(key_set):
            raise ValueError(f"{percentage}% split contains duplicate expression keys.")
        if not key_set <= source_key_set:
            raise ValueError(f"{percentage}% split contains expressions outside train.")
        if not previous <= key_set:
            raise ValueError("Few-shot splits are not nested within a seed.")
        previous = key_set


def build_union(
    per_seed_subsets: dict[int, dict[int, Sequence[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for subsets in per_seed_subsets.values():
        largest = subsets[max(subsets)]
        for sample in largest:
            by_key[sample_key(sample)] = dict(sample)
    return [by_key[key] for key in sorted(by_key)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create expression-level target-type-stratified few-shot splits. "
            "Within each seed, smaller fractions are strict subsets of larger ones."
        )
    )
    parser.add_argument("--grefs", default=str(DEFAULT_GREFS))
    parser.add_argument("--output-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        default=list(DEFAULT_PERCENTAGES),
    )
    parser.add_argument(
        "--union-output",
        default="",
        help="Optional output containing the union of every seed's largest split.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional JSON manifest. Defaults to OUTPUT_DIR/fewshot_manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    percentages = validate_percentages(args.percentages)
    seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds:
        raise ValueError("At least one seed is required.")

    grefs_path = Path(args.grefs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_samples = expand_train_expressions(load_json(grefs_path))
    print("Full training set:", summarize(train_samples))

    per_seed_subsets: dict[int, dict[int, list[dict[str, Any]]]] = {}
    split_records = []
    for seed in seeds:
        subsets = build_nested_subsets(train_samples, seed, percentages)
        per_seed_subsets[seed] = subsets
        for percentage, samples in subsets.items():
            output_path = output_dir / f"train_{percentage}pct_seed{seed}.json"
            save_json(samples, output_path)
            record = {
                "seed": seed,
                "percentage": percentage,
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                **summarize(samples),
            }
            split_records.append(record)
            print(f"seed={seed} {percentage}%:", record)

    union_record = None
    if args.union_output:
        union_samples = build_union(per_seed_subsets)
        union_path = Path(args.union_output)
        save_json(union_samples, union_path)
        union_record = {
            "path": str(union_path),
            "sha256": sha256_file(union_path),
            **summarize(union_samples),
        }
        print("Stage 5 union:", union_record)

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_dir / "fewshot_manifest.json"
    )
    manifest = {
        "protocol": "expression-level target-type-stratified nested few-shot splits",
        "grefs": str(grefs_path),
        "grefs_sha256": sha256_file(grefs_path),
        "seeds": list(seeds),
        "percentages": list(percentages),
        "full_train": summarize(train_samples),
        "splits": split_records,
        "largest-split-union": union_record,
        "command": shlex.join(sys.argv),
    }
    save_json(manifest, manifest_path)
    print(f"Nested split validation passed; manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
