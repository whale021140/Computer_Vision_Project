"""Audit few-shot nesting, uniqueness, and train/evaluation separation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence


TRAIN_NAME = re.compile(r"train_(?P<percentage>\d+)pct_seed(?P<seed>-?\d+)\.json$")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expression_key(sample: dict[str, Any]) -> tuple[int, int]:
    return int(sample["ref_id"]), int(sample["sent_id"])


def describe(path: str | Path) -> dict[str, Any]:
    samples = load_json(path)
    expression_keys = [expression_key(sample) for sample in samples]
    if len(expression_keys) != len(set(expression_keys)):
        raise ValueError(f"Duplicate expression keys in {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "samples": len(samples),
        "expression_keys": set(expression_keys),
        "ref_ids": {int(sample["ref_id"]) for sample in samples},
        "image_ids": {int(sample["image_id"]) for sample in samples},
    }


def audit_splits(
    train_split_files: Sequence[str | Path],
    eval_split_files: Sequence[str | Path],
) -> dict[str, Any]:
    train = {}
    for path in train_split_files:
        match = TRAIN_NAME.search(Path(path).name)
        if match is None:
            raise ValueError(f"Unrecognized train split name: {path}")
        key = (int(match.group("seed")), int(match.group("percentage")))
        if key in train:
            raise ValueError(f"Duplicate seed/percentage cell: {key}")
        train[key] = describe(path)
    evaluation = {Path(path).stem: describe(path) for path in eval_split_files}

    nesting = []
    for seed in sorted({seed for seed, _ in train}):
        percentages = sorted(pct for item_seed, pct in train if item_seed == seed)
        previous: set[tuple[int, int]] = set()
        for percentage in percentages:
            keys = train[(seed, percentage)]["expression_keys"]
            nested = previous <= keys
            nesting.append(
                {
                    "seed": seed,
                    "percentage": percentage,
                    "contains_previous": nested,
                }
            )
            if not nested:
                raise ValueError(f"Non-nested split for seed={seed}, {percentage}%")
            previous = keys

    train_eval_overlap = []
    for (seed, percentage), train_record in sorted(train.items()):
        for eval_name, eval_record in evaluation.items():
            row = {
                "seed": seed,
                "percentage": percentage,
                "evaluation_split": eval_name,
                "expression_overlap": len(
                    train_record["expression_keys"] & eval_record["expression_keys"]
                ),
                "ref_overlap": len(train_record["ref_ids"] & eval_record["ref_ids"]),
                "image_overlap": len(
                    train_record["image_ids"] & eval_record["image_ids"]
                ),
            }
            train_eval_overlap.append(row)
            if any(row[key] for key in ("expression_overlap", "ref_overlap", "image_overlap")):
                raise ValueError(f"Train/evaluation leakage detected: {row}")

    eval_overlap = []
    eval_items = sorted(evaluation.items())
    for index, (left_name, left) in enumerate(eval_items):
        for right_name, right in eval_items[index + 1 :]:
            eval_overlap.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "expression_overlap": len(
                        left["expression_keys"] & right["expression_keys"]
                    ),
                    "ref_overlap": len(left["ref_ids"] & right["ref_ids"]),
                    "image_overlap": len(left["image_ids"] & right["image_ids"]),
                }
            )

    def public(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key not in {"expression_keys", "ref_ids", "image_ids"}
        } | {
            "unique_refs": len(record["ref_ids"]),
            "unique_images": len(record["image_ids"]),
        }

    return {
        "status": "passed",
        "train": [
            {"seed": seed, "percentage": percentage, **public(record)}
            for (seed, percentage), record in sorted(train.items())
        ],
        "evaluation": {
            name: public(record) for name, record in sorted(evaluation.items())
        },
        "nesting": nesting,
        "train_evaluation_overlap": train_eval_overlap,
        "evaluation_overlap": eval_overlap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-split-files", nargs="+", required=True)
    parser.add_argument("--eval-split-files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_splits(args.train_split_files, args.eval_split_files)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
