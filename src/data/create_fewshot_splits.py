from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import random
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = PROJECT_ROOT / "data" / "grefcoco" / "annotations"
SPLIT_DIR = PROJECT_ROOT / "splits"

GREF_PATH = ANNOTATION_DIR / "grefs(unc).json"

SEED = 0
FRACTIONS = [0.01, 0.05, 0.10]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def get_target_type(ref: dict[str, Any]) -> str:
    if ref["no_target"]:
        return "no-target"

    num_targets = len(ref["ann_id"])

    if num_targets == 1:
        return "single-target"

    return "multi-target"


def expand_train_expressions(
    refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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


def summarize(samples: list[dict[str, Any]]) -> Counter:
    counts = Counter(sample["target_type"] for sample in samples)
    counts["total"] = len(samples)
    return counts


refs = load_json(GREF_PATH)
train_samples = expand_train_expressions(refs)

groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

for sample in train_samples:
    groups[sample["target_type"]].append(sample)

# One fixed shuffle per group ensures nested subsets.
for target_type, group in groups.items():
    random.Random(f"{SEED}-{target_type}").shuffle(group)

print("Full training set:", summarize(train_samples))

all_subsets = {}

for fraction in FRACTIONS:
    subset = []

    for target_type in [
        "no-target",
        "single-target",
        "multi-target",
    ]:
        group = groups[target_type]
        sample_count = round(len(group) * fraction)

        subset.extend(group[:sample_count])

    random.Random(SEED).shuffle(subset)

    percentage = int(fraction * 100)
    output_path = (
        SPLIT_DIR
        / f"train_{percentage}pct_seed{SEED}.json"
    )

    save_json(subset, output_path)
    all_subsets[percentage] = subset

    print(
        f"{percentage}% subset:",
        summarize(subset),
        "saved to",
        output_path,
    )


# Verify nested structure.
def sample_key(sample: dict[str, Any]) -> tuple[int, int]:
    return sample["ref_id"], sample["sent_id"]


keys_1 = {
    sample_key(sample)
    for sample in all_subsets[1]
}

keys_5 = {
    sample_key(sample)
    for sample in all_subsets[5]
}

keys_10 = {
    sample_key(sample)
    for sample in all_subsets[10]
}

assert keys_1 <= keys_5
assert keys_5 <= keys_10

print("Nested subset validation passed.")