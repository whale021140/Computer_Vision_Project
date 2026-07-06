from pathlib import Path
from collections import Counter, defaultdict
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = PROJECT_ROOT / "data" / "grefcoco" / "annotations"

GREF_PATH = ANNOTATION_DIR / "grefs(unc).json"
INSTANCE_PATH = ANNOTATION_DIR / "instances.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


refs = load_json(GREF_PATH)
instances = load_json(INSTANCE_PATH)

print("=" * 100)
print("BASIC FILE STATISTICS")
print("=" * 100)

print("Number of references:", len(refs))
print("Number of COCO images:", len(instances["images"]))
print("Number of COCO annotations:", len(instances["annotations"]))
print("Number of COCO categories:", len(instances["categories"]))


# ------------------------------------------------------------------
# Build COCO lookup tables
# ------------------------------------------------------------------

image_by_id = {
    image["id"]: image
    for image in instances["images"]
}

annotation_by_id = {
    annotation["id"]: annotation
    for annotation in instances["annotations"]
}

category_by_id = {
    category["id"]: category["name"]
    for category in instances["categories"]
}


# ------------------------------------------------------------------
# Expand references into expression-level samples
# ------------------------------------------------------------------

samples = []

for ref in refs:
    if ref["no_target"]:
        num_targets = 0
    else:
        num_targets = len(ref["ann_id"])

    if num_targets == 0:
        target_type = "no-target"
    elif num_targets == 1:
        target_type = "single-target"
    else:
        target_type = "multi-target"

    for sentence in ref["sentences"]:
        samples.append(
            {
                "ref_id": ref["ref_id"],
                "sent_id": sentence["sent_id"],
                "image_id": ref["image_id"],
                "file_name": ref["file_name"],
                "expression": sentence["sent"],
                "tokens": sentence["tokens"],
                "ann_ids": ref["ann_id"],
                "category_ids": ref["category_id"],
                "no_target": ref["no_target"],
                "num_targets": num_targets,
                "target_type": target_type,
                "split": ref["split"],
            }
        )


print("\n" + "=" * 100)
print("EXPRESSION-LEVEL STATISTICS")
print("=" * 100)

print("Number of expression-level samples:", len(samples))
print("Available splits:", sorted(set(sample["split"] for sample in samples)))


# ------------------------------------------------------------------
# Per-split statistics
# ------------------------------------------------------------------

split_stats = defaultdict(Counter)
split_images = defaultdict(set)
split_refs = defaultdict(set)

for sample in samples:
    split = sample["split"]

    split_stats[split]["expressions"] += 1
    split_stats[split][sample["target_type"]] += 1

    split_images[split].add(sample["image_id"])
    split_refs[split].add(sample["ref_id"])


print("\n" + "=" * 100)
print("PER-SPLIT STATISTICS")
print("=" * 100)

for split in sorted(split_stats):
    print(f"\nSplit: {split}")
    print("  Unique images:", len(split_images[split]))
    print("  References:", len(split_refs[split]))
    print("  Expressions:", split_stats[split]["expressions"])
    print("  No-target:", split_stats[split]["no-target"])
    print("  Single-target:", split_stats[split]["single-target"])
    print("  Multi-target:", split_stats[split]["multi-target"])


# ------------------------------------------------------------------
# Overall target distribution
# ------------------------------------------------------------------

target_type_counts = Counter(
    sample["target_type"]
    for sample in samples
)

target_count_distribution = Counter(
    sample["num_targets"]
    for sample in samples
)

print("\n" + "=" * 100)
print("OVERALL TARGET-TYPE DISTRIBUTION")
print("=" * 100)

for target_type in ["no-target", "single-target", "multi-target"]:
    count = target_type_counts[target_type]
    percentage = 100.0 * count / len(samples)

    print(
        f"{target_type}: "
        f"{count} expressions "
        f"({percentage:.2f}%)"
    )


print("\n" + "=" * 100)
print("EXACT TARGET-COUNT DISTRIBUTION")
print("=" * 100)

for num_targets in sorted(target_count_distribution):
    print(
        f"{num_targets} targets: "
        f"{target_count_distribution[num_targets]}"
    )


# ------------------------------------------------------------------
# Expression length
# ------------------------------------------------------------------

expression_lengths = [
    len(sample["tokens"])
    for sample in samples
]

print("\n" + "=" * 100)
print("EXPRESSION LENGTH")
print("=" * 100)

print("Minimum length:", min(expression_lengths))
print("Maximum length:", max(expression_lengths))
print(
    "Mean length:",
    sum(expression_lengths) / len(expression_lengths)
)


# ------------------------------------------------------------------
# Validate IDs
# ------------------------------------------------------------------

missing_images = 0
missing_annotations = 0

for ref in refs:
    if ref["image_id"] not in image_by_id:
        missing_images += 1

    if not ref["no_target"]:
        for annotation_id in ref["ann_id"]:
            if annotation_id not in annotation_by_id:
                missing_annotations += 1


print("\n" + "=" * 100)
print("ID VALIDATION")
print("=" * 100)

print("References with missing image IDs:", missing_images)
print("Missing annotation IDs:", missing_annotations)


# ------------------------------------------------------------------
# Target category frequencies
# ------------------------------------------------------------------

category_counts = Counter()

for sample in samples:
    if sample["no_target"]:
        continue

    for category_id in sample["category_ids"]:
        category_name = category_by_id.get(
            category_id,
            f"unknown-{category_id}",
        )
        category_counts[category_name] += 1


print("\n" + "=" * 100)
print("TOP 10 TARGET CATEGORIES")
print("=" * 100)

for category_name, count in category_counts.most_common(10):
    print(f"{category_name}: {count}")

print("\n" + "=" * 100)
print("IMAGE OVERLAP BETWEEN SPLITS")
print("=" * 100)

split_names = sorted(split_images.keys())

for i, split_a in enumerate(split_names):
    for split_b in split_names[i + 1:]:
        overlap = split_images[split_a] & split_images[split_b]

        print(
            f"{split_a} vs {split_b}: "
            f"{len(overlap)} overlapping images"
        )

        if overlap:
            print("  Example image IDs:", sorted(overlap)[:10])