from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import time

from torch.utils.data import DataLoader

from grefcoco_dataset import (
    GRefCOCODataset,
    grefcoco_collate_fn,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_DIR = PROJECT_ROOT / "data" / "grefcoco" / "annotations"
IMAGE_DIR = PROJECT_ROOT / "data" / "coco" / "train2014"
SPLIT_PATH = PROJECT_ROOT / "splits" / "train_1pct_seed0.json"

GREF_PATH = ANNOTATION_DIR / "grefs(unc).json"
INSTANCE_PATH = ANNOTATION_DIR / "instances.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


refs = load_json(GREF_PATH)
instances = load_json(INSTANCE_PATH)
fewshot_ids = load_json(SPLIT_PATH)

annotation_by_id = {
    annotation["id"]: annotation
    for annotation in instances["annotations"]
}


def get_target_type(ref):
    if ref["no_target"]:
        return "no-target"

    if len(ref["ann_id"]) == 1:
        return "single-target"

    return "multi-target"


selected_keys = {
    (
        sample["ref_id"],
        sample["sent_id"],
    )
    for sample in fewshot_ids
}


samples = []

for ref in refs:
    if ref["split"] != "train":
        continue

    target_type = get_target_type(ref)

    for sentence in ref["sentences"]:
        key = (
            ref["ref_id"],
            sentence["sent_id"],
        )

        if key not in selected_keys:
            continue

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
                "target_type": target_type,
                "split": ref["split"],
            }
        )


dataset = GRefCOCODataset(
    samples=samples,
    image_root=IMAGE_DIR,
    annotation_by_id=annotation_by_id,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=grefcoco_collate_fn,
    pin_memory=True,
)


print("=" * 100)
print("DATASET SUMMARY")
print("=" * 100)

print("Dataset size:", len(dataset))

print(
    "Target-type counts:",
    Counter(
        sample["target_type"]
        for sample in samples
    ),
)


print("\n" + "=" * 100)
print("SINGLE SAMPLE TEST")
print("=" * 100)

sample = dataset[0]

print("image_id:", sample["image_id"])
print("expression:", sample["expression"])
print("boxes shape:", sample["boxes"].shape)
print("normalized_boxes:", sample["normalized_boxes"])
print("num_targets:", sample["num_targets"])
print("target_type:", sample["target_type"])
print("original_size:", sample["original_size"])


print("\n" + "=" * 100)
print("BATCH TEST")
print("=" * 100)

batch = next(iter(loader))

print("Batch size:", len(batch["images"]))
print("Image IDs:", batch["image_ids"])
print("Expressions:", batch["expressions"])
print("Box shapes:", [box.shape for box in batch["boxes"]])
print("Target counts:", batch["num_targets"])
print("Target types:", batch["target_types"])
print("Original sizes:", batch["original_sizes"])


print("\n" + "=" * 100)
print("FULL LOADER VALIDATION")
print("=" * 100)

start_time = time.perf_counter()

num_batches = 0
num_samples = 0
num_failures = 0

target_counts = Counter()

try:
    for current_batch in loader:
        num_batches += 1
        num_samples += len(current_batch["images"])

        target_counts.update(
            current_batch["target_types"]
        )

except Exception as exception:
    num_failures += 1
    print("Loader failure:", repr(exception))

elapsed_time = time.perf_counter() - start_time

print("Successfully loaded batches:", num_batches)
print("Successfully loaded samples:", num_samples)
print("Loading failures:", num_failures)
print("Elapsed time:", elapsed_time)

if num_batches > 0:
    print(
        "Average time per batch:",
        elapsed_time / num_batches,
    )

print("Observed target counts:", target_counts)