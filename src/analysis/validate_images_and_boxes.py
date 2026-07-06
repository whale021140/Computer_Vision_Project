from pathlib import Path
from collections import Counter
import json
import statistics

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "grefcoco"
    / "annotations"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "coco"
    / "train2014"
)

GREF_PATH = ANNOTATION_DIR / "grefs(unc).json"
INSTANCE_PATH = ANNOTATION_DIR / "instances.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


refs = load_json(GREF_PATH)
instances = load_json(INSTANCE_PATH)

image_by_id = {
    image["id"]: image
    for image in instances["images"]
}

annotation_by_id = {
    annotation["id"]: annotation
    for annotation in instances["annotations"]
}


validation = Counter()

widths = []
heights = []

checked_image_ids = set()


for image_id, image_info in image_by_id.items():
    file_name = image_info["file_name"]
    image_path = IMAGE_DIR / file_name

    if not image_path.exists():
        validation["missing_image_files"] += 1
        continue

    validation["existing_image_files"] += 1

    metadata_width = image_info["width"]
    metadata_height = image_info["height"]

    widths.append(metadata_width)
    heights.append(metadata_height)

    try:
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size

        if (
            actual_width != metadata_width
            or actual_height != metadata_height
        ):
            validation["image_size_mismatch"] += 1
        else:
            validation["image_size_match"] += 1

    except Exception:
        validation["unreadable_images"] += 1


for ref in refs:
    if ref["no_target"]:
        if ref["ann_id"] != [-1]:
            validation["unexpected_no_target_ann_id"] += 1

        continue

    for ann_id in ref["ann_id"]:
        ann = annotation_by_id.get(ann_id)

        if ann is None:
            validation["missing_annotation"] += 1
            continue

        image_info = image_by_id[ann["image_id"]]

        image_width = image_info["width"]
        image_height = image_info["height"]

        x, y, width, height = ann["bbox"]

        if width <= 0 or height <= 0:
            validation["nonpositive_boxes"] += 1
            continue

        x2 = x + width
        y2 = y + height

        if (
            x < 0
            or y < 0
            or x2 > image_width
            or y2 > image_height
        ):
            validation["out_of_bounds_boxes"] += 1
        else:
            validation["valid_boxes"] += 1


print("=" * 100)
print("IMAGE AND BOUNDING-BOX VALIDATION")
print("=" * 100)

for key, value in sorted(validation.items()):
    print(f"{key}: {value}")


print("\n" + "=" * 100)
print("IMAGE RESOLUTION STATISTICS")
print("=" * 100)

if widths and heights:
    print("Width minimum:", min(widths))
    print("Width maximum:", max(widths))
    print("Width mean:", statistics.mean(widths))
    print("Width median:", statistics.median(widths))

    print("Height minimum:", min(heights))
    print("Height maximum:", max(heights))
    print("Height mean:", statistics.mean(heights))
    print("Height median:", statistics.median(heights))