from __future__ import annotations

from pathlib import Path
import json
import random
import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_DIR = PROJECT_ROOT / "data" / "grefcoco" / "annotations"
IMAGE_DIR = PROJECT_ROOT / "data" / "coco" / "train2014"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

GREF_PATH = ANNOTATION_DIR / "grefs(unc).json"
INSTANCE_PATH = ANNOTATION_DIR / "instances.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 0


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def coco_xywh_to_xyxy(box):
    x, y, width, height = box
    return x, y, x + width, y + height


refs = load_json(GREF_PATH)
instances = load_json(INSTANCE_PATH)

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


groups = {
    "no-target": [],
    "single-target": [],
    "multi-target": [],
}

for ref in refs:
    target_type = get_target_type(ref)

    for sentence in ref["sentences"]:
        groups[target_type].append(
            {
                "ref": ref,
                "sentence": sentence,
            }
        )


random_generator = random.Random(RANDOM_SEED)

selected = []

for target_type in [
    "no-target",
    "single-target",
    "multi-target",
]:
    selected.extend(
        random_generator.sample(groups[target_type], 2)
    )


fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(14, 16),
)

for axis, sample in zip(axes.flat, selected):
    ref = sample["ref"]
    sentence = sample["sentence"]

    image_path = IMAGE_DIR / ref["file_name"]
    image = Image.open(image_path).convert("RGB")

    axis.imshow(image)
    axis.axis("off")

    if not ref["no_target"]:
        for annotation_id in ref["ann_id"]:
            annotation = annotation_by_id[annotation_id]
            x1, y1, x2, y2 = coco_xywh_to_xyxy(
                annotation["bbox"]
            )

            rectangle = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )

            axis.add_patch(rectangle)

    target_type = get_target_type(ref)
    num_targets = 0 if ref["no_target"] else len(ref["ann_id"])

    wrapped_expression = "\n".join(
        textwrap.wrap(sentence["sent"], width=45)
    )

    axis.set_title(
        f"{target_type} | targets={num_targets}\n"
        f"{wrapped_expression}",
        fontsize=11,
    )

plt.tight_layout()

output_path = (
    OUTPUT_DIR
    / "representative_grefcoco_samples.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Saved:", output_path)