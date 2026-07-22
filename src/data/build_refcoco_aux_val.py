"""Build a single-target auxiliary validation split from RefCOCO UNC val.

The source parquet is an annotation-only mirror of the cleaned RefCOCO data.
Every target annotation and bounding box is checked against the gRefCOCO COCO
instances file before any project artifact is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_URL = (
    "https://huggingface.co/datasets/jxu124/refcoco/resolve/main/data/"
    "validation-00000-of-00001-bfeafdc84ca37aa2.parquet"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable_sentence(sentence: Any) -> dict[str, Any]:
    sentence = dict(sentence)
    return {
        "raw": str(sentence["raw"]),
        "sent": str(sentence["sent"]),
        "sent_id": int(sentence["sent_id"]),
        "tokens": [str(token) for token in sentence["tokens"]],
    }


def bbox_matches(left: list[float], right: list[float], tolerance: float = 1e-5) -> bool:
    return len(left) == len(right) and all(
        abs(float(x) - float(y)) <= tolerance for x, y in zip(left, right)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument(
        "--grefs",
        default="data/grefcoco/annotations/grefs(unc).json",
    )
    parser.add_argument(
        "--instances",
        default="data/grefcoco/annotations/instances.json",
    )
    parser.add_argument(
        "--output-grefs",
        default="data/refcoco_aux/grefs_refcoco_unc_val.json",
    )
    parser.add_argument(
        "--output-split",
        default="splits/refcoco_unc_val.json",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/stage5/refcoco_aux/provenance.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_parquet)
    frame = pd.read_parquet(source_path)
    required = {
        "ann_id",
        "ref_id",
        "image_id",
        "split",
        "sentences",
        "category_id",
        "raw_anns",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Source parquet is missing columns: {missing_columns}")
    if set(frame["split"].astype(str)) != {"val"}:
        raise ValueError("Expected only the RefCOCO UNC val split.")
    if frame["ref_id"].duplicated().any():
        raise ValueError("RefCOCO source contains duplicate ref_id values.")

    instances = load_json(args.instances)
    ann_by_id = {int(ann["id"]): ann for ann in instances["annotations"]}
    image_by_id = {int(image["id"]): image for image in instances["images"]}
    grefs = load_json(args.grefs)
    gref_images = {
        split: {
            int(ref["image_id"])
            for ref in grefs
            if str(ref.get("split")) == split
        }
        for split in ("train", "val", "testA", "testB")
    }

    converted_refs: list[dict[str, Any]] = []
    split_samples: list[dict[str, Any]] = []
    seen_sent_ids: set[int] = set()
    target_ann_ids: set[int] = set()
    image_ids: set[int] = set()

    for row in frame.itertuples(index=False):
        ann_id = int(row.ann_id)
        ref_id = int(row.ref_id)
        image_id = int(row.image_id)
        if ann_id not in ann_by_id:
            raise KeyError(f"ann_id={ann_id} is absent from {args.instances}")
        if image_id not in image_by_id:
            raise KeyError(f"image_id={image_id} is absent from {args.instances}")
        ann = ann_by_id[ann_id]
        if int(ann["image_id"]) != image_id:
            raise ValueError(f"ann_id={ann_id} belongs to a different image.")
        raw_ann = json.loads(str(row.raw_anns))
        if not bbox_matches(ann["bbox"], raw_ann["bbox"]):
            raise ValueError(f"Bounding-box mismatch for ann_id={ann_id}.")

        sentences = [jsonable_sentence(sentence) for sentence in row.sentences]
        if not sentences:
            raise ValueError(f"ref_id={ref_id} has no expressions.")
        for sentence in sentences:
            sent_id = int(sentence["sent_id"])
            if sent_id in seen_sent_ids:
                raise ValueError(f"Duplicate sent_id={sent_id}.")
            seen_sent_ids.add(sent_id)
            split_samples.append(
                {
                    "ref_id": ref_id,
                    "sent_id": sent_id,
                    "image_id": image_id,
                    "target_type": "single-target",
                    "num_targets": 1,
                }
            )

        converted_refs.append(
            {
                "ref_id": ref_id,
                "ann_id": [ann_id],
                "category_id": [int(row.category_id)],
                "image_id": image_id,
                "file_name": str(image_by_id[image_id]["file_name"]),
                "split": "refcoco_unc_val",
                "no_target": False,
                "sent_ids": [int(sentence["sent_id"]) for sentence in sentences],
                "sentences": sentences,
            }
        )
        target_ann_ids.add(ann_id)
        image_ids.add(image_id)

    output_grefs = Path(args.output_grefs)
    output_split = Path(args.output_split)
    manifest_path = Path(args.manifest)
    for path in (output_grefs, output_split, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_grefs.write_text(
        json.dumps(converted_refs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_split.write_text(
        json.dumps(split_samples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    overlap = {
        split: len(image_ids & split_images)
        for split, split_images in gref_images.items()
    }
    if overlap["train"] or overlap["testA"] or overlap["testB"]:
        raise ValueError(f"Unexpected RefCOCO val image leakage: {overlap}")
    if overlap["val"] != len(image_ids):
        raise ValueError(
            "RefCOCO UNC val images are not a subset of current gRefCOCO val."
        )

    manifest = {
        "dataset": "RefCOCO",
        "split_by": "unc",
        "source_split": "val",
        "role": "single-target auxiliary validation; not a replacement for gRefCOCO val",
        "source_url": SOURCE_URL,
        "source_parquet": str(source_path),
        "source_sha256": sha256_file(source_path),
        "grefcoco_annotation_file": args.grefs,
        "grefcoco_annotation_sha256": sha256_file(args.grefs),
        "instances_file": args.instances,
        "instances_sha256": sha256_file(args.instances),
        "counts": {
            "refs": len(converted_refs),
            "expressions": len(split_samples),
            "images": len(image_ids),
            "unique_target_annotations": len(target_ann_ids),
            "target_type": {"single-target": len(split_samples)},
        },
        "integrity": {
            "all_target_ids_in_instances": True,
            "all_target_bboxes_match_instances": True,
            "unique_ref_ids": True,
            "unique_sent_ids": True,
            "image_overlap_with_grefcoco": overlap,
            "current_grefcoco_val_images": len(gref_images["val"]),
        },
        "output_grefs": str(output_grefs),
        "output_grefs_sha256": sha256_file(output_grefs),
        "output_split": str(output_split),
        "output_split_sha256": sha256_file(output_split),
        "command": shlex.join(sys.argv),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
