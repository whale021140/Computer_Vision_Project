import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from src.utils.boxes import xywh_to_xyxy, normalize_xyxy, count_to_class


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_sentence(ref: dict, sent_id: int) -> str:
    for sent in ref["sentences"]:
        if sent["sent_id"] == sent_id:
            return sent["sent"]
    raise KeyError(f"sent_id={sent_id} not found in ref_id={ref['ref_id']}")


def clean_target_ann_ids(ref: dict) -> list[int]:
    """Return valid target annotation ids. No-target refs use ann_id=[-1]."""
    return [int(aid) for aid in ref["ann_id"] if int(aid) != -1]


def build_lookups(grefs: list[dict], instances: dict):
    ref_by_id = {int(ref["ref_id"]): ref for ref in grefs}

    image_by_id = {
        int(image["id"]): image
        for image in instances["images"]
    }

    ann_by_id = {
        int(ann["id"]): ann
        for ann in instances["annotations"]
    }

    anns_by_image = defaultdict(list)
    for ann in instances["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    return ref_by_id, image_by_id, ann_by_id, anns_by_image


def build_candidate_sample(
    split_sample: dict,
    ref_by_id: dict,
    image_by_id: dict,
    ann_by_id: dict,
    anns_by_image: dict,
    sample_index: int,
) -> dict:
    ref_id = int(split_sample["ref_id"])
    sent_id = int(split_sample["sent_id"])
    image_id = int(split_sample["image_id"])

    ref = ref_by_id[ref_id]
    image = image_by_id[image_id]

    width = int(image["width"])
    height = int(image["height"])
    file_name = ref.get("file_name", image["file_name"])

    expression = find_sentence(ref, sent_id)

    target_ann_ids = clean_target_ann_ids(ref)
    target_ann_id_set = set(target_ann_ids)

    target_boxes_xyxy = []
    target_boxes_norm = []

    for ann_id in target_ann_ids:
        ann = ann_by_id[ann_id]
        box_xyxy = xywh_to_xyxy(ann["bbox"])
        target_boxes_xyxy.append(box_xyxy)
        target_boxes_norm.append(normalize_xyxy(box_xyxy, width, height))

    candidate_anns = anns_by_image.get(image_id, [])

    candidate_ann_ids = []
    candidate_boxes_xyxy = []
    candidate_boxes_norm = []
    candidate_labels = []

    for ann in candidate_anns:
        ann_id = int(ann["id"])
        box_xyxy = xywh_to_xyxy(ann["bbox"])

        candidate_ann_ids.append(ann_id)
        candidate_boxes_xyxy.append(box_xyxy)
        candidate_boxes_norm.append(normalize_xyxy(box_xyxy, width, height))
        candidate_labels.append(1 if ann_id in target_ann_id_set else 0)

    num_targets = len(target_ann_ids)

    return {
        "sample_id": f"{sample_index:07d}",
        "ref_id": ref_id,
        "sent_id": sent_id,
        "image_id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "expression": expression,
        "target_type": split_sample["target_type"],
        "target_ann_ids": target_ann_ids,
        "target_boxes_xyxy": target_boxes_xyxy,
        "target_boxes_norm": target_boxes_norm,
        "candidate_ann_ids": candidate_ann_ids,
        "candidate_boxes_xyxy": candidate_boxes_xyxy,
        "candidate_boxes_norm": candidate_boxes_norm,
        "candidate_labels": candidate_labels,
        "num_targets": num_targets,
        "count_class": count_to_class(num_targets),
        "num_candidates": len(candidate_ann_ids),
    }


def write_jsonl(samples: list[dict], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_stats(samples: list[dict], stats_file: Path):
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    target_type_counter = Counter(sample["target_type"] for sample in samples)
    count_class_counter = Counter(sample["count_class"] for sample in samples)

    candidate_counts = [sample["num_candidates"] for sample in samples]
    positive_counts = [sum(sample["candidate_labels"]) for sample in samples]
    target_counts = [sample["num_targets"] for sample in samples]

    zero_candidate_samples = sum(1 for c in candidate_counts if c == 0)

    no_target_samples = [
        sample for sample in samples
        if sample["target_type"] == "no-target"
    ]
    single_target_samples = [
        sample for sample in samples
        if sample["target_type"] == "single-target"
    ]
    multi_target_samples = [
        sample for sample in samples
        if sample["target_type"] == "multi-target"
    ]

    no_target_positive_total = sum(
        sum(sample["candidate_labels"])
        for sample in no_target_samples
    )

    single_exact_one = sum(
        1 for sample in single_target_samples
        if sum(sample["candidate_labels"]) == 1
    )

    multi_exact_count = sum(
        1 for sample in multi_target_samples
        if sum(sample["candidate_labels"]) == sample["num_targets"]
    )

    all_exact_count = sum(
        1 for sample in samples
        if sum(sample["candidate_labels"]) == sample["num_targets"]
    )

    total_targets = sum(target_counts)
    total_matched_targets = sum(positive_counts)
    ann_id_recall = (
        total_matched_targets / total_targets
        if total_targets > 0
        else 0.0
    )

    with stats_file.open("w", encoding="utf-8") as f:
        f.write("Candidate Sample Statistics\n")
        f.write("===========================\n\n")

        f.write(f"Number of samples: {len(samples)}\n")
        f.write(f"No-target samples: {target_type_counter['no-target']}\n")
        f.write(f"Single-target samples: {target_type_counter['single-target']}\n")
        f.write(f"Multi-target samples: {target_type_counter['multi-target']}\n\n")

        f.write("Count-class distribution:\n")
        for cls in sorted(count_class_counter):
            f.write(f"  class {cls}: {count_class_counter[cls]}\n")
        f.write("\n")

        f.write("Candidate count statistics:\n")
        f.write(f"  average candidates per sample: {mean(candidate_counts):.4f}\n")
        f.write(f"  min candidates per sample: {min(candidate_counts)}\n")
        f.write(f"  max candidates per sample: {max(candidate_counts)}\n")
        f.write(f"  samples with zero candidates: {zero_candidate_samples}\n\n")

        f.write("Candidate-label sanity checks:\n")
        f.write(f"  no-target positive labels: {no_target_positive_total}\n")
        f.write(
            "  single-target samples with exactly one positive: "
            f"{single_exact_one} / {len(single_target_samples)}\n"
        )
        f.write(
            "  multi-target samples with positive count equal to target count: "
            f"{multi_exact_count} / {len(multi_target_samples)}\n"
        )
        f.write(
            "  all samples with positive count equal to target count: "
            f"{all_exact_count} / {len(samples)}\n"
        )
        f.write(f"  total target boxes: {total_targets}\n")
        f.write(f"  total matched positive candidates: {total_matched_targets}\n")
        f.write(f"  COCO-candidate ann-id recall: {ann_id_recall:.6f}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build COCO-instance candidate samples for gRefCOCO."
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Path to few-shot split JSON file.",
    )
    parser.add_argument(
        "--gref-file",
        type=Path,
        default=Path("data/grefcoco/annotations/grefs(unc).json"),
        help="Path to grefs(unc).json.",
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=Path("data/grefcoco/annotations/instances.json"),
        help="Path to instances.json.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        required=True,
        help="Output statistics TXT file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    split_samples = load_json(args.split_file)
    grefs = load_json(args.gref_file)
    instances = load_json(args.instances_file)

    ref_by_id, image_by_id, ann_by_id, anns_by_image = build_lookups(
        grefs,
        instances,
    )

    output_samples = []

    for idx, split_sample in enumerate(split_samples):
        candidate_sample = build_candidate_sample(
            split_sample=split_sample,
            ref_by_id=ref_by_id,
            image_by_id=image_by_id,
            ann_by_id=ann_by_id,
            anns_by_image=anns_by_image,
            sample_index=idx,
        )
        output_samples.append(candidate_sample)

    write_jsonl(output_samples, args.output_file)
    write_stats(output_samples, args.stats_file)

    print(f"Built {len(output_samples)} candidate samples.")
    print(f"Saved JSONL to: {args.output_file}")
    print(f"Saved stats to: {args.stats_file}")


if __name__ == "__main__":
    main()
