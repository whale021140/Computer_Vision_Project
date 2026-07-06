import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen CLIP text and candidate-region features."
    )
    parser.add_argument(
        "--candidate-file",
        type=str,
        required=True,
        help="Path to candidate JSONL file.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Directory containing COCO train2014 images.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output .pt feature cache.",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        required=True,
        help="Path to text stats file.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        help="OpenAI CLIP model name. Default: ViT-B/32.",
    )
    parser.add_argument(
        "--region-batch-size",
        type=int,
        default=64,
        help="Number of cropped regions encoded per CLIP forward pass.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debug limit.",
    )
    return parser.parse_args()


def load_jsonl(path: Path, max_samples: int | None = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_samples is not None and line_idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def clamp_box_xyxy(
    box: List[float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def encode_regions(
    image: Image.Image,
    boxes_xyxy: List[List[float]],
    preprocess,
    model,
    device: torch.device,
    region_batch_size: int,
) -> torch.Tensor:
    crops = []
    width, height = image.size

    for box in boxes_xyxy:
        x1, y1, x2, y2 = clamp_box_xyxy(box, width, height)
        crop = image.crop((x1, y1, x2, y2)).convert("RGB")
        crops.append(preprocess(crop))

    if len(crops) == 0:
        return torch.empty(0, model.visual.output_dim, dtype=torch.float32)

    features = []
    for start in range(0, len(crops), region_batch_size):
        batch = torch.stack(crops[start : start + region_batch_size]).to(device)
        region_features = model.encode_image(batch)
        region_features = region_features.float()
        region_features = region_features / region_features.norm(dim=-1, keepdim=True)
        features.append(region_features.cpu())

    return torch.cat(features, dim=0)


def encode_text(
    expression: str,
    model,
    clip_module,
    device: torch.device,
) -> torch.Tensor:
    tokens = clip_module.tokenize([expression], truncate=True).to(device)
    text_feature = model.encode_text(tokens)
    text_feature = text_feature.float()
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    return text_feature.squeeze(0).cpu()


def main() -> None:
    args = parse_args()

    candidate_file = Path(args.candidate_file)
    image_root = Path(args.image_root)
    output_file = Path(args.output_file)
    stats_file = Path(args.stats_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    import clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()

    records = load_jsonl(candidate_file, args.max_samples)

    feature_records: List[Dict[str, Any]] = []

    failed_images = 0
    failed_samples = 0
    total_candidates = 0
    total_positive_labels = 0
    no_target_count = 0
    single_target_count = 0
    multi_target_count = 0

    start_time = time.time()

    with torch.no_grad():
        for record in tqdm(records, desc="Extracting CLIP features"):
            file_name = record["file_name"]
            image_path = image_root / file_name

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                failed_images += 1
                failed_samples += 1
                continue

            try:
                expression = record["expression"]
                boxes_xyxy = record["candidate_boxes_xyxy"]

                text_feature = encode_text(
                    expression=expression,
                    model=model,
                    clip_module=clip,
                    device=device,
                )

                candidate_features = encode_regions(
                    image=image,
                    boxes_xyxy=boxes_xyxy,
                    preprocess=preprocess,
                    model=model,
                    device=device,
                    region_batch_size=args.region_batch_size,
                )

                if candidate_features.shape[0] != len(boxes_xyxy):
                    raise RuntimeError(
                        f"Feature count mismatch: "
                        f"{candidate_features.shape[0]} vs {len(boxes_xyxy)}"
                    )

                if candidate_features.numel() == 0:
                    similarity = torch.empty(0, dtype=torch.float32)
                else:
                    similarity = candidate_features @ text_feature

                candidate_labels = torch.tensor(
                    record["candidate_labels"],
                    dtype=torch.float32,
                )
                candidate_boxes_norm = torch.tensor(
                    record["candidate_boxes_norm"],
                    dtype=torch.float32,
                )
                target_boxes_xyxy = torch.tensor(
                    record["target_boxes_xyxy"],
                    dtype=torch.float32,
                )
                target_boxes_norm = torch.tensor(
                    record["target_boxes_norm"],
                    dtype=torch.float32,
                )
                count_class = torch.tensor(
                    record["count_class"],
                    dtype=torch.long,
                )

                target_type = record["target_type"]
                if target_type == "no-target":
                    no_target_count += 1
                elif target_type == "single-target":
                    single_target_count += 1
                elif target_type == "multi-target":
                    multi_target_count += 1

                total_candidates += len(boxes_xyxy)
                total_positive_labels += int(candidate_labels.sum().item())

                metadata = {
                    "sample_id": record.get("sample_id"),
                    "ref_id": record.get("ref_id"),
                    "sent_id": record.get("sent_id"),
                    "image_id": record.get("image_id"),
                    "file_name": record.get("file_name"),
                    "width": record.get("width"),
                    "height": record.get("height"),
                    "target_type": record.get("target_type"),
                    "num_targets": record.get("num_targets"),
                    "num_candidates": record.get("num_candidates"),
                    "target_ann_ids": record.get("target_ann_ids"),
                    "candidate_ann_ids": record.get("candidate_ann_ids"),
                }

                feature_records.append(
                    {
                        "sample_id": record.get("sample_id"),
                        "metadata": metadata,
                        "expression": expression,
                        "text_feature": text_feature,
                        "candidate_features": candidate_features,
                        "candidate_text_similarity": similarity.cpu(),
                        "candidate_boxes_norm": candidate_boxes_norm,
                        "candidate_labels": candidate_labels,
                        "count_class": count_class,
                        "target_boxes_xyxy": target_boxes_xyxy,
                        "target_boxes_norm": target_boxes_norm,
                    }
                )

            except Exception as exc:
                failed_samples += 1
                print(f"[WARN] failed sample {record.get('sample_id')}: {exc}")
                continue

    elapsed = time.time() - start_time

    feature_dim = None
    if len(feature_records) > 0:
        feature_dim = int(feature_records[0]["text_feature"].shape[-1])

    cache = {
        "clip_model": args.clip_model,
        "feature_dim": feature_dim,
        "candidate_file": str(candidate_file),
        "image_root": str(image_root),
        "num_samples": len(feature_records),
        "records": feature_records,
    }

    torch.save(cache, output_file)

    avg_candidates = (
        total_candidates / len(feature_records)
        if len(feature_records) > 0
        else 0.0
    )

    stats_lines = [
        f"CLIP model: {args.clip_model}",
        f"Device: {device}",
        f"Feature dimension: {feature_dim}",
        f"Input candidate file: {candidate_file}",
        f"Output feature file: {output_file}",
        f"Records read: {len(records)}",
        f"Samples processed: {len(feature_records)}",
        f"Failed image loads: {failed_images}",
        f"Failed samples: {failed_samples}",
        f"No-target samples: {no_target_count}",
        f"Single-target samples: {single_target_count}",
        f"Multi-target samples: {multi_target_count}",
        f"Total candidate regions encoded: {total_candidates}",
        f"Total positive candidate labels: {total_positive_labels}",
        f"Average candidates per processed sample: {avg_candidates:.4f}",
        f"Total runtime seconds: {elapsed:.2f}",
    ]

    stats_file.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    print("\n".join(stats_lines))


if __name__ == "__main__":
    main()
