from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm


CACHE_FORMAT = "clip_shared_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen CLIP features while storing candidate-region "
            "features once per unique image."
        )
    )
    parser.add_argument("--candidate-file", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--stats-file", type=str, required=True)
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--region-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument(
        "--storage-dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_storage_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported storage dtype: {name!r}")


def box_tensor(boxes: Any) -> torch.Tensor:
    if boxes is None or len(boxes) == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def clamp_box_xyxy(
    box: Sequence[float],
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
    boxes_xyxy: Sequence[Sequence[float]],
    preprocess: Any,
    model: torch.nn.Module,
    device: torch.device,
    region_batch_size: int,
) -> torch.Tensor:
    if region_batch_size <= 0:
        raise ValueError("region_batch_size must be positive.")
    width, height = image.size
    crops = [
        preprocess(
            image.crop(clamp_box_xyxy(box, width, height)).convert("RGB")
        )
        for box in boxes_xyxy
    ]
    if not crops:
        raise ValueError("Every shared image must contain at least one candidate.")

    features = []
    for start in range(0, len(crops), region_batch_size):
        batch = torch.stack(crops[start : start + region_batch_size]).to(device)
        region_features = model.encode_image(batch).float()
        region_features = torch.nn.functional.normalize(region_features, dim=-1)
        features.append(region_features.cpu())
    return torch.cat(features, dim=0)


def encode_text_batch(
    expressions: Sequence[str],
    model: torch.nn.Module,
    clip_module: Any,
    device: torch.device,
) -> torch.Tensor:
    tokens = clip_module.tokenize(list(expressions), truncate=True).to(device)
    text_features = model.encode_text(tokens).float()
    return torch.nn.functional.normalize(text_features, dim=-1).cpu()


def _image_spec(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_id": int(record["image_id"]),
        "file_name": str(record["file_name"]),
        "width": int(record["width"]),
        "height": int(record["height"]),
        "candidate_source": record.get("candidate_source", "unknown"),
        "proposal_config": record.get("proposal_config"),
        "candidate_boxes_xyxy": record["candidate_boxes_xyxy"],
        "candidate_boxes_norm": box_tensor(record["candidate_boxes_norm"]),
        "candidate_scores": torch.as_tensor(
            record.get("candidate_scores", []),
            dtype=torch.float32,
        ),
        "candidate_detector_labels": torch.as_tensor(
            record.get("candidate_detector_labels", []),
            dtype=torch.long,
        ),
    }


def _validate_shared_image(
    existing: Dict[str, Any],
    candidate: Dict[str, Any],
) -> None:
    fields = (
        "file_name",
        "width",
        "height",
        "candidate_source",
        "proposal_config",
        "candidate_boxes_xyxy",
    )
    for field in fields:
        if existing[field] != candidate[field]:
            raise ValueError(
                f"Image {existing['image_id']} has inconsistent {field} "
                "across expression records."
            )


def _expression_record(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "sample_id": record.get("sample_id"),
        "ref_id": record.get("ref_id"),
        "sent_id": record.get("sent_id"),
        "image_id": int(record["image_id"]),
        "file_name": record.get("file_name"),
        "width": int(record["width"]),
        "height": int(record["height"]),
        "target_type": record.get("target_type"),
        "num_targets": int(record.get("num_targets", 0)),
        "num_candidates": int(record["num_candidates"]),
        "target_ann_ids": record.get("target_ann_ids"),
        "candidate_source": record.get("candidate_source", "unknown"),
    }
    return {
        "sample_id": record.get("sample_id"),
        "image_id": int(record["image_id"]),
        "metadata": metadata,
        "expression": str(record["expression"]),
        "candidate_labels": torch.as_tensor(
            record["candidate_labels"],
            dtype=torch.float32,
        ),
        "count_class": torch.as_tensor(record["count_class"], dtype=torch.long),
        "target_boxes_xyxy": box_tensor(record.get("target_boxes_xyxy", [])),
        "target_boxes_norm": box_tensor(record.get("target_boxes_norm", [])),
        "target_best_proposal_ious": torch.as_tensor(
            record.get("target_best_proposal_ious", []),
            dtype=torch.float32,
        ),
    }


def load_shared_inputs(
    path: str | Path,
    max_samples: int | None = None,
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    images: Dict[str, Dict[str, Any]] = {}
    expressions: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if max_samples is not None and len(expressions) >= max_samples:
                break
            record = json.loads(line)
            image = _image_spec(record)
            image_key = str(image["image_id"])
            if image_key in images:
                _validate_shared_image(images[image_key], image)
            else:
                images[image_key] = image
            expression = _expression_record(record)
            if expression["candidate_labels"].numel() != len(
                image["candidate_boxes_xyxy"]
            ):
                raise ValueError(
                    f"Candidate label length mismatch for sample "
                    f"{expression['sample_id']}."
                )
            expressions.append(expression)
    if not expressions:
        raise ValueError(f"No candidate records loaded from {path}")
    return images, expressions


def extract_shared_features(
    image_specs: Dict[str, Dict[str, Any]],
    expression_records: List[Dict[str, Any]],
    image_root: Path,
    model: torch.nn.Module,
    preprocess: Any,
    clip_module: Any,
    device: torch.device,
    region_batch_size: int,
    text_batch_size: int,
    storage_dtype: torch.dtype,
) -> tuple[Dict[str, Dict[str, Any]], int]:
    if text_batch_size <= 0:
        raise ValueError("text_batch_size must be positive.")

    images: Dict[str, Dict[str, Any]] = {}
    total_regions = 0
    for image_key, spec in tqdm(
        image_specs.items(),
        desc="Encoding unique images",
    ):
        image_path = image_root / spec["file_name"]
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Failed to load image: {image_path}") from error

        features = encode_regions(
            image=image,
            boxes_xyxy=spec["candidate_boxes_xyxy"],
            preprocess=preprocess,
            model=model,
            device=device,
            region_batch_size=region_batch_size,
        )
        if features.shape[0] != len(spec["candidate_boxes_xyxy"]):
            raise RuntimeError(
                f"Feature count mismatch for image {image_key}: "
                f"{features.shape[0]} vs {len(spec['candidate_boxes_xyxy'])}"
            )
        total_regions += int(features.shape[0])
        images[image_key] = {
            key: value
            for key, value in spec.items()
            if key != "candidate_boxes_xyxy"
        }
        images[image_key]["candidate_features"] = features.to(storage_dtype)

    for start in tqdm(
        range(0, len(expression_records), text_batch_size),
        desc="Encoding expressions",
    ):
        batch_records = expression_records[start : start + text_batch_size]
        text_features = encode_text_batch(
            [record["expression"] for record in batch_records],
            model=model,
            clip_module=clip_module,
            device=device,
        ).to(storage_dtype)
        for record, text_feature in zip(batch_records, text_features):
            record["text_feature"] = text_feature
    return images, total_regions


def summarize_inputs(
    images: Dict[str, Dict[str, Any]],
    expressions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    type_counts = Counter(
        str(record["metadata"]["target_type"]) for record in expressions
    )
    candidate_references = sum(
        int(record["candidate_labels"].numel()) for record in expressions
    )
    unique_candidates = sum(
        len(image["candidate_boxes_xyxy"]) for image in images.values()
    )
    return {
        "num_samples": len(expressions),
        "num_unique_images": len(images),
        "candidate_references": candidate_references,
        "unique_candidate_regions": unique_candidates,
        "region_reuse_factor": (
            candidate_references / unique_candidates if unique_candidates else 0.0
        ),
        "target_type_counts": dict(sorted(type_counts.items())),
    }


def main() -> None:
    args = parse_args()
    candidate_file = Path(args.candidate_file)
    image_root = Path(args.image_root)
    output_file = Path(args.output_file)
    stats_file = Path(args.stats_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    import clip

    device = get_device(args.device)
    storage_dtype = get_storage_dtype(args.storage_dtype)
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    image_specs, expression_records = load_shared_inputs(
        candidate_file,
        max_samples=args.max_samples,
    )
    input_stats = summarize_inputs(image_specs, expression_records)
    start_time = time.time()
    with torch.inference_mode():
        images, encoded_regions = extract_shared_features(
            image_specs=image_specs,
            expression_records=expression_records,
            image_root=image_root,
            model=model,
            preprocess=preprocess,
            clip_module=clip,
            device=device,
            region_batch_size=args.region_batch_size,
            text_batch_size=args.text_batch_size,
            storage_dtype=storage_dtype,
        )
    elapsed = time.time() - start_time
    if encoded_regions != input_stats["unique_candidate_regions"]:
        raise RuntimeError("Encoded-region total disagrees with input summary.")

    feature_dim = int(next(iter(images.values()))["candidate_features"].shape[1])
    cache = {
        "cache_format": CACHE_FORMAT,
        "clip_model": args.clip_model,
        "feature_dim": feature_dim,
        "storage_dtype": args.storage_dtype,
        "candidate_file": str(candidate_file),
        "candidate_file_sha256": sha256_file(candidate_file),
        "image_root": str(image_root),
        "num_samples": len(expression_records),
        "num_images": len(images),
        "images": images,
        "records": expression_records,
    }
    torch.save(cache, output_file)

    stats = {
        "cache_format": CACHE_FORMAT,
        "clip_model": args.clip_model,
        "device": str(device),
        "feature_dim": feature_dim,
        "storage_dtype": args.storage_dtype,
        "candidate_file": str(candidate_file),
        "candidate_file_sha256": cache["candidate_file_sha256"],
        "output_file": str(output_file),
        **input_stats,
        "encoded_candidate_regions": encoded_regions,
        "elapsed_seconds": elapsed,
    }
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
