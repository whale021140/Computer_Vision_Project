"""Extract image-shared features for any Stage 4 frozen representation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from tqdm import tqdm

from src.features.extract_clip_features import (
    clamp_box_xyxy,
    get_device,
    get_storage_dtype,
    load_shared_inputs,
    sha256_file,
    summarize_inputs,
)
from src.features.frozen_encoders import FrozenRegionTextEncoder, build_encoder


CACHE_FORMAT = "frozen_representation_v1"


def shard_encoder_signature(encoder: FrozenRegionTextEncoder) -> Dict[str, Any]:
    """Identity fields that must match before a resumed shard is reusable."""
    metadata = dict(encoder.metadata())
    metadata.pop("encoder_parameters", None)
    return {
        **metadata,
        "similarity_spec": encoder.similarity_spec,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract image-shared frozen region/text representation features."
    )
    parser.add_argument(
        "--representation",
        choices=["clip", "clip_dinov2", "siglip2"],
        required=True,
    )
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument(
        "--siglip2-model", default="google/siglip2-base-patch16-224"
    )
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument(
        "--storage-dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated per-image feature shards from an interrupted run.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def _crop_regions(
    image: Image.Image,
    boxes_xyxy: Sequence[Sequence[float]],
) -> List[Image.Image]:
    width, height = image.size
    return [
        image.crop(clamp_box_xyxy(box, width, height)).convert("RGB")
        for box in boxes_xyxy
    ]


def _batched_image_features(
    encoder: FrozenRegionTextEncoder,
    images: Sequence[Image.Image],
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("region_batch_size must be positive.")
    batches = [
        encoder.encode_images(images[start : start + batch_size])
        for start in range(0, len(images), batch_size)
    ]
    if not batches:
        raise ValueError("Every image must contain at least one candidate region.")
    return torch.cat(batches, dim=0)


def extract_features(
    image_specs: Dict[str, Dict[str, Any]],
    expression_records: List[Dict[str, Any]],
    image_root: Path,
    encoder: FrozenRegionTextEncoder,
    region_batch_size: int,
    text_batch_size: int,
    storage_dtype: torch.dtype,
    shard_dir: Path | None = None,
    resume: bool = False,
) -> tuple[Dict[str, Dict[str, Any]], int, int]:
    if text_batch_size <= 0:
        raise ValueError("text_batch_size must be positive.")

    images: Dict[str, Dict[str, Any]] = {}
    encoded_regions = 0
    resumed_images = 0
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)
    encoder_signature = shard_encoder_signature(encoder)
    for image_key, spec in tqdm(image_specs.items(), desc="Encoding unique images"):
        expected_boxes = spec["candidate_boxes_norm"].float()
        shard_path = shard_dir / f"{image_key}.pt" if shard_dir else None
        features = None
        if resume and shard_path is not None and shard_path.exists():
            shard = torch.load(shard_path, map_location="cpu")
            if (
                shard.get("candidate_feature_dim")
                != encoder.candidate_feature_dim
                or not torch.equal(shard["candidate_boxes_norm"], expected_boxes)
                or shard.get("encoder_signature") != encoder_signature
            ):
                raise ValueError(
                    f"Resume shard does not match image {image_key}; remove "
                    f"{shard_path} or restart without --resume."
                )
            features = shard["candidate_features"]
            resumed_images += 1

        if features is None:
            image_path = image_root / spec["file_name"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            crops = _crop_regions(image, spec["candidate_boxes_xyxy"])
            features = _batched_image_features(
                encoder,
                crops,
                batch_size=region_batch_size,
            ).to(storage_dtype)
            if shard_path is not None:
                temporary_path = shard_path.with_suffix(".tmp")
                torch.save(
                    {
                        "candidate_feature_dim": encoder.candidate_feature_dim,
                        "encoder_signature": encoder_signature,
                        "candidate_boxes_norm": expected_boxes,
                        "candidate_features": features,
                    },
                    temporary_path,
                )
                temporary_path.replace(shard_path)

        expected_shape = (
            int(expected_boxes.shape[0]),
            encoder.candidate_feature_dim,
        )
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected candidate feature shape for image {image_key}: "
                f"{tuple(features.shape)} vs {expected_shape}."
            )
        images[image_key] = {
            key: value
            for key, value in spec.items()
            if key != "candidate_boxes_xyxy"
        }
        images[image_key]["candidate_features"] = features.to(storage_dtype)
        encoded_regions += expected_shape[0]

    for start in tqdm(
        range(0, len(expression_records), text_batch_size),
        desc="Encoding expressions",
    ):
        records = expression_records[start : start + text_batch_size]
        features = encoder.encode_texts([record["expression"] for record in records])
        if features.shape != (len(records), encoder.text_feature_dim):
            raise RuntimeError(f"Unexpected text feature shape: {tuple(features.shape)}.")
        for record, feature in zip(records, features):
            record["text_feature"] = feature.to(storage_dtype)
    return images, encoded_regions, resumed_images


def main() -> None:
    args = parse_args()
    if args.amp and args.device and not args.device.startswith("cuda"):
        raise ValueError("--amp requires a CUDA device.")
    device = get_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device.")
    storage_dtype = get_storage_dtype(args.storage_dtype)
    encoder = build_encoder(
        representation=args.representation,
        device=device,
        clip_model_id=args.clip_model,
        dinov2_model_id=args.dinov2_model,
        siglip2_model_id=args.siglip2_model,
    )

    candidate_file = Path(args.candidate_file)
    output_file = Path(args.output_file)
    stats_file = Path(args.stats_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    image_specs, expression_records = load_shared_inputs(
        candidate_file,
        max_samples=args.max_samples,
    )
    input_stats = summarize_inputs(image_specs, expression_records)
    start = time.time()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp,
        ),
    ):
        images, encoded_regions, resumed_images = extract_features(
            image_specs=image_specs,
            expression_records=expression_records,
            image_root=Path(args.image_root),
            encoder=encoder,
            region_batch_size=args.region_batch_size,
            text_batch_size=args.text_batch_size,
            storage_dtype=storage_dtype,
            shard_dir=output_file.with_suffix(output_file.suffix + ".parts"),
            resume=args.resume,
        )
    elapsed = time.time() - start
    representation = encoder.metadata()
    cache = {
        "cache_format": CACHE_FORMAT,
        "representation": representation,
        "clip_model": args.clip_model if args.representation != "siglip2" else "none",
        "feature_dim": encoder.candidate_feature_dim,
        "candidate_feature_dim": encoder.candidate_feature_dim,
        "text_feature_dim": encoder.text_feature_dim,
        "similarity_spec": encoder.similarity_spec,
        "storage_dtype": args.storage_dtype,
        "candidate_file": str(candidate_file),
        "candidate_file_sha256": sha256_file(candidate_file),
        "image_root": args.image_root,
        "num_samples": len(expression_records),
        "num_images": len(images),
        "images": images,
        "records": expression_records,
    }
    temporary_output = output_file.with_suffix(output_file.suffix + ".tmp")
    torch.save(cache, temporary_output)
    temporary_output.replace(output_file)
    stats = {
        "cache_format": CACHE_FORMAT,
        "representation": representation,
        "device": str(device),
        "amp": args.amp,
        "storage_dtype": args.storage_dtype,
        "candidate_file": str(candidate_file),
        "candidate_file_sha256": cache["candidate_file_sha256"],
        "output_file": str(output_file),
        **input_stats,
        "encoded_candidate_regions": encoded_regions,
        "resumed_images": resumed_images,
        "elapsed_seconds": elapsed,
    }
    stats_file.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
