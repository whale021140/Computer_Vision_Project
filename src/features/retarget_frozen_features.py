"""Reuse frozen image-region features while encoding a new expression set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.features.extract_clip_features import (
    get_device,
    get_storage_dtype,
    load_shared_inputs,
    sha256_file,
    summarize_inputs,
)
from src.features.extract_frozen_features import CACHE_FORMAT
from src.features.frozen_encoders import build_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a new text feature bank by reusing matching image features."
    )
    parser.add_argument(
        "--representation",
        choices=["clip", "clip_dinov2", "siglip2"],
        required=True,
    )
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--reuse-image-feature-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--siglip2-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument(
        "--storage-dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--device", default="")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def validate_reusable_image(
    image_key: str,
    expected: dict[str, Any],
    existing: dict[str, Any],
) -> None:
    for field in (
        "image_id",
        "file_name",
        "width",
        "height",
        "candidate_source",
        "proposal_config",
    ):
        if existing.get(field) != expected.get(field):
            raise ValueError(f"Image {image_key} has incompatible {field}.")
    for field in (
        "candidate_boxes_norm",
        "candidate_scores",
        "candidate_detector_labels",
    ):
        if not torch.equal(existing[field].cpu(), expected[field].cpu()):
            raise ValueError(f"Image {image_key} has incompatible {field}.")


def main() -> None:
    args = parse_args()
    if args.text_batch_size <= 0:
        raise ValueError("--text-batch-size must be positive.")
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
    source_file = Path(args.reuse_image_feature_file)
    output_file = Path(args.output_file)
    stats_file = Path(args.stats_file)
    image_specs, expression_records = load_shared_inputs(candidate_file)
    input_stats = summarize_inputs(image_specs, expression_records)

    source = torch.load(source_file, map_location="cpu")
    source_format = source.get("cache_format")
    legacy_clip = source_format == "clip_shared_v2" and args.representation == "clip"
    if source_format != CACHE_FORMAT and not legacy_clip:
        raise ValueError(f"Unsupported source cache format in {source_file}.")
    source_representation = source.get("representation", {})
    source_name = "clip" if legacy_clip else source_representation.get("name")
    if source_name != args.representation:
        raise ValueError(
            f"Source representation {source_name!r} does not "
            f"match {args.representation!r}."
        )
    source_candidate_dim = int(
        source.get("candidate_feature_dim", source["feature_dim"])
    )
    source_text_dim = int(source.get("text_feature_dim", source["feature_dim"]))
    if source_candidate_dim != encoder.candidate_feature_dim:
        raise ValueError("Source candidate feature dimension does not match encoder.")
    if source_text_dim != encoder.text_feature_dim:
        raise ValueError("Source text feature dimension does not match encoder.")
    if not legacy_clip and source.get("similarity_spec") != encoder.similarity_spec:
        raise ValueError("Source similarity specification does not match encoder.")

    source_images = source.get("images")
    if not isinstance(source_images, dict):
        raise ValueError("Source cache does not contain shared image features.")
    images: dict[str, dict[str, Any]] = {}
    for image_key, spec in image_specs.items():
        if image_key not in source_images:
            raise KeyError(f"Source cache is missing image_id={image_key}.")
        existing = source_images[image_key]
        validate_reusable_image(image_key, spec, existing)
        images[image_key] = {
            key: value for key, value in spec.items() if key != "candidate_boxes_xyxy"
        }
        images[image_key]["candidate_features"] = existing[
            "candidate_features"
        ].to(storage_dtype)

    start = time.time()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp,
        ),
    ):
        for offset in tqdm(
            range(0, len(expression_records), args.text_batch_size),
            desc="Encoding expressions",
        ):
            records = expression_records[offset : offset + args.text_batch_size]
            features = encoder.encode_texts(
                [record["expression"] for record in records]
            )
            expected_shape = (len(records), encoder.text_feature_dim)
            if tuple(features.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected text feature shape: {tuple(features.shape)}."
                )
            for record, feature in zip(records, features):
                record["text_feature"] = feature.to(storage_dtype)
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
        "reused_image_feature_file": str(source_file),
        "reused_image_feature_file_sha256": sha256_file(source_file),
        "num_samples": len(expression_records),
        "num_images": len(images),
        "images": images,
        "records": expression_records,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
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
        "reused_image_feature_file": str(source_file),
        "reused_image_feature_file_sha256": cache[
            "reused_image_feature_file_sha256"
        ],
        "output_file": str(output_file),
        **input_stats,
        "reused_image_feature_images": len(images),
        "encoded_candidate_regions": 0,
        "encoded_expressions": len(expression_records),
        "elapsed_seconds": elapsed,
    }
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
