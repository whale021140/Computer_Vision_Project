from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision import __version__ as torchvision_version
from tqdm import tqdm

from src.proposals.fasterrcnn import (
    ProposalConfig,
    filter_detector_output,
    load_fasterrcnn,
)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_image_ids(split_files: Sequence[str | Path]) -> List[int]:
    image_ids = {
        int(sample["image_id"])
        for split_file in split_files
        for sample in load_json(split_file)
    }
    return sorted(image_ids)


def build_image_lookup(instances: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(image["id"]): image for image in instances["images"]}


class ProposalImageDataset(Dataset):
    def __init__(
        self,
        image_ids: Sequence[int],
        image_lookup: Dict[int, Dict[str, Any]],
        image_root: str | Path,
    ) -> None:
        self.image_ids = list(image_ids)
        self.image_lookup = image_lookup
        self.image_root = Path(image_root)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id = self.image_ids[index]
        image_info = self.image_lookup[image_id]
        image_path = self.image_root / image_info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            tensor = pil_to_tensor(rgb).float().div_(255.0)
            width, height = rgb.size
        return {
            "image_id": image_id,
            "file_name": image_info["file_name"],
            "width": int(width),
            "height": int(height),
            "image": tensor,
        }


def proposal_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def load_existing_records(
    output_file: Path,
    config: ProposalConfig,
    resume: bool,
) -> tuple[List[Dict[str, Any]], set[int]]:
    if not output_file.exists() or not resume:
        return [], set()

    records: List[Dict[str, Any]] = []
    seen: set[int] = set()
    with output_file.open("rb+") as handle:
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if handle.read().strip():
                    raise ValueError(
                        "Malformed proposal-cache record before the final line."
                    ) from error
                handle.seek(line_start)
                handle.truncate()
                break
            if record.get("proposal_config") != config.to_dict():
                raise ValueError(
                    "Existing proposal cache uses a different configuration; "
                    "choose a new output file or disable --resume."
                )
            image_id = int(record["image_id"])
            if image_id in seen:
                raise ValueError(f"Duplicate image_id={image_id} in proposal cache.")
            seen.add(image_id)
            records.append(record)
    return records, seen


def summarize_records(
    records: Sequence[Dict[str, Any]],
    requested_images: int,
    elapsed_seconds: float,
    config: ProposalConfig,
) -> Dict[str, Any]:
    counts = [int(record["num_proposals"]) for record in records]
    fallback_counts: Dict[str, int] = {}
    for record in records:
        fallback = str(record["fallback"])
        fallback_counts[fallback] = fallback_counts.get(fallback, 0) + 1
    return {
        "proposal_config": config.to_dict(),
        "requested_images": requested_images,
        "cached_images": len(records),
        "average_proposals_per_image": mean(counts) if counts else 0.0,
        "min_proposals_per_image": min(counts) if counts else 0,
        "max_proposals_per_image": max(counts) if counts else 0,
        "fallback_counts": fallback_counts,
        "elapsed_seconds_this_run": elapsed_seconds,
    }


def get_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate shared frozen Faster R-CNN candidate boxes per image."
    )
    parser.add_argument("--split-files", nargs="+", required=True)
    parser.add_argument(
        "--instances-json",
        default="data/grefcoco/annotations/instances.json",
    )
    parser.add_argument("--image-root", default="data/coco/train2014")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--max-proposals", type=int, default=100)
    parser.add_argument("--detector-output-limit", type=int, default=300)
    parser.add_argument("--min-box-size", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA float16 autocast; recorded in the proposal cache config.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = ProposalConfig(
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_proposals=args.max_proposals,
        detector_output_limit=args.detector_output_limit,
        min_box_size=args.min_box_size,
        inference_precision="float16" if args.amp else "float32",
    )
    config.validate()

    instances = load_json(args.instances_json)
    image_lookup = build_image_lookup(instances)
    requested_image_ids = collect_image_ids(args.split_files)
    if args.max_images is not None:
        requested_image_ids = requested_image_ids[: args.max_images]

    missing_metadata = [image_id for image_id in requested_image_ids if image_id not in image_lookup]
    if missing_metadata:
        raise KeyError(f"Missing image metadata for image IDs: {missing_metadata[:10]}")

    output_file = Path(args.output_file)
    stats_file = Path(args.stats_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    existing_records, completed_ids = load_existing_records(
        output_file,
        config,
        resume=args.resume,
    )
    pending_ids = [image_id for image_id in requested_image_ids if image_id not in completed_ids]

    dataset = ProposalImageDataset(pending_ids, image_lookup, args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=proposal_collate_fn,
    )
    device = get_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device.")
    print(f"Device: {device}")
    print(f"Requested unique images: {len(requested_image_ids)}")
    print(f"Already cached: {len(completed_ids)}")
    print(f"Pending: {len(pending_ids)}")
    model = load_fasterrcnn(config, device)

    mode = "a" if args.resume and output_file.exists() else "w"
    new_records: List[Dict[str, Any]] = []
    start = time.time()
    with (
        output_file.open(mode, encoding="utf-8") as handle,
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp,
        ),
    ):
        for batch in tqdm(loader, desc="Generating proposals"):
            images = [sample["image"].to(device) for sample in batch]
            outputs = model(images)
            for sample, output in zip(batch, outputs):
                filtered = filter_detector_output(
                    output,
                    image_height=sample["height"],
                    image_width=sample["width"],
                    config=config,
                )
                record = {
                    "image_id": sample["image_id"],
                    "file_name": sample["file_name"],
                    "width": sample["width"],
                    "height": sample["height"],
                    "proposal_config": config.to_dict(),
                    **filtered,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                new_records.append(record)

    elapsed = time.time() - start
    all_records = existing_records + new_records
    stats = summarize_records(
        all_records,
        requested_images=len(requested_image_ids),
        elapsed_seconds=elapsed,
        config=config,
    )
    stats.update(
        {
            "device": str(device),
            "split_files": [str(path) for path in args.split_files],
            "split_sha256": {
                str(path): sha256_file(path) for path in args.split_files
            },
            "output_file": str(output_file),
            "seed": args.seed,
            "environment": {
                "torch": torch.__version__,
                "torchvision": torchvision_version,
            },
            "command": shlex.join(sys.argv),
        }
    )
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
