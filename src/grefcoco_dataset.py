from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class GRefCOCODataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        image_root: str | Path,
        annotation_by_id: dict[int, dict[str, Any]],
        transform: Callable | None = None,
    ):
        self.samples = samples
        self.image_root = Path(image_root)
        self.annotation_by_id = annotation_by_id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def coco_xywh_to_xyxy(box):
        x, y, width, height = box

        return [
            x,
            y,
            x + width,
            y + height,
        ]

    def get_boxes(self, sample):
        if sample["no_target"]:
            return torch.empty(
                (0, 4),
                dtype=torch.float32,
            )

        boxes = []

        for annotation_id in sample["ann_ids"]:
            annotation = self.annotation_by_id[annotation_id]

            boxes.append(
                self.coco_xywh_to_xyxy(
                    annotation["bbox"]
                )
            )

        return torch.tensor(
            boxes,
            dtype=torch.float32,
        ).reshape(-1, 4)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image_path = (
            self.image_root
            / sample["file_name"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        boxes = self.get_boxes(sample)

        normalized_boxes = boxes.clone()

        if len(normalized_boxes) > 0:
            normalized_boxes[:, [0, 2]] /= original_width
            normalized_boxes[:, [1, 3]] /= original_height

        processed_image = image

        if self.transform is not None:
            processed_image = self.transform(image)

        return {
            "image": processed_image,
            "image_id": sample["image_id"],
            "file_name": sample["file_name"],
            "expression": sample["expression"],
            "ref_id": sample["ref_id"],
            "sent_id": sample["sent_id"],
            "boxes": boxes,
            "normalized_boxes": normalized_boxes,
            "num_targets": boxes.shape[0],
            "target_type": sample["target_type"],
            "split": sample["split"],
            "original_size": torch.tensor(
                [original_width, original_height],
                dtype=torch.long,
            ),
        }


def grefcoco_collate_fn(batch):
    return {
        "images": [
            item["image"]
            for item in batch
        ],
        "image_ids": torch.tensor(
            [
                item["image_id"]
                for item in batch
            ],
            dtype=torch.long,
        ),
        "file_names": [
            item["file_name"]
            for item in batch
        ],
        "expressions": [
            item["expression"]
            for item in batch
        ],
        "ref_ids": torch.tensor(
            [
                item["ref_id"]
                for item in batch
            ],
            dtype=torch.long,
        ),
        "sent_ids": torch.tensor(
            [
                item["sent_id"]
                for item in batch
            ],
            dtype=torch.long,
        ),
        "boxes": [
            item["boxes"]
            for item in batch
        ],
        "normalized_boxes": [
            item["normalized_boxes"]
            for item in batch
        ],
        "num_targets": torch.tensor(
            [
                item["num_targets"]
                for item in batch
            ],
            dtype=torch.long,
        ),
        "target_types": [
            item["target_type"]
            for item in batch
        ],
        "splits": [
            item["split"]
            for item in batch
        ],
        "original_sizes": torch.stack(
            [
                item["original_size"]
                for item in batch
            ]
        ),
    }