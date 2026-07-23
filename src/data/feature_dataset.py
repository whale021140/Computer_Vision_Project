from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.evaluation.grec_metrics import greedy_one_to_one_matches, pairwise_iou


def _record_key(record: Dict[str, Any]) -> tuple[int, int]:
    metadata = record.get("metadata", {})
    return int(metadata["ref_id"]), int(metadata["sent_id"])


def _load_split_keys(split_file: str) -> list[tuple[int, int]]:
    with open(split_file, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    keys = [(int(sample["ref_id"]), int(sample["sent_id"])) for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Split contains duplicate expression keys: {split_file}")
    return keys


class ClipFeatureDataset(Dataset):
    def __init__(
        self,
        feature_file: str,
        max_samples: Optional[int] = None,
        split_file: str = "",
        label_policy: str = "cached",
        cache: Optional[Dict[str, Any]] = None,
    ):
        if label_policy not in {"cached", "one-to-one"}:
            raise ValueError("label_policy must be 'cached' or 'one-to-one'.")
        self.feature_file = feature_file
        self.split_file = split_file
        self.label_policy = label_policy
        # Train and validation splits can select different records from the
        # same multi-GB feature bank. Allow callers to share the immutable
        # loaded cache instead of deserializing and retaining it twice.
        self.cache = (
            torch.load(feature_file, map_location="cpu")
            if cache is None
            else cache
        )
        self.records = self.cache["records"]

        if split_file:
            requested_keys = _load_split_keys(split_file)
            records_by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
            for record in self.records:
                key = _record_key(record)
                if key in records_by_key:
                    raise ValueError(
                        f"Feature cache contains duplicate expression key {key}."
                    )
                records_by_key[key] = record
            missing = [key for key in requested_keys if key not in records_by_key]
            if missing:
                raise KeyError(
                    f"Feature cache is missing {len(missing)} expressions requested "
                    f"by {split_file}; first missing key: {missing[0]}"
                )
            # Follow split order so each representation sees the same sample order.
            self.records = [records_by_key[key] for key in requested_keys]

        if max_samples is not None:
            self.records = self.records[:max_samples]

        legacy_feature_dim = int(self.cache["feature_dim"])
        self.candidate_feature_dim = int(
            self.cache.get("candidate_feature_dim", legacy_feature_dim)
        )
        self.text_feature_dim = int(
            self.cache.get("text_feature_dim", legacy_feature_dim)
        )
        # Retained for older callers and checkpoints where image/text dimensions
        # are identical. New code should use the explicit dimensions above.
        self.feature_dim = self.candidate_feature_dim
        self.clip_model = self.cache.get("clip_model", "unknown")
        self.representation = self.cache.get(
            "representation",
            {"name": "clip", "model_ids": [self.clip_model]},
        )
        self.cache_format = self.cache.get("cache_format", "clip_legacy_v1")
        self.images = self.cache.get("images")
        self.similarity_spec = self.cache.get("similarity_spec")

    def _candidate_text_similarity(
        self,
        candidate_features: torch.Tensor,
        text_feature: torch.Tensor,
    ) -> torch.Tensor:
        if self.similarity_spec is None:
            if candidate_features.shape[1] != text_feature.shape[0]:
                raise ValueError(
                    "Candidate and text dimensions differ but the feature cache "
                    "does not define similarity_spec."
                )
            return candidate_features @ text_feature

        candidate_start, candidate_end = self.similarity_spec["candidate_slice"]
        text_start, text_end = self.similarity_spec["text_slice"]
        candidate_aligned = candidate_features[:, candidate_start:candidate_end]
        text_aligned = text_feature[text_start:text_end]
        if candidate_aligned.shape[1] != text_aligned.shape[0]:
            raise ValueError("similarity_spec selects incompatible dimensions.")
        return candidate_aligned @ text_aligned

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]

        if self.images is not None:
            image_id = str(int(r["image_id"]))
            if image_id not in self.images:
                raise KeyError(f"Missing shared image features for image_id={image_id}")
            image = self.images[image_id]
            text_feature = r["text_feature"].float()
            candidate_features = image["candidate_features"].float()
            candidate_boxes_norm = image["candidate_boxes_norm"].float()
            candidate_text_similarity = self._candidate_text_similarity(
                candidate_features,
                text_feature,
            )
        else:
            text_feature = r["text_feature"].float()
            candidate_features = r["candidate_features"].float()
            candidate_boxes_norm = r["candidate_boxes_norm"].float()
            candidate_text_similarity = r["candidate_text_similarity"].float()

        candidate_labels = r["candidate_labels"].float()
        if self.label_policy == "one-to-one":
            target_boxes_norm = r["target_boxes_norm"].float().reshape(-1, 4)
            candidate_labels = torch.zeros(
                candidate_boxes_norm.shape[0], dtype=torch.float32
            )
            overlaps = pairwise_iou(candidate_boxes_norm, target_boxes_norm)
            for candidate_index, _, _ in greedy_one_to_one_matches(
                overlaps, threshold=0.5
            ):
                candidate_labels[candidate_index] = 1.0

        return {
            "sample_id": r["sample_id"],
            "metadata": r["metadata"],
            "expression": r["expression"],
            "text_feature": text_feature,
            "candidate_features": candidate_features,
            "candidate_text_similarity": candidate_text_similarity,
            "candidate_boxes_norm": candidate_boxes_norm,
            "candidate_labels": candidate_labels,
            "count_class": torch.as_tensor(r["count_class"], dtype=torch.long),
            "target_boxes_xyxy": r["target_boxes_xyxy"].float(),
            "target_boxes_norm": r["target_boxes_norm"].float(),
        }


def clip_feature_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_ids": [b["sample_id"] for b in batch],
        "metadata": [b["metadata"] for b in batch],
        "expressions": [b["expression"] for b in batch],
        "text_features": [b["text_feature"] for b in batch],
        "candidate_features": [b["candidate_features"] for b in batch],
        "candidate_text_similarity": [b["candidate_text_similarity"] for b in batch],
        "candidate_boxes_norm": [b["candidate_boxes_norm"] for b in batch],
        "candidate_labels": [b["candidate_labels"] for b in batch],
        "count_class": torch.stack([b["count_class"] for b in batch], dim=0),
        "target_boxes_xyxy": [b["target_boxes_xyxy"] for b in batch],
        "target_boxes_norm": [b["target_boxes_norm"] for b in batch],
    }
