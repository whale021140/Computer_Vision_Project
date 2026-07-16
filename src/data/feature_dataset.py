from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class ClipFeatureDataset(Dataset):
    def __init__(self, feature_file: str, max_samples: Optional[int] = None):
        self.feature_file = feature_file
        self.cache = torch.load(feature_file, map_location="cpu")
        self.records = self.cache["records"]

        if max_samples is not None:
            self.records = self.records[:max_samples]

        self.feature_dim = int(self.cache["feature_dim"])
        self.clip_model = self.cache.get("clip_model", "unknown")
        self.cache_format = self.cache.get("cache_format", "clip_legacy_v1")
        self.images = self.cache.get("images")

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
            candidate_text_similarity = candidate_features @ text_feature
        else:
            text_feature = r["text_feature"].float()
            candidate_features = r["candidate_features"].float()
            candidate_boxes_norm = r["candidate_boxes_norm"].float()
            candidate_text_similarity = r["candidate_text_similarity"].float()

        return {
            "sample_id": r["sample_id"],
            "metadata": r["metadata"],
            "expression": r["expression"],
            "text_feature": text_feature,
            "candidate_features": candidate_features,
            "candidate_text_similarity": candidate_text_similarity,
            "candidate_boxes_norm": candidate_boxes_norm,
            "candidate_labels": r["candidate_labels"].float(),
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
