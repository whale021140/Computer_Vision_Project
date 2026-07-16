from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class ClipCandidateBaseline(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_count_classes: int = 4,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.input_dim = feature_dim * 2 + 1 + 4

        self.candidate_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.membership_head = nn.Linear(hidden_dim, 1)

        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_count_classes),
        )

    def _build_candidate_input(
        self,
        text_feature: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_text_similarity: torch.Tensor,
        candidate_boxes_norm: torch.Tensor,
    ) -> torch.Tensor:
        n = candidate_features.shape[0]
        text_repeated = text_feature.unsqueeze(0).expand(n, -1)
        sim = candidate_text_similarity.view(n, 1)

        return torch.cat(
            [
                candidate_features,
                text_repeated,
                sim,
                candidate_boxes_norm,
            ],
            dim=1,
        )

    def forward(self, batch: Dict[str, List[torch.Tensor]]) -> Dict[str, object]:
        device = next(self.parameters()).device

        membership_logits = []
        pooled_features = []

        batch_size = len(batch["text_features"])

        for i in range(batch_size):
            text_feature = batch["text_features"][i].to(device)
            candidate_features = batch["candidate_features"][i].to(device)
            candidate_text_similarity = batch["candidate_text_similarity"][i].to(device)
            candidate_boxes_norm = batch["candidate_boxes_norm"][i].to(device)

            if candidate_features.shape[0] == 0:
                raise ValueError(
                    f"Sample {i} has no candidate features; every sample must "
                    "contain at least one candidate box."
                )

            x = self._build_candidate_input(
                text_feature=text_feature,
                candidate_features=candidate_features,
                candidate_text_similarity=candidate_text_similarity,
                candidate_boxes_norm=candidate_boxes_norm,
            )

            h = self.candidate_mlp(x)
            logits = self.membership_head(h).squeeze(-1)

            membership_logits.append(logits)
            pooled_features.append(h.mean(dim=0))

        pooled_features = torch.stack(pooled_features, dim=0)
        count_logits = self.count_head(pooled_features)

        return {
            "membership_logits": membership_logits,
            "count_logits": count_logits,
        }
