from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class ClipCandidateBaseline(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        candidate_feature_dim: int | None = None,
        text_feature_dim: int | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_count_classes: int = 4,
        pooling: str = "mean",
        hierarchical_cardinality: bool = False,
        membership_only: bool = False,
        use_box_coordinates: bool = True,
        use_explicit_similarity: bool = True,
    ):
        super().__init__()

        if pooling not in {"mean", "mean_max_stats"}:
            raise ValueError("pooling must be 'mean' or 'mean_max_stats'.")
        self.pooling = pooling
        self.hierarchical_cardinality = bool(hierarchical_cardinality)
        self.membership_only = bool(membership_only)
        self.use_box_coordinates = bool(use_box_coordinates)
        self.use_explicit_similarity = bool(use_explicit_similarity)
        if self.membership_only and self.hierarchical_cardinality:
            raise ValueError(
                "membership_only and hierarchical_cardinality are mutually exclusive."
            )

        self.candidate_feature_dim = int(candidate_feature_dim or feature_dim)
        self.text_feature_dim = int(text_feature_dim or feature_dim)
        # Backward-compatible alias for legacy CLIP-only code.
        self.feature_dim = self.candidate_feature_dim
        self.input_dim = (
            self.candidate_feature_dim
            + self.text_feature_dim
            + int(self.use_explicit_similarity)
            + 4 * int(self.use_box_coordinates)
        )

        self.candidate_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.membership_head = nn.Linear(hidden_dim, 1)

        pooled_dim = hidden_dim if pooling == "mean" else hidden_dim * 2 + 4
        if self.membership_only:
            self.presence_head = None
            self.positive_count_head = None
            self.count_head = None
        elif self.hierarchical_cardinality:
            self.presence_head = nn.Sequential(
                nn.Linear(pooled_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            self.positive_count_head = nn.Sequential(
                nn.Linear(pooled_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3),
            )
            self.count_head = None
        else:
            self.count_head = nn.Sequential(
                nn.Linear(pooled_dim, hidden_dim),
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
        if candidate_features.shape[1] != self.candidate_feature_dim:
            raise ValueError(
                "Candidate feature dimension mismatch: "
                f"expected {self.candidate_feature_dim}, got "
                f"{candidate_features.shape[1]}."
            )
        if text_feature.shape[0] != self.text_feature_dim:
            raise ValueError(
                "Text feature dimension mismatch: "
                f"expected {self.text_feature_dim}, got {text_feature.shape[0]}."
            )
        text_repeated = text_feature.unsqueeze(0).expand(n, -1)
        sim = candidate_text_similarity.view(n, 1)

        components = [candidate_features, text_repeated]
        if self.use_explicit_similarity:
            components.append(sim)
        if self.use_box_coordinates:
            components.append(candidate_boxes_norm)
        return torch.cat(components, dim=1)

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
            if self.pooling == "mean":
                pooled_features.append(h.mean(dim=0))
            else:
                probabilities = torch.sigmoid(logits)
                statistics = torch.stack(
                    [
                        probabilities.mean(),
                        probabilities.max(),
                        probabilities.std(unbiased=False),
                        torch.log1p(
                            torch.as_tensor(
                                float(probabilities.numel()), device=device
                            )
                        )
                        / torch.log(torch.as_tensor(101.0, device=device)),
                    ]
                )
                pooled_features.append(
                    torch.cat([h.mean(dim=0), h.max(dim=0).values, statistics])
                )

        pooled_features = torch.stack(pooled_features, dim=0)
        if self.membership_only:
            presence_logits = None
            positive_count_logits = None
            count_logits = torch.zeros(
                (batch_size, 4), dtype=pooled_features.dtype, device=device
            )
        elif self.hierarchical_cardinality:
            presence_logits = self.presence_head(pooled_features)
            positive_count_logits = self.positive_count_head(pooled_features)
            presence_log_probs = torch.log_softmax(presence_logits, dim=1)
            positive_log_probs = torch.log_softmax(positive_count_logits, dim=1)
            count_logits = torch.cat(
                [
                    presence_log_probs[:, :1],
                    presence_log_probs[:, 1:] + positive_log_probs,
                ],
                dim=1,
            )
        else:
            presence_logits = None
            positive_count_logits = None
            count_logits = self.count_head(pooled_features)

        return {
            "membership_logits": membership_logits,
            "count_logits": count_logits,
            "presence_logits": presence_logits,
            "positive_count_logits": positive_count_logits,
        }
