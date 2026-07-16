from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.ops import clip_boxes_to_image, nms, remove_small_boxes


DETECTOR_ID = "torchvision/fasterrcnn_resnet50_fpn_v2:coco_v1"


@dataclass(frozen=True)
class ProposalConfig:
    score_threshold: float = 0.05
    nms_threshold: float = 0.7
    max_proposals: int = 100
    detector_output_limit: int = 300
    min_box_size: float = 1.0
    inference_precision: str = "float32"
    detector_id: str = DETECTOR_ID

    def validate(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1.")
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1.")
        if self.max_proposals <= 0:
            raise ValueError("max_proposals must be positive.")
        if self.detector_output_limit < self.max_proposals:
            raise ValueError("detector_output_limit must be at least max_proposals.")
        if self.min_box_size < 0:
            raise ValueError("min_box_size must be non-negative.")
        if self.inference_precision not in {"float32", "float16"}:
            raise ValueError("inference_precision must be 'float32' or 'float16'.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_fasterrcnn(config: ProposalConfig, device: torch.device) -> torch.nn.Module:
    """Load the frozen detector with permissive internal postprocessing.

    Final class predictions are converted into a shared class-agnostic proposal
    pool by :func:`filter_detector_output`.
    """
    config.validate()
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        box_score_thresh=0.0,
        box_nms_thresh=1.0,
        box_detections_per_img=config.detector_output_limit,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def filter_detector_output(
    output: Dict[str, torch.Tensor],
    image_height: int,
    image_width: int,
    config: ProposalConfig,
) -> Dict[str, Any]:
    """Create a deterministic class-agnostic proposal pool from detections."""
    config.validate()
    boxes = output["boxes"].detach().cpu().float().reshape(-1, 4)
    scores = output["scores"].detach().cpu().float().reshape(-1)
    labels = output["labels"].detach().cpu().long().reshape(-1)
    if not (boxes.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("Detector boxes, scores, and labels must have equal lengths.")

    raw_count = int(boxes.shape[0])
    if raw_count:
        boxes = clip_boxes_to_image(boxes, (image_height, image_width))
        valid = remove_small_boxes(boxes, min_size=config.min_box_size)
        boxes = boxes[valid]
        scores = scores[valid]
        labels = labels[valid]

    fallback = "none"
    above_threshold = torch.nonzero(
        scores >= config.score_threshold,
        as_tuple=False,
    ).flatten()
    if above_threshold.numel() > 0:
        boxes = boxes[above_threshold]
        scores = scores[above_threshold]
        labels = labels[above_threshold]
    elif scores.numel() > 0:
        best = int(torch.argmax(scores).item())
        boxes = boxes[best : best + 1]
        scores = scores[best : best + 1]
        labels = labels[best : best + 1]
        fallback = "highest-score-below-threshold"
    else:
        boxes = torch.tensor(
            [[0.0, 0.0, float(image_width), float(image_height)]],
            dtype=torch.float32,
        )
        scores = torch.zeros(1, dtype=torch.float32)
        labels = torch.zeros(1, dtype=torch.long)
        fallback = "full-image"

    keep = nms(boxes, scores, config.nms_threshold)
    keep = keep[: config.max_proposals]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    order = torch.argsort(scores, descending=True, stable=True)
    boxes = boxes[order]
    scores = scores[order]
    labels = labels[order]

    return {
        "proposal_boxes_xyxy": boxes.tolist(),
        "proposal_scores": scores.tolist(),
        "detector_labels": labels.tolist(),
        "num_raw_detections": raw_count,
        "num_proposals": int(boxes.shape[0]),
        "fallback": fallback,
    }
