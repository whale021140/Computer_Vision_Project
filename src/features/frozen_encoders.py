"""Frozen region/text encoder adapters used by the shared feature cache."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F
from PIL import Image


def prepare_transformers_image_backend() -> None:
    """Let Transformers use PIL when an optional OpenCV install is broken."""
    import transformers.utils.import_utils as import_utils

    if not import_utils.is_cv2_available():
        return
    try:
        import cv2  # noqa: F401
    except (ImportError, OSError):
        # Transformers checks module presence rather than importability. Some
        # Conda environments contain a cv2 extension linked against a different
        # libjpeg/libtiff build; OpenCV is optional for all encoders used here.
        import_utils._cv2_available = False


def parameter_counts(module: torch.nn.Module) -> Dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in module.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
    }


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class FrozenRegionTextEncoder(ABC):
    name: str
    candidate_feature_dim: int
    text_feature_dim: int
    similarity_spec: Dict[str, list[int]]

    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Return normalized/aligned features for one image batch on CPU."""

    @abstractmethod
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Return normalized/aligned features for one text batch on CPU."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return model IDs, dimensions, normalization, and parameter counts."""


class ClipEncoder(FrozenRegionTextEncoder):
    name = "clip"

    def __init__(self, device: torch.device, model_id: str = "ViT-B/32"):
        super().__init__(device)
        import clip

        self.clip_module = clip
        self.model_id = model_id
        self.model, self.preprocess = clip.load(model_id, device=device)
        freeze_module(self.model)
        self.candidate_feature_dim = int(self.model.visual.output_dim)
        self.text_feature_dim = self.candidate_feature_dim
        self.similarity_spec = {
            "candidate_slice": [0, self.candidate_feature_dim],
            "text_slice": [0, self.text_feature_dim],
        }

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        pixels = torch.stack([self.preprocess(image) for image in images]).to(
            self.device
        )
        features = self.model.encode_image(pixels).float()
        return F.normalize(features, dim=-1).cpu()

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.clip_module.tokenize(list(texts), truncate=True).to(self.device)
        features = self.model.encode_text(tokens).float()
        return F.normalize(features, dim=-1).cpu()

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_ids": {"clip": self.model_id},
            "candidate_feature_dim": self.candidate_feature_dim,
            "text_feature_dim": self.text_feature_dim,
            "component_normalization": {"clip_image": "l2", "clip_text": "l2"},
            "fusion": "none",
            "encoder_parameters": {"clip": parameter_counts(self.model)},
        }


class ClipDinov2Encoder(FrozenRegionTextEncoder):
    name = "clip_dinov2"

    def __init__(
        self,
        device: torch.device,
        clip_model_id: str = "ViT-B/32",
        dinov2_model_id: str = "facebook/dinov2-base",
    ):
        super().__init__(device)
        prepare_transformers_image_backend()
        from transformers import AutoImageProcessor, AutoModel

        self.clip = ClipEncoder(device=device, model_id=clip_model_id)
        self.dinov2_model_id = dinov2_model_id
        self.dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_id)
        self.dinov2_model = AutoModel.from_pretrained(dinov2_model_id).to(device)
        freeze_module(self.dinov2_model)
        self.dinov2_feature_dim = int(self.dinov2_model.config.hidden_size)
        self.candidate_feature_dim = (
            self.clip.candidate_feature_dim + self.dinov2_feature_dim
        )
        self.text_feature_dim = self.clip.text_feature_dim
        self.similarity_spec = {
            "candidate_slice": [0, self.clip.candidate_feature_dim],
            "text_slice": [0, self.clip.text_feature_dim],
        }

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        clip_features = self.clip.encode_images(images)
        inputs = self.dinov2_processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.dinov2_model(pixel_values=pixel_values)
        dinov2_features = getattr(outputs, "pooler_output", None)
        if dinov2_features is None:
            dinov2_features = outputs.last_hidden_state[:, 0]
        dinov2_features = F.normalize(dinov2_features.float(), dim=-1).cpu()
        # Each branch is normalized independently so neither representation is
        # suppressed only because of its dimensionality.
        return torch.cat([clip_features, dinov2_features], dim=-1)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        return self.clip.encode_texts(texts)

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_ids": {
                "clip": self.clip.model_id,
                "dinov2": self.dinov2_model_id,
            },
            "candidate_feature_dim": self.candidate_feature_dim,
            "text_feature_dim": self.text_feature_dim,
            "component_normalization": {
                "clip_image": "l2",
                "clip_text": "l2",
                "dinov2_cls": "l2",
            },
            "fusion": "concatenate normalized CLIP image and DINOv2 CLS features",
            "similarity_source": "normalized CLIP image/text subspace",
            "encoder_parameters": {
                "clip": parameter_counts(self.clip.model),
                "dinov2": parameter_counts(self.dinov2_model),
            },
        }


class Siglip2Encoder(FrozenRegionTextEncoder):
    name = "siglip2"

    def __init__(
        self,
        device: torch.device,
        model_id: str = "google/siglip2-base-patch16-224",
    ):
        super().__init__(device)
        prepare_transformers_image_backend()
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        freeze_module(self.model)
        self.candidate_feature_dim = int(self.model.config.vision_config.hidden_size)
        self.text_feature_dim = int(self.model.config.text_config.hidden_size)
        if self.candidate_feature_dim != self.text_feature_dim:
            raise ValueError(
                "SigLIP 2 image and text projection dimensions must agree for "
                "region-text similarity."
            )
        self.similarity_spec = {
            "candidate_slice": [0, self.candidate_feature_dim],
            "text_slice": [0, self.text_feature_dim],
        }

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        features = self.model.get_image_features(
            pixel_values=inputs["pixel_values"].to(self.device)
        )
        return F.normalize(features.float(), dim=-1).cpu()

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask"}
        }
        features = self.model.get_text_features(**model_inputs)
        return F.normalize(features.float(), dim=-1).cpu()

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_ids": {"siglip2": self.model_id},
            "candidate_feature_dim": self.candidate_feature_dim,
            "text_feature_dim": self.text_feature_dim,
            "component_normalization": {
                "siglip2_image": "l2",
                "siglip2_text": "l2",
            },
            "fusion": "none",
            "similarity_source": "normalized SigLIP 2 image/text embeddings",
            "encoder_parameters": {"siglip2": parameter_counts(self.model)},
        }


def build_encoder(
    representation: str,
    device: torch.device,
    clip_model_id: str = "ViT-B/32",
    dinov2_model_id: str = "facebook/dinov2-base",
    siglip2_model_id: str = "google/siglip2-base-patch16-224",
) -> FrozenRegionTextEncoder:
    if representation == "clip":
        return ClipEncoder(device=device, model_id=clip_model_id)
    if representation == "clip_dinov2":
        return ClipDinov2Encoder(
            device=device,
            clip_model_id=clip_model_id,
            dinov2_model_id=dinov2_model_id,
        )
    if representation == "siglip2":
        return Siglip2Encoder(device=device, model_id=siglip2_model_id)
    raise ValueError(f"Unknown representation: {representation!r}")
