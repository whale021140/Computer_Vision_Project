import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data.feature_dataset import ClipFeatureDataset
from src.features.extract_frozen_features import extract_features
from src.features.frozen_encoders import (
    FrozenRegionTextEncoder,
    freeze_module,
    prepare_transformers_image_backend,
)
from src.models.baseline_heads import ClipCandidateBaseline


class FakeEncoder(FrozenRegionTextEncoder):
    name = "fake_multimodal"
    candidate_feature_dim = 5
    text_feature_dim = 2
    similarity_spec = {
        "candidate_slice": [0, 2],
        "text_slice": [0, 2],
    }

    def __init__(self):
        super().__init__(torch.device("cpu"))

    def encode_images(self, images):
        rows = []
        for index, _ in enumerate(images):
            rows.append([1.0, 0.0, float(index), 2.0, 3.0])
        return torch.tensor(rows)

    def encode_texts(self, texts):
        return torch.tensor([[1.0, 0.0] for _ in texts])

    def metadata(self):
        return {
            "name": self.name,
            "candidate_feature_dim": self.candidate_feature_dim,
            "text_feature_dim": self.text_feature_dim,
            "encoder_parameters": {},
        }


class FrozenRepresentationTests(unittest.TestCase):
    def test_dataset_uses_aligned_similarity_subspace(self) -> None:
        cache = {
            "cache_format": "frozen_representation_v1",
            "feature_dim": 5,
            "candidate_feature_dim": 5,
            "text_feature_dim": 2,
            "similarity_spec": FakeEncoder.similarity_spec,
            "representation": {"name": "clip_dinov2"},
            "images": {
                "7": {
                    "candidate_features": torch.tensor(
                        [[1.0, 0.0, 9.0, 9.0, 9.0], [0.0, 1.0, 8.0, 8.0, 8.0]]
                    ),
                    "candidate_boxes_norm": torch.zeros((2, 4)),
                }
            },
            "records": [
                {
                    "sample_id": "sample",
                    "image_id": 7,
                    "metadata": {"image_id": 7},
                    "expression": "object",
                    "text_feature": torch.tensor([1.0, 0.0]),
                    "candidate_labels": torch.tensor([1.0, 0.0]),
                    "count_class": torch.tensor(1),
                    "target_boxes_xyxy": torch.zeros((1, 4)),
                    "target_boxes_norm": torch.zeros((1, 4)),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.pt"
            torch.save(cache, path)
            dataset = ClipFeatureDataset(str(path))
            sample = dataset[0]

        self.assertEqual(dataset.candidate_feature_dim, 5)
        self.assertEqual(dataset.text_feature_dim, 2)
        self.assertEqual(sample["candidate_text_similarity"].tolist(), [1.0, 0.0])
        model = ClipCandidateBaseline(
            candidate_feature_dim=5,
            text_feature_dim=2,
            hidden_dim=4,
        )
        self.assertEqual(model.input_dim, 12)

    def test_generic_extraction_separates_images_and_expressions(self) -> None:
        image_specs = {
            "7": {
                "image_id": 7,
                "file_name": "image.png",
                "width": 20,
                "height": 10,
                "candidate_source": "fake",
                "proposal_config": None,
                "candidate_boxes_xyxy": [[0, 0, 10, 10], [10, 0, 20, 10]],
                "candidate_boxes_norm": torch.tensor(
                    [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]
                ),
                "candidate_scores": torch.ones(2),
                "candidate_detector_labels": torch.ones(2, dtype=torch.long),
            }
        }
        records = [
            {"expression": "first object"},
            {"expression": "second object"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            Image.new("RGB", (20, 10), color="white").save(
                Path(tmpdir) / "image.png"
            )
            images, count, resumed = extract_features(
                image_specs=image_specs,
                expression_records=records,
                image_root=Path(tmpdir),
                encoder=FakeEncoder(),
                region_batch_size=1,
                text_batch_size=2,
                storage_dtype=torch.float16,
            )

        self.assertEqual(count, 2)
        self.assertEqual(resumed, 0)
        self.assertEqual(images["7"]["candidate_features"].shape, (2, 5))
        self.assertNotIn("candidate_boxes_xyxy", images["7"])
        self.assertEqual(records[0]["text_feature"].shape, (2,))

    def test_generic_extraction_resumes_validated_image_shards(self) -> None:
        image_specs = {
            "7": {
                "image_id": 7,
                "file_name": "image.png",
                "width": 10,
                "height": 10,
                "candidate_source": "fake",
                "proposal_config": None,
                "candidate_boxes_xyxy": [[0, 0, 10, 10]],
                "candidate_boxes_norm": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "candidate_scores": torch.ones(1),
                "candidate_detector_labels": torch.ones(1, dtype=torch.long),
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            Image.new("RGB", (10, 10), color="white").save(root / "image.png")
            shard_dir = root / "parts"
            first_records = [{"expression": "object"}]
            extract_features(
                image_specs,
                first_records,
                root,
                FakeEncoder(),
                region_batch_size=1,
                text_batch_size=1,
                storage_dtype=torch.float16,
                shard_dir=shard_dir,
            )
            second_records = [{"expression": "object"}]
            _, count, resumed = extract_features(
                image_specs,
                second_records,
                root,
                FakeEncoder(),
                region_batch_size=1,
                text_batch_size=1,
                storage_dtype=torch.float16,
                shard_dir=shard_dir,
                resume=True,
            )

        self.assertEqual(count, 1)
        self.assertEqual(resumed, 1)

    def test_freeze_module_reports_no_trainable_parameters(self) -> None:
        module = torch.nn.Linear(3, 2)
        freeze_module(module)
        self.assertFalse(module.training)
        self.assertFalse(any(parameter.requires_grad for parameter in module.parameters()))

    def test_transformers_dinov2_and_siglip_forward_contracts(self) -> None:
        prepare_transformers_image_backend()
        from transformers import (
            Dinov2Config,
            Dinov2Model,
            SiglipConfig,
            SiglipModel,
            SiglipTextConfig,
            SiglipVisionConfig,
        )

        dino = Dinov2Model(
            Dinov2Config(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=64,
                image_size=28,
                patch_size=14,
            )
        )
        dino_features = dino(torch.randn(2, 3, 28, 28)).pooler_output
        self.assertEqual(dino_features.shape, (2, 32))

        siglip = SiglipModel(
            SiglipConfig(
                text_config=SiglipTextConfig(
                    vocab_size=100,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    max_position_embeddings=8,
                ).to_dict(),
                vision_config=SiglipVisionConfig(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    image_size=16,
                    patch_size=8,
                ).to_dict(),
            )
        )
        image_features = siglip.get_image_features(
            pixel_values=torch.randn(2, 3, 16, 16)
        )
        text_features = siglip.get_text_features(
            input_ids=torch.randint(0, 100, (2, 8)),
            attention_mask=torch.ones((2, 8), dtype=torch.long),
        )
        self.assertEqual(image_features.shape, (2, 32))
        self.assertEqual(text_features.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()
