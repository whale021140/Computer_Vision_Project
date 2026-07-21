import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.data.feature_dataset import ClipFeatureDataset
from src.features.extract_frozen_features import extract_features
from src.evaluation.summarize_representation_results import build_row
from src.features.frozen_encoders import (
    ClipDinov2Encoder,
    FrozenRegionTextEncoder,
    Siglip2Encoder,
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
    def test_comparison_row_records_legacy_clip_encoder_metadata(self) -> None:
        evaluation = {
            "feature_dim": 2,
            "official": {"F1_score": 0.5, "T_acc": 0.6, "N_acc": 0.7},
            "diagnostics": {
                "mean_f1": 0.4,
                "cardinality_accuracy": 0.3,
                "false_grounding_rate": 0.2,
                "multi_target_mean_f1": 0.1,
            },
        }
        row = build_row(
            "clip",
            evaluation,
            encoder_parameters={"frozen": 123, "trainable": 0},
            model_ids={"clip": "ViT-B/32"},
        )
        self.assertEqual(row["encoder_parameters"], {"frozen": 123, "trainable": 0})
        self.assertEqual(row["model_ids"], {"clip": "ViT-B/32"})

    def test_siglip2_text_preprocessing_matches_training_contract(self) -> None:
        class RecordingProcessor:
            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {
                    "input_ids": torch.ones((2, 64), dtype=torch.long),
                    "attention_mask": torch.ones((2, 64), dtype=torch.long),
                }

        class FakeModel:
            @staticmethod
            def get_text_features(**model_inputs):
                return torch.ones((model_inputs["input_ids"].shape[0], 3))

        encoder = object.__new__(Siglip2Encoder)
        encoder.device = torch.device("cpu")
        encoder.processor = RecordingProcessor()
        encoder.model = FakeModel()
        features = encoder.encode_texts(["Red CAR", "BLUE bus"])

        self.assertEqual(encoder.processor.kwargs["text"], ["red car", "blue bus"])
        self.assertEqual(encoder.processor.kwargs["padding"], "max_length")
        self.assertTrue(encoder.processor.kwargs["truncation"])
        self.assertEqual(encoder.processor.kwargs["max_length"], 64)
        self.assertEqual(features.shape, (2, 3))
        self.assertTrue(torch.allclose(features.norm(dim=-1), torch.ones(2)))

    def test_transformer_processors_receive_explicit_channel_order(self) -> None:
        class RecordingProcessor:
            def __init__(self):
                self.kwargs = None

            @property
            def image_processor(self):
                return self

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {"pixel_values": torch.zeros((1, 3, 8, 8))}

        class FakeModel:
            @staticmethod
            def get_image_features(pixel_values):
                return torch.ones((pixel_values.shape[0], 3))

        encoder = object.__new__(Siglip2Encoder)
        encoder.device = torch.device("cpu")
        encoder.processor = RecordingProcessor()
        encoder.model = FakeModel()
        features = encoder.encode_images([Image.new("RGB", (3, 5))])

        self.assertEqual(
            encoder.processor.kwargs["input_data_format"], "channels_last"
        )
        self.assertEqual(features.shape, (1, 3))

    def test_dinov2_processor_receives_explicit_channel_order(self) -> None:
        class FakeClip:
            @staticmethod
            def encode_images(images):
                return torch.ones((len(images), 2))

        class RecordingProcessor:
            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return {"pixel_values": torch.zeros((1, 3, 8, 8))}

        class FakeDinoModel:
            @staticmethod
            def __call__(pixel_values):
                return type("Outputs", (), {"pooler_output": torch.ones((1, 3))})()

        encoder = object.__new__(ClipDinov2Encoder)
        encoder.device = torch.device("cpu")
        encoder.clip = FakeClip()
        encoder.dinov2_processor = RecordingProcessor()
        encoder.dinov2_model = FakeDinoModel()
        features = encoder.encode_images([Image.new("RGB", (3, 5))])

        self.assertEqual(
            encoder.dinov2_processor.kwargs["input_data_format"], "channels_last"
        )
        self.assertEqual(features.shape, (1, 5))

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

    def test_generic_extraction_rejects_shard_from_other_encoder(self) -> None:
        class OtherFakeEncoder(FakeEncoder):
            def metadata(self):
                return {**super().metadata(), "name": "other_fake"}

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
            extract_features(
                image_specs,
                [{"expression": "object"}],
                root,
                FakeEncoder(),
                region_batch_size=1,
                text_batch_size=1,
                storage_dtype=torch.float16,
                shard_dir=shard_dir,
            )
            with self.assertRaises(ValueError):
                extract_features(
                    image_specs,
                    [{"expression": "object"}],
                    root,
                    OtherFakeEncoder(),
                    region_batch_size=1,
                    text_batch_size=1,
                    storage_dtype=torch.float16,
                    shard_dir=shard_dir,
                    resume=True,
                )

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
