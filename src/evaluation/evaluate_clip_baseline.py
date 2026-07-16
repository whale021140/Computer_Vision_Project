from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.evaluation.grec_metrics import (
    PredictionRecord,
    evaluate_records,
    evaluate_sample,
)
from src.evaluation.metrics import (
    select_cardinality_gated_indices,
    select_topk_indices as select_legacy_topk_indices,
)
from src.models.baseline_heads import ClipCandidateBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-txt", type=str, required=True)
    parser.add_argument(
        "--output-predictions",
        type=str,
        default="",
        help="Optional representation-independent prediction-record JSON file.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument(
        "--selection-policy",
        choices=["cardinality-threshold", "legacy-topk"],
        default="cardinality-threshold",
    )
    parser.add_argument(
        "--membership-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used when the predicted count class is 3+.",
    )
    parser.add_argument(
        "--calibration-json",
        default="",
        help=(
            "Optional validation-calibration JSON. Its best membership threshold "
            "overrides --membership-threshold."
        ),
    )
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-metric", choices=["iou", "giou"], default="iou")
    parser.add_argument("--prediction-score-threshold", type=float, default=None)
    parser.add_argument("--image-f1-threshold", type=float, default=1.0)
    return parser.parse_args()


def get_device(device_arg: str = "") -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def positive_indices(candidate_labels: torch.Tensor) -> Set[int]:
    labels = candidate_labels.detach().cpu()
    idx = torch.nonzero(labels > 0.5, as_tuple=False).flatten().tolist()
    return {int(i) for i in idx}


def select_topk_indices(
    membership_logits: torch.Tensor,
    pred_count_class: int,
) -> Set[int]:
    """Legacy wrapper used by existing Milestone 2 visualizations."""
    return select_legacy_topk_indices(membership_logits, pred_count_class)


def set_metrics(pred: Set[int], gt: Set[int]) -> Dict[str, Any]:
    """Legacy candidate-index metrics used by existing visualizations."""
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    if len(pred) == 0 and len(gt) == 0:
        precision = recall = 1.0
    else:
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gt) if gt else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": float(pred == gt),
    }


def load_model(
    checkpoint_path: str,
    candidate_feature_dim: int,
    text_feature_dim: int,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> ClipCandidateBaseline:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(ckpt)}")

    ckpt_args = ckpt.get("args", {})
    if isinstance(ckpt_args, dict):
        hidden_dim = int(ckpt_args.get("hidden_dim", hidden_dim))
        dropout = float(ckpt_args.get("dropout", dropout))
    candidate_feature_dim = int(
        ckpt.get("candidate_feature_dim", ckpt.get("feature_dim", candidate_feature_dim))
    )
    text_feature_dim = int(
        ckpt.get("text_feature_dim", ckpt.get("feature_dim", text_feature_dim))
    )

    model = ClipCandidateBaseline(
        candidate_feature_dim=candidate_feature_dim,
        text_feature_dim=text_feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device)
    model.eval()
    return model


def normalized_boxes_to_xyxy(
    normalized_boxes: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    boxes = normalized_boxes.detach().cpu().float().reshape(-1, 4).clone()
    if boxes.numel() == 0:
        return boxes
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    return boxes


def select_prediction_indices(
    membership_logits: torch.Tensor,
    pred_count_class: int,
    selection_policy: str,
    membership_threshold: float,
) -> Set[int]:
    if selection_policy == "legacy-topk":
        return select_legacy_topk_indices(membership_logits, pred_count_class)
    if selection_policy == "cardinality-threshold":
        return select_cardinality_gated_indices(
            membership_logits,
            pred_count_class,
            membership_threshold=membership_threshold,
        )
    raise ValueError(f"Unknown selection policy: {selection_policy!r}")


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = get_device(args.device)
    dataset = ClipFeatureDataset(args.feature_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=clip_feature_collate_fn,
    )
    model = load_model(
        checkpoint_path=args.checkpoint,
        candidate_feature_dim=dataset.candidate_feature_dim,
        text_feature_dim=dataset.text_feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=device,
    )

    prediction_records: List[PredictionRecord] = []
    examples: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="Evaluating"):
        outputs = model(batch)
        membership_logits_list: List[torch.Tensor] = outputs["membership_logits"]
        count_logits: torch.Tensor = outputs["count_logits"]
        pred_count_classes = count_logits.argmax(dim=1).detach().cpu()
        true_count_classes = batch["count_class"].detach().cpu()

        for i, membership_logits in enumerate(membership_logits_list):
            metadata = batch["metadata"][i]
            target_type = metadata.get("target_type", "unknown")
            pred_count_class = int(pred_count_classes[i].item())
            true_count_class = int(true_count_classes[i].item())
            pred_indices = select_prediction_indices(
                membership_logits,
                pred_count_class,
                selection_policy=args.selection_policy,
                membership_threshold=args.membership_threshold,
            )

            width = int(metadata["width"])
            height = int(metadata["height"])
            candidate_boxes = normalized_boxes_to_xyxy(
                batch["candidate_boxes_norm"][i],
                width=width,
                height=height,
            )
            sorted_indices = sorted(pred_indices)
            predicted_boxes = (
                candidate_boxes[sorted_indices]
                if sorted_indices
                else torch.empty((0, 4), dtype=torch.float32)
            )
            probabilities = torch.sigmoid(membership_logits.detach().cpu())
            predicted_scores = (
                probabilities[sorted_indices]
                if sorted_indices
                else torch.empty(0, dtype=torch.float32)
            )
            record = PredictionRecord(
                sample_id=str(batch["sample_ids"][i]),
                predicted_boxes=predicted_boxes,
                predicted_scores=predicted_scores,
                target_boxes=batch["target_boxes_xyxy"][i],
                target_type=target_type,
                predicted_count_class=pred_count_class,
            )
            prediction_records.append(record)

            if len(examples) < args.num_examples:
                sample_metrics = evaluate_sample(
                    record,
                    match_threshold=args.match_threshold,
                    overlap_metric=args.overlap_metric,
                    prediction_score_threshold=args.prediction_score_threshold,
                    image_f1_threshold=args.image_f1_threshold,
                )
                examples.append(
                    {
                        "sample_id": record.sample_id,
                        "image_id": metadata.get("image_id"),
                        "file_name": metadata.get("file_name"),
                        "target_type": target_type,
                        "expression": batch["expressions"][i],
                        "true_count_class": true_count_class,
                        "pred_count_class": pred_count_class,
                        "gt_candidate_indices": sorted(
                            positive_indices(batch["candidate_labels"][i])
                        ),
                        "pred_candidate_indices": sorted_indices,
                        "num_predictions": sample_metrics.num_predictions,
                        "num_targets": sample_metrics.num_targets,
                        "precision": sample_metrics.precision,
                        "recall": sample_metrics.recall,
                        "f1": sample_metrics.f1,
                        "exact": sample_metrics.exact_set,
                    }
                )

    metrics = evaluate_records(
        prediction_records,
        match_threshold=args.match_threshold,
        overlap_metric=args.overlap_metric,
        prediction_score_threshold=args.prediction_score_threshold,
        image_f1_threshold=args.image_f1_threshold,
    )
    result = {
        "feature_file": args.feature_file,
        "checkpoint": args.checkpoint,
        "clip_model": dataset.clip_model,
        "feature_dim": dataset.candidate_feature_dim,
        "candidate_feature_dim": dataset.candidate_feature_dim,
        "text_feature_dim": dataset.text_feature_dim,
        "representation": dataset.representation,
        "device": str(device),
        "batch_size": args.batch_size,
        "selection_policy": args.selection_policy,
        "membership_threshold": args.membership_threshold,
        **metrics,
        "examples": examples,
    }

    if args.output_predictions:
        prediction_path = Path(args.output_predictions)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(
            json.dumps(
                {"records": [record.to_dict() for record in prediction_records]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result["prediction_record_file"] = str(prediction_path)
    return result


def write_text_summary(result: Dict[str, Any], output_txt: str) -> None:
    official = result["official"]
    diagnostics = result["diagnostics"]
    config = result["config"]
    lines = [
        "Frozen Representation Baseline GREC Evaluation",
        "==============================================",
        f"Feature file: {result['feature_file']}",
        f"Checkpoint: {result['checkpoint']}",
        f"Representation: {result.get('representation', result['clip_model'])}",
        f"Device: {result['device']}",
        f"Selection policy: {result['selection_policy']}",
        f"Membership threshold: {result['membership_threshold']}",
        f"Overlap metric: {config['overlap_metric']}",
        f"Match threshold: {config['match_threshold']}",
        f"Prediction score threshold: {config['prediction_score_threshold']}",
        "",
        "[released GREC metrics]",
        f"F1_score: {official['F1_score']:.6f}",
        f"T_acc: {official['T_acc']:.6f}",
        f"N_acc: {official['N_acc']:.6f}",
        "",
        "[diagnostics]",
    ]
    for key, value in diagnostics.items():
        lines.append(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    for target_type, group in result["by_target_type"].items():
        lines.extend(["", f"[{target_type}]"])
        for key, value in group.items():
            lines.append(
                f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}"
            )

    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.calibration_json:
        calibration = json.loads(
            Path(args.calibration_json).read_text(encoding="utf-8")
        )
        args.membership_threshold = float(
            calibration["best"]["membership_threshold"]
        )
    result = evaluate(args)
    if args.calibration_json:
        result["calibration_json"] = args.calibration_json
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_text_summary(result, args.output_txt)
    print(Path(args.output_txt).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
