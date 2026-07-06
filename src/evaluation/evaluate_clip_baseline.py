from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.models.baseline_heads import ClipCandidateBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-txt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-examples", type=int, default=20)
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


def count_class_to_k(count_class: int, num_candidates: int) -> int:
    if count_class <= 0:
        return 0
    if count_class == 1:
        return min(1, num_candidates)
    if count_class == 2:
        return min(2, num_candidates)
    return min(3, num_candidates)


def select_topk_indices(membership_logits: torch.Tensor, pred_count_class: int) -> Set[int]:
    logits = membership_logits.detach().cpu()
    num_candidates = int(logits.numel())
    k = count_class_to_k(pred_count_class, num_candidates)

    if k <= 0 or num_candidates == 0:
        return set()

    topk = torch.topk(logits, k=k).indices.tolist()
    return {int(i) for i in topk}


def set_metrics(pred: Set[int], gt: Set[int]) -> Dict[str, Any]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    if len(pred) == 0 and len(gt) == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = tp / len(pred) if len(pred) > 0 else 0.0
        recall = tp / len(gt) if len(gt) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    exact = 1.0 if pred == gt else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": exact,
    }


def mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def load_model(
    checkpoint_path: str,
    feature_dim: int,
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

    feature_dim = int(ckpt.get("feature_dim", feature_dim))

    model = ClipCandidateBaseline(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def new_accumulator() -> Dict[str, Any]:
    return {
        "num_samples": 0,
        "count_correct": 0,
        "count_total": 0,
        "precision_values": [],
        "recall_values": [],
        "f1_values": [],
        "exact_values": [],
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
        "no_target_total": 0,
        "no_target_correct": 0,
        "false_grounding": 0,
        "single_target_total": 0,
        "single_target_correct": 0,
        "multi_target_total": 0,
        "multi_target_exact": 0,
    }


def update_accumulator(
    acc: Dict[str, Any],
    target_type: str,
    true_count_class: int,
    pred_count_class: int,
    pred: Set[int],
    gt: Set[int],
    m: Dict[str, Any],
) -> None:
    acc["num_samples"] += 1
    acc["count_total"] += 1
    acc["count_correct"] += int(pred_count_class == true_count_class)

    acc["precision_values"].append(float(m["precision"]))
    acc["recall_values"].append(float(m["recall"]))
    acc["f1_values"].append(float(m["f1"]))
    acc["exact_values"].append(float(m["exact"]))

    acc["total_tp"] += int(m["tp"])
    acc["total_fp"] += int(m["fp"])
    acc["total_fn"] += int(m["fn"])

    if target_type == "no-target":
        acc["no_target_total"] += 1
        acc["no_target_correct"] += int(len(pred) == 0 and len(gt) == 0)
        acc["false_grounding"] += int(len(pred) > 0)

    if target_type == "single-target":
        acc["single_target_total"] += 1
        acc["single_target_correct"] += int(m["exact"] == 1.0)

    if target_type == "multi-target":
        acc["multi_target_total"] += 1
        acc["multi_target_exact"] += int(m["exact"] == 1.0)


def finalize_accumulator(acc: Dict[str, Any]) -> Dict[str, Any]:
    tp = acc["total_tp"]
    fp = acc["total_fp"]
    fn = acc["total_fn"]

    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    no_target_total = acc["no_target_total"]
    single_total = acc["single_target_total"]
    multi_total = acc["multi_target_total"]

    return {
        "num_samples": acc["num_samples"],
        "count_accuracy": (
            acc["count_correct"] / acc["count_total"]
            if acc["count_total"] > 0
            else 0.0
        ),
        "mean_precision": mean(acc["precision_values"]),
        "mean_recall": mean(acc["recall_values"]),
        "mean_f1": mean(acc["f1_values"]),
        "exact_set_accuracy": mean(acc["exact_values"]),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "total_tp": tp,
        "total_fp": fp,
        "total_fn": fn,
        "no_target_total": no_target_total,
        "no_target_accuracy": (
            acc["no_target_correct"] / no_target_total
            if no_target_total > 0
            else 0.0
        ),
        "false_grounding_rate": (
            acc["false_grounding"] / no_target_total
            if no_target_total > 0
            else 0.0
        ),
        "single_target_total": single_total,
        "single_target_exact_accuracy": (
            acc["single_target_correct"] / single_total
            if single_total > 0
            else 0.0
        ),
        "multi_target_total": multi_total,
        "multi_target_exact_accuracy": (
            acc["multi_target_exact"] / multi_total
            if multi_total > 0
            else 0.0
        ),
    }


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
        feature_dim=dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=device,
    )

    overall = new_accumulator()
    by_target_type = defaultdict(new_accumulator)
    examples: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="Evaluating"):
        # Important: this model's forward signature is forward(batch), not keyword args.
        outputs = model(batch)

        membership_logits_list: List[torch.Tensor] = outputs["membership_logits"]
        count_logits: torch.Tensor = outputs["count_logits"]

        pred_count_classes = count_logits.argmax(dim=1).detach().cpu()
        true_count_classes = batch["count_class"].detach().cpu()

        for i, membership_logits in enumerate(membership_logits_list):
            metadata = batch["metadata"][i]
            target_type = metadata.get("target_type", "unknown")

            true_count_class = int(true_count_classes[i].item())
            pred_count_class = int(pred_count_classes[i].item())

            gt = positive_indices(batch["candidate_labels"][i])
            pred = select_topk_indices(
                membership_logits=membership_logits,
                pred_count_class=pred_count_class,
            )

            m = set_metrics(pred=pred, gt=gt)

            update_accumulator(
                acc=overall,
                target_type=target_type,
                true_count_class=true_count_class,
                pred_count_class=pred_count_class,
                pred=pred,
                gt=gt,
                m=m,
            )

            update_accumulator(
                acc=by_target_type[target_type],
                target_type=target_type,
                true_count_class=true_count_class,
                pred_count_class=pred_count_class,
                pred=pred,
                gt=gt,
                m=m,
            )

            if len(examples) < args.num_examples:
                examples.append(
                    {
                        "sample_id": batch["sample_ids"][i],
                        "image_id": metadata.get("image_id"),
                        "file_name": metadata.get("file_name"),
                        "target_type": target_type,
                        "expression": batch["expressions"][i],
                        "true_count_class": true_count_class,
                        "pred_count_class": pred_count_class,
                        "gt_candidate_indices": sorted(gt),
                        "pred_candidate_indices": sorted(pred),
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "f1": m["f1"],
                        "exact": m["exact"],
                    }
                )

    result = {
        "feature_file": args.feature_file,
        "checkpoint": args.checkpoint,
        "clip_model": dataset.clip_model,
        "feature_dim": dataset.feature_dim,
        "device": str(device),
        "batch_size": args.batch_size,
        "inference_rule": (
            "Predicted count class selects top-k membership logits. "
            "Count class 0 selects no boxes; 1 selects top-1; "
            "2 selects top-2; 3 selects top-3."
        ),
        "overall": finalize_accumulator(overall),
        "by_target_type": {
            k: finalize_accumulator(v)
            for k, v in sorted(by_target_type.items())
        },
        "examples": examples,
    }

    return result


def write_text_summary(result: Dict[str, Any], output_txt: str) -> None:
    lines: List[str] = []

    lines.append("CLIP Baseline Evaluation Summary")
    lines.append("================================")
    lines.append(f"Feature file: {result['feature_file']}")
    lines.append(f"Checkpoint: {result['checkpoint']}")
    lines.append(f"CLIP model: {result['clip_model']}")
    lines.append(f"Feature dimension: {result['feature_dim']}")
    lines.append(f"Device: {result['device']}")
    lines.append(f"Batch size: {result['batch_size']}")
    lines.append(f"Inference rule: {result['inference_rule']}")
    lines.append("")

    def add_group(name: str, group: Dict[str, Any]) -> None:
        lines.append(f"[{name}]")
        lines.append(f"num_samples: {group['num_samples']}")
        lines.append(f"count_accuracy: {group['count_accuracy']:.6f}")
        lines.append(f"mean_precision: {group['mean_precision']:.6f}")
        lines.append(f"mean_recall: {group['mean_recall']:.6f}")
        lines.append(f"mean_f1: {group['mean_f1']:.6f}")
        lines.append(f"exact_set_accuracy: {group['exact_set_accuracy']:.6f}")
        lines.append(f"micro_precision: {group['micro_precision']:.6f}")
        lines.append(f"micro_recall: {group['micro_recall']:.6f}")
        lines.append(f"micro_f1: {group['micro_f1']:.6f}")
        lines.append(f"total_tp: {group['total_tp']}")
        lines.append(f"total_fp: {group['total_fp']}")
        lines.append(f"total_fn: {group['total_fn']}")
        lines.append(f"no_target_total: {group['no_target_total']}")
        lines.append(f"no_target_accuracy: {group['no_target_accuracy']:.6f}")
        lines.append(f"false_grounding_rate: {group['false_grounding_rate']:.6f}")
        lines.append(f"single_target_total: {group['single_target_total']}")
        lines.append(
            f"single_target_exact_accuracy: "
            f"{group['single_target_exact_accuracy']:.6f}"
        )
        lines.append(f"multi_target_total: {group['multi_target_total']}")
        lines.append(
            f"multi_target_exact_accuracy: "
            f"{group['multi_target_exact_accuracy']:.6f}"
        )
        lines.append("")

    add_group("overall", result["overall"])

    for target_type, group in result["by_target_type"].items():
        add_group(target_type, group)

    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    result = evaluate(args)

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
