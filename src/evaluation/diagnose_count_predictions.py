from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.models.baseline_heads import ClipCandidateBaseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ClipFeatureDataset(args.feature_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=clip_feature_collate_fn,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ClipCandidateBaseline(
        candidate_feature_dim=ckpt.get(
            "candidate_feature_dim", ckpt["feature_dim"]
        ),
        text_feature_dim=ckpt.get("text_feature_dim", ckpt["feature_dim"]),
        hidden_dim=ckpt["args"]["hidden_dim"],
        dropout=ckpt["args"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    matrix = defaultdict(Counter)
    overall_pred = Counter()
    overall_true = Counter()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Diagnosing"):
            outputs = model(batch)
            pred = outputs["count_logits"].argmax(dim=1).cpu()
            true = batch["count_class"].cpu()

            for i in range(len(pred)):
                target_type = batch["metadata"][i].get("target_type", "unknown")
                t = int(true[i].item())
                p = int(pred[i].item())

                matrix[target_type][(t, p)] += 1
                overall_pred[p] += 1
                overall_true[t] += 1

    lines = []
    lines.append("Count Prediction Diagnosis")
    lines.append("==========================")
    lines.append("")
    lines.append(f"Feature file: {args.feature_file}")
    lines.append(f"Checkpoint: {args.checkpoint}")
    lines.append("")
    lines.append("Overall true count-class distribution:")
    for k in sorted(overall_true):
        lines.append(f"  true class {k}: {overall_true[k]}")
    lines.append("")
    lines.append("Overall predicted count-class distribution:")
    for k in sorted(overall_pred):
        lines.append(f"  pred class {k}: {overall_pred[k]}")
    lines.append("")
    lines.append("By target type: entries are true_class -> pred_class")
    for target_type in sorted(matrix):
        lines.append("")
        lines.append(f"[{target_type}]")
        for (t, p), n in sorted(matrix[target_type].items()):
            lines.append(f"  true {t} -> pred {p}: {n}")

    text = "\n".join(lines)
    print(text)

    if args.output_file:
        path = Path(args.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
