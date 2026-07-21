from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from src.data.feature_dataset import ClipFeatureDataset, clip_feature_collate_fn
from src.evaluation.evaluate_clip_baseline import (
    load_model,
    positive_indices,
    select_topk_indices,
    set_metrics,
)


CATEGORY_ORDER = [
    "correct_no_target",
    "false_grounding_no_target",
    "correct_single_target",
    "failed_single_target",
    "correct_multi_target",
    "failed_multi_target",
]

CATEGORY_TITLES = {
    "correct_no_target": "Correct no-target",
    "false_grounding_no_target": "False grounding on no-target",
    "correct_single_target": "Correct single-target",
    "failed_single_target": "Failed single-target",
    "correct_multi_target": "Correct multi-target",
    "failed_multi_target": "Failed multi-target",
}

COUNT_CLASS_NAMES = {
    0: "0 / empty",
    1: "1",
    2: "2",
    3: "3+",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize qualitative examples from the CLIP candidate baseline. "
            "The script selects correct and failed cases for no-target, "
            "single-target, and multi-target expressions."
        )
    )
    parser.add_argument("--feature-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Directory containing COCO train2014 images, or a parent directory that contains train2014.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/milestone2/qualitative/testA",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=1,
        help="Number of examples to save for each qualitative category.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on feature records loaded from the feature cache.",
    )
    parser.add_argument(
        "--max-scan-samples",
        type=int,
        default=0,
        help="Optional limit on scanned samples. Use 0 to scan until quotas are filled or dataset ends.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip a selected example if its image file cannot be found.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI used when saving PNG files.",
    )
    parser.add_argument(
        "--show-box-indices",
        action="store_true",
        help="Draw GT/pred candidate indices beside boxes.",
    )
    return parser.parse_args()


def get_device(device_arg: str = "") -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sanitize_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:120] if text else "sample"


def resolve_image_path(image_root: Path, file_name: str) -> Path:
    """Resolve COCO image path robustly for common local layouts."""
    file_path = Path(file_name)
    candidates = []
    if file_path.is_absolute():
        candidates.append(file_path)
    else:
        candidates.extend(
            [
                image_root / file_name,
                image_root / "train2014" / file_name,
                image_root / "images" / "train2014" / file_name,
                image_root / "grefcoco" / "images" / "train2014" / file_name,
                image_root / "data" / "images" / "train2014" / file_name,
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    candidate_text = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find image file for file_name={file_name!r}. Tried:\n{candidate_text}"
    )


def normalized_boxes_to_xyxy(
    boxes_norm: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    if boxes_norm.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    boxes = boxes_norm.detach().cpu().float().clone().reshape(-1, 4)
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    return boxes * scale


def count_class_name(count_class: int) -> str:
    return COUNT_CLASS_NAMES.get(int(count_class), str(count_class))


def classify_example(target_type: str, pred: Set[int], gt: Set[int]) -> Optional[str]:
    exact = pred == gt
    if target_type == "no-target":
        if exact:
            return "correct_no_target"
        if len(pred) > 0:
            return "false_grounding_no_target"
        return None
    if target_type == "single-target":
        return "correct_single_target" if exact else "failed_single_target"
    if target_type == "multi-target":
        return "correct_multi_target" if exact else "failed_multi_target"
    return None


def draw_box_set(
    ax,
    boxes_xyxy: torch.Tensor,
    label_prefix: str,
    edgecolor: str,
    linewidth: float,
    linestyle: str,
    indices: Optional[Sequence[int]] = None,
    show_indices: bool = False,
) -> None:
    if boxes_xyxy.numel() == 0:
        return
    boxes = boxes_xyxy.detach().cpu().float().reshape(-1, 4)
    if indices is None:
        indices = list(range(boxes.shape[0]))
    for local_idx, (box, original_idx) in enumerate(zip(boxes, indices)):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        ax.add_patch(rect)
        if show_indices:
            ax.text(
                x1,
                max(0.0, y1 - 3),
                f"{label_prefix}{original_idx}",
                fontsize=7,
                color=edgecolor,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
            )


def save_visualization(
    image_path: Path,
    output_path: Path,
    category: str,
    expression: str,
    target_type: str,
    sample_id: str,
    file_name: str,
    true_count_class: int,
    pred_count_class: int,
    gt_candidate_indices: List[int],
    pred_candidate_indices: List[int],
    gt_boxes_xyxy: torch.Tensor,
    pred_boxes_xyxy: torch.Tensor,
    metrics: Dict[str, Any],
    dpi: int,
    show_box_indices: bool,
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Keep the figure compact but readable for a 2--3 page milestone report.
    base_width = 8.0
    fig_height = max(5.5, base_width * height / max(width, 1) + 1.5)
    fig, ax = plt.subplots(figsize=(base_width, fig_height))
    ax.imshow(image)
    ax.axis("off")

    draw_box_set(
        ax=ax,
        boxes_xyxy=gt_boxes_xyxy,
        label_prefix="GT ",
        edgecolor="red",
        linewidth=2.2,
        linestyle="-",
        indices=gt_candidate_indices,
        show_indices=show_box_indices,
    )
    draw_box_set(
        ax=ax,
        boxes_xyxy=pred_boxes_xyxy,
        label_prefix="Pred ",
        edgecolor="blue",
        linewidth=2.0,
        linestyle="--",
        indices=pred_candidate_indices,
        show_indices=show_box_indices,
    )

    wrapped_expression = textwrap.fill(expression, width=82)
    title = (
        f"{CATEGORY_TITLES.get(category, category)} | {target_type} | sample={sample_id}\n"
        f"true count={count_class_name(true_count_class)} | "
        f"pred count={count_class_name(pred_count_class)} | "
        f"F1={metrics['f1']:.3f} | exact={int(metrics['exact'])}\n"
        f"{wrapped_expression}"
    )
    ax.set_title(title, fontsize=9)

    legend_handles = [
        Line2D([0], [0], color="red", lw=2.2, label="Ground truth"),
        Line2D([0], [0], color="blue", lw=2.0, linestyle="--", label="Prediction"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def selected_quota_is_full(selected_counts: Dict[str, int], examples_per_category: int) -> bool:
    return all(selected_counts.get(k, 0) >= examples_per_category for k in CATEGORY_ORDER)


@torch.no_grad()
def visualize(args: argparse.Namespace) -> Dict[str, Any]:
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ClipFeatureDataset(args.feature_file, max_samples=args.max_samples)
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

    selected_counts = {category: 0 for category in CATEGORY_ORDER}
    candidate_counts = {category: 0 for category in CATEGORY_ORDER}
    manifest_examples: List[Dict[str, Any]] = []
    scanned = 0
    skipped_missing_images = 0

    for batch in tqdm(loader, desc="Selecting qualitative examples"):
        outputs = model(batch)
        membership_logits_list: List[torch.Tensor] = outputs["membership_logits"]
        count_logits: torch.Tensor = outputs["count_logits"]
        pred_count_classes = count_logits.argmax(dim=1).detach().cpu()
        true_count_classes = batch["count_class"].detach().cpu()

        for i, membership_logits in enumerate(membership_logits_list):
            scanned += 1
            if args.max_scan_samples and scanned > args.max_scan_samples:
                break

            metadata = batch["metadata"][i]
            target_type = metadata.get("target_type", "unknown")
            true_count_class = int(true_count_classes[i].item())
            pred_count_class = int(pred_count_classes[i].item())
            gt = positive_indices(batch["candidate_labels"][i])
            pred = select_topk_indices(
                membership_logits=membership_logits,
                pred_count_class=pred_count_class,
            )
            category = classify_example(target_type=target_type, pred=pred, gt=gt)
            if category is None:
                continue

            candidate_counts[category] += 1
            if selected_counts[category] >= args.examples_per_category:
                continue

            file_name = str(metadata.get("file_name"))
            try:
                image_path = resolve_image_path(image_root=image_root, file_name=file_name)
            except FileNotFoundError:
                if args.skip_missing_images:
                    skipped_missing_images += 1
                    continue
                raise

            width = int(metadata.get("width") or Image.open(image_path).size[0])
            height = int(metadata.get("height") or Image.open(image_path).size[1])
            candidate_boxes_xyxy = normalized_boxes_to_xyxy(
                batch["candidate_boxes_norm"][i], width=width, height=height
            )
            target_boxes_xyxy = batch["target_boxes_xyxy"][i].detach().cpu().float()

            gt_indices = sorted(gt)
            pred_indices = sorted(pred)
            pred_boxes_xyxy = (
                candidate_boxes_xyxy[pred_indices]
                if len(pred_indices) > 0
                else torch.empty((0, 4), dtype=torch.float32)
            )

            metrics = set_metrics(pred=pred, gt=gt)
            sample_id = str(batch["sample_ids"][i])
            selected_counts[category] += 1
            output_name = (
                f"{category}_{selected_counts[category]:02d}_"
                f"{sanitize_filename(sample_id)}_{sanitize_filename(Path(file_name).stem)}.png"
            )
            output_path = output_dir / output_name

            save_visualization(
                image_path=image_path,
                output_path=output_path,
                category=category,
                expression=batch["expressions"][i],
                target_type=target_type,
                sample_id=sample_id,
                file_name=file_name,
                true_count_class=true_count_class,
                pred_count_class=pred_count_class,
                gt_candidate_indices=gt_indices,
                pred_candidate_indices=pred_indices,
                gt_boxes_xyxy=target_boxes_xyxy,
                pred_boxes_xyxy=pred_boxes_xyxy,
                metrics=metrics,
                dpi=args.dpi,
                show_box_indices=args.show_box_indices,
            )

            manifest_examples.append(
                {
                    "category": category,
                    "category_title": CATEGORY_TITLES.get(category, category),
                    "output_file": str(output_path),
                    "sample_id": sample_id,
                    "image_id": metadata.get("image_id"),
                    "file_name": file_name,
                    "target_type": target_type,
                    "expression": batch["expressions"][i],
                    "true_count_class": true_count_class,
                    "pred_count_class": pred_count_class,
                    "gt_candidate_indices": gt_indices,
                    "pred_candidate_indices": pred_indices,
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "exact": float(metrics["exact"]),
                }
            )

            if selected_quota_is_full(selected_counts, args.examples_per_category):
                break

        if args.max_scan_samples and scanned >= args.max_scan_samples:
            break
        if selected_quota_is_full(selected_counts, args.examples_per_category):
            break

    summary = {
        "feature_file": args.feature_file,
        "checkpoint": args.checkpoint,
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "clip_model": dataset.clip_model,
        "feature_dim": dataset.feature_dim,
        "examples_per_category": args.examples_per_category,
        "scanned_samples": scanned,
        "selected_counts": selected_counts,
        "candidate_counts_seen": candidate_counts,
        "skipped_missing_images": skipped_missing_images,
        "examples": manifest_examples,
    }

    manifest_json = output_dir / "manifest.json"
    manifest_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "CLIP Baseline Qualitative Visualization Summary",
        "================================================",
        f"Feature file: {args.feature_file}",
        f"Checkpoint: {args.checkpoint}",
        f"Image root: {image_root}",
        f"Output dir: {output_dir}",
        f"Device: {device}",
        f"CLIP model: {dataset.clip_model}",
        f"Feature dimension: {dataset.feature_dim}",
        f"Scanned samples: {scanned}",
        f"Skipped missing images: {skipped_missing_images}",
        "",
        "Selected examples:",
    ]
    for category in CATEGORY_ORDER:
        lines.append(f"  {category}: {selected_counts[category]} / {args.examples_per_category}")
    lines.append("")
    lines.append("Files:")
    for ex in manifest_examples:
        lines.append(
            f"  - [{ex['category']}] {ex['output_file']} | "
            f"sample={ex['sample_id']} | exact={ex['exact']:.0f} | F1={ex['f1']:.3f}"
        )

    manifest_txt = output_dir / "manifest.txt"
    manifest_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary


def main() -> None:
    args = parse_args()
    summary = visualize(args)
    print(json.dumps(summary["selected_counts"], indent=2))
    print(f"Saved qualitative visualizations to: {summary['output_dir']}")
    print(f"Saved manifest to: {Path(summary['output_dir']) / 'manifest.json'}")
    print(f"Saved summary to: {Path(summary['output_dir']) / 'manifest.txt'}")


if __name__ == "__main__":
    main()
