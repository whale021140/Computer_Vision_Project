"""Render calibrated Stage 6 qualitative examples saved by the failure audit."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples-per-category", type=int, default=1)
    return parser.parse_args()


def draw_boxes(ax: Any, boxes: list[list[float]], color: str, style: str) -> None:
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.2,
                linestyle=style,
            )
        )
        ax.text(
            x1,
            y1,
            str(index),
            color="white",
            fontsize=7,
            bbox={"facecolor": color, "alpha": 0.8, "pad": 1},
        )


def main() -> None:
    args = parse_args()
    diagnosis = json.loads(Path(args.diagnosis_json).read_text())
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for category, rows in sorted(diagnosis["qualitative_examples"].items()):
        for index, row in enumerate(rows[: args.examples_per_category], start=1):
            image_path = image_root / row["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image = Image.open(image_path).convert("RGB")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image)
            draw_boxes(ax, row["target_boxes_xyxy"], "red", "-")
            draw_boxes(ax, row["predicted_boxes_xyxy"], "blue", "--")
            ax.axis("off")
            expression = "\n".join(
                textwrap.wrap(str(row["expression"]), width=65)
            )
            ax.set_title(
                f"{category} | targets={row['num_targets']} | "
                f"predictions={row['num_predictions']} | "
                f"matched={row['matched_targets']}\n{expression}",
                fontsize=9,
            )
            output = output_dir / f"{category}_{index:02d}.png"
            fig.savefig(output, dpi=150, bbox_inches="tight")
            plt.close(fig)
            manifest.append({**row, "output_file": str(output)})

    payload = {
        "stage": "6 compact calibrated qualitative visualization",
        "source": args.diagnosis_json,
        "legend": {"ground_truth": "red solid", "prediction": "blue dashed"},
        "examples": manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "Stage 6 Calibrated Qualitative Examples",
        "=======================================",
        "Ground truth: red solid; prediction: blue dashed.",
    ]
    lines.extend(
        f"{row['category']}: {row['output_file']} | {row['expression']}"
        for row in manifest
    )
    (output_dir / "manifest.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Rendered {len(manifest)} calibrated examples to {output_dir}")


if __name__ == "__main__":
    main()
