from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.grec_metrics import PredictionRecord, evaluate_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate representation-independent GREC box predictions."
    )
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-metric", choices=["iou", "giou"], default="iou")
    parser.add_argument("--prediction-score-threshold", type=float, default=None)
    parser.add_argument("--image-f1-threshold", type=float, default=1.0)
    parser.add_argument("--include-sample-metrics", action="store_true")
    return parser.parse_args()


def load_records(path: str | Path) -> List[PredictionRecord]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("records", value.get("predictions"))
    if not isinstance(value, list):
        raise ValueError(
            "Prediction file must contain a list, or an object with a "
            "'records' or 'predictions' list."
        )
    return [PredictionRecord.from_dict(record) for record in value]


def format_summary(result: Dict[str, Any]) -> str:
    official = result["official"]
    diagnostics = result["diagnostics"]
    config = result["config"]
    lines = [
        "GREC Box Evaluation",
        "===================",
        f"Overlap metric: {config['overlap_metric']}",
        f"Match threshold: {config['match_threshold']}",
        f"Prediction score threshold: {config['prediction_score_threshold']}",
        f"Image F1 threshold: {config['image_f1_threshold']}",
        "",
        "[released GREC metrics]",
        f"F1_score: {official['F1_score']:.6f}",
        f"T_acc: {official['T_acc']:.6f}",
        f"N_acc: {official['N_acc']:.6f}",
        "",
        "[diagnostics]",
    ]
    for key, value in diagnostics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")

    for target_type, group in result["by_target_type"].items():
        lines.extend(["", f"[{target_type}]"])
        for key, value in group.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    records = load_records(args.prediction_file)
    result = evaluate_records(
        records,
        match_threshold=args.match_threshold,
        overlap_metric=args.overlap_metric,
        prediction_score_threshold=args.prediction_score_threshold,
        image_f1_threshold=args.image_f1_threshold,
        include_sample_metrics=args.include_sample_metrics,
    )

    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = format_summary(result)
    output_txt.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
