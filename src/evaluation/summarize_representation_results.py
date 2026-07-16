"""Create a compact Stage 4 comparison from evaluation and cache reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.models.baseline_heads import ClipCandidateBaseline


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def head_parameter_count(candidate_dim: int, text_dim: int) -> int:
    model = ClipCandidateBaseline(
        candidate_feature_dim=candidate_dim,
        text_feature_dim=text_dim,
        hidden_dim=256,
        dropout=0.1,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def encoder_parameter_totals(stats: Dict[str, Any]) -> Dict[str, int]:
    components = stats["representation"]["encoder_parameters"]
    return {
        "frozen": sum(int(values["total"]) for values in components.values()),
        "trainable": sum(
            int(values["trainable"]) for values in components.values()
        ),
    }


def build_row(
    name: str,
    evaluation: Dict[str, Any],
    stats: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    candidate_dim = int(evaluation.get("candidate_feature_dim", evaluation["feature_dim"]))
    text_dim = int(evaluation.get("text_feature_dim", evaluation["feature_dim"]))
    row = {
        "representation": name,
        "candidate_feature_dim": candidate_dim,
        "text_feature_dim": text_dim,
        "trainable_head_parameters": head_parameter_count(candidate_dim, text_dim),
        "official": evaluation["official"],
        "diagnostics": {
            key: evaluation["diagnostics"][key]
            for key in (
                "mean_f1",
                "cardinality_accuracy",
                "false_grounding_rate",
                "multi_target_mean_f1",
            )
        },
    }
    if stats is not None:
        row["encoder_parameters"] = encoder_parameter_totals(stats)
        row["model_ids"] = stats["representation"]["model_ids"]
    return row


def format_report(rows: list[Dict[str, Any]]) -> str:
    lines = [
        "Stage 4 frozen representation comparison",
        "========================================",
        "representation        F1_score     T_acc     N_acc    mean_f1   head_params",
    ]
    for row in rows:
        official = row["official"]
        lines.append(
            f"{row['representation']:<20} "
            f"{official['F1_score']:>8.6f}  {official['T_acc']:>8.6f}  "
            f"{official['N_acc']:>8.6f}  {row['diagnostics']['mean_f1']:>8.6f}  "
            f"{row['trainable_head_parameters']:>11d}"
        )
    lines.extend(["", "[encoder parameter counts]"])
    for row in rows:
        counts = row.get("encoder_parameters")
        if counts is not None:
            lines.append(
                f"{row['representation']}: frozen={counts['frozen']}, "
                f"trainable={counts['trainable']}"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-eval", required=True)
    parser.add_argument("--clip-dinov2-stats", required=True)
    parser.add_argument("--clip-dinov2-eval", required=True)
    parser.add_argument("--siglip2-stats", required=True)
    parser.add_argument("--siglip2-eval", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        build_row("clip", load_json(args.clip_eval)),
        build_row(
            "clip_dinov2",
            load_json(args.clip_dinov2_eval),
            load_json(args.clip_dinov2_stats),
        ),
        build_row(
            "siglip2",
            load_json(args.siglip2_eval),
            load_json(args.siglip2_stats),
        ),
    ]
    result = {"split": "val", "rows": rows}
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = format_report(rows)
    output_txt.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
