"""Compare oracle- and detector-candidate GREC evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OFFICIAL_KEYS = ("F1_score", "T_acc", "N_acc")
DIAGNOSTIC_KEYS = (
    "mean_f1",
    "exact_set_accuracy",
    "cardinality_accuracy",
    "false_grounding_rate",
    "multi_target_mean_f1",
    "multi_target_exact_accuracy",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_comparison(
    oracle: dict[str, Any],
    detector: dict[str, Any],
    proposal_recall: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact, machine-readable candidate-source comparison."""
    oracle_samples = oracle["diagnostics"]["num_samples"]
    detector_samples = detector["diagnostics"]["num_samples"]
    if oracle_samples != detector_samples:
        raise ValueError(
            "Oracle and detector evaluations must cover the same number of samples: "
            f"{oracle_samples} != {detector_samples}."
        )

    metrics: dict[str, dict[str, float]] = {}
    for key in OFFICIAL_KEYS:
        oracle_value = float(oracle["official"][key])
        detector_value = float(detector["official"][key])
        metrics[key] = {
            "oracle": oracle_value,
            "detector": detector_value,
            "detector_minus_oracle": detector_value - oracle_value,
        }
    for key in DIAGNOSTIC_KEYS:
        oracle_value = float(oracle["diagnostics"][key])
        detector_value = float(detector["diagnostics"][key])
        metrics[key] = {
            "oracle": oracle_value,
            "detector": detector_value,
            "detector_minus_oracle": detector_value - oracle_value,
        }

    unique_recall = float(proposal_recall["unique_target_recall"])
    expression_recall = float(proposal_recall["overall"]["target_recall"])
    official_f1_gap = metrics["F1_score"]["detector_minus_oracle"]
    return {
        "num_samples": oracle_samples,
        "oracle_feature_file": oracle.get("feature_file"),
        "detector_feature_file": detector.get("feature_file"),
        "metrics": metrics,
        "proposal_recall": {
            "iou_threshold": float(proposal_recall["iou_threshold"]),
            "unique_target_recall": unique_recall,
            "expression_weighted_target_recall": expression_recall,
            "full_target_coverage": float(
                proposal_recall["overall"]["full_target_coverage"]
            ),
        },
        "interpretation": {
            "official_f1_gap": official_f1_gap,
            "unique_target_miss_rate": 1.0 - unique_recall,
            "proposal_recall_is_not_direct_error_decomposition": True,
            "summary": (
                "Detector proposals retain nearly all validation targets, while the "
                "detector-candidate model loses substantially more official F1 than the "
                "proposal miss rate alone. The remaining gap therefore includes the "
                "harder detector distractor pool, duplicate/overlapping proposals, "
                "candidate ranking, and cardinality errors; it cannot be assigned only "
                "to missing proposals."
            ),
        },
    }


def format_report(comparison: dict[str, Any]) -> str:
    lines = [
        "Oracle vs detector candidate comparison",
        "=======================================",
        f"Samples: {comparison['num_samples']}",
        "",
        "metric                              oracle    detector       delta",
    ]
    for key, values in comparison["metrics"].items():
        lines.append(
            f"{key:<34} {values['oracle']:>8.6f}  "
            f"{values['detector']:>8.6f}  "
            f"{values['detector_minus_oracle']:>+10.6f}"
        )

    recall = comparison["proposal_recall"]
    lines.extend(
        [
            "",
            "[validation proposal recall at IoU 0.5]",
            f"unique_target_recall: {recall['unique_target_recall']:.6f}",
            "expression_weighted_target_recall: "
            f"{recall['expression_weighted_target_recall']:.6f}",
            f"full_target_coverage: {recall['full_target_coverage']:.6f}",
            "",
            "[interpretation]",
            comparison["interpretation"]["summary"],
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare oracle and detector candidate evaluation reports."
    )
    parser.add_argument("--oracle-json", required=True)
    parser.add_argument("--detector-json", required=True)
    parser.add_argument("--proposal-recall-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = build_comparison(
        _load_json(args.oracle_json),
        _load_json(args.detector_json),
        _load_json(args.proposal_recall_json),
    )

    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    report = format_report(comparison)
    output_txt.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
