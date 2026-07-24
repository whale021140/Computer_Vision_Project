"""Aggregate the single-policy Stage 5.6 testA/testB benchmark."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any

from src.evaluation.summarize_stage5_grid import MAIN_METRICS, aggregate, nested


REPRESENTATIONS = ("clip", "clip_dinov2", "siglip2")
PERCENTAGES = (1, 5, 10)
SEEDS = (0, 1, 2)
SPLITS = ("testA", "testB")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    pilot = load_json("outputs/stage5_6/pilot_selection.json")
    calibration_protocol = load_json(
        "outputs/stage5_6/calibration_revision_v2_protocol.json"
    )
    calibration_audit = load_json(
        "outputs/stage5_6/calibration_v2_audit.json"
    )
    if not calibration_audit.get("test_gate_passed"):
        raise ValueError("Stage 5.6 v2 calibration audit did not pass")
    variant = pilot["selected_variant"]
    evaluations = {}
    calibration_revisions = set()
    for representation in REPRESENTATIONS:
        for percentage in PERCENTAGES:
            for seed in SEEDS:
                tag = f"{representation}_{percentage}pct_seed{seed}_{variant}"
                calibration_path = (
                    Path("outputs/stage5_6/cells")
                    / tag
                    / "calibration_dev.json"
                )
                calibration = load_json(calibration_path)
                calibration_revisions.add(
                    calibration.get("calibration_revision")
                )
                for split in SPLITS:
                    path = (
                        Path("outputs/stage5_6/cells")
                        / tag
                        / f"evaluation_{split}.json"
                    )
                    if not path.exists():
                        raise FileNotFoundError(path)
                    row = load_json(path)
                    if row["representation"]["name"] != representation:
                        raise ValueError(f"Representation mismatch in {path}")
                    evaluations[(representation, percentage, seed, split)] = row
    if calibration_revisions != {"v2_wide"}:
        raise ValueError(
            "Expected only v2_wide calibrations, found "
            f"{sorted(str(value) for value in calibration_revisions)}"
        )

    summary = []
    for split in SPLITS:
        for representation in REPRESENTATIONS:
            for percentage in PERCENTAGES:
                rows = [
                    evaluations[(representation, percentage, seed, split)]
                    for seed in SEEDS
                ]
                summary.append(
                    {
                        "split": split,
                        "representation": representation,
                        "percentage": percentage,
                        "seeds": list(SEEDS),
                        "metrics": {
                            name: aggregate(nested(row, path) for row in rows)
                            for name, path in MAIN_METRICS.items()
                        },
                        "by_target_type": {
                            target_type: {
                                metric: aggregate(
                                    row["by_target_type"][target_type][metric]
                                    for row in rows
                                )
                                for metric in (
                                    "mean_f1",
                                    "exact_set_accuracy",
                                    "cardinality_accuracy",
                                )
                            }
                            for target_type in (
                                "no-target",
                                "single-target",
                                "multi-target",
                            )
                        },
                        "by_count_group": {
                            group: {
                                metric: aggregate(
                                    row["by_count_group"][group][metric]
                                    for row in rows
                                )
                                for metric in (
                                    "mean_f1",
                                    "exact_set_accuracy",
                                    "cardinality_accuracy",
                                )
                            }
                            for group in ("0", "1", "2", "3+")
                        },
                    }
                )

    paired = []
    for split in SPLITS:
        for percentage in PERCENTAGES:
            for left, right in combinations(REPRESENTATIONS, 2):
                left_rows = [
                    evaluations[(left, percentage, seed, split)] for seed in SEEDS
                ]
                right_rows = [
                    evaluations[(right, percentage, seed, split)] for seed in SEEDS
                ]
                paired.append(
                    {
                        "split": split,
                        "percentage": percentage,
                        "left_representation": left,
                        "right_representation": right,
                        "difference": "left_minus_right",
                        "metrics": {
                            name: aggregate(
                                float(nested(left_row, path))
                                - float(nested(right_row, path))
                                for left_row, right_row in zip(left_rows, right_rows)
                            )
                            for name, path in MAIN_METRICS.items()
                        },
                    }
                )

    result = {
        "stage": "5.6",
        "protocol": (
            "single development-selected policy with revised wide "
            "development-only calibration"
        ),
        "calibration_revision": "v2_wide",
        "calibration_protocol": calibration_protocol,
        "calibration_audit": {
            "path": "outputs/stage5_6/calibration_v2_audit.json",
            "num_unique_calibrations": calibration_audit[
                "num_unique_calibrations"
            ],
            "num_validated": calibration_audit["num_validated"],
            "test_gate_passed": calibration_audit["test_gate_passed"],
            "best_setting_counts": calibration_audit[
                "best_setting_counts"
            ],
        },
        "selected_variant": variant,
        "representations": list(REPRESENTATIONS),
        "percentages": list(PERCENTAGES),
        "seeds": list(SEEDS),
        "splits": list(SPLITS),
        "num_evaluations": len(evaluations),
        "standard_deviation": "sample standard deviation (n-1 denominator)",
        "summary": summary,
        "paired_differences": paired,
    }
    output_json = Path("outputs/stage5_6/test_summary.json")
    output_txt = Path("outputs/stage5_6/test_summary.txt")
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Stage 5.6 Final Benchmark",
        "=========================",
        f"Selected recipe: {variant}",
        "Calibration revision: v2_wide",
        (
            "Calibration audit: "
            f"{calibration_audit['num_validated']}/"
            f"{calibration_audit['num_unique_calibrations']} valid; "
            f"test gate passed={calibration_audit['test_gate_passed']}"
        ),
        f"Evaluations: {len(evaluations)}",
    ]
    for split in SPLITS:
        lines.extend(
            [
                "",
                f"[{split}]",
                "representation | fraction | F1_score | T_acc | N_acc | mean_f1",
                "--- | ---: | ---: | ---: | ---: | ---:",
            ]
        )
        for row in summary:
            if row["split"] != split:
                continue
            rendered = [
                f"{row['metrics'][name]['mean']:.6f} ± "
                f"{row['metrics'][name]['std']:.6f}"
                for name in ("F1_score", "T_acc", "N_acc", "mean_f1")
            ]
            lines.append(
                f"{row['representation']} | {row['percentage']}% | "
                + " | ".join(rendered)
            )
    lines.extend(
        [
            "",
            "[paired differences; left minus right]",
            "split | fraction | comparison | F1_score | mean_f1",
            "--- | ---: | --- | ---: | ---:",
        ]
    )
    for row in paired:
        metrics = row["metrics"]
        lines.append(
            f"{row['split']} | {row['percentage']}% | "
            f"{row['left_representation']} - {row['right_representation']} | "
            f"{metrics['F1_score']['mean']:+.6f} ± "
            f"{metrics['F1_score']['std']:.6f} | "
            f"{metrics['mean_f1']['mean']:+.6f} ± "
            f"{metrics['mean_f1']['std']:.6f}"
        )
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_txt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
