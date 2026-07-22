"""Validate and aggregate the Stage 5 multi-seed validation grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence


MAIN_METRICS = {
    "F1_score": ("official", "F1_score"),
    "T_acc": ("official", "T_acc"),
    "N_acc": ("official", "N_acc"),
    "mean_f1": ("diagnostics", "mean_f1"),
    "cardinality_accuracy": ("diagnostics", "cardinality_accuracy"),
    "false_grounding_rate": ("diagnostics", "false_grounding_rate"),
    "multi_target_mean_f1": ("diagnostics", "multi_target_mean_f1"),
    "multi_target_exact_accuracy": (
        "diagnostics",
        "multi_target_exact_accuracy",
    ),
}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested(record: dict[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for key in path:
        value = value[key]
    return value


def aggregate(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "values": values,
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "num_seeds": len(values),
    }


def summarize_grid(
    manifests: Sequence[dict[str, Any]],
    representations: Sequence[str],
    percentages: Sequence[int],
    seeds: Sequence[int],
) -> dict[str, Any]:
    by_cell = {}
    for manifest in manifests:
        cell = manifest["cell"]
        key = (
            str(cell["representation"]),
            int(cell["percentage"]),
            int(cell["seed"]),
        )
        if key in by_cell:
            raise ValueError(f"Duplicate Stage 5 cell: {key}")
        by_cell[key] = manifest

    expected = {
        (representation, int(percentage), int(seed))
        for representation in representations
        for percentage in percentages
        for seed in seeds
    }
    missing = sorted(expected - set(by_cell))
    unexpected = sorted(set(by_cell) - expected)
    if missing or unexpected:
        raise ValueError(
            f"Grid cells do not match the contract; missing={missing}, "
            f"unexpected={unexpected}"
        )

    candidate_hashes = {
        manifest["features"]["candidate_file_sha256"]
        for manifest in by_cell.values()
    }
    if len(candidate_hashes) != 1:
        raise ValueError("Representations do not share one candidate-file hash.")
    for percentage in percentages:
        for seed in seeds:
            split_hashes = {
                by_cell[(representation, percentage, seed)]["split"]["sha256"]
                for representation in representations
            }
            if len(split_hashes) != 1:
                raise ValueError(
                    f"Split hash mismatch for {percentage}% seed {seed}."
                )

    summary = []
    for representation in representations:
        for percentage in percentages:
            cells = [
                by_cell[(representation, percentage, seed)] for seed in seeds
            ]
            evaluations = [cell["validation_evaluation"] for cell in cells]
            metrics = {
                name: aggregate(nested(evaluation, path) for evaluation in evaluations)
                for name, path in MAIN_METRICS.items()
            }
            by_count_group = {}
            for group in ("0", "1", "2", "3+"):
                by_count_group[group] = {
                    metric: aggregate(
                        evaluation["by_count_group"][group][metric]
                        for evaluation in evaluations
                    )
                    for metric in (
                        "mean_f1",
                        "exact_set_accuracy",
                        "cardinality_accuracy",
                    )
                }
            summary.append(
                {
                    "representation": representation,
                    "percentage": percentage,
                    "seeds": list(seeds),
                    "metrics": metrics,
                    "by_count_group": by_count_group,
                }
            )

    return {
        "stage": 5,
        "split": "validation",
        "standard_deviation": "sample standard deviation (n-1 denominator)",
        "representations": list(representations),
        "percentages": list(percentages),
        "seeds": list(seeds),
        "candidate_file_sha256": next(iter(candidate_hashes)),
        "num_cells": len(by_cell),
        "git_commits": sorted(
            {manifest["git_commit"] for manifest in by_cell.values()}
        ),
        "summary": summary,
    }


def format_summary(result: dict[str, Any]) -> str:
    lines = [
        "Stage 5 Multi-Seed Validation Summary",
        "=====================================",
        f"Cells: {result['num_cells']}",
        f"Seeds: {result['seeds']}",
        "Std: sample standard deviation (n-1)",
        "",
        "representation | fraction | F1_score | T_acc | N_acc | mean_f1",
        "--- | ---: | ---: | ---: | ---: | ---:",
    ]
    for row in result["summary"]:
        metrics = row["metrics"]
        rendered = [
            f"{metrics[name]['mean']:.6f} ± {metrics[name]['std']:.6f}"
            for name in ("F1_score", "T_acc", "N_acc", "mean_f1")
        ]
        lines.append(
            f"{row['representation']} | {row['percentage']}% | "
            + " | ".join(rendered)
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", default="outputs/stage5/grid")
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["clip", "clip_dinov2", "siglip2"],
    )
    parser.add_argument("--percentages", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output-json",
        default="outputs/stage5/grid_summary_val.json",
    )
    parser.add_argument(
        "--output-txt",
        default="outputs/stage5/grid_summary_val.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.grid_dir).glob("*/manifest.json"))
    result = summarize_grid(
        [load_json(path) for path in paths],
        representations=args.representations,
        percentages=args.percentages,
        seeds=args.seeds,
    )
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = format_summary(result)
    output_txt.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
