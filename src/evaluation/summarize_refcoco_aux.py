"""Aggregate the Stage 5 RefCOCO single-target auxiliary validation audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPRESENTATIONS = ("siglip2", "clip", "clip_dinov2")
PERCENTAGES = (1, 5, 10)
SEEDS = (0, 1, 2)
CHECKPOINTS = ("best", "last")
METRICS = {
    "F1_score": ("official", "F1_score"),
    "T_acc": ("official", "T_acc"),
    "mean_f1": ("diagnostics", "mean_f1"),
    "localization_accuracy": (
        "diagnostics",
        "single_target_localization_accuracy",
    ),
    "cardinality_accuracy": ("diagnostics", "cardinality_accuracy"),
}


def metric(result: dict[str, Any], path: tuple[str, str]) -> float:
    return float(result[path[0]][path[1]])


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="outputs/stage5/refcoco_aux/grid",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/stage5/refcoco_aux/summary.json",
    )
    parser.add_argument(
        "--output-txt",
        default="outputs/stage5/refcoco_aux/summary.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    rows: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    raw: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)

    for representation in REPRESENTATIONS:
        for percentage in PERCENTAGES:
            for seed in SEEDS:
                tag = f"{representation}_{percentage}pct_seed{seed}"
                for checkpoint in CHECKPOINTS:
                    path = input_root / tag / f"evaluation_{checkpoint}.json"
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    result = json.loads(path.read_text(encoding="utf-8"))
                    if int(result["diagnostics"]["single_target_total"]) != 10834:
                        raise ValueError(f"Unexpected sample count in {path}.")
                    raw[(representation, percentage, checkpoint)].append(
                        {
                            "seed": seed,
                            "path": str(path),
                            "metrics": {
                                name: metric(result, metric_path)
                                for name, metric_path in METRICS.items()
                            },
                        }
                    )
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} auxiliary evaluations; first: {missing[0]}"
        )

    for representation in REPRESENTATIONS:
        for percentage in PERCENTAGES:
            checkpoint_rows: dict[str, dict[str, Any]] = {}
            for checkpoint in CHECKPOINTS:
                records = sorted(
                    raw[(representation, percentage, checkpoint)],
                    key=lambda record: record["seed"],
                )
                checkpoint_rows[checkpoint] = {
                    "seeds": [record["seed"] for record in records],
                    "files": [record["path"] for record in records],
                    "metrics": {
                        name: summary(
                            [record["metrics"][name] for record in records]
                        )
                        for name in METRICS
                    },
                }
            deltas = {
                name: summary(
                    [
                        last["metrics"][name] - best["metrics"][name]
                        for best, last in zip(
                            sorted(
                                raw[(representation, percentage, "best")],
                                key=lambda record: record["seed"],
                            ),
                            sorted(
                                raw[(representation, percentage, "last")],
                                key=lambda record: record["seed"],
                            ),
                        )
                    ]
                )
                for name in METRICS
            }
            rows.append(
                {
                    "representation": representation,
                    "percentage": percentage,
                    "checkpoints": checkpoint_rows,
                    "paired_delta_last_minus_best": deltas,
                }
            )

            suite_checkpoint_rows: dict[str, dict[str, Any]] = {}
            suite_seed_records: dict[str, list[dict[str, Any]]] = {}
            for checkpoint in CHECKPOINTS:
                records = []
                for seed in SEEDS:
                    tag = f"{representation}_{percentage}pct_seed{seed}"
                    aux_path = input_root / tag / f"evaluation_{checkpoint}.json"
                    if checkpoint == "best":
                        gref_path = (
                            Path("outputs/stage5/grid") / tag / "evaluation_val.json"
                        )
                    else:
                        gref_path = (
                            input_root / tag / "evaluation_gref_val_last.json"
                        )
                    if not gref_path.exists():
                        raise FileNotFoundError(
                            f"Missing composite-audit input: {gref_path}"
                        )
                    aux_result = json.loads(aux_path.read_text(encoding="utf-8"))
                    gref_result = json.loads(gref_path.read_text(encoding="utf-8"))
                    no_group = gref_result["by_target_type"]["no-target"]
                    single_group = aux_result["by_target_type"]["single-target"]
                    multi_group = gref_result["by_target_type"]["multi-target"]
                    per_type = {
                        "no-target": {
                            "mean_f1": float(no_group["mean_f1"]),
                            "exact_accuracy": float(no_group["exact_set_accuracy"]),
                            "cardinality_accuracy": float(
                                no_group["cardinality_accuracy"]
                            ),
                        },
                        "single-target": {
                            "mean_f1": float(single_group["mean_f1"]),
                            "exact_accuracy": float(
                                single_group["exact_set_accuracy"]
                            ),
                            "cardinality_accuracy": float(
                                single_group["cardinality_accuracy"]
                            ),
                        },
                        "multi-target": {
                            "mean_f1": float(multi_group["mean_f1"]),
                            "exact_accuracy": float(
                                multi_group["exact_set_accuracy"]
                            ),
                            "cardinality_accuracy": float(
                                multi_group["cardinality_accuracy"]
                            ),
                        },
                    }
                    records.append(
                        {
                            "seed": seed,
                            "gref_file": str(gref_path),
                            "single_file": str(aux_path),
                            "per_target_type": per_type,
                            "target_type_macro_mean_f1": mean(
                                group["mean_f1"] for group in per_type.values()
                            ),
                            "target_type_macro_exact_accuracy": mean(
                                group["exact_accuracy"] for group in per_type.values()
                            ),
                            "target_type_macro_cardinality_accuracy": mean(
                                group["cardinality_accuracy"]
                                for group in per_type.values()
                            ),
                        }
                    )
                suite_seed_records[checkpoint] = records
                suite_checkpoint_rows[checkpoint] = {
                    "seeds": list(SEEDS),
                    "target_type_macro_mean_f1": summary(
                        [record["target_type_macro_mean_f1"] for record in records]
                    ),
                    "target_type_macro_exact_accuracy": summary(
                        [
                            record["target_type_macro_exact_accuracy"]
                            for record in records
                        ]
                    ),
                    "target_type_macro_cardinality_accuracy": summary(
                        [
                            record["target_type_macro_cardinality_accuracy"]
                            for record in records
                        ]
                    ),
                    "records": records,
                }
            suite_rows.append(
                {
                    "representation": representation,
                    "percentage": percentage,
                    "checkpoints": suite_checkpoint_rows,
                    "paired_delta_last_minus_best": {
                        name: summary(
                            [
                                last[name] - best[name]
                                for best, last in zip(
                                    suite_seed_records["best"],
                                    suite_seed_records["last"],
                                )
                            ]
                        )
                        for name in (
                            "target_type_macro_mean_f1",
                            "target_type_macro_exact_accuracy",
                            "target_type_macro_cardinality_accuracy",
                        )
                    },
                }
            )

    output = {
        "dataset": "RefCOCO UNC val",
        "role": "single-target auxiliary validation audit",
        "selection_policy": "cardinality-threshold",
        "membership_threshold": 0.5,
        "cells": 54,
        "rows": rows,
        "composite_validation_suite": {
            "definition": (
                "gRefCOCO current val supplies no-target and multi-target; "
                "RefCOCO UNC val supplies single-target; target types are "
                "macro-averaged with equal weight"
            ),
            "rows": suite_rows,
        },
    }
    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "RefCOCO UNC Val Single-Target Auxiliary Audit",
        "==============================================",
        "Values are mean ± sample standard deviation across seeds.",
        "Delta is paired last.pt minus best.pt.",
        "",
        "representation | fraction | checkpoint | F1_score | T_acc | mean_f1 | localization | cardinality",
        "--- | ---: | --- | ---: | ---: | ---: | ---: | ---:",
    ]
    for row in rows:
        for checkpoint in CHECKPOINTS:
            metrics = row["checkpoints"][checkpoint]["metrics"]
            values = [
                f"{metrics[name]['mean']:.6f} ± {metrics[name]['sample_std']:.6f}"
                for name in METRICS
            ]
            lines.append(
                f"{row['representation']} | {row['percentage']}% | {checkpoint} | "
                + " | ".join(values)
            )
        deltas = row["paired_delta_last_minus_best"]
        values = [
            f"{deltas[name]['mean']:+.6f} ± {deltas[name]['sample_std']:.6f}"
            for name in METRICS
        ]
        lines.append(
            f"{row['representation']} | {row['percentage']}% | delta | "
            + " | ".join(values)
        )
    lines.extend(
        [
            "",
            "Composite Validation Suite (equal-weight no/single/multi)",
            "----------------------------------------------------------",
            "representation | fraction | checkpoint | macro mean F1 | macro exact | macro cardinality",
            "--- | ---: | --- | ---: | ---: | ---:",
        ]
    )
    suite_metric_names = (
        "target_type_macro_mean_f1",
        "target_type_macro_exact_accuracy",
        "target_type_macro_cardinality_accuracy",
    )
    for row in suite_rows:
        for checkpoint in CHECKPOINTS:
            values = [
                f"{row['checkpoints'][checkpoint][name]['mean']:.6f} ± "
                f"{row['checkpoints'][checkpoint][name]['sample_std']:.6f}"
                for name in suite_metric_names
            ]
            lines.append(
                f"{row['representation']} | {row['percentage']}% | {checkpoint} | "
                + " | ".join(values)
            )
        values = [
            f"{row['paired_delta_last_minus_best'][name]['mean']:+.6f} ± "
            f"{row['paired_delta_last_minus_best'][name]['sample_std']:.6f}"
            for name in suite_metric_names
        ]
        lines.append(
            f"{row['representation']} | {row['percentage']}% | delta | "
            + " | ".join(values)
        )
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
