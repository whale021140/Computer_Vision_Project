"""Validate and aggregate the complete Stage 5.5 evaluation grid."""

from __future__ import annotations

import json
import math
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


REPRESENTATIONS = ("clip", "clip_dinov2", "siglip2")
SEEDS = (0, 1, 2)
SPLITS = ("shadow_dev", "gref_val", "refcoco_aux", "testA", "testB")
COUNT_GROUPS = ("0", "1", "2", "3+")
TARGET_TYPES = ("no-target", "single-target", "multi-target")
AVAILABILITY_NOTES = {
    "gref_val": "count-1/single-target unavailable (zero samples)",
    "refcoco_aux": "only count-1/single-target available; N_acc is not applicable",
}
METRICS = {
    "F1_score": ("official", "F1_score"),
    "T_acc": ("official", "T_acc"),
    "N_acc": ("official", "N_acc"),
    "mean_f1": ("diagnostics", "mean_f1"),
    "cardinality_accuracy": ("diagnostics", "cardinality_accuracy"),
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(row: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = row
    for key in path:
        value = value[key]
    return float(value)


def aggregate(values: Iterable[float]) -> dict[str, Any]:
    items = [float(value) for value in values]
    if not items or not all(math.isfinite(value) for value in items):
        raise ValueError(f"Invalid values: {items}")
    return {
        "values": items,
        "mean": statistics.fmean(items),
        "std": statistics.stdev(items) if len(items) > 1 else 0.0,
        "num_seeds": len(items),
    }


def main() -> None:
    pilot = load(Path("outputs/stage5_5/pilot_selection.json"))
    variant = str(pilot["selected_variant"])
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    calibrations: dict[tuple[str, int], dict[str, Any]] = {}

    for representation in REPRESENTATIONS:
        for seed in SEEDS:
            tag = f"{representation}_10pct_seed{seed}_{variant}"
            root = Path("outputs/stage5_5/cells") / tag
            calibration = load(root / "calibration_shadow.json")
            shadow = calibration["best"]
            rows[(representation, seed, "shadow_dev")] = shadow
            calibrations[(representation, seed)] = {
                "selected_epoch": load(root / "selection_shadow.json")["selected"]["epoch"],
                "membership_threshold": shadow["membership_threshold"],
                "count_logit_bias": shadow["count_logit_bias"],
                "count_macro_mean_f1": shadow["count_macro_mean_f1"],
            }
            for split in SPLITS[1:]:
                evaluation = load(root / f"evaluation_{split}.json")
                if evaluation["representation"]["name"] != representation:
                    raise ValueError(f"Representation mismatch for {tag}/{split}")
                if int(evaluation["diagnostics"]["num_samples"]) <= 0:
                    raise ValueError(f"Empty evaluation for {tag}/{split}")
                rows[(representation, seed, split)] = evaluation

    expected = {
        (representation, seed, split)
        for representation in REPRESENTATIONS
        for seed in SEEDS
        for split in SPLITS
    }
    if set(rows) != expected:
        raise ValueError("Stage 5.5 grid is incomplete.")

    summary = []
    for split in SPLITS:
        for representation in REPRESENTATIONS:
            group = [rows[(representation, seed, split)] for seed in SEEDS]
            entry: dict[str, Any] = {
                "split": split,
                "representation": representation,
                "seeds": list(SEEDS),
                "metrics": {
                    name: aggregate(nested(row, path) for row in group)
                    for name, path in METRICS.items()
                },
            }
            count_key_sets = [
                {
                    count
                    for count, metrics in row.get("by_count_group", {}).items()
                    if int(metrics.get("num_samples", 0)) > 0
                }
                for row in group
            ]
            if any(keys != count_key_sets[0] for keys in count_key_sets[1:]):
                raise ValueError(
                    f"Count-group mismatch across seeds for {split}/{representation}: "
                    f"{count_key_sets}"
                )
            available_counts = [count for count in COUNT_GROUPS if count in count_key_sets[0]]
            entry["available_count_groups"] = available_counts
            entry["unavailable_count_groups"] = [
                count for count in COUNT_GROUPS if count not in count_key_sets[0]
            ]
            entry["by_count_group"] = {
                count: {
                    metric: aggregate(
                        float(row["by_count_group"][count][metric]) for row in group
                    )
                    for metric in ("mean_f1", "exact_set_accuracy", "cardinality_accuracy")
                }
                for count in available_counts
            }
            target_key_sets = [
                {
                    target
                    for target, metrics in row.get("by_target_type", {}).items()
                    if int(metrics.get("num_samples", 0)) > 0
                }
                for row in group
            ]
            if any(keys != target_key_sets[0] for keys in target_key_sets[1:]):
                raise ValueError(
                    f"Target-type mismatch across seeds for {split}/{representation}: "
                    f"{target_key_sets}"
                )
            available_targets = [
                target for target in TARGET_TYPES if target in target_key_sets[0]
            ]
            entry["available_target_types"] = available_targets
            entry["unavailable_target_types"] = [
                target for target in TARGET_TYPES if target not in target_key_sets[0]
            ]
            entry["by_target_type"] = {
                target: {
                    metric: aggregate(
                        float(row["by_target_type"][target][metric]) for row in group
                    )
                    for metric in ("mean_f1", "exact_set_accuracy", "cardinality_accuracy")
                }
                for target in available_targets
            }
            summary.append(entry)

    paired = []
    for split in SPLITS:
        for left, right in combinations(REPRESENTATIONS, 2):
            paired.append(
                {
                    "split": split,
                    "left": left,
                    "right": right,
                    "difference": "left_minus_right",
                    "metrics": {
                        name: aggregate(
                            nested(rows[(left, seed, split)], path)
                            - nested(rows[(right, seed, split)], path)
                            for seed in SEEDS
                        )
                        for name, path in METRICS.items()
                    },
                }
            )

    stage5 = load(Path("outputs/stage5/test_grid_summary.json"))
    comparison = []
    for split in ("testA", "testB"):
        for representation in REPRESENTATIONS:
            old = next(
                row for row in stage5["summary"]
                if row["split"] == split
                and row["checkpoint_policy"] == "last"
                and row["representation"] == representation
                and int(row["percentage"]) == 10
            )
            new = next(
                row for row in summary
                if row["split"] == split and row["representation"] == representation
            )
            comparison.append(
                {
                    "split": split,
                    "representation": representation,
                    "note": "aggregate comparison; Stage 5.5 uses shadow-dev-excluded train splits",
                    "stage5": {name: old["metrics"][name] for name in METRICS},
                    "stage5_5": {name: new["metrics"][name] for name in METRICS},
                    "mean_delta": {
                        name: new["metrics"][name]["mean"] - old["metrics"][name]["mean"]
                        for name in METRICS
                    },
                }
            )

    result = {
        "stage": "5.5",
        "selected_variant": variant,
        "selection_and_calibration_source": "locked shadow-dev only",
        "test_usage": "post-hoc audit; Stage 5 locked test grid remains primary",
        "standard_deviation": "sample standard deviation (n-1 denominator)",
        "representations": list(REPRESENTATIONS),
        "seeds": list(SEEDS),
        "splits": list(SPLITS),
        "num_models": len(REPRESENTATIONS) * len(SEEDS),
        "num_external_evaluations": len(REPRESENTATIONS) * len(SEEDS) * (len(SPLITS) - 1),
        "calibrations": {f"{rep}_seed{seed}": value for (rep, seed), value in calibrations.items()},
        "pilot_selection": pilot,
        "summary": summary,
        "paired_differences": paired,
        "stage5_comparison": comparison,
    }
    output_json = Path("outputs/stage5_5/summary.json")
    output_txt = Path("outputs/stage5_5/summary.txt")
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Stage 5.5 Enhanced-System Summary",
        "=================================",
        f"Selected variant: {variant}",
        "selection/calibration: locked shadow-dev only",
        "metric format: mean +/- sample std across three seeds",
    ]
    for split in SPLITS:
        lines.extend(["", f"[{split}]", "representation | F1_score | T_acc | N_acc | mean_f1 | card_acc", "--- | ---: | ---: | ---: | ---: | ---:"])
        for row in summary:
            if row["split"] != split:
                continue
            values = [
                f"{row['metrics'][name]['mean']:.6f} +/- {row['metrics'][name]['std']:.6f}"
                for name in METRICS
            ]
            lines.append(f"{row['representation']} | " + " | ".join(values))
        if split in AVAILABILITY_NOTES:
            lines.append(f"availability note: {AVAILABILITY_NOTES[split]}")
    lines.extend(["", "[Stage 5.5 minus Stage 5, aggregate means; not strictly paired]", "split | representation | dF1 | dMeanF1 | dCardAcc", "--- | --- | ---: | ---: | ---:"])
    for row in comparison:
        delta = row["mean_delta"]
        lines.append(
            f"{row['split']} | {row['representation']} | {delta['F1_score']:+.6f} | "
            f"{delta['mean_f1']:+.6f} | {delta['cardinality_accuracy']:+.6f}"
        )
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_txt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
