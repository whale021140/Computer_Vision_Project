"""Validate and aggregate locked Stage 5 testA/testB evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.summarize_stage5_grid import MAIN_METRICS, aggregate, nested


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_test_grid(
    manifests: Sequence[dict[str, Any]],
    evaluations: Mapping[tuple[str, int, int, str, str], dict[str, Any]],
    feature_stats: Mapping[tuple[str, str], dict[str, Any]],
    representations: Sequence[str],
    percentages: Sequence[int],
    seeds: Sequence[int],
    splits: Sequence[str],
    checkpoint_policies: Sequence[str],
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
            raise ValueError(f"Duplicate validation manifest: {key}")
        by_cell[key] = manifest

    expected_cells = {
        (representation, percentage, seed)
        for representation in representations
        for percentage in percentages
        for seed in seeds
    }
    if set(by_cell) != expected_cells:
        raise ValueError("Locked validation manifests are incomplete.")
    expected_evaluations = {
        (*cell, split, checkpoint)
        for cell in expected_cells
        for split in splits
        for checkpoint in checkpoint_policies
    }
    if set(evaluations) != expected_evaluations:
        missing = sorted(expected_evaluations - set(evaluations))
        unexpected = sorted(set(evaluations) - expected_evaluations)
        raise ValueError(
            f"Test evaluations are incomplete; missing={missing}, "
            f"unexpected={unexpected}"
        )

    candidate_hashes = {}
    for split in splits:
        hashes = {
            feature_stats[(representation, split)]["candidate_file_sha256"]
            for representation in representations
        }
        if len(hashes) != 1:
            raise ValueError(f"Candidate hash mismatch across representations: {split}")
        candidate_hashes[split] = next(iter(hashes))

    expected_samples = {}
    for split in splits:
        counts = {
            int(feature_stats[(representation, split)]["num_samples"])
            for representation in representations
        }
        if len(counts) != 1:
            raise ValueError(f"Feature sample-count mismatch for {split}.")
        expected_samples[split] = next(iter(counts))

    for key, evaluation in evaluations.items():
        representation, percentage, seed, split, checkpoint_policy = key
        expected_checkpoint = str(
            Path("checkpoints/stage5")
            / f"{representation}_{percentage}pct_seed{seed}"
            / f"{checkpoint_policy}.pt"
        )
        if evaluation["checkpoint"] != expected_checkpoint:
            raise ValueError(f"Checkpoint mismatch for locked test cell {key}.")
        if float(evaluation["membership_threshold"]) != 0.5:
            raise ValueError(f"Threshold mismatch for locked test cell {key}.")
        if evaluation["representation"]["name"] != representation:
            raise ValueError(f"Representation mismatch for test cell {key}.")
        if int(evaluation["diagnostics"]["num_samples"]) != expected_samples[split]:
            raise ValueError(f"Full-split sample count mismatch for test cell {key}.")

    summary = []
    for split in splits:
        for checkpoint_policy in checkpoint_policies:
            for representation in representations:
                for percentage in percentages:
                    rows = [
                        evaluations[
                            (
                                representation,
                                percentage,
                                seed,
                                split,
                                checkpoint_policy,
                            )
                        ]
                        for seed in seeds
                    ]
                    metrics = {
                        name: aggregate(nested(row, path) for row in rows)
                        for name, path in MAIN_METRICS.items()
                    }
                    by_target_type = {
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
                    }
                    by_count_group = {
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
                    }
                    summary.append(
                        {
                            "split": split,
                            "checkpoint_policy": checkpoint_policy,
                            "representation": representation,
                            "percentage": percentage,
                            "seeds": list(seeds),
                            "metrics": metrics,
                            "by_target_type": by_target_type,
                            "by_count_group": by_count_group,
                        }
                    )

    return {
        "stage": 5,
        "evaluation_policy": (
            "full locked test evaluation; primary fixed epoch-20 last.pt and "
            "historical current-gRefCOCO-val best.pt sensitivity are both "
            "reported; membership threshold fixed at 0.5; no test selection"
        ),
        "standard_deviation": "sample standard deviation (n-1 denominator)",
        "representations": list(representations),
        "percentages": list(percentages),
        "seeds": list(seeds),
        "splits": list(splits),
        "checkpoint_policies": list(checkpoint_policies),
        "num_evaluations": len(evaluations),
        "expected_samples": expected_samples,
        "candidate_file_sha256": candidate_hashes,
        "git_commits": sorted(
            {manifest["git_commit"] for manifest in manifests}
        ),
        "summary": summary,
    }


def format_summary(result: dict[str, Any]) -> str:
    lines = [
        "Stage 5 Locked Test Summary",
        "===========================",
        f"Evaluations: {result['num_evaluations']}",
        f"Seeds: {result['seeds']}",
        "Std: sample standard deviation (n-1)",
    ]
    for split in result["splits"]:
        for checkpoint_policy in result["checkpoint_policies"]:
            lines.extend(
                [
                    "",
                    f"[{split} / {checkpoint_policy}]",
                    "representation | fraction | F1_score | T_acc | N_acc | mean_f1",
                    "--- | ---: | ---: | ---: | ---: | ---:",
                ]
            )
            for row in result["summary"]:
                if (
                    row["split"] != split
                    or row["checkpoint_policy"] != checkpoint_policy
                ):
                    continue
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
        default=["siglip2", "clip", "clip_dinov2"],
    )
    parser.add_argument("--percentages", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--splits", nargs="+", default=["testA", "testB"])
    parser.add_argument(
        "--checkpoint-policies", nargs="+", default=["last", "best"]
    )
    parser.add_argument(
        "--output-json",
        default="outputs/stage5/test_grid_summary.json",
    )
    parser.add_argument(
        "--output-txt",
        default="outputs/stage5/test_grid_summary.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_dir = Path(args.grid_dir)
    manifests = [load_json(path) for path in sorted(grid_dir.glob("*/manifest.json"))]
    evaluations = {}
    for representation in args.representations:
        for percentage in args.percentages:
            for seed in args.seeds:
                cell_dir = grid_dir / f"{representation}_{percentage}pct_seed{seed}"
                for split in args.splits:
                    for checkpoint in args.checkpoint_policies:
                        path = cell_dir / f"evaluation_{split}_{checkpoint}.json"
                        if path.exists():
                            evaluations[
                                (
                                    representation,
                                    percentage,
                                    seed,
                                    split,
                                    checkpoint,
                                )
                            ] = load_json(path)
    feature_stats = {
        (representation, split): load_json(
            f"outputs/stage5/{representation}_{split}_feature_stats.json"
        )
        for representation in args.representations
        for split in args.splits
    }
    result = summarize_test_grid(
        manifests,
        evaluations,
        feature_stats,
        args.representations,
        args.percentages,
        args.seeds,
        args.splits,
        args.checkpoint_policies,
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
