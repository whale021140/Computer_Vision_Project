from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/final-project-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "stage7"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

SOURCE_PATHS = [
    "outputs/stage5_6/test_summary.json",
    "outputs/stage5_6/proposal_recall_feature_union.json",
    "outputs/stage3/oracle_vs_detector_val.json",
    "outputs/stage6/stage6_final_summary.json",
    "outputs/stage6/stage6_1_three_seed_confirmation.json",
    "outputs/stage6/stage6_2_input_ablation_summary.json",
    "outputs/stage6/candidate_cap_audit_dev.json",
    "outputs/stage6/multitarget_failure_diagnosis_dev.json",
    "outputs/stage6/counterfactual_local_audit.json",
    "outputs/stage6/final_test_lock.json",
]

REPRESENTATION_LABELS = {
    "clip": "CLIP",
    "clip_dinov2": "CLIP+DINOv2",
    "siglip2": "SigLIP 2",
}
FAMILY_LABELS = {
    "membership_only": "Membership only",
    "flat_lambda1": "Flat, λ=1",
    "hierarchical_lambda1_stage5_6_baseline": "Hierarchical, λ=1",
    "hierarchical_lambda010": "Final: hierarchical, λ=0.10 + pre-NMS",
}
COUNT_GROUPS = ["0", "1", "2", "3+"]


def load_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected object in {path}")
    assert_finite(value, str(path))
    return value


def assert_finite(value: Any, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {context}: {value}")
    if isinstance(value, dict):
        for key, nested in value.items():
            assert_finite(nested, f"{context}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_finite(nested, f"{context}[{index}]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_pm(metric: dict[str, Any], sd_key: str = "std") -> str:
    return f"{metric['mean']:.4f} ± {metric[sd_key]:.4f}"


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    materialized = [[str(value) for value in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in materialized)
    return "\n".join(lines) + "\n"


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "Matplotlib"},
    )
    fig.savefig(
        FIGURES / f"{name}.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "Final_Project Stage 7",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def validate_sources(
    stage5: dict[str, Any],
    stage6: dict[str, Any],
    lock: dict[str, Any],
) -> None:
    if stage5.get("calibration_revision") != "v2_wide":
        raise ValueError("Stage 7 requires the Stage 5.6 v2_wide summary")
    expected_representations = ["clip", "clip_dinov2", "siglip2"]
    if stage5.get("representations") != expected_representations:
        raise ValueError("Unexpected Stage 5.6 representations")
    if stage5.get("percentages") != [1, 5, 10]:
        raise ValueError("Unexpected Stage 5.6 fractions")
    if stage5.get("seeds") != [0, 1, 2]:
        raise ValueError("Unexpected Stage 5.6 seeds")
    if stage5.get("num_evaluations") != 54 or len(stage5.get("summary", [])) != 18:
        raise ValueError("Stage 5.6 main grid is incomplete")
    expected_families = {
        "membership_only",
        "flat_lambda1",
        "hierarchical_lambda010",
        "hierarchical_lambda1_stage5_6_baseline",
    }
    if set(stage6.get("families", {})) != expected_families:
        raise ValueError("Stage 6 final family set is incomplete")
    if "complete" not in stage6.get("status", ""):
        raise ValueError("Stage 6 final summary is not complete")
    if len(lock.get("new_test_cells", [])) != 9:
        raise ValueError("Stage 6 lock must contain nine new test cells")
    for cell in lock["new_test_cells"]:
        if cell["family"] == "hierarchical_lambda010":
            if cell["pre_nms_threshold"] != 0.3:
                raise ValueError("Final Stage 6 pre-NMS threshold changed")
            if cell["membership_threshold"] != 0.5:
                raise ValueError("Final Stage 6 membership threshold changed")


def build_main_table(stage5: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in stage5["summary"]:
        metrics = row["metrics"]
        rows.append(
            {
                "split": row["split"],
                "representation": REPRESENTATION_LABELS[row["representation"]],
                "fraction": row["percentage"],
                "f1_mean": metrics["F1_score"]["mean"],
                "f1_sd": metrics["F1_score"]["std"],
                "mean_set_f1_mean": metrics["mean_f1"]["mean"],
                "mean_set_f1_sd": metrics["mean_f1"]["std"],
                "t_acc_mean": metrics["T_acc"]["mean"],
                "t_acc_sd": metrics["T_acc"]["std"],
                "n_acc_mean": metrics["N_acc"]["mean"],
                "n_acc_sd": metrics["N_acc"]["std"],
            }
        )
    rows.sort(
        key=lambda item: (
            item["split"],
            ["CLIP", "CLIP+DINOv2", "SigLIP 2"].index(item["representation"]),
            item["fraction"],
        )
    )
    csv_rows = [
        [
            row["split"],
            row["representation"],
            row["fraction"],
            row["f1_mean"],
            row["f1_sd"],
            row["mean_set_f1_mean"],
            row["mean_set_f1_sd"],
            row["t_acc_mean"],
            row["t_acc_sd"],
            row["n_acc_mean"],
            row["n_acc_sd"],
        ]
        for row in rows
    ]
    headers = [
        "split",
        "representation",
        "fraction_percent",
        "F1_mean",
        "F1_sample_sd",
        "mean_set_F1_mean",
        "mean_set_F1_sample_sd",
        "T_acc_mean",
        "T_acc_sample_sd",
        "N_acc_mean",
        "N_acc_sample_sd",
    ]
    write_csv(TABLES / "stage5_6_main_results.csv", headers, csv_rows)
    md_rows = [
        [
            row["split"],
            row["representation"],
            f"{row['fraction']}%",
            f"{row['f1_mean']:.4f} ± {row['f1_sd']:.4f}",
            f"{row['mean_set_f1_mean']:.4f} ± {row['mean_set_f1_sd']:.4f}",
            f"{row['t_acc_mean']:.4f} ± {row['t_acc_sd']:.4f}",
            f"{row['n_acc_mean']:.4f} ± {row['n_acc_sd']:.4f}",
        ]
        for row in rows
    ]
    (TABLES / "stage5_6_main_results.md").write_text(
        markdown_table(
            [
                "Split",
                "Representation",
                "Fraction",
                "Official F1",
                "Mean set F1",
                "T_acc",
                "N_acc",
            ],
            md_rows,
        ),
        encoding="utf-8",
    )
    return rows


def build_final_system_table(stage6: dict[str, Any]) -> list[dict[str, Any]]:
    family_order = [
        "membership_only",
        "flat_lambda1",
        "hierarchical_lambda1_stage5_6_baseline",
        "hierarchical_lambda010",
    ]
    rows: list[dict[str, Any]] = []
    for family in family_order:
        for split in ("testA", "testB"):
            aggregate = stage6["families"][family][split]["aggregate"]
            rows.append(
                {
                    "family": family,
                    "label": FAMILY_LABELS[family],
                    "split": split,
                    "f1": aggregate["F1_score"],
                    "mean_set_f1": aggregate["mean_set_f1"],
                    "t_acc": aggregate["T_acc"],
                    "n_acc": aggregate["N_acc"],
                }
            )
    headers = [
        "family",
        "split",
        "F1_mean",
        "F1_sample_sd",
        "mean_set_F1_mean",
        "mean_set_F1_sample_sd",
        "T_acc_mean",
        "T_acc_sample_sd",
        "N_acc_mean",
        "N_acc_sample_sd",
    ]
    csv_rows = [
        [
            row["family"],
            row["split"],
            row["f1"]["mean"],
            row["f1"]["sample_sd"],
            row["mean_set_f1"]["mean"],
            row["mean_set_f1"]["sample_sd"],
            row["t_acc"]["mean"],
            row["t_acc"]["sample_sd"],
            row["n_acc"]["mean"],
            row["n_acc"]["sample_sd"],
        ]
        for row in rows
    ]
    write_csv(TABLES / "stage6_final_system_results.csv", headers, csv_rows)
    md_rows = [
        [
            row["label"],
            row["split"],
            format_pm(row["f1"], "sample_sd"),
            format_pm(row["mean_set_f1"], "sample_sd"),
            format_pm(row["t_acc"], "sample_sd"),
            format_pm(row["n_acc"], "sample_sd"),
        ]
        for row in rows
    ]
    (TABLES / "stage6_final_system_results.md").write_text(
        markdown_table(
            ["System", "Split", "Official F1", "Mean set F1", "T_acc", "N_acc"],
            md_rows,
        ),
        encoding="utf-8",
    )
    return rows


def aggregate_final_target_types(
    stage6: dict[str, Any],
    source_paths: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family = stage6["families"]["hierarchical_lambda010"]
    for split in ("testA", "testB"):
        files = family[split]["per_seed_files"]
        if len(files) != 3:
            raise ValueError(f"Expected three final files for {split}")
        evaluations = []
        for path in files:
            source_paths.append(path)
            evaluations.append(load_json(path))
        for target_type in ("no-target", "single-target", "multi-target"):
            sample_counts = {
                int(item["by_target_type"][target_type]["num_samples"])
                for item in evaluations
            }
            if len(sample_counts) != 1:
                raise ValueError(f"Inconsistent sample count for {split}/{target_type}")
            row: dict[str, Any] = {
                "split": split,
                "target_type": target_type,
                "num_samples": sample_counts.pop(),
            }
            for metric in ("mean_f1", "exact_set_accuracy", "cardinality_accuracy"):
                values = [
                    float(item["by_target_type"][target_type][metric])
                    for item in evaluations
                ]
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_sd"] = stdev(values)
            rows.append(row)
    headers = [
        "split",
        "target_type",
        "num_samples",
        "mean_F1_mean",
        "mean_F1_sample_sd",
        "exact_set_accuracy_mean",
        "exact_set_accuracy_sample_sd",
        "cardinality_accuracy_mean",
        "cardinality_accuracy_sample_sd",
    ]
    csv_rows = [
        [
            row["split"],
            row["target_type"],
            row["num_samples"],
            row["mean_f1_mean"],
            row["mean_f1_sd"],
            row["exact_set_accuracy_mean"],
            row["exact_set_accuracy_sd"],
            row["cardinality_accuracy_mean"],
            row["cardinality_accuracy_sd"],
        ]
        for row in rows
    ]
    write_csv(TABLES / "final_target_type_breakdown.csv", headers, csv_rows)
    md_rows = [
        [
            row["split"],
            row["target_type"],
            row["num_samples"],
            f"{row['mean_f1_mean']:.4f} ± {row['mean_f1_sd']:.4f}",
            f"{row['exact_set_accuracy_mean']:.4f} ± {row['exact_set_accuracy_sd']:.4f}",
            f"{row['cardinality_accuracy_mean']:.4f} ± {row['cardinality_accuracy_sd']:.4f}",
        ]
        for row in rows
    ]
    (TABLES / "final_target_type_breakdown.md").write_text(
        markdown_table(
            [
                "Split",
                "Target type",
                "Samples",
                "Mean F1",
                "Exact-set accuracy",
                "Cardinality accuracy",
            ],
            md_rows,
        ),
        encoding="utf-8",
    )
    return rows


def build_diagnostic_table(
    proposal: dict[str, Any],
    oracle: dict[str, Any],
    failures: dict[str, Any],
) -> None:
    rows: list[list[Any]] = [
        ["Feature-union unique target recall", proposal["unique_target_recall"]],
        ["Feature-union expression-weighted target recall", proposal["overall"]["target_recall"]],
        ["Feature-union full-target coverage", proposal["overall"]["full_target_coverage"]],
        ["Stage 3 oracle official F1", oracle["metrics"]["F1_score"]["oracle"]],
        ["Stage 3 detector official F1", oracle["metrics"]["F1_score"]["detector"]],
        [
            "Stage 3 detector minus oracle F1",
            oracle["metrics"]["F1_score"]["detector_minus_oracle"],
        ],
    ]
    original = failures["exact_count_failures_by_policy"]["none"]
    repaired = failures["exact_count_failures_by_policy"]["pre_nms_0.3_thr_0.5"]
    for group in ("3", "4", "5", "6+"):
        rows.append(
            [
                f"Exact {group} success, original",
                original[group]["fractions"]["success"],
            ]
        )
        rows.append(
            [
                f"Exact {group} success, repaired",
                repaired[group]["fractions"]["success"],
            ]
        )
    write_csv(TABLES / "proposal_and_exact_count_diagnostics.csv", ["diagnostic", "value"], rows)
    (TABLES / "proposal_and_exact_count_diagnostics.md").write_text(
        markdown_table(
            ["Diagnostic", "Value"],
            [[label, f"{float(value):.6f}"] for label, value in rows],
        ),
        encoding="utf-8",
    )


def plot_scaling(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    colors = {"CLIP": "#4C78A8", "CLIP+DINOv2": "#F58518", "SigLIP 2": "#54A24B"}
    for axis, split in zip(axes, ("testA", "testB")):
        for representation in ("CLIP", "CLIP+DINOv2", "SigLIP 2"):
            selected = [
                row
                for row in rows
                if row["split"] == split and row["representation"] == representation
            ]
            selected.sort(key=lambda item: item["fraction"])
            axis.errorbar(
                [row["fraction"] for row in selected],
                [row["f1_mean"] for row in selected],
                yerr=[row["f1_sd"] for row in selected],
                marker="o",
                linewidth=2,
                capsize=3,
                label=representation,
                color=colors[representation],
            )
        axis.set_title(split)
        axis.set_xlabel("Training supervision (%)")
        axis.set_xticks([1, 5, 10])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Official GREC F1")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Stage 5.6: supervision scaling across frozen representations")
    save_figure(fig, "supervision_scaling")


def plot_final_system(rows: list[dict[str, Any]]) -> None:
    family_order = [
        "membership_only",
        "flat_lambda1",
        "hierarchical_lambda1_stage5_6_baseline",
        "hierarchical_lambda010",
    ]
    labels = [FAMILY_LABELS[family] for family in family_order]
    x = np.arange(len(family_order))
    width = 0.36
    fig, axis = plt.subplots(figsize=(11, 4.8))
    for offset, split, color in ((-width / 2, "testA", "#4C78A8"), (width / 2, "testB", "#F58518")):
        selected = {
            row["family"]: row
            for row in rows
            if row["split"] == split
        }
        values = [selected[family]["f1"]["mean"] for family in family_order]
        errors = [selected[family]["f1"]["sample_sd"] for family in family_order]
        axis.bar(x + offset, values, width, yerr=errors, capsize=3, label=split, color=color)
    axis.set_xticks(x, labels, rotation=12, ha="right")
    axis.set_ylabel("Official GREC F1")
    axis.set_title("Stage 6: locked final-system comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save_figure(fig, "final_system_comparison")


def plot_count_groups(stage6: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    width = 0.36
    x = np.arange(len(COUNT_GROUPS))
    for axis, split in zip(axes, ("testA", "testB")):
        for offset, family, label, color in (
            (-width / 2, "hierarchical_lambda1_stage5_6_baseline", "Stage 5.6 baseline", "#9C9C9C"),
            (width / 2, "hierarchical_lambda010", "Stage 6 final", "#54A24B"),
        ):
            metrics = stage6["families"][family][split]["aggregate"]["by_count_group_mean_f1"]
            values = [metrics[group]["mean"] for group in COUNT_GROUPS]
            errors = [metrics[group]["sample_sd"] for group in COUNT_GROUPS]
            axis.bar(x + offset, values, width, yerr=errors, capsize=3, label=label, color=color)
        axis.set_title(split)
        axis.set_xticks(x, COUNT_GROUPS)
        axis.set_xlabel("True target-count group")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean set F1")
    axes[1].legend(frameon=False)
    fig.suptitle("Count-group performance before and after the Stage 6 repair")
    save_figure(fig, "count_group_breakdown")


def plot_target_types(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    target_types = ["no-target", "single-target", "multi-target"]
    colors = ["#E45756", "#4C78A8", "#54A24B"]
    for axis, split in zip(axes, ("testA", "testB")):
        selected = {row["target_type"]: row for row in rows if row["split"] == split}
        values = [selected[target]["mean_f1_mean"] for target in target_types]
        errors = [selected[target]["mean_f1_sd"] for target in target_types]
        axis.bar(target_types, values, yerr=errors, capsize=3, color=colors)
        axis.set_title(split)
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean set F1")
    fig.suptitle("Stage 6 final system by target type")
    save_figure(fig, "final_target_type_breakdown")


def plot_proposal_and_repair(
    proposal: dict[str, Any],
    failures: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    groups = ["1", "2", "3+"]
    target_recall = [proposal["by_count_group"][group]["target_recall"] for group in groups]
    coverage = [proposal["by_count_group"][group]["full_target_coverage"] for group in groups]
    x = np.arange(len(groups))
    width = 0.36
    axes[0].bar(x - width / 2, target_recall, width, label="Target recall", color="#4C78A8")
    axes[0].bar(x + width / 2, coverage, width, label="Full-target coverage", color="#F58518")
    axes[0].set_xticks(x, groups)
    axes[0].set_ylim(0.9, 1.005)
    axes[0].set_xlabel("True target-count group")
    axes[0].set_ylabel("Proposal metric")
    axes[0].set_title("Frozen proposal coverage (IoU ≥ 0.5)")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    exact_groups = ["3", "4", "5", "6+"]
    original = failures["exact_count_failures_by_policy"]["none"]
    repaired = failures["exact_count_failures_by_policy"]["pre_nms_0.3_thr_0.5"]
    original_values = [original[group]["fractions"]["success"] for group in exact_groups]
    repaired_values = [repaired[group]["fractions"]["success"] for group in exact_groups]
    x = np.arange(len(exact_groups))
    axes[1].bar(x - width / 2, original_values, width, label="Original inference", color="#9C9C9C")
    axes[1].bar(x + width / 2, repaired_values, width, label="Pre-NMS repair", color="#54A24B")
    axes[1].set_xticks(x, exact_groups)
    axes[1].set_xlabel("Exact true target count")
    axes[1].set_ylabel("Exact-set success rate")
    axes[1].set_title("Seed-0 dev exact-count repair")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    save_figure(fig, "proposal_and_exact_count_diagnostics")


def plot_core_ablations(
    confirmation: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    labels = [
        "Membership only",
        "Flat λ=1",
        "Hierarchical λ=1",
        "Hierarchical λ=0.10",
        "No coordinates\n(seed 0)",
        "No explicit similarity\n(seed 0)",
    ]
    values = [
        confirmation["families"]["C0_membership_only"]["aggregate"]["count_macro_mean_f1"]["mean"],
        confirmation["families"]["C1_flat_lambda1"]["aggregate"]["count_macro_mean_f1"]["mean"],
        confirmation["families"]["C2_hierarchical_lambda1"]["aggregate"]["count_macro_mean_f1"]["mean"],
        confirmation["families"]["C5_hierarchical_lambda010"]["aggregate"]["count_macro_mean_f1"]["mean"],
        inputs["variants"]["no_box_coordinates"]["count_macro_mean_f1"],
        inputs["variants"]["no_explicit_similarity"]["count_macro_mean_f1"],
    ]
    colors = ["#9C9C9C", "#F58518", "#72B7B2", "#54A24B", "#E45756", "#B279A2"]
    fig, axis = plt.subplots(figsize=(11, 4.8))
    axis.bar(np.arange(len(labels)), values, color=colors)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=12, ha="right")
    axis.set_ylabel("Development count-macro mean F1")
    axis.set_title("Core Stage 6 mechanism and input ablations")
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, "core_ablations")


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_manifest(source_paths: list[str]) -> None:
    generated_paths = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "stage": "7 final deliverables",
        "policy": (
            "Generated only from frozen Stage 5.6/6 artifacts; no model training, "
            "calibration, or test inference."
        ),
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_branch": git_value("branch", "--show-current"),
        "working_tree_dirty_during_build": bool(git_value("status", "--porcelain")),
        "command": (
            "conda run --no-capture-output -n ece485 "
            "python -m src.reporting.build_stage7_deliverables"
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "inputs": [
            {"path": path, "sha256": sha256(ROOT / path)}
            for path in sorted(set(source_paths))
        ],
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in generated_paths
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = {path: load_json(path) for path in SOURCE_PATHS}
    stage5 = data["outputs/stage5_6/test_summary.json"]
    stage6 = data["outputs/stage6/stage6_final_summary.json"]
    lock = data["outputs/stage6/final_test_lock.json"]
    validate_sources(stage5, stage6, lock)

    main_rows = build_main_table(stage5)
    final_rows = build_final_system_table(stage6)
    all_sources = list(SOURCE_PATHS)
    target_rows = aggregate_final_target_types(stage6, all_sources)
    build_diagnostic_table(
        data["outputs/stage5_6/proposal_recall_feature_union.json"],
        data["outputs/stage3/oracle_vs_detector_val.json"],
        data["outputs/stage6/multitarget_failure_diagnosis_dev.json"],
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_scaling(main_rows)
    plot_final_system(final_rows)
    plot_count_groups(stage6)
    plot_target_types(target_rows)
    plot_proposal_and_repair(
        data["outputs/stage5_6/proposal_recall_feature_union.json"],
        data["outputs/stage6/multitarget_failure_diagnosis_dev.json"],
    )
    plot_core_ablations(
        data["outputs/stage6/stage6_1_three_seed_confirmation.json"],
        data["outputs/stage6/stage6_2_input_ablation_summary.json"],
    )
    write_manifest(all_sources)
    print(f"Stage 7 deliverables written to {OUTPUT}")


if __name__ == "__main__":
    main()
