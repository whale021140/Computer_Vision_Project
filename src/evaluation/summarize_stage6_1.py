"""Summarize Stage 6.1 development-only mechanism pilots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compact(name: str, source: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "policy": row.get("policy", "cardinality-gated"),
        "membership_threshold": row["membership_threshold"],
        "count_logit_bias": row.get("count_logit_bias"),
        "count_macro_mean_f1": row["count_macro_mean_f1"],
        "official": row["official"],
        "by_count_group": row["by_count_group"],
    }


def main() -> None:
    inference = load(
        "outputs/stage6/inference_audit_siglip2_10pct_seed0.json"
    )
    baseline_calibration = load(
        "outputs/stage5_6/cells/"
        "siglip2_10pct_seed0_hierarchical/calibration_dev.json"
    )
    baseline = compact(
        "C2_hierarchical_lambda1",
        "accepted Stage 5.6 checkpoint",
        baseline_calibration["best"],
    )

    pilot_tags = {
        "C0_membership_only_lambda0": "membership_only_lambda0",
        "C1_flat_lambda1": "flat_lambda1",
        "hierarchical_lambda005": "hierarchical_lambda005",
        "hierarchical_lambda010": "hierarchical_lambda010",
        "hierarchical_lambda0125": "hierarchical_lambda0125",
        "hierarchical_lambda025": "hierarchical_lambda025",
        "hierarchical_lambda050": "hierarchical_lambda050",
        "hierarchical_lambda200": "hierarchical_lambda200",
    }
    trained = {}
    selections = {}
    for name, tag in pilot_tags.items():
        root = Path("outputs/stage6/pilots") / tag
        calibration = load(root / "calibration_dev.json")
        active = [
            key
            for key, value in calibration["best_on_boundary"].items()
            if value
        ]
        if active:
            raise ValueError(f"{tag} has active calibration boundaries: {active}")
        trained[name] = compact(
            name, f"new Stage 6 pilot {tag}", calibration["best"]
        )
        selections[name] = load(root / "selection_dev.json")["selected"][
            "epoch"
        ]

    inference_rows = {
        "C3_baseline_checkpoint_membership_only": compact(
            "C3_baseline_checkpoint_membership_only",
            "zero-training inference ablation",
            inference["policies"]["C3_membership_only"],
        ),
        "C4a_no_bias_best_threshold": compact(
            "C4a_no_bias_best_threshold",
            "zero-training inference ablation",
            inference["policies"]["C4a_no_bias_best_threshold"],
        ),
        "C4b_fully_neutral": compact(
            "C4b_fully_neutral",
            "zero-training inference ablation",
            inference["policies"]["C4b_fully_neutral"],
        ),
    }
    lambda_rows = {
        "0_membership_only_control": trained["C0_membership_only_lambda0"],
        "0.05": trained["hierarchical_lambda005"],
        "0.10": trained["hierarchical_lambda010"],
        "0.125": trained["hierarchical_lambda0125"],
        "0.25": trained["hierarchical_lambda025"],
        "0.5": trained["hierarchical_lambda050"],
        "1.0_baseline": baseline,
        "2.0": trained["hierarchical_lambda200"],
    }
    numeric_lambda_rows = {
        float(key.split("_", maxsplit=1)[0]): value
        for key, value in lambda_rows.items()
        if not key.startswith("0_membership")
    }
    best_lambda = max(
        numeric_lambda_rows.items(),
        key=lambda item: (
            item[1]["count_macro_mean_f1"],
            item[1]["official"]["F1_score"],
        ),
    )
    tested_lambdas = sorted(numeric_lambda_rows)
    best_index = tested_lambdas.index(best_lambda[0])
    search_is_bracketed = 0 < best_index < len(tested_lambdas) - 1
    if not search_is_bracketed:
        raise ValueError(
            f"best positive lambda {best_lambda[0]} is still on the search boundary"
        )
    confirmation = [
        "C0_membership_only_lambda0",
        "C1_flat_lambda1",
        "C2_hierarchical_lambda1",
    ]
    if best_lambda[0] != 1.0:
        confirmation.append(f"hierarchical_lambda{best_lambda[0]:g}")

    result = {
        "stage": "6.1",
        "scope": "development-only pilot; no Stage 6 test access",
        "trained_pilots": trained,
        "selected_epochs": selections,
        "accepted_baseline": baseline,
        "inference_ablations": inference_rows,
        "lambda_sweep": lambda_rows,
        "best_positive_lambda_on_development": best_lambda[0],
        "positive_lambda_search_bracketed": search_is_bracketed,
        "draft_three_seed_confirmation": confirmation,
        "status": "awaiting decision-gate review before confirmation training",
    }
    output_json = Path("outputs/stage6/stage6_1_pilot_summary.json")
    output_txt = Path("outputs/stage6/stage6_1_pilot_summary.txt")
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    all_rows = {
        **trained,
        "C2_hierarchical_lambda1": baseline,
        **inference_rows,
    }
    lines = [
        "Stage 6.1 Development-Only Pilot Summary",
        "========================================",
        "name | epoch | macro F1 | F1_score | T_acc | N_acc",
        "--- | ---: | ---: | ---: | ---: | ---:",
    ]
    for name, row in all_rows.items():
        epoch = selections.get(name, "-")
        lines.append(
            f"{name} | {epoch} | {row['count_macro_mean_f1']:.6f} | "
            f"{row['official']['F1_score']:.6f} | "
            f"{row['official']['T_acc']:.6f} | "
            f"{row['official']['N_acc']:.6f}"
        )
    lines.extend(
        [
            "",
            f"Best positive lambda on dev: {best_lambda[0]:g}",
            f"Positive-lambda search bracketed: {search_is_bracketed}",
            "Draft confirmation set: " + ", ".join(confirmation),
            "Status: stop for decision-gate review before test or confirmation.",
        ]
    )
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_txt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
