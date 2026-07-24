"""Summarize the compact Stage 6.2 input-information ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compact(name: str, calibration: Path, selection: Path) -> dict[str, Any]:
    payload = load(calibration)
    active = [
        key for key, value in payload["best_on_boundary"].items() if value
    ]
    if active:
        raise ValueError(f"{name} calibration boundaries: {active}")
    best = payload["best"]
    return {
        "name": name,
        "selected_epoch": load(selection)["selected"]["epoch"],
        "count_macro_mean_f1": best["count_macro_mean_f1"],
        "official": best["official"],
        "by_count_group": best["by_count_group"],
        "membership_threshold": best["membership_threshold"],
        "count_logit_bias": best["count_logit_bias"],
    }


def main() -> None:
    full = compact(
        "full_input",
        Path(
            "outputs/stage6/pilots/"
            "hierarchical_lambda010/calibration_dev.json"
        ),
        Path(
            "outputs/stage6/pilots/"
            "hierarchical_lambda010/selection_dev.json"
        ),
    )
    variants = {}
    for tag in ("no_box_coordinates", "no_explicit_similarity"):
        root = Path("outputs/stage6/input_ablations") / tag
        variants[tag] = compact(
            tag,
            root / "calibration_dev.json",
            root / "selection_dev.json",
        )

    deltas = {
        name: {
            "count_macro_mean_f1": (
                row["count_macro_mean_f1"] - full["count_macro_mean_f1"]
            ),
            "official_F1_score": (
                row["official"]["F1_score"] - full["official"]["F1_score"]
            ),
            "by_count_group_mean_f1": {
                group: (
                    row["by_count_group"][group]["mean_f1"]
                    - full["by_count_group"][group]["mean_f1"]
                )
                for group in ("0", "1", "2", "3+")
            },
        }
        for name, row in variants.items()
    }
    result = {
        "stage": "6.2 compact input-information ablation",
        "scope": "SigLIP 2, 10%, seed 0, development only",
        "full_input": full,
        "variants": variants,
        "variant_minus_full": deltas,
        "status": "complete; no Stage 6 test access",
    }
    output_json = Path("outputs/stage6/stage6_2_input_ablation_summary.json")
    output_txt = Path("outputs/stage6/stage6_2_input_ablation_summary.txt")
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Stage 6.2 Compact Input Ablations",
        "=================================",
        "variant | epoch | macro F1 | official F1 | delta macro | delta F1",
        "--- | ---: | ---: | ---: | ---: | ---:",
    ]
    rows = {"full_input": full, **variants}
    for name, row in rows.items():
        delta = (
            {"count_macro_mean_f1": 0.0, "official_F1_score": 0.0}
            if name == "full_input"
            else deltas[name]
        )
        lines.append(
            f"{name} | {row['selected_epoch']} | "
            f"{row['count_macro_mean_f1']:.6f} | "
            f"{row['official']['F1_score']:.6f} | "
            f"{delta['count_macro_mean_f1']:+.6f} | "
            f"{delta['official_F1_score']:+.6f}"
        )
    lines.append("Status: development-only result; Stage 6 test remains locked.")
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_txt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
