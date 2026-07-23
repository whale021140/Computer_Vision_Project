"""Select the locked Stage 5.5 pilot recipe from shadow-dev results."""

from __future__ import annotations

import json
from pathlib import Path


VARIANTS = [
    "selection_only",
    "balanced",
    "hierarchical",
    "one_to_one",
    "combined",
]


def main() -> None:
    rows = []
    for order, variant in enumerate(VARIANTS):
        root = Path("outputs/stage5_5/cells") / f"siglip2_10pct_seed0_{variant}"
        calibration = json.loads(
            (root / "calibration_shadow.json").read_text(encoding="utf-8")
        )
        selection = json.loads(
            (root / "selection_shadow.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "variant": variant,
                "declared_order": order,
                "selected_epoch": selection["selected"]["epoch"],
                "uncalibrated_count_macro_mean_f1": selection["selected"][
                    "count_macro_mean_f1"
                ],
                "best": calibration["best"],
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            float(row["best"]["count_macro_mean_f1"]),
            float(row["best"]["official"]["F1_score"]),
            -int(row["declared_order"]),
        ),
    )
    result = {
        "stage": "5.5",
        "selection_source": "locked shadow-dev only",
        "criterion": [
            "calibrated count_macro_mean_f1",
            "official.F1_score",
            "earlier declared variant",
        ],
        "selected_variant": selected["variant"],
        "selected": selected,
        "pilots": rows,
    }
    output_json = Path("outputs/stage5_5/pilot_selection.json")
    output_txt = Path("outputs/stage5_5/pilot_selection.txt")
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "Stage 5.5 Pilot Selection",
        "=========================",
        "variant | epoch | uncalibrated_macro | calibrated_macro | F1_score",
        "--- | ---: | ---: | ---: | ---:",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']} | {row['selected_epoch']} | "
            f"{row['uncalibrated_count_macro_mean_f1']:.6f} | "
            f"{row['best']['count_macro_mean_f1']:.6f} | "
            f"{row['best']['official']['F1_score']:.6f}"
        )
    lines.extend(["", f"Selected variant: {selected['variant']}"])
    summary = "\n".join(lines) + "\n"
    output_txt.write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
