"""Audit the revised Stage 5.6 calibration grid before revised test access."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


PILOT_VARIANTS = (
    "selection_only",
    "balanced",
    "hierarchical",
    "one_to_one",
    "combined",
)
REPRESENTATIONS = ("clip", "clip_dinov2", "siglip2")
PERCENTAGES = (1, 5, 10)
SEEDS = (0, 1, 2)


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    protocol = load(
        "outputs/stage5_6/calibration_revision_v2_protocol.json"
    )
    pilot = load("outputs/stage5_6/pilot_selection.json")
    selected_variant = str(pilot["selected_variant"])
    paths = {
        Path("outputs/stage5_6/cells")
        / f"siglip2_10pct_seed0_{variant}"
        / "calibration_dev.json"
        for variant in PILOT_VARIANTS
    }
    paths.update(
        Path("outputs/stage5_6/cells")
        / f"{representation}_{percentage}pct_seed{seed}_{selected_variant}"
        / "calibration_dev.json"
        for representation in REPRESENTATIONS
        for percentage in PERCENTAGES
        for seed in SEEDS
    )
    expected_grid = protocol["grid"]
    records = []
    failures = []
    for path in sorted(paths):
        if not path.exists():
            failures.append(f"missing: {path}")
            continue
        row = load(path)
        if row.get("calibration_revision") != "v2_wide":
            failures.append(f"wrong revision: {path}")
        for key in (
            "class0_biases",
            "class3_biases",
            "membership_thresholds",
            "num_settings",
        ):
            if row.get("grid", {}).get(key) != expected_grid[
                "num_settings_per_checkpoint" if key == "num_settings" else key
            ]:
                failures.append(f"grid mismatch ({key}): {path}")
        boundary = row.get("best_on_boundary", {})
        active_boundaries = sorted(
            key for key, value in boundary.items() if bool(value)
        )
        if active_boundaries:
            failures.append(
                f"boundary optimum {active_boundaries}: {path}"
            )
        best = row["best"]
        records.append(
            {
                "path": str(path),
                "class0_logit_bias": best["class0_logit_bias"],
                "class3_logit_bias": best["class3_logit_bias"],
                "membership_threshold": best["membership_threshold"],
                "count_macro_mean_f1": best["count_macro_mean_f1"],
                "boundary": active_boundaries,
            }
        )
    result = {
        "stage": "5.6",
        "revision": "v2_wide_calibration",
        "selected_variant": selected_variant,
        "num_unique_calibrations": len(paths),
        "num_validated": len(records),
        "test_gate_passed": not failures,
        "failures": failures,
        "best_setting_counts": {
            "class0_logit_bias": dict(
                sorted(Counter(row["class0_logit_bias"] for row in records).items())
            ),
            "class3_logit_bias": dict(
                sorted(Counter(row["class3_logit_bias"] for row in records).items())
            ),
            "membership_threshold": dict(
                sorted(
                    Counter(
                        row["membership_threshold"] for row in records
                    ).items()
                )
            ),
        },
        "records": records,
    }
    output_json = Path("outputs/stage5_6/calibration_v2_audit.json")
    output_txt = Path("outputs/stage5_6/calibration_v2_audit.txt")
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "Stage 5.6 Wide Calibration Audit",
        "================================",
        f"Selected variant: {selected_variant}",
        f"Validated calibrations: {len(records)}/{len(paths)}",
        f"Test gate passed: {not failures}",
        f"Class-0 settings: {result['best_setting_counts']['class0_logit_bias']}",
        f"Class-3 settings: {result['best_setting_counts']['class3_logit_bias']}",
        (
            "Threshold settings: "
            f"{result['best_setting_counts']['membership_threshold']}"
        ),
    ]
    if failures:
        lines.extend(["", "Failures:", *[f"- {value}" for value in failures]])
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_txt.read_text(encoding="utf-8"), end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
